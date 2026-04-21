import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_custom.model.abstract_recommender import SequentialRecommender
from recbole_custom.model.loss import BPRLoss
from recbole_custom.model.init import xavier_uniform_initialization

from recbole_custom.utils.mmhcl_utils import build_sim, build_knn_normalized_graph, get_u2u_mat

class SoftKMeans(nn.Module):
    def __init__(self, n_clusters, hidden_size, temp):
        super().__init__()
        self.n_clusters = n_clusters
        self.centers = nn.Parameter(torch.randn(n_clusters, hidden_size))
        self.temp = temp

    def forward(self, x):
        dist = torch.cdist(x, self.centers)
        return F.softmax(-dist / self.temp, dim=-1)


class MMHyperHawkes(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MMHyperHawkes, self).__init__(config, dataset)
        self.dataset = dataset
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.device = config['device']

        self.embedding_size = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']
        self.top_k = config['top_k']

        self.n_clusters = config['n_clusters']  # 意图数量
        self.temp_cluster = config['temp_cluster']
        self.time_scalar = config['time_scalar']  # 时间缩放因子
        self.sub_time_delta = config['sub_time_delta']
        self.emb_dropout_prob = config['emb_dropout_prob']
        self.intent_kl_threshold = config['intent_kl_threshold'] if 'intent_kl_threshold' in config else 0.5
        self.full_sort_chunk_size = config['full_sort_chunk_size'] if 'full_sort_chunk_size' in config else 256
# ablation switches
        self.use_u2u = config['use_u2u'] if 'use_u2u' in config else True
        self.use_i2i = config['use_i2i'] if 'use_i2i' in config else True
        self.use_hgcn = config['use_hgcn'] if 'use_hgcn' in config else True
        self.use_soft_cluster = config['use_soft_cluster'] if 'use_soft_cluster' in config else True
        self.use_intent_excitation = config['use_intent_excitation'] if 'use_intent_excitation' in config else True
        self.use_short_term_attention = config['use_short_term_attention'] if 'use_short_term_attention' in config else True
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        image_input_dim = dataset.image_features.size(1)
        text_input_dim = dataset.text_features.size(1)
        self.image_mlp = nn.Linear(image_input_dim, self.embedding_size)
        self.text_mlp = nn.Linear(text_input_dim, self.embedding_size)

        self.soft_kmeans = SoftKMeans(self.n_clusters, self.embedding_size, self.temp_cluster)
        self.register_buffer('cluster_prob', torch.zeros(self.n_items, self.n_clusters))

        self.global_alpha = nn.Parameter(torch.tensor(0.1))
        self.intent_dist = nn.Sequential(
            nn.Linear(self.n_clusters + self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 5)  # 输出 alpha, beta, mu, sigma, pi
        )

        self.W_q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_k = nn.Linear(self.embedding_size, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.embedding_size)

        self._build_mmhcl_graphs(dataset)

        self.loss_fct = BPRLoss()
        self.apply(xavier_uniform_initialization)

    def _build_mmhcl_graphs(self, dataset):
        """
        构建 MMHCL 所需的 U2U 和 I2I 图
        """
        sim_img = build_sim(dataset.image_features)
        graph_img = build_knn_normalized_graph(sim_img, self.top_k)
        # 文本图
        sim_txt = build_sim(dataset.text_features)
        graph_txt = build_knn_normalized_graph(sim_txt, self.top_k)
        self.i2i_mat = ((graph_img + graph_txt) / 2).to(self.device)

        self.u2u_mat = get_u2u_mat(dataset.inter_feat, self.n_users, self.n_items).to(self.device)

    def mmhcl_encoder(self):

        img_emb = self.image_mlp(self.dataset.image_features.to(self.device))
        txt_emb = self.text_mlp(self.dataset.text_features.to(self.device))

        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + img_emb + txt_emb  # 简单的多模态融合

        if not self.use_hgcn:
            return user_emb, item_emb

        if self.use_u2u:
            u_embs = [user_emb]
            ego_u = user_emb
            for _ in range(self.n_ui_layers):
                ego_u = torch.sparse.mm(self.u2u_mat, ego_u)
                u_embs.append(ego_u)
            u_final = torch.stack(u_embs, dim=1).mean(dim=1)
        else:
            u_final = user_emb

        if self.use_i2i:
            i_embs = [item_emb]
            ego_i = item_emb
            for _ in range(self.n_ui_layers):
                ego_i = torch.sparse.mm(self.i2i_mat, ego_i)
                i_embs.append(ego_i)
            i_final = torch.stack(i_embs, dim=1).mean(dim=1)
        else:
            i_final = item_emb

        return u_final, i_final

    def e_step(self):
        with torch.no_grad():
            _, item_embs = self.mmhcl_encoder()
            if self.use_soft_cluster:
                self.cluster_prob = self.soft_kmeans(item_embs.detach())
            else:
                # disable soft clustering -> uniform intent assignment
                self.cluster_prob = torch.full(
                    (self.n_items, self.n_clusters),
                    1.0 / self.n_clusters,
                    device=self.device,
                    dtype=item_embs.dtype,
                )
                

    def _get_time_tensors(self, interaction):
        if 'timestamp_list' not in interaction or 'timestamp' not in interaction:
            raise KeyError('MMHyperHawkes requires `timestamp_list` and `timestamp` fields in interaction.')
        return interaction['timestamp_list'], interaction['timestamp']

    def get_hawkes_excitation(self, target_item, item_seq, time_seq, target_time, user_rep):
        if not self.use_intent_excitation:
            return torch.zeros(target_item.size(0), device=target_item.device)
        eps = 1e-10
        target_cluster_probs = self.cluster_prob[target_item].clamp_min(eps)
        target_cluster_probs = target_cluster_probs / target_cluster_probs.sum(dim=-1, keepdim=True)
        seq_cluster_probs = self.cluster_prob[item_seq].clamp_min(eps)
        seq_cluster_probs = seq_cluster_probs / seq_cluster_probs.sum(dim=-1, keepdim=True)

        seq_cluster_probs = self.cluster_prob[item_seq].clamp_min(eps)
        seq_cluster_probs = seq_cluster_probs / seq_cluster_probs.sum(dim=-1, keepdim=True)

        kl_div = F.kl_div(
            seq_cluster_probs.log(),
            target_cluster_probs.unsqueeze(1).log(),
            reduction='none',
            log_target=True,
        ).sum(dim=2)

        intent_mask = (kl_div < self.intent_kl_threshold) & (item_seq > 0) # [B, L]

        delta_t = target_time.unsqueeze(1) - time_seq  # [B, L]
        delta_mask = (delta_t > self.sub_time_delta) & intent_mask
        delta_t = (delta_t / self.time_scalar) * delta_mask.float()  # [B, L]

        mask = (delta_t > 0).float()  # [B, L]

        # ===== 4. Distribution parameter prediction =====
        delta_t = (delta_t / self.time_scalar) * delta_mask.float()
        mask = (delta_t > 0).float()
        target_item_emb = self.item_embedding(target_item)
        dist_input = torch.cat((target_cluster_probs, target_item_emb, user_rep), dim=1)
        dist_params = self.intent_dist(dist_input)
        mus = dist_params[:, 0].clamp(1e-10, 10).unsqueeze(1)
        sigmas = dist_params[:, 1].clamp(1e-10, 10).unsqueeze(1)
        alphas = (self.global_alpha + dist_params[:, 2]).unsqueeze(1)
        betas = (dist_params[:, 3] + 1).clamp(1e-10, 10).unsqueeze(1)
        pis = (dist_params[:, 4] + 0.5).clamp(1e-10, 1).unsqueeze(1) # [B, *]

        # ===== 5. Hawkes kernel mixture =====
        exp_dist = torch.distributions.Exponential(betas)
        norm_dist = torch.distributions.Normal(mus, sigmas)

        excitation = pis * exp_dist.log_prob(delta_t + 1e-9).exp() + (1 - pis) * norm_dist.log_prob(
            delta_t + 1e-9).exp()
        excitation = alphas * excitation * mask
        return excitation.sum(dim=1)

    # --- 来源：HyperHawkes (hyperhawkes.py short-term attention) ---
    def get_short_term_rep(self, item_seq_emb, mask):
        if not self.use_short_term_attention:
            valid_len = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (item_seq_emb * mask.unsqueeze(-1)).sum(dim=1) / valid_len

        q = self.W_q(item_seq_emb)
        k = self.W_k(item_seq_emb)
        v = item_seq_emb

        # Scaled Dot-Product
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.embedding_size ** 0.5)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        return self.LayerNorm(output.mean(dim=1))  # Mean pooling of attention output

    def forward(self, item_seq, item_seq_len, target_item, time_seq, target_time):
        # 1. 获取图增强表示 (MMHCL Encoder)
        _, i_g_embeddings = self.mmhcl_encoder()
        seq_emb = i_g_embeddings[item_seq]
        target_emb = i_g_embeddings[target_item]

        # 2. 短期兴趣 (HyperHawkes Short-term)
        mask = (item_seq > 0).float()
        short_term_rep = self.get_short_term_rep(seq_emb, mask)

        # 3. 基础强度 AURA (Base Intensity)
        # 用 user_embedding (经 U2U 增强) 代表用户长期静态偏好
        # 这里为了简化，假设 batch 内每个序列对应一个用户，实际应用需传入 user_id
        # 暂时用 short_term_rep 作为 User Query 的近似
        base_score = torch.mul(short_term_rep, target_emb).sum(dim=1)

        # 4. 意图激发 (Hawkes Process)
        # 假设我们有 user_rep (可以是 short_term_rep + static user emb)
        hawkes_score = self.get_hawkes_excitation(target_item, item_seq, time_seq, target_time, short_term_rep)

        # 5. 融合分数
        final_score = base_score + hawkes_score
        return final_score

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # 获取时间信息 (RecBole 字段)
        time_seq = interaction['timestamp_list']  # 假设 dataset 处理了时间序列
        target_time = interaction['timestamp']

        pos_score = self.forward(item_seq, item_seq_len, pos_items, time_seq, target_time)
        neg_score = self.forward(item_seq, item_seq_len, neg_items, time_seq, target_time)

        return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_item = interaction[self.ITEM_ID]
        time_seq, target_time = self._get_time_tensors(interaction)
        return self.forward(item_seq, item_seq_len, target_item, time_seq, target_time)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        time_seq, target_time = self._get_time_tensors(interaction)
        bsz, seq_len = item_seq.size()

        _, i_g_embeddings = self.mmhcl_encoder()
        seq_emb = i_g_embeddings[item_seq]
        mask = (item_seq > 0).float()
        short_term_rep = self.get_short_term_rep(seq_emb, mask)

        base_scores = torch.matmul(short_term_rep, i_g_embeddings.transpose(0, 1))
        scores = base_scores.clone()

        chunk = self.full_sort_chunk_size
        for start in range(0, self.n_items, chunk):
            end = min(start + chunk, self.n_items)
            item_range = torch.arange(start, end, device=self.device)
            target_item = item_range.unsqueeze(0).expand(bsz, -1).reshape(-1)
            rep_expand = short_term_rep.unsqueeze(1).expand(bsz, end - start, -1).reshape(-1, self.embedding_size)
            seq_expand = item_seq.unsqueeze(1).expand(bsz, end - start, seq_len).reshape(-1, seq_len)
            time_expand = time_seq.unsqueeze(1).expand(bsz, end - start, seq_len).reshape(-1, seq_len)
            target_time_expand = target_time.unsqueeze(1).expand(bsz, end - start).reshape(-1)

            hawkes_scores = self.get_hawkes_excitation(
                target_item=target_item,
                item_seq=seq_expand,
                time_seq=time_expand,
                target_time=target_time_expand,
                user_rep=rep_expand,
            ).view(bsz, end - start)
            scores[:, start:end] += hawkes_scores

        scores[:, 0] = -1e12
        return scores