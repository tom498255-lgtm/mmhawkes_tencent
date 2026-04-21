import argparse
import os
import time
from recbole_custom.quick_start import run_recbole

def _auto_cast(v):
    low = v.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _parse_unknown_args(unknown_args):
    extra = {}
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if token.startswith('--'):
            key = token[2:].replace('-', '_')
            if '=' in key:
                k, val = key.split('=', 1)
                extra[k] = _auto_cast(val)
                i += 1
                continue
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                extra[key] = _auto_cast(unknown_args[i + 1])
                i += 2
            else:
                extra[key] = True
                i += 1
        else:
            i += 1
    return extra
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认模型修改为 MMHP
    parser.add_argument('--model', '-m', type=str, default='MMHyperHawkes', help='name of models')
    # 默认数据集修改为 tiktok (MMHCL 常用数据集)
    parser.add_argument('--dataset', '-d', type=str, default='tiktok', help='name of dataset')
    # 允许传入自定义配置文件路径
    parser.add_argument('--config_files', type=str, default='', help='extra config files')
    parser.add_argument('--show-progress', '-sp', type=int, default=0)
    parser.add_argument('--seed', '-s', type=int, default=2023)

    args, unknown_args = parser.parse_known_args()

    config_file_list = ["./configs/general_full.yaml"]

    config_file_list.append(f"./configs/dataset/{args.dataset}.yaml")
    model_config_path = f"./configs/model/{args.model}/{args.dataset}.yaml"
    if os.path.exists(model_config_path):
        config_file_list.append(model_config_path)
    if args.config_files:
        extra_config_files = [i.strip() for i in args.config_files.split(',') if i.strip()]
        for config_path in extra_config_files:
            if os.path.exists(config_path):
                config_file_list.append(config_path)
            else:
                raise FileNotFoundError(f'Config file not found: {config_path}')

    print(config_file_list)

    if args.seed is None or args.seed == 0:
        args.seed = int(time.time() // 1000)

    config_dict = vars(args)
    config_dict.update(_parse_unknown_args(unknown_args))
                       
    run_recbole(model=args.model, dataset=args.dataset,
                config_file_list=config_file_list,
                config_dict=config_dict)
    print(args.dataset)
    print(args)