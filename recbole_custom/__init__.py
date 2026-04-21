from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = "1.1.1"
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in __path__:
    __path__.append(str(_repo_root))