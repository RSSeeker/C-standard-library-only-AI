# -*- coding: utf-8 -*-
"""定位 Python 主仓库（AI.py）并加入 sys.path。

c/tools/ 下的参考导出脚本统一 `import _py_path` 即可，避免每份脚本各自
硬编码仓库路径。可用环境变量 AI_PY_DIR 覆盖（CI / 其他机器）。

仓库布局约定（两个仓库同级）：
    <parent>/Python-standard-library-only-AI-main/AI.py
    <parent>/AI_C/c/tools/_py_path.py
"""
import os
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))          # AI_C/c/tools
_C_DIR = os.path.dirname(_TOOLS)                              # AI_C/c
_REPO = os.path.dirname(_C_DIR)                               # AI_C
PY_REPO = os.environ.get("AI_PY_DIR") or os.path.join(
    os.path.dirname(_REPO), "Python-standard-library-only-AI-main")

if PY_REPO not in sys.path:
    sys.path.insert(0, PY_REPO)
