"""生成 LLaMA C 移植的对拍参考（权重 + 前向 logits）。

用法： PYTHONPATH=<python repo> python tools/llama_reference.py llama_reference.json
"""
import json
import os
import sys

REPO = os.environ.get("AI_PY_REPO", r"D:\vscode\Python-standard-library-only-AI-main")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import AI  # noqa: E402


def flatten(obj):
    out = []

    def rec(v):
        if isinstance(v, (list, tuple)):
            for item in v:
                rec(item)
        else:
            out.append(float(v))

    rec(obj)
    return out


def shape_of(obj):
    shape = []
    cur = obj
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "llama_reference.json"
    model = AI.LlamaModel(
        vocab_size=6, d_model=8, num_heads=2, d_ff=12, num_layers=2,
        num_kv_heads=1, max_len=8, dropout=0.0, rope=True)
    ids = [1, 3, 2, 4]
    logits = model.forward(ids)
    ids2 = [0, 5]
    logits2 = model.forward(ids2)

    weights = []
    for name, param, _grad in model._named_params():
        weights.append({"name": name, "shape": shape_of(param), "data": flatten(param)})

    payload = {
        "config": {
            "vocab_size": 6,
            "d_model": 8,
            "num_heads": 2,
            "num_kv_heads": 1,
            "d_ff": 12,
            "num_layers": 2,
            "max_len": 8,
            "rope": True,
            "param_count": model.param_count(),
        },
        "weights": weights,
        "cases": [
            {"name": "short", "ids": ids, "logits": logits},
            {"name": "tiny", "ids": ids2, "logits": logits2},
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote", path, "params", model.param_count(), "T", len(ids), len(ids2))


if __name__ == "__main__":
    main()
