"""生成 BERT 前向的跨语言对拍参考（C: tests/compare_bert.c）。

- 用 Python AI.BertModel 构造小模型（vocab6, d=8, heads2, d_ff12, layers2, max_len8）；
- random.seed(11) 固定初始化（BertModel 构造器没有 seed 参数，但 weight 初始化走全局 random）；
- dump 按 _named_params() 顺序的全部权重、forward 的 hidden、MLM/NSP logits 与 param_count。
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

import AI  # noqa: E402


def dump_value(value):
    """把参数（可能是 list of list 或 list）转成 {shape,data}。"""
    if value and isinstance(value[0], list):
        rows = len(value)
        cols = len(value[0])
        data = []
        for row in value:
            data.extend(float(v) for v in row)
        return {"shape": [rows, cols], "data": data}
    return {"shape": [len(value)], "data": [float(v) for v in value]}


def main():
    random.seed(11)
    model = AI.BertModel(vocab_size=6, d_model=8, num_heads=2, d_ff=12,
                         num_layers=2, max_len=8, dropout=0.0)

    ids = [1, 3, 2, 4, 5]
    seg = [0, 0, 1, 1, 1]
    hidden = model.forward(ids, seg)
    mlm = model.masked_lm_logits()
    nsp = model.nsp_logits()

    weights = []
    for name, param, _grad in model._named_params():
        entry = {"name": name}
        entry.update(dump_value(param))
        weights.append(entry)

    payload = {
        "config": {
            "vocab_size": 6, "d_model": 8, "num_heads": 2, "d_ff": 12,
            "num_layers": 2, "max_len": 8, "param_count": model.param_count(),
        },
        "weights": weights,
        "cases": [
            {
                "name": "main",
                "ids": ids,
                "seg": seg,
                "hidden": [[float(v) for v in row] for row in hidden],
                "mlm": [[float(v) for v in row] for row in mlm],
                "nsp": [float(v) for v in nsp],
                "tolerance": 1e-12,
            },
        ],
    }
    out = os.path.join(HERE, "..", "bert_reference.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out, "param_count", payload["config"]["param_count"])
    print("hidden[0][:4]", hidden[0][:4])
    print("nsp", nsp)


if __name__ == "__main__":
    main()
