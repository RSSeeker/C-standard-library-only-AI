#!/usr/bin/env python3
"""生成投机解码（Medusa / EAGLE）的 C 侧对拍参考.

用法：cd c && python tools/spec_decode_reference.py spec_decode_reference.json
输出包含：目标模型参数块（按 llm_param_blocks 名字对齐）、草稿头/草稿器权重、
以及若干解码用例的 ids 与 stats。C 侧逐项比较。
"""
import json
import sys

import _py_path  # noqa: F401  (把 Python 主仓库加入 sys.path)
import AI  # noqa: E402


def flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(flat(v))
        return out
    return [float(x)]


def shape_of(x):
    dims = []
    while isinstance(x, (list, tuple)):
        dims.append(len(x))
        x = x[0]
    return dims


def blocks(model):
    out = []
    for name, param, _grad in model._named_params():
        out.append({"name": name, "shape": shape_of(param), "data": flat(param)})
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "spec_decode_reference.json"
    import random
    state = random.getstate()
    random.seed(3)
    target = AI.CausalLM(vocab_size=6, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=10, dropout=0.0)
    random.setstate(state)
    heads = AI.MedusaHeads(8, 6, 2, seed=5)
    drafter = AI.EagleDrafter(8, 6, hidden=4, seed=7)

    head_weights = []
    for h in heads.heads:
        head_weights.append({"w": [list(row) for row in h.weights],
                             "b": list(h.biases)})

    cases = []

    def add_medusa(name, prompt, max_new, eos_id):
        r = AI.medusa_decode(target, list(prompt), heads,
                             max_new_tokens=max_new, eos_id=eos_id)
        cases.append({"name": name, "func": "medusa", "prompt": list(prompt),
                      "max_new": max_new, "eos_id": eos_id,
                      "ids": r["ids"], "tokens": r["tokens"],
                      "stats": r["stats"]})

    def add_eagle(name, prompt, max_new, gamma, eos_id):
        r = AI.eagle_decode(target, list(prompt), drafter,
                            max_new_tokens=max_new, gamma=gamma, eos_id=eos_id)
        cases.append({"name": name, "func": "eagle", "prompt": list(prompt),
                      "max_new": max_new, "gamma": gamma, "eos_id": eos_id,
                      "ids": r["ids"], "tokens": r["tokens"],
                      "stats": r["stats"]})

    add_medusa("medusa_plain", [1, 2], 4, None)
    add_medusa("medusa_eos0", [2, 4, 1], 3, 0)
    add_medusa("medusa_quick", [1, 2, 3, 4, 5, 0, 1, 2], 2, None)
    add_eagle("eagle_gamma3", [1, 2], 4, 3, None)
    add_eagle("eagle_gamma1", [2, 3], 3, 1, 0)
    add_eagle("eagle_quick", [1, 2, 3, 4, 5, 0, 1], 2, 3, None)

    payload = {
        "target": {
            "vocab_size": 6, "d_model": 8, "num_heads": 2, "d_ff": 16,
            "num_layers": 1, "max_len": 10, "dropout": 0.0, "seed": 3,
            "blocks": blocks(target),
        },
        "heads": {"d_model": 8, "vocab_size": 6, "num_heads": 2,
                  "weights": head_weights},
        "drafter": {"d_model": 8, "vocab_size": 6, "hidden": 4,
                    "Wf": [list(r) for r in drafter.Wf],
                    "We": [list(r) for r in drafter.We],
                    "bh": list(drafter.bh),
                    "Wh": [list(r) for r in drafter.Wh],
                    "bo": list(drafter.bo)},
        "cases": cases,
        "tolerance": 1e-15,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", path, "cases:", len(cases))


if __name__ == "__main__":
    main()
