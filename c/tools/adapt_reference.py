# -*- coding: utf-8 -*-
"""导出 LoRA 的黄金值（C 侧 compare_adapt 用）。

用法：python adapt_reference.py adapt_reference.json
"""
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.join(
    _HERE, "..", "..", "..", "Python-standard-library-only-AI-main")
sys.path.insert(0, os.path.abspath(_PY_DIR))

import AI  # noqa: E402


def flat(mat):
    out = []
    for row in mat:
        out.extend(float(v) for v in row)
    return out


def case(name, fi, fo, rank, alpha, seed):
    rnd = random.Random(seed)
    W = [[rnd.gauss(0.0, 0.3) for _ in range(fi)] for _ in range(fo)]
    b = [rnd.gauss(0.0, 0.1) for _ in range(fo)]
    A = [[rnd.gauss(0.0, 0.5) for _ in range(fi)] for _ in range(rank)]
    B = [[rnd.gauss(0.0, 0.5) for _ in range(rank)] for _ in range(fo)]
    x = [rnd.gauss(0.0, 1.0) for _ in range(fi)]
    dout = [rnd.gauss(0.0, 1.0) for _ in range(fo)]

    layer = AI.Layer(fi, fo, activation="linear")
    layer.weights = [row[:] for row in W]
    layer.biases = b[:]
    lora = AI.LoRA(layer, rank=rank, alpha=alpha)
    lora.A = [row[:] for row in A]
    lora.B = [row[:] for row in B]

    y = lora.forward(x)
    dx = lora.backward(dout)
    gA = flat(lora.grad_A)
    gB = flat(lora.grad_B)

    # merge 之后：前向只用 W，反向只走原层
    merged = AI.LoRA(layer, rank=rank, alpha=alpha)
    merged.A = [row[:] for row in A]
    merged.B = [row[:] for row in B]
    merged.merge()
    y_merged = merged.forward(x)
    dx_merged = merged.backward(dout)

    return {
        "name": name,
        "fan_in": fi, "fan_out": fo, "rank": rank, "alpha": alpha,
        "scaling": lora.scaling,
        "w": flat(W), "b": b,
        "a": flat(A), "bmat": flat(B),
        "x": x, "dout": dout,
        "y": y, "dx": dx, "grad_a": gA, "grad_b": gB,
        "merged_w": flat(layer.weights),
        "y_merged": y_merged,
        "dx_merged": dx_merged,
        "trainable": lora.trainable_param_count(),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "adapt_reference.json"
    cases = [
        case("basic", 4, 3, 2, 2.0, 11),
        case("alpha_nondefault", 5, 4, 3, 6.0, 23),
        case("rank_one", 6, 2, 1, 1.0, 37),
        case("rank_eq_in", 3, 5, 3, 3.0, 41),
    ]
    payload = {
        "meta": {
            "source": "AI.LoRA (Python)",
            "note": "y = W x + b + scaling * B (A x)；dx 含 A^T dh；grad 只在 A/B 上",
        },
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote %s (%d cases)" % (out_path, len(cases)))


if __name__ == "__main__":
    main()
