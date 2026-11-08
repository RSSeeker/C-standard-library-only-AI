# -*- coding: utf-8 -*-
"""导出 FeedForward 的参考数据（8 种中间层激活）。

对照 AI.py 的 FeedForward：本脚本构造同样参数、覆写权重，跑一次 forward +
backward，把 out / dx / 四个梯度写成 JSON；compare_ffn.c 逐位比对。
"""
import io
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.environ.get("AI_PY_DIR") or os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "Python-standard-library-only-AI-main"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from AI import FeedForward  # noqa: E402

ACTS = ["relu", "gelu", "sigmoid", "linear", "silu", "mish", "elu", "selu"]


def flat(a):
    out = []
    for row in a:
        out.extend(row)
    return out


def build(d_model, d_ff, activation, T, seed):
    rnd = random.Random(seed)
    ff = FeedForward(d_model, d_ff, activation)
    ff.W1 = [[rnd.uniform(-0.6, 0.6) for _ in range(d_model)] for _ in range(d_ff)]
    ff.b1 = [rnd.uniform(-0.4, 0.4) for _ in range(d_ff)]
    ff.W2 = [[rnd.uniform(-0.6, 0.6) for _ in range(d_ff)] for _ in range(d_model)]
    ff.b2 = [rnd.uniform(-0.4, 0.4) for _ in range(d_model)]
    x = [[rnd.uniform(-1.5, 1.5) for _ in range(d_model)] for _ in range(T)]
    dout = [[rnd.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(T)]
    return ff, x, dout


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "ffn_reference.json"
    cases = []
    index = 0
    for activation in ACTS:
        for (d_model, d_ff, T) in ((4, 6, 3), (3, 5, 1)):
            index += 1
            ff, x, dout = build(d_model, d_ff, activation, T, 20000 + index)
            out = ff.forward(x)
            dx = ff.backward(dout)
            # libm 差异：只有 relu / linear 是纯四则运算，其余经 exp/tanh/log1p
            tol = 0.0 if activation in ("relu", "linear") else 1e-12
            note = ("纯四则运算" if tol == 0.0 else
                    "exp/tanh/log1p 在 msvcrt 与 UCRT 上可差 1 ULP")
            cases.append({
                "name": "%s_T%d" % (activation, T),
                "activation": activation,
                "d_model": d_model,
                "d_ff": d_ff,
                "T": T,
                "W1": flat(ff.W1),
                "b1": list(ff.b1),
                "W2": flat(ff.W2),
                "b2": list(ff.b2),
                "x": flat(x),
                "dout": flat(dout),
                "out": flat(out),
                "dx": flat(dx),
                "grad_W1": flat(ff.grads["W1"]),
                "grad_b1": list(ff.grads["b1"]),
                "grad_W2": flat(ff.grads["W2"]),
                "grad_b2": list(ff.grads["b2"]),
                "tol": tol,
                "note": note,
            })
            print("%-10s d_model=%d d_ff=%d T=%d  out[0]=%.6f" %
                  (activation, d_model, d_ff, T, out[0][0]))
    data = {
        "meta": {
            "source": "AI.py FeedForward（8 种 activation）",
            "cases": len(cases),
            "python": sys.version.split()[0],
        },
        "cases": cases,
    }
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote %s (%d cases)" % (out_path, len(cases)))


if __name__ == "__main__":
    main()
