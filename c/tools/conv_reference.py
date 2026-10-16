# -*- coding: utf-8 -*-
"""导出 Conv2d / MaxPool2d / Flatten 的参考数据，供 C 实现逐位比对

用法：python tools/conv_reference.py [输出路径]
默认写到 c/conv_reference.json
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)
from AI import Conv2d, MaxPool2d, Flatten          # noqa: E402


def make_image(C, H, W, rnd):
    return [[[rnd.uniform(-1, 1) for _ in range(W)] for _ in range(H)] for _ in range(C)]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.abspath(os.path.join(_here, "..")), "conv_reference.json")

    rnd = random.Random(20240707)
    cases = []

    # ---------- Conv2d：两种配置 ----------
    for (C, OC, K, S, P, H, W) in ((1, 2, 2, 1, 0, 5, 5), (2, 3, 3, 2, 1, 6, 6)):
        conv = Conv2d(C, OC, K, S, P)
        # 灌入确定权重（按 Python 的 weights[oc][ic][kh][kw] 展平顺序）
        conv.weights = [[[[rnd.uniform(-0.5, 0.5) for _ in range(K)]
                          for _ in range(K)] for _ in range(C)] for _ in range(OC)]
        conv.biases = [rnd.uniform(-0.2, 0.2) for _ in range(OC)]

        x = make_image(C, H, W, rnd)
        out = conv.forward([[[v for v in row] for row in ch] for ch in x])
        ho, wo = len(out[0]), len(out[0][0])
        dout = [[[rnd.uniform(-1, 1) for _ in range(wo)] for _ in range(ho)] for _ in range(OC)]
        dx = conv.backward([[ [v for v in row] for row in ch] for ch in dout])

        flat_w = [conv.weights[oc][ic][kh][kw]
                  for oc in range(OC) for ic in range(C)
                  for kh in range(K) for kw in range(K)]
        cases.append({
            "C": C, "OC": OC, "K": K, "S": S, "P": P, "H": H, "W": W,
            "ho": ho, "wo": wo,
            # 注意：键名不能叫 "W"——那会和输入宽度 W 撞名（Python dict 后者覆盖前者）
            "weights": flat_w, "b": list(conv.biases),
            "x": [v for ch in x for row in ch for v in row],
            "out": [v for ch in out for row in ch for v in row],
            "dout": [v for ch in dout for row in ch for v in row],
            "dx": [v for ch in dx for row in ch for v in row],
            "grad_w": [conv.grad_w[oc][ic][kh][kw]
                       for oc in range(OC) for ic in range(C)
                       for kh in range(K) for kw in range(K)],
            "grad_b": list(conv.grad_b),
        })

    # ---------- MaxPool2d ----------
    pools = []
    for (C, K, S, H, W) in ((2, 2, 2, 6, 5), (1, 3, 1, 5, 5)):
        pool = MaxPool2d(K, S)
        x = make_image(C, H, W, rnd)
        out = pool.forward([[[v for v in row] for row in ch] for ch in x])
        ho, wo = len(out[0]), len(out[0][0])
        dout = [[[rnd.uniform(-1, 1) for _ in range(wo)] for _ in range(ho)] for _ in range(C)]
        dx = pool.backward([[[v for v in row] for row in ch] for ch in dout])
        pools.append({
            "C": C, "K": K, "S": S, "H": H, "W": W, "ho": ho, "wo": wo,
            "x": [v for ch in x for row in ch for v in row],
            "out": [v for ch in out for row in ch for v in row],
            "dout": [v for ch in dout for row in ch for v in row],
            "dx": [v for ch in dx for row in ch for v in row],
        })

    # ---------- Flatten ----------
    C, H, W = 2, 3, 4
    flat = Flatten()
    x = make_image(C, H, W, rnd)
    out = flat.forward([[[v for v in row] for row in ch] for ch in x])
    dout = [rnd.uniform(-1, 1) for _ in range(C * H * W)]
    dx = flat.backward(list(dout))

    data = {
        "conv": cases,
        "pool": pools,
        "flatten": {
            "C": C, "H": H, "W": W,
            "x": [v for ch in x for row in ch for v in row],
            "out": list(out),
            "dout": dout,
            "dx": [v for ch in dx for row in ch for v in row],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"参考数据已写入 {out_path}")
    for i, c in enumerate(cases):
        print(f"  Conv2d[{i}] C={c['C']} OC={c['OC']} K={c['K']} S={c['S']} P={c['P']} "
              f"{c['H']}x{c['W']} -> {c['ho']}x{c['wo']}  前向首值 {c['out'][0]:.12g}")
    for i, p in enumerate(pools):
        print(f"  MaxPool2d[{i}] K={p['K']} S={p['S']} {p['H']}x{p['W']} -> {p['ho']}x{p['wo']}")
    print(f"  Flatten ({C},{H},{W}) -> ({C*H*W},)")


if __name__ == "__main__":
    main()
