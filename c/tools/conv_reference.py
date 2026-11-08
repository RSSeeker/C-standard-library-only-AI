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
from AI import Conv2d, Conv1d, ConvTranspose2d, MaxPool2d, Flatten          # noqa: E402


def make_image(C, H, W, rnd):
    return [[[rnd.uniform(-1, 1) for _ in range(W)] for _ in range(H)] for _ in range(C)]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.abspath(os.path.join(_here, "..")), "conv_reference.json")

    rnd = random.Random(20240707)
    cases = []

    # ---------- Conv2d：前两条是默认路径，后四条覆盖 groups / dilation ----------
    # 元组含义：(C, OC, K, S, P, H, W, groups, dilation)
    conv_cfgs = (
        (1, 2, 2, 1, 0, 5, 5, 1, 1),
        (2, 3, 3, 2, 1, 6, 6, 1, 1),
        (4, 6, 3, 1, 1, 7, 7, 2, 1),     # 分组卷积
        (4, 4, 3, 1, 1, 6, 6, 4, 1),     # 深度卷积（groups == C）
        (2, 3, 3, 1, 2, 7, 7, 1, 2),     # 膨胀卷积
        (4, 6, 3, 2, 2, 9, 9, 2, 2),     # 分组 + 膨胀 + stride
    )
    for (C, OC, K, S, P, H, W, G, D) in conv_cfgs:
        conv = Conv2d(C, OC, K, S, P, groups=G, dilation=D)
        cg = C // G
        # 灌入确定权重（按 Python 的 weights[oc][ic][kh][kw] 展平顺序，ic 只到 cg）
        conv.weights = [[[[rnd.uniform(-0.5, 0.5) for _ in range(K)]
                          for _ in range(K)] for _ in range(cg)] for _ in range(OC)]
        conv.biases = [rnd.uniform(-0.2, 0.2) for _ in range(OC)]

        x = make_image(C, H, W, rnd)
        out = conv.forward([[[v for v in row] for row in ch] for ch in x])
        ho, wo = len(out[0]), len(out[0][0])
        dout = [[[rnd.uniform(-1, 1) for _ in range(wo)] for _ in range(ho)] for _ in range(OC)]
        dx = conv.backward([[ [v for v in row] for row in ch] for ch in dout])

        flat_w = [conv.weights[oc][ic][kh][kw]
                  for oc in range(OC) for ic in range(cg)
                  for kh in range(K) for kw in range(K)]
        cases.append({
            "C": C, "OC": OC, "K": K, "S": S, "P": P, "H": H, "W": W,
            "groups": G, "dilation": D,
            "ho": ho, "wo": wo,
            # 注意：键名不能叫 "W"——那会和输入宽度 W 撞名（Python dict 后者覆盖前者）
            "weights": flat_w, "b": list(conv.biases),
            "x": [v for ch in x for row in ch for v in row],
            "out": [v for ch in out for row in ch for v in row],
            "dout": [v for ch in dout for row in ch for v in row],
            "dx": [v for ch in dx for row in ch for v in row],
            "grad_w": [conv.grad_w[oc][ic][kh][kw]
                       for oc in range(OC) for ic in range(cg)
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

    # 立刻存成独立变量：后面的 Conv1d/ConvTranspose2d 循环会覆盖 x/out/dout/dx
    flat_x = [[[v for v in row] for row in ch] for ch in x]
    flat_out = list(out)
    flat_dout = list(dout)
    flat_dx = [[[v for v in row] for row in ch] for ch in dx]
    flat_c, flat_h, flat_w = C, H, W


    # ---------- Conv1d ----------
    # 元组含义：(C, OC, K, S, P, L, groups)
    one_d = []
    conv1d_cfgs = (
        (1, 2, 3, 1, 0, 7, 1),      # 默认路径
        (2, 3, 2, 2, 1, 8, 1),      # stride + padding
        (4, 6, 3, 1, 1, 9, 2),      # 分组
        (3, 3, 3, 2, 1, 10, 3),     # 深度卷积（groups == C == OC）
        (2, 4, 4, 3, 2, 11, 2),     # 分组 + 大 stride + 大 padding
    )
    for (C, OC, K, S, P, L, G) in conv1d_cfgs:
        conv = Conv1d(C, OC, K, S, P, groups=G)
        cg = C // G
        conv.weights = [[[rnd.uniform(-0.5, 0.5) for _ in range(K)]
                         for _ in range(cg)] for _ in range(OC)]
        conv.biases = [rnd.uniform(-0.2, 0.2) for _ in range(OC)]

        x = [[rnd.uniform(-1, 1) for _ in range(L)] for _ in range(C)]
        out = conv.forward([[v for v in row] for row in x])
        lo = len(out[0])
        dout = [[rnd.uniform(-1, 1) for _ in range(lo)] for _ in range(OC)]
        dx = conv.backward([[v for v in row] for row in dout])

        one_d.append({
            "C": C, "OC": OC, "K": K, "S": S, "P": P, "L": L, "groups": G, "lo": lo,
            "weights": [conv.weights[oc][ic][kk]
                        for oc in range(OC) for ic in range(cg) for kk in range(K)],
            "b": list(conv.biases),
            "x": [v for row in x for v in row],
            "out": [v for row in out for v in row],
            "dout": [v for row in dout for v in row],
            "dx": [v for row in dx for v in row],
            "grad_w": [conv.grad_w[oc][ic][kk]
                       for oc in range(OC) for ic in range(cg) for kk in range(K)],
            "grad_b": list(conv.grad_b),
        })

    # ---------- ConvTranspose2d ----------
    # 元组含义：(C, OC, K, S, P, OP, H, W, groups)
    t_conv = []
    convt_cfgs = (
        (1, 2, 2, 1, 0, 0, 4, 4, 1),    # 默认路径（上采样 1 倍）
        (2, 3, 3, 2, 1, 1, 5, 5, 1),    # stride 2 + output_padding 1
        (4, 6, 3, 2, 1, 0, 6, 6, 2),    # 分组
        (3, 3, 3, 1, 1, 0, 5, 5, 3),    # 深度（groups == C == OC）
        (2, 4, 4, 2, 2, 1, 5, 5, 2),    # 分组 + stride + output_padding
    )
    for (C, OC, K, S, P, OP, H, W, G) in convt_cfgs:
        conv = ConvTranspose2d(C, OC, K, S, P, OP, groups=G)
        og = OC // G
        # 权重布局是 [cin][og][kh][kw]（PyTorch 的 (in, out/groups, kh, kw)）
        conv.weights = [[[[rnd.uniform(-0.5, 0.5) for _ in range(K)]
                          for _ in range(K)] for _ in range(og)] for _ in range(C)]
        conv.biases = [rnd.uniform(-0.2, 0.2) for _ in range(OC)]

        x = make_image(C, H, W, rnd)
        out = conv.forward([[[v for v in row] for row in ch] for ch in x])
        ho, wo = len(out[0]), len(out[0][0])
        dout = [[[rnd.uniform(-1, 1) for _ in range(wo)] for _ in range(ho)] for _ in range(OC)]
        dx = conv.backward([[[v for v in row] for row in ch] for ch in dout])

        t_conv.append({
            "C": C, "OC": OC, "K": K, "S": S, "P": P, "OP": OP,
            "H": H, "W": W, "groups": G, "ho": ho, "wo": wo,
            "weights": [conv.weights[ci][co][kh][kw]
                        for ci in range(C) for co in range(og)
                        for kh in range(K) for kw in range(K)],
            "b": list(conv.biases),
            "x": [v for ch in x for row in ch for v in row],
            "out": [v for ch in out for row in ch for v in row],
            "dout": [v for ch in dout for row in ch for v in row],
            "dx": [v for ch in dx for row in ch for v in row],
            "grad_w": [conv.grad_w[ci][co][kh][kw]
                       for ci in range(C) for co in range(og)
                       for kh in range(K) for kw in range(K)],
            "grad_b": list(conv.grad_b),
        })

    # Flatten 段的形状就是下面这三个常量；上面的 Conv1d/ConvTranspose2d 循环
    # 复用过同名变量，这里显式复位，避免 flatten 元数据与数组长度不一致
    C, H, W = 2, 3, 4

    data = {
        "conv": cases,
        "conv1d": one_d,
        "convt2d": t_conv,
        "pool": pools,
        "flatten": {
            "C": flat_c, "H": flat_h, "W": flat_w,
            "x": [v for ch in flat_x for row in ch for v in row],
            "out": flat_out,
            "dout": flat_dout,
            "dx": [v for ch in flat_dx for row in ch for v in row],
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
    for i, c in enumerate(one_d):
        print(f"  Conv1d[{i}] C={c['C']} OC={c['OC']} K={c['K']} S={c['S']} P={c['P']} "
              f"L={c['L']} groups={c['groups']} -> lo={c['lo']}  前向首值 {c['out'][0]:.12g}")
    for i, c in enumerate(t_conv):
        print(f"  ConvTranspose2d[{i}] C={c['C']} OC={c['OC']} K={c['K']} S={c['S']} "
              f"P={c['P']} OP={c['OP']} groups={c['groups']} "
              f"{c['H']}x{c['W']} -> {c['ho']}x{c['wo']}")
    print(f"  Flatten ({C},{H},{W}) -> ({C*H*W},)")


if __name__ == "__main__":
    main()
