"""生成 ViT 前向的跨语言对拍参考（C: tests/compare_vit.c）。

小模型：in_channels=1, image_size=8, patch_size=4, d_model=8, heads=2, d_ff=12,
layers=1, num_classes=3；输入是固定图案 8×8。
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

import AI  # noqa: E402


def flt(x):
    return float(x)


def dump_matrix(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    data = []
    for row in mat:
        data.extend(flt(v) for v in row)
    return {"shape": [rows, cols], "data": data}


def main():
    random.seed(5)
    vit = AI.ViT(in_channels=1, image_size=8, patch_size=4, d_model=8,
                 num_heads=2, d_ff=12, num_layers=1, num_classes=3)

    # 固定图案：checkerboard + 梯度
    image = [[[0.0] * 8 for _ in range(8)]]
    for r in range(8):
        for c in range(8):
            image[0][r][c] = 0.1 + 0.05 * r - 0.02 * c + (0.3 if (r + c) % 2 == 0 else 0.0)

    logits = vit.forward(image)

    weights = []
    conv = vit.patch_embed.conv
    Wc = conv.weights  # [out][in][kh][kw]
    out_c = len(Wc)
    in_c = len(Wc[0])
    kh = len(Wc[0][0])
    kw = len(Wc[0][0][0])
    data = []
    for oc in range(out_c):
        for ic in range(in_c):
            for i in range(kh):
                for j in range(kw):
                    data.append(flt(Wc[oc][ic][i][j]))
    weights.append({"name": "patch.weight", "shape": [out_c, in_c, kh, kw], "data": data})
    weights.append({"name": "patch.bias", "shape": [conv.biases.__len__()],
                    "data": [flt(v) for v in conv.biases]})

    weights.append({"name": "cls", "shape": [len(vit.cls)],
                    "data": [flt(v) for v in vit.cls]})
    weights.append({"name": "pos", "shape": [len(vit.pos), len(vit.pos[0])],
                    "data": [flt(v) for row in vit.pos for v in row]})

    for li, layer in enumerate(vit.encoder.layers):
        attn = layer.self_attn
        for key in ("W_q", "W_k", "W_v", "W_o"):
            mat = getattr(attn, key)
            weights.append({"name": f"enc.L{li}.attn.{key}", **dump_matrix(mat)})
        for key in ("b_q", "b_k", "b_v", "b_o"):
            vec = getattr(attn, key)
            weights.append({"name": f"enc.L{li}.attn.{key}", "shape": [len(vec)],
                            "data": [flt(v) for v in vec]})
        ffn = layer.ffn
        for key in ("W1", "W2"):
            weights.append({"name": f"enc.L{li}.ffn.{key}", **dump_matrix(getattr(ffn, key))})
        for key in ("b1", "b2"):
            vec = getattr(ffn, key)
            weights.append({"name": f"enc.L{li}.ffn.{key}", "shape": [len(vec)],
                            "data": [flt(v) for v in vec]})
        for ln_name in ("ln1", "ln2"):
            ln = getattr(layer, ln_name)
            weights.append({"name": f"enc.L{li}.{ln_name}.gamma", "shape": [len(ln.gamma)],
                            "data": [flt(v) for v in ln.gamma]})
            weights.append({"name": f"enc.L{li}.{ln_name}.beta", "shape": [len(ln.beta)],
                            "data": [flt(v) for v in ln.beta]})

    weights.append({"name": "norm.gamma", "shape": [len(vit.norm.gamma)],
                    "data": [flt(v) for v in vit.norm.gamma]})
    weights.append({"name": "norm.beta", "shape": [len(vit.norm.beta)],
                    "data": [flt(v) for v in vit.norm.beta]})
    weights.append({"name": "head.W", **dump_matrix(vit.head.weights)})
    weights.append({"name": "head.b", "shape": [len(vit.head.biases)],
                    "data": [flt(v) for v in vit.head.biases]})

    payload = {
        "config": {
            "in_channels": 1, "image_h": 8, "image_w": 8, "patch_size": 4,
            "d_model": 8, "num_heads": 2, "d_ff": 12, "num_layers": 1,
            "num_classes": 3, "param_count": vit.param_count(),
        },
        "weights": weights,
        "case": {
            "image": [flt(v) for ch in image for row in ch for v in row],
            "logits": [flt(v) for v in logits],
            "tolerance": 1e-12,
        },
    }
    out = os.path.join(HERE, "..", "vit_reference.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out, "param_count", payload["config"]["param_count"])
    print("logits", logits)


if __name__ == "__main__":
    main()
