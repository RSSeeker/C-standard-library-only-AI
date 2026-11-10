"""为 C 侧 Mamba 前向对拍生成参考数据。

对应 C 模块：c/mamba.h / c/mamba.c（只前向）。
"""
import json
import random
import sys

sys.path.insert(0, r"D:\vscode\Python-standard-library-only-AI-main")
import AI  # noqa: E402


VOCAB = 6
D_MODEL = 8
D_STATE = 4
D_CONV = 4
EXPAND = 2
LAYERS = 1
MAX_LEN = 8
SEED = 3


def flat2(m):
    return [[float(v) for v in row] for row in m]


def flat1(v):
    return [float(x) for x in v]


def dump_layer(lay):
    return {"weights": flat2(lay.weights), "biases": flat1(lay.biases)}


def main():
    random.seed(SEED)
    model = AI.MambaModel(VOCAB, D_MODEL, D_STATE, D_CONV, EXPAND, LAYERS, MAX_LEN,
                          seed=SEED)

    blocks = []
    for b in model.blocks:
        cw = [[[float(v) for v in kk] for kk in ic] for ic in b.conv.weights]
        blocks.append({
            "in_proj": dump_layer(b.in_proj),
            "conv_w": cw,
            "conv_b": flat1(b.conv.biases),
            "delta_proj": dump_layer(b.delta_proj),
            "b_proj": dump_layer(b.b_proj),
            "c_proj": dump_layer(b.c_proj),
            "A": flat2(b.A),
            "D": flat1(b.D),
            "out_proj": dump_layer(b.out_proj),
        })

    ids = [1, 3, 2, 4, 5]
    logits = model.forward(ids)

    payload = {
        "config": {
            "vocab_size": VOCAB,
            "d_model": D_MODEL,
            "d_state": D_STATE,
            "d_conv": D_CONV,
            "expand": EXPAND,
            "num_layers": LAYERS,
            "max_len": MAX_LEN,
            "param_count": model.param_count(),
        },
        "weights": {
            "emb": flat2(model.embedding.weight),
            "blocks": blocks,
            "head": dump_layer(model.head),
        },
        "cases": [
            {"ids": ids, "logits": flat2(logits)},
        ],
    }
    out = r"D:\vscode\AI_C\c\mamba_reference.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("wrote", out, "param_count", payload["config"]["param_count"])


if __name__ == "__main__":
    main()
