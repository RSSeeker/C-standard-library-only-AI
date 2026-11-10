"""为 C 侧 xLSTM 前向对拍生成参考数据。

对应 C 模块：c/xlstm.h / c/xlstm.c（只前向）。
"""
import json
import random
import sys

sys.path.insert(0, r"D:\vscode\Python-standard-library-only-AI-main")
import AI  # noqa: E402


VOCAB = 6
D_MODEL = 8
D_FF = 12
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
    model = AI.XLSTMModel(VOCAB, D_MODEL, D_FF, LAYERS, MAX_LEN, seed=SEED)

    blocks = []
    for b in model.blocks:
        blocks.append({
            "ln1_gamma": flat1(b.ln1.gamma),
            "ln1_beta": flat1(b.ln1.beta),
            "Wg": dump_layer(b.cell.Wg),
            "Wo": dump_layer(b.cell.Wo),
            "ln2_gamma": flat1(b.ln2.gamma),
            "ln2_beta": flat1(b.ln2.beta),
            "W1": dump_layer(b.W1),
            "W2": dump_layer(b.W2),
        })

    ids = [1, 3, 2, 4, 5]
    logits = model.forward(ids)

    payload = {
        "config": {
            "vocab_size": VOCAB,
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "num_layers": LAYERS,
            "max_len": MAX_LEN,
            "param_count": model.param_count(),
        },
        "weights": {
            "emb": flat2(model._emb),
            "blocks": blocks,
            "ln_out_gamma": flat1(model.ln_out.gamma),
            "ln_out_beta": flat1(model.ln_out.beta),
            "W_out": dump_layer(model.W_out),
        },
        "cases": [
            {"ids": ids, "logits": flat2(logits)},
        ],
    }
    out = r"D:\vscode\AI_C\c\xlstm_reference.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("wrote", out, "param_count", payload["config"]["param_count"])


if __name__ == "__main__":
    main()
