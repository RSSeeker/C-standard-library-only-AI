"""Generate DeBERTa forward reference for tests/compare_deberta.c."""
import json
import sys

sys.path.insert(0, "D:/vscode/Python-standard-library-only-AI-main")
import AI  # noqa: E402


def mat(m):
    return [[float(v) for v in row] for row in m]


def dump_dense(layer):
    return {"weights": mat(layer.weights), "biases": [float(v) for v in layer.biases]}


def dump_ln(ln):
    return {"gamma": [float(v) for v in ln.gamma], "beta": [float(v) for v in ln.beta]}


def main():
    vocab, d_model, d_ff, layers, max_len, max_rel = 6, 8, 12, 2, 8, 4
    model = AI.DeBERTaModel(vocab_size=vocab, d_model=d_model, d_ff=d_ff,
                            num_layers=layers, max_len=max_len,
                            max_rel=max_rel, seed=21)
    ids = [1, 3, 2, 4, 5]

    blocks = []
    for b in model.blocks:
        blocks.append({
            "ln1": dump_ln(b.ln1),
            "ln2": dump_ln(b.ln2),
            "Wq": dump_dense(b.Wq),
            "Wk": dump_dense(b.Wk),
            "Wv": dump_dense(b.Wv),
            "Wo": dump_dense(b.Wo),
            "W1": dump_dense(b.W1),
            "W2": dump_dense(b.W2),
            "Rk": mat(b.Rk),
            "Rq": mat(b.Rq),
        })

    payload = {
        "config": {
            "vocab_size": vocab,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": layers,
            "max_len": max_len,
            "max_rel": max_rel,
            "param_count": model.param_count(),
        },
        "weights": {
            "emb": mat(model._emb),
            "blocks": blocks,
            "ln_out": dump_ln(model.ln_out),
            "W_out": dump_dense(model.W_out),
        },
        "cases": [
            {"ids": ids, "logits": mat(model.forward(ids))},
        ],
    }

    out = "deberta_reference.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out, "param_count", payload["config"]["param_count"])


if __name__ == "__main__":
    main()
