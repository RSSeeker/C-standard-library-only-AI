"""生成 ELECTRA 判别器前向的跨语言对拍参考（C: tests/compare_electra.c）。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

import AI  # noqa: E402


def dump_matrix(mat):
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    data = []
    for row in mat:
        data.extend(float(v) for v in row)
    return {"shape": [rows, cols], "data": data}


def main():
    m = AI.ElectraDiscriminator(vocab_size=6, d_model=8, num_heads=2, d_ff=12,
                                num_layers=2, max_len=8, seed=3)
    ids = [1, 6, 2, 4, 5]      # 6 = mask token（词表外那一行）
    score = m.forward(ids)
    prob = m.probabilities(ids)

    weights = []
    emb = m.embedding.weight
    weights.append({"name": "emb.weight", **dump_matrix(emb)})
    for li, layer in enumerate(m.encoder.layers):
        attn = layer.self_attn
        for key in ("W_q", "W_k", "W_v", "W_o"):
            weights.append({"name": f"enc.L{li}.attn.{key}", **dump_matrix(getattr(attn, key))})
        for key in ("b_q", "b_k", "b_v", "b_o"):
            vec = getattr(attn, key)
            weights.append({"name": f"enc.L{li}.attn.{key}", "shape": [len(vec)],
                            "data": [float(v) for v in vec]})
        ffn = layer.ffn
        for key in ("W1", "W2"):
            weights.append({"name": f"enc.L{li}.ffn.{key}", **dump_matrix(getattr(ffn, key))})
        for key in ("b1", "b2"):
            vec = getattr(ffn, key)
            weights.append({"name": f"enc.L{li}.ffn.{key}", "shape": [len(vec)],
                            "data": [float(v) for v in vec]})
        for ln_name in ("ln1", "ln2"):
            ln = getattr(layer, ln_name)
            weights.append({"name": f"enc.L{li}.{ln_name}.gamma", "shape": [len(ln.gamma)],
                            "data": [float(v) for v in ln.gamma]})
            weights.append({"name": f"enc.L{li}.{ln_name}.beta", "shape": [len(ln.beta)],
                            "data": [float(v) for v in ln.beta]})
    weights.append({"name": "W_d", **dump_matrix(m.W_d)})
    weights.append({"name": "b_d", "shape": [len(m.b_d)], "data": [float(v) for v in m.b_d]})

    payload = {
        "config": {
            "vocab_size": 6, "d_model": 8, "num_heads": 2, "d_ff": 12,
            "num_layers": 2, "max_len": 8, "param_count": m.param_count(),
        },
        "weights": weights,
        "case": {
            "ids": ids,
            "score": [float(v) for v in score],
            "prob": [float(v) for v in prob],
            "tolerance": 1e-12,
        },
    }
    out = os.path.join(HERE, "..", "electra_reference.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out, "param_count", payload["config"]["param_count"])
    print("score", score)


if __name__ == "__main__":
    main()
