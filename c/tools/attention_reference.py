# -*- coding: utf-8 -*-
"""导出 MultiHeadAttention / TransformerEncoderLayer 的参考数值，供
C 实现（tests/compare_attention.c）逐位比对。

用法：python c/tools/attention_reference.py [输出路径]
默认写到 c/attention_reference.json

覆盖：普通因果 MHA、滑动窗口、RoPE、GQA、MQA、三者组合，以及 TEL 的
forward + backward（含 LN / FFN / 注意力全部参数梯度）；MHA 另导出
forward_step 的逐 token 输出（增量 KV cache 的跨语言对照）。

权重不写死：固定种子在 Python 侧建模型，把初始参数一起导出，C 侧灌入后
从同一组 double 出发，任何差异都只能来自实现本身。dropout 一律为 0，
因为两边采样 RNG 不同，带 dropout 的前向本就不在逐位契约内。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)

import random  # noqa: E402

import AI  # noqa: E402


def flat(x):
    if isinstance(x, list):
        out = []
        for row in x:
            out.extend(flat(row))
        return out
    return [float(x)]


def make_x(T, d, scale):
    """确定性的输入（都是二进制可精确表示的整数倍，避免十进制解析歧义）。"""
    return [[float(((t * 7 + j * 3) % 11 - 5) * scale) for j in range(d)]
            for t in range(T)]


def mha_case(name, d, h, kv, T, causal, window, rope, seed):
    random.seed(seed)
    m = AI.MultiHeadAttention(d, h, dropout=0.0, causal=bool(causal),
                              rope=bool(rope), rope_base=10000.0,
                              window=(window if window else None),
                              num_kv_heads=kv)
    m.eval()   # dropout 关闭；本组用例 p=0，数值与 train 相同
    x = make_x(T, d, 0.125)
    dout = make_x(T, d, 0.25)
    out = m.forward(x)
    m.zero_grad()
    dx = m.backward(dout)

    # 增量前向（cache 从零开始，逐 token 调用）
    cache = {}
    steps = [m.forward_step(x[t], cache) for t in range(T)]

    params = {
        "W_q": flat(m.W_q), "b_q": list(m.b_q),
        "W_k": flat(m.W_k), "b_k": list(m.b_k),
        "W_v": flat(m.W_v), "b_v": list(m.b_v),
        "W_o": flat(m.W_o), "b_o": list(m.b_o),
    }
    grads = {k: flat(v) for k, v in m.grads.items()}
    return {
        "name": name,
        "kind": "mha",
        "config": {"d_model": d, "num_heads": h, "num_kv_heads": kv,
                   "max_seq": max(T, 6), "causal": int(bool(causal)),
                   "window": int(window), "rope": int(bool(rope)),
                   "rope_base": 10000.0, "dropout": 0.0},
        "params": params,
        "x": flat(x),
        "dout": flat(dout),
        "out": flat(out),
        "dx": flat(dx),
        "grads": grads,
        "steps": flat(steps),
    }


def tel_case(name, d, h, ff, T, seed):
    random.seed(seed)
    layer = AI.TransformerEncoderLayer(d, h, ff, dropout=0.0, causal=True)
    layer.eval()
    x = make_x(T, d, 0.125)
    dout = make_x(T, d, 0.25)
    out = layer.forward(x)
    layer.zero_grad()
    dx = layer.backward(dout)

    params = {
        "W_q": flat(layer.self_attn.W_q), "b_q": list(layer.self_attn.b_q),
        "W_k": flat(layer.self_attn.W_k), "b_k": list(layer.self_attn.b_k),
        "W_v": flat(layer.self_attn.W_v), "b_v": list(layer.self_attn.b_v),
        "W_o": flat(layer.self_attn.W_o), "b_o": list(layer.self_attn.b_o),
        "W1": flat(layer.ffn.W1), "b1": list(layer.ffn.b1),
        "W2": flat(layer.ffn.W2), "b2": list(layer.ffn.b2),
        "ln1_gamma": list(layer.ln1.gamma), "ln1_beta": list(layer.ln1.beta),
        "ln2_gamma": list(layer.ln2.gamma), "ln2_beta": list(layer.ln2.beta),
    }
    grads = {}
    for k, v in layer.self_attn.grads.items():
        grads[k] = flat(v)
    for k, v in layer.ffn.grads.items():
        grads[k] = flat(v)
    for k in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
        grads[k] = flat(layer.grads[k])

    return {
        "name": name,
        "kind": "tel",
        "config": {"d_model": d, "num_heads": h, "d_ff": ff,
                   "max_seq": max(T, 6), "causal": 1, "dropout": 0.0},
        "params": params,
        "x": flat(x),
        "dout": flat(dout),
        "out": flat(out),
        "dx": flat(dx),
        "grads": grads,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.abspath(os.path.join(_here, "..")), "attention_reference.json")

    cases = [
        mha_case("mha_causal", d=8, h=2, kv=2, T=4, causal=1, window=0,
                 rope=0, seed=1001),
        mha_case("mha_window2", d=8, h=2, kv=2, T=6, causal=0, window=2,
                 rope=0, seed=1002),
        mha_case("mha_rope", d=8, h=2, kv=2, T=5, causal=1, window=0,
                 rope=1, seed=1003),
        mha_case("mha_gqa", d=8, h=4, kv=2, T=4, causal=1, window=0,
                 rope=0, seed=1004),
        mha_case("mha_mqa", d=8, h=4, kv=1, T=5, causal=1, window=0,
                 rope=0, seed=1005),
        mha_case("mha_combo", d=8, h=4, kv=2, T=6, causal=0, window=2,
                 rope=1, seed=1006),
        tel_case("tel_causal", d=8, h=2, ff=16, T=4, seed=1007),
    ]

    out = {"cases": cases}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    total = sum(len(c["params"][k]) + len(c["grads"][k])
                for c in cases for k in c["params"])
    print(f"已写出 {out_path}（{len(cases)} 个用例，约 {total} 个参数/梯度值）")


if __name__ == "__main__":
    main()
