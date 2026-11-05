# -*- coding: utf-8 -*-
"""导出 MultiHeadAttention / TransformerEncoderLayer 的参考数值，供
C 实现（tests/compare_attention.c）逐位比对。

用法：python c/tools/attention_reference.py [输出路径]
默认写到 c/attention_reference.json

覆盖：普通因果 MHA、滑动窗口、RoPE、GQA、MQA、三者组合，以及 TEL 的
forward + backward（含 LN / FFN / 注意力全部参数梯度）；MHA 另导出
forward_step 的逐 token 输出（增量 KV cache 的跨语言对照）。

注意力权重 dropout 单独用「注入 mask」的方式导出：给模型换一个只有 .random()
的假 RNG（按预生成列表返回），于是 Python 侧用的 mask 完全确定，再把它一起
导出；C 侧通过 mha_set_attn_mask() 灌入同一份 mask，就能绕开「两边 RNG 不同源」
做逐位对拍。同一组权重另导出一份 dropout=0 的对照，用来确认 C 侧默认路径没变。

权重不写死：固定种子在 Python 侧建模型，把初始参数一起导出，C 侧灌入后
从同一组 double 出发，任何差异都只能来自实现本身。
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


def mha_case(name, d, h, kv, T, causal, window, rope, seed, alibi=False):
    random.seed(seed)
    m = AI.MultiHeadAttention(d, h, dropout=0.0, causal=bool(causal),
                              rope=bool(rope), rope_base=10000.0,
                              window=(window if window else None),
                              num_kv_heads=kv, alibi=bool(alibi))
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
                   "rope_base": 10000.0, "dropout": 0.0,
                   "alibi": int(bool(alibi))},
        "params": params,
        "x": flat(x),
        "dout": flat(dout),
        "out": flat(out),
        "dx": flat(dx),
        "grads": grads,
        "steps": flat(steps),
    }


class _FixedRng:
    """只有 .random() 的假 RNG，按预生成列表逐项返回。

    用它替换 MultiHeadAttention.rng 后，Python 侧生成的 dropout mask 完全确定。
    列表被耗尽时直接抛错（而不是回绕或补 0）：那说明实现消耗的随机数比预期多，
    导出的 mask 就不再等于 Python 实际用的 mask，必须立刻暴露。
    """

    def __init__(self, values):
        self._values = list(values)
        self._next = 0

    def random(self):
        if self._next >= len(self._values):
            raise AssertionError("假 RNG 的随机数不够用：mask 消耗量超出预生成列表")
        value = self._values[self._next]
        self._next += 1
        return value


def _dropout_randoms(h, T, p):
    """按 (head, t, j) 行主序预生成 h*T*T 个 [0,1) 随机数。

    0.25 与 0.9 都远离 keep=1-p 的边界（p 取 0.5 时 keep=0.5），因此
    「< keep 就保留」的判定不受浮点舍入影响；(i + head) % 3 让每头、每行
    都同时包含保留与丢弃两种位置，mask 里既有 0 也有 1/(1-p)。
    """
    keep = 1.0 - p
    assert 0.25 < keep <= 0.9, "预生成值固定为 0.25/0.9，只适用于 keep 落在 (0.25,0.9] 的 p"
    values = []
    for head in range(h):
        for i in range(T * T):
            values.append(0.9 if (i + head) % 3 == 0 else 0.25)
    return values


def mha_dropout_case(name, d, h, kv, T, p, training, seed, causal=1):
    """注意力权重 dropout 用例：固定权重 + 注入 mask（training=1）或对照（p=0）。"""
    random.seed(seed)
    m = AI.MultiHeadAttention(d, h, dropout=p, causal=bool(causal),
                              rope=False, rope_base=10000.0,
                              window=None, num_kv_heads=kv)
    if training:
        m.train()
        if p > 0.0:
            # 同一份预生成随机数只够一次 forward：导出前先注入，再 forward
            m.rng = _FixedRng(_dropout_randoms(h, T, p))
    else:
        m.eval()

    x = make_x(T, d, 0.125)
    dout = make_x(T, d, 0.25)
    out = m.forward(x)
    m.zero_grad()
    dx = m.backward(dout)

    cache = {}
    steps = [m.forward_step(x[t], cache) for t in range(T)]

    mask = []
    if training and p > 0.0:
        for head in range(h):
            row = m.attn_masks[head]
            assert row is not None, "training 且 p>0 时必须记录 mask"
            mask.extend(flat(row))
        assert len(mask) == h * T * T

    params = {
        "W_q": flat(m.W_q), "b_q": list(m.b_q),
        "W_k": flat(m.W_k), "b_k": list(m.b_k),
        "W_v": flat(m.W_v), "b_v": list(m.b_v),
        "W_o": flat(m.W_o), "b_o": list(m.b_o),
    }
    grads = {k: flat(v) for k, v in m.grads.items()}
    case = {
        "name": name,
        "kind": "mha",
        "config": {"d_model": d, "num_heads": h, "num_kv_heads": kv,
                   "max_seq": max(T, 6), "causal": int(bool(causal)),
                   "window": 0, "rope": 0, "rope_base": 10000.0,
                   "dropout": p, "training": 1 if training else 0},
        "params": params,
        "x": flat(x),
        "dout": flat(dout),
        "out": flat(out),
        "dx": flat(dx),
        "grads": grads,
        "steps": flat(steps),
    }
    if mask:
        # training=1 且 p>0 时是 (num_heads, T, T) 行主序的 inverted mask；
        # 对照用例（p=0）不写这个键，C 侧据此走默认路径。
        case["mask"] = mask
    return case


def tel_case(name, d, h, ff, T, seed, alibi=False):
    random.seed(seed)
    layer = AI.TransformerEncoderLayer(d, h, ff, dropout=0.0, causal=True,
                                       alibi=bool(alibi))
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
                   "max_seq": max(T, 6), "causal": 1, "dropout": 0.0,
                   "alibi": int(bool(alibi))},
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
        # ALiBi：slope_h = 2^(-8*(h+1)/num_heads)，h=2/4 时是 2 的整数次幂，
        # 两种语言都能精确表示；out / dx / 8 组梯度 / forward_step 都要求逐位相同。
        mha_case("mha_alibi", d=8, h=2, kv=2, T=4, causal=1, window=0,
                 rope=0, seed=1009, alibi=1),
        mha_case("mha_alibi_win", d=8, h=4, kv=2, T=6, causal=0, window=2,
                 rope=0, seed=1010, alibi=1),
        mha_case("mha_alibi_rope", d=8, h=2, kv=2, T=5, causal=1, window=0,
                 rope=1, seed=1011, alibi=1),
        tel_case("tel_alibi", d=8, h=2, ff=16, T=4, seed=1012, alibi=1),
        # 注意力权重 dropout：同一 seed/权重下，training=1 + p=0.5（注入固定 mask）
        # 与 p=0 对照。C 侧对前者调 mha_set_training(1) + mha_set_attn_mask()，
        # 对后者保持默认；两者除 dropout 外完全同源，差异只可能来自 dropout 实现。
        mha_dropout_case("mha_dropout_p50", d=8, h=2, kv=2, T=4, p=0.5,
                         training=1, seed=1008),
        mha_dropout_case("mha_dropout_ctrl", d=8, h=2, kv=2, T=4, p=0.0,
                         training=1, seed=1008),
    ]

    out = {"cases": cases}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    total = sum(len(c["params"][k]) + len(c["grads"][k])
                for c in cases for k in c["params"])
    print(f"已写出 {out_path}（{len(cases)} 个用例，约 {total} 个参数/梯度值）")


if __name__ == "__main__":
    main()
