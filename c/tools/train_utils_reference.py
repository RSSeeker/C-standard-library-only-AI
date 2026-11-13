# -*- coding: utf-8 -*-
"""从 AI.py 导出训练辅助工具的参考值 -> c/train_utils_reference.json

覆盖 NF4 量化 / 双重量化 / WeightStandardization / GradScaler / MoE 负载均衡损失，
供 c/tests/compare_train_utils.c 逐位（容差 0）对拍。
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(C_DIR)
PY_REPO = os.path.join(os.path.dirname(REPO), "Python-standard-library-only-AI-main")
sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

CASES = []


def add(**kw):
    CASES.append(kw)


def flat(matrix):
    out = []
    for row in matrix:
        out.extend(float(v) for v in row)
    return out


def rand_rows(rng, rows, cols, lo=-2.0, hi=2.0):
    return [[rng.uniform(lo, hi) for _ in range(cols)] for _ in range(rows)]


# ---------------- NF4 ----------------
NF4_RNG = random.Random(20261111)
for block_size in (4, 8, 64):
    n = 10
    values = [round(NF4_RNG.uniform(-1.5, 1.5), 6) for _ in range(n)]
    quant = AI.nf4_quantize(values, block_size=block_size)
    deq = AI.nf4_dequantize(quant)
    fake = AI.nf4_fake_quantize(values, block_size=block_size)
    dq = AI.double_quantize(quant["scales"], block_size=4)
    ddeq = AI.double_dequantize(dq)
    dfake = AI.nf4_fake_quantize_double(values, block_size=block_size,
                                        outer_block_size=4)
    add(kind="nf4", name="nf4_%d" % block_size, x=values,
        block_size=block_size, outer_block_size=4,
        codes=list(quant["codes"]), scales=list(quant["scales"]),
        deq=deq, fake=fake, double_codes=list(dq["codes"]),
        double_outers=list(dq["outers"]), double_deq=ddeq, double_fake=dfake,
        tol=0.0)

# ---------------- WeightStandardization ----------------
WS_RNG = random.Random(20261112)
for (fan_out, fan_in, n, eps) in ((2, 3, 4, 1e-5), (3, 2, 3, 1e-3)):
    w = rand_rows(WS_RNG, fan_out, fan_in)
    b = [WS_RNG.uniform(-0.5, 0.5) for _ in range(fan_out)]
    x = rand_rows(WS_RNG, n, fan_in)
    dy = rand_rows(WS_RNG, n, fan_out)
    layer = AI.Layer(fan_in, fan_out, "linear")
    wrap = AI.WeightStandardization(layer, eps)
    wrap.w = [list(row) for row in w]
    wrap.b = list(b)
    wrap._sync_inner()
    ws, mu, sigma = wrap._effective()
    y = [wrap.forward(row) for row in x]
    dw = [[0.0] * fan_in for _ in range(fan_out)]
    db = [0.0] * fan_out
    dx = []
    for r in range(n):
        wrap.forward(x[r])
        dxr = wrap.backward(dy[r])
        for i in range(fan_out):
            db[i] += wrap.grad_b[i]
            for j in range(fan_in):
                dw[i][j] += wrap.grad_w[i][j]
        dx.append(dxr)
    add(kind="ws", name="ws_%dx%d" % (fan_out, fan_in), w=flat(w), b=b,
        x=flat(x), dy=flat(dy), fan_out=fan_out, fan_in=fan_in, n=n, eps=eps,
        ws=flat(ws), mu=mu, sigma=sigma, y=flat(y), dw=flat(dw), db=db,
        dx=flat(dx), tol=1e-15)

# ---------------- GradScaler ----------------
GS_RNG = random.Random(20261113)
configs = (
    dict(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
         growth_interval=3, n_steps=7,
         force_inf=[0, 0, 0, 1, 0, 0, 0]),
    dict(init_scale=1024.0, growth_factor=4.0, backoff_factor=0.25,
         growth_interval=2, n_steps=6,
         force_inf=[0, 1, 0, 1, 0, 0]),
)
for ci, cfg in enumerate(configs):
    scaler = AI.GradScaler(cfg["init_scale"], cfg["growth_factor"],
                           cfg["backoff_factor"], cfg["growth_interval"])
    steps = []
    for t in range(cfg["n_steps"]):
        loss = round(GS_RNG.uniform(0.1, 5.0), 6)
        grads = [round(GS_RNG.uniform(-1.0, 1.0), 6) for _ in range(3)]
        force_inf = cfg["force_inf"][t]
        scaled = scaler.scale_loss(loss)
        unscaled = scaler.unscale([list(grads)])
        can_step = scaler.update(force_inf if force_inf else None)
        steps.append(dict(loss=loss, grads=grads, force_inf=force_inf,
                          scaled=scaled, unscaled=unscaled[0],
                          can_step=1 if can_step else 0))
    state = scaler.state_dict()
    add(kind="gradscaler", name="gradscaler_%d" % ci,
        init_scale=cfg["init_scale"], growth_factor=cfg["growth_factor"],
        backoff_factor=cfg["backoff_factor"],
        growth_interval=cfg["growth_interval"], steps=steps,
        state=dict(scale=state["scale"],
                   growth_tracker=state["growth_tracker"],
                   steps_skipped=state["steps_skipped"],
                   steps_taken=state["steps_taken"],
                   total_steps=state["total_steps"]),
        tol=0.0)

# ---------------- MoE 负载均衡 ----------------
moe = AI.MoE(4, 8, 4, 2)
tokens = 5
x_seq = rand_rows(WS_RNG, tokens, 4)
moe.forward(x_seq)
topk_idx = []
for route in moe.routes:
    for e, _w in route:
        topk_idx.append(int(e))
gate_probs = flat(moe._gate_all_probs)
for coef in (0.01, 0.0, 0.1):
    add(kind="moe_aux", name="moe_aux_%g" % coef, tokens=tokens, experts=4,
        top_k=2, topk_idx=topk_idx, gate_probs=gate_probs,
        coefficient=coef, value=moe.load_balancing_loss(coef), tol=1e-12)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "train_utils_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
