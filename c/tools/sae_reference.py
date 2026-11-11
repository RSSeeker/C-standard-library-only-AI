# -*- coding: utf-8 -*-
"""从 AI.py 导出 SAE（稀疏自编码器）参考值 -> c/sae_reference.json

导出：初始参数、前向（recon/features/pre）、损失与梯度、SGD 一步后的
参数与 feature_stats，供 c/tests/compare_sae.c 逐位对拍。
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


def flat(matrix):
    out = []
    for row in matrix:
        out.extend(float(v) for v in row)
    return out


def add(**kw):
    CASES.append(kw)


CFG = ((4, 8, 1e-3, 1), (3, 5, 0.01, 3), (2, 3, 0.0, 7))
for (d_in, d_hidden, l1_coeff, seed) in CFG:
    rng = random.Random(20261113 + seed)
    sae = AI.SAE(d_in, d_hidden, l1_coeff=l1_coeff, seed=seed)
    xs = [[round(rng.uniform(-1.5, 1.5), 6) for _ in range(d_in)]
          for _ in range(4)]
    init = {
        "W_enc": flat(sae.W_enc), "b_enc": [float(v) for v in sae.b_enc],
        "W_dec": flat(sae.W_dec), "b_dec": [float(v) for v in sae.b_dec],
    }
    fwd = sae.forward(xs)
    sae.zero_grad()
    info = sae.loss_and_backward(xs)
    grads = {
        "gW_enc": flat(sae.grad_W_enc),
        "gb_enc": [float(v) for v in sae.grad_b_enc],
        "gW_dec": flat(sae.grad_W_dec),
        "gb_dec": [float(v) for v in sae.grad_b_dec],
    }
    sae.sgd_step(0.05)
    post = {
        "W_enc": flat(sae.W_enc), "b_enc": [float(v) for v in sae.b_enc],
        "W_dec": flat(sae.W_dec), "b_dec": [float(v) for v in sae.b_dec],
    }
    stats = sae.feature_stats(xs)
    enc = sae.encode(xs[0])
    dec = sae.decode(enc)
    add(kind="sae", name="sae_%dx%d_%g" % (d_in, d_hidden, l1_coeff),
        d_in=d_in, d_hidden=d_hidden, l1_coeff=l1_coeff, xs=xs,
        param_count=sae.param_count(), sgd_lr=0.05,
        init=init, post=post,
        recon=fwd["recon"], features=fwd["features"], pre=fwd["pre"],
        loss=info["loss"], mse=info["mse"], l1=info["l1"], grads=grads,
        active_fraction=stats["active_fraction"],
        dead_features=stats["dead_features"],
        mean_active=stats["mean_active"],
        encode=enc, decode=dec, tol=1e-15)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "sae_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
