# -*- coding: utf-8 -*-
"""从 AI.py 导出 RL/μP 工具参考值 -> c/rl_utils_reference.json

覆盖 group_relative_advantages / best_of_n / rejection_sampling_filter /
self_consistency（整数样本）/ mup_target_std / mup_layer_kinds / mup_lr /
MuPAdam 的单步更新，供 c/tests/compare_rl_utils.c 逐位（容差 0）对拍。
"""
import json
import math
import os
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


# ---------------- GRPO 组相对优势 ----------------
for name, rewards, group_size, normalize in (
    ("adv_single", [1.0, 2.0, 3.0, 4.0], 0, 1),
    ("adv_single_raw", [1.0, 2.0, 3.0, 4.0], 0, 0),
    ("adv_two_groups", [1.0, 3.0, 2.0, 6.0], 2, 1),
    ("adv_two_groups_raw", [1.0, 3.0, 2.0, 6.0], 2, 0),
    ("adv_constant", [2.5, 2.5, 2.5], 3, 1),
    ("adv_single_reward", [7.0], 1, 1),
):
    out = AI.group_relative_advantages(rewards, group_size=group_size or None,
                                       normalize=bool(normalize))
    add(kind="group_adv", name=name, rewards=rewards, group_size=group_size,
        eps=1e-8, normalize=normalize, out=out, tol=0.0)

# ---------------- Best-of-N ----------------
for name, rewards in (
    ("bon_1", [0.1, 0.9, 0.5, 0.9]),
    ("bon_negative", [-3.0, -1.0, -2.0]),
    ("bon_single", [4.5]),
):
    res = AI.best_of_n(["a", "b", "c", "d"][:len(rewards)], rewards)
    add(kind="best_of_n", name=name, rewards=rewards, index=res["index"],
        reward=res["reward"], tol=0.0)

# ---------------- 拒绝采样 ----------------
for name, rewards, threshold, top_ratio in (
    ("rej_threshold", [0.1, 0.9, 0.5, 0.7], 0.5, 0),
    ("rej_threshold_none", [0.1, 0.2], 0.99, 0),
    ("rej_ratio", [0.1, 0.9, 0.5, 0.7], 0, 0.5),
    ("rej_ratio_ties", [0.5, 0.5, 0.5, 0.5], 0, 0.5),
    ("rej_ratio_small", [0.2, 0.8], 0, 0.1),
):
    kept, idx = AI.rejection_sampling_filter(
        list(range(len(rewards))), rewards,
        threshold=None if threshold in (0, 0.0) and top_ratio else threshold,
        top_ratio=top_ratio or None)
    add(kind="rejection", name=name, rewards=rewards,
        mode="threshold" if top_ratio == 0 else "top_ratio",
        threshold=float(threshold), ratio=float(top_ratio),
        kept=idx, count=len(idx), tol=0.0)

# ---------------- 自洽性（整数样本） ----------------
for name, samples in (
    ("sc_basic", [1, 2, 2, 3, 2]),
    ("sc_tie", [5, 5, 6, 6, 7]),
    ("sc_single", [9]),
    ("sc_all_same", [4, 4, 4]),
):
    best, tally, conf = AI.self_consistency(samples)
    keys = list(tally.keys())
    counts = [tally[k] for k in keys]
    add(kind="self_consistency", name=name, samples=samples, best=best,
        confidence=conf, unique=len(keys), keys=keys, counts=counts, tol=0.0)

# ---------------- μP：target_std ----------------
for kind in ("input", "hidden", "readout"):
    for fan_in in (4, 32):
        for width in (None, 16.0):
            res = AI.mup_target_std(kind, fan_in, base_width=8.0,
                                    width=width, scale=1.0)
            add(kind="mup_target_std", name="%s_%d_%s" % (kind, fan_in, width),
                mup_kind=kind, fan_in=float(fan_in), base_width=8.0,
                width=0.0 if width is None else width, scale=1.0,
                value=res, tol=1e-15)

# ---------------- μP：layer kinds ----------------
for n in (1, 2, 3, 5):
    kinds = AI._mup_layer_kinds(n)
    add(kind="mup_layer_kinds", name="kinds_%d" % n, num_layers=n,
        kinds=kinds, tol=0.0)

# ---------------- μP：分层学习率 ----------------
for name, layers, base_lr, width, base_width in (
    ("lr_1", 1, 1e-3, 0.0, 1.0),
    ("lr_2", 2, 1e-3, 0.0, 1.0),
    ("lr_4", 4, 1e-3, 0.0, 1.0),
    ("lr_explicit", 4, 2e-3, 10.0, 2.0),
):
    model = AI.MLP([3] + [4] * (layers - 1) + [2],
                   ["relu"] * (layers - 1) + ["linear"])
    fan_ins = [layer.fan_in for layer in model.layers]
    lrs = AI.mup_lr(model, base_lr=base_lr,
                    width=None if width == 0.0 else width,
                    base_width=base_width)
    add(kind="mup_lr", name=name, fan_ins=fan_ins, base_lr=base_lr,
        width=width, base_width=base_width, lrs=lrs, tol=0.0)

# ---------------- μP：MuPAdam 单步 ----------------
for name, step_count, lr in (("adam_step1", 1, 0.01), ("adam_step3", 3, 0.01)):
    params = [0.5, -1.0, 2.0, 0.0]
    grads = [0.1, -0.2, 0.3, 0.4]
    m = [0.0] * 4
    v = [0.0] * 4
    c1 = 1.0 - 0.9 ** step_count
    c2 = 1.0 - 0.999 ** step_count
    out = []
    for i in range(4):
        mi = 0.9 * m[i] + 0.1 * grads[i]
        vi = 0.999 * v[i] + 0.001 * grads[i] * grads[i]
        m[i] = mi
        v[i] = vi
        out.append(params[i] - lr * (mi / c1) / (math.sqrt(vi / c2) + 1e-8))
    add(kind="mupadam", name=name, params=params, grads=grads, lr=lr,
        beta1=0.9, beta2=0.999, eps=1e-8, step_count=step_count,
        out=out, tol=1e-15)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "rl_utils_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
