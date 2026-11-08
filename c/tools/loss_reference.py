# -*- coding: utf-8 -*-
"""生成 focal / bce / bce_with_logits / huber 的跨语言对拍数据

用法：python tools/loss_reference.py [输出路径]
      默认写到 c/loss_reference.json

数据来源是 AI.py 的四个损失函数（它们都返回 (loss, grad)）：
  focal_loss / bce_loss / bce_with_logits_loss / huber_loss

对齐约定：
  - 只导出 C 侧支持的形态：focal 的 alpha 用标量或 Python 的 None（C 里 -1 表示
    None）；bce / huber 的 target 是与输入等长的列表（C 只支持等长，不支持标量）。
  - tol 一律 0.0：这批输入只用到 + - * / 与 log/exp/log1p/pow，本机实测 C 与
    Python 逐位相同（有差异会在 compare_loss 里报出来，届时再按实测标注，
    不许预先放宽容差）。
"""
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main")
)
if not os.path.isdir(AI_PY_DIR):
    sys.exit("找不到 Python 参考实现目录: %s（可用环境变量 AI_PY_DIR 指定）" % AI_PY_DIR)
sys.path.insert(0, AI_PY_DIR)

from AI import (  # noqa: E402
    bce_loss,
    bce_with_logits_loss,
    focal_loss,
    huber_loss,
    smooth_l1_loss,
)

CASES = []


def add_focal(name, pred, target_idx, gamma=2.0, alpha=None):
    loss, grad = focal_loss(pred, target_idx, gamma=gamma, alpha=alpha)
    CASES.append({
        "name": name,
        "func": "focal",
        "pred": [float(p) for p in pred],
        "target_idx": int(target_idx),
        "gamma": float(gamma),
        "alpha": -1.0 if alpha is None else float(alpha),
        "loss": float(loss),
        "grad": [float(g) for g in grad],
    })


def add_bce(name, pred, target, eps=1e-12):
    loss, grad = bce_loss(pred, target, eps=eps)
    CASES.append({
        "name": name,
        "func": "bce",
        "pred": [float(p) for p in pred],
        "target": [float(t) for t in target],
        "eps": float(eps),
        "loss": float(loss),
        "grad": [float(g) for g in grad],
    })


def add_bce_logits(name, logits, target):
    loss, grad = bce_with_logits_loss(logits, target)
    CASES.append({
        "name": name,
        "func": "bce_logits",
        "logits": [float(z) for z in logits],
        "target": [float(t) for t in target],
        "loss": float(loss),
        "grad": [float(g) for g in grad],
    })


def add_huber(name, pred, target, delta=1.0):
    loss, grad = huber_loss(pred, target, delta=delta)
    s_loss, s_grad = smooth_l1_loss(pred, target, delta=delta)
    assert loss == s_loss and grad == s_grad, "smooth_l1 与 huber 必须逐位相同"
    CASES.append({
        "name": name,
        "func": "huber",
        "pred": [float(p) for p in pred],
        "target": [float(t) for t in target],
        "delta": float(delta),
        "loss": float(loss),
        "grad": [float(g) for g in grad],
    })


# ---------- focal：gamma / alpha / 端点 ----------
F_PRED = [0.1, 0.6, 0.2, 0.1]
add_focal("gamma2_alpha_none", F_PRED, 1)
add_focal("gamma0_alpha_none", F_PRED, 1, gamma=0.0)
add_focal("gamma2_alpha_scalar", F_PRED, 0, gamma=2.0, alpha=0.25)
add_focal("gamma0.5_alpha_scalar", F_PRED, 2, gamma=0.5, alpha=0.5)
add_focal("gamma2_p_t_is_one", [0.0, 1.0, 0.0, 0.0], 1)
add_focal("gamma2_p_t_tiny", [1e-16, 1.0 - 1e-16], 0)
add_focal("gamma2_confident_correct", [0.999, 0.001], 0, alpha=0.75)
add_focal("gamma3_alpha2", [0.3, 0.4, 0.3], 1, gamma=3.0, alpha=2.0)

# ---------- bce：概率端点 + eps ----------
add_bce("basic", [0.9, 0.1, 0.5, 0.999], [1.0, 0.0, 1.0, 0.0])
add_bce("endpoints_clamped", [0.0, 1.0, 0.5], [1.0, 0.0, 0.0])
add_bce("eps_1e-3", [0.0, 1.0, 0.9999], [1.0, 0.0, 1.0], eps=1e-3)
add_bce("soft_targets", [0.2, 0.8, 0.6], [0.1, 0.9, 0.35])
add_bce("single", [0.25], [1.0])
add_bce("large_batch", [i / 20.0 for i in range(1, 20)],
        [float(i % 2) for i in range(1, 20)])

# ---------- bce_with_logits：数值稳定分支 ----------
add_bce_logits("basic", [-2.0, -1.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.5, 1.0, 1.0])
add_bce_logits("large_magnitude", [-50.0, -40.0, 40.0, 50.0], [0.0, 1.0, 0.0, 1.0])
add_bce_logits("extreme", [-800.0, 800.0], [0.0, 1.0])
add_bce_logits("all_zero_target", [0.3, -0.7, 2.5], [0.0, 0.0, 0.0])
add_bce_logits("all_one_target", [0.3, -0.7, 2.5], [1.0, 1.0, 1.0])
add_bce_logits("single", [1.25], [1.0])

# ---------- huber：边界 |e| == delta ----------
add_huber("boundary", [1.0, -2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0])
add_huber("delta_half", [1.0, -0.5, 0.25, 0.75], [0.0, 0.0, 0.0, 0.0], delta=0.5)
add_huber("large_errors", [10.0, -7.5], [0.0, 0.0], delta=1.0)
add_huber("delta_2", [3.0, -3.0, 1.0], [0.0, 0.0, 1.0], delta=2.0)


# ---------- 实测容差表 ----------
# 默认 tol = 0.0（要求逐位相同）。下面这条是 2026-11-08 在本机
#   gcc 16.1.0 (MinGW-W64, msvcrt.dll) × CPython 3.14.6 (UCRT)
# 上跑 tests/compare_loss 的实测结果：只有 loss 差 1 ULP（5.551e-17，
# 相对 1.4e-16），梯度逐位相同。原因是 pow(0.6, 3.0) 在 msvcrt 返回
# 0.216（精确立方），UCRT 返回 0.21599999999999997。这是 libm 差异，
# 不是公式差异——所以按「实测结果」标注，而不是预先放宽所有 focal 用例。
TOL_OVERRIDE = {
    "gamma3_alpha2": (1e-12, "pow(0.6,3.0) 命中 msvcrt/UCRT 的 1 ULP 差异："
                             "loss 差 5.551e-17，梯度逐位相同"),
}

for _c in CASES:
    tol, note = TOL_OVERRIDE.get(_c["name"], (0.0, None))
    _c["tol"] = tol
    if note:
        _c["note"] = note



def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(_here), "loss_reference.json")
    payload = {
        "meta": {
            "source": "AI.py: focal_loss / bce_loss / bce_with_logits_loss / huber_loss",
            "note": "tol 一律 0.0（逐位）；alpha=-1 表示 Python 的 alpha=None",
            "cases": len(CASES),
        },
        "cases": CASES,
    }
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    counts = {}
    for c in CASES:
        counts[c["func"]] = counts.get(c["func"], 0) + 1
    print("写出 %s：%d 个用例 %s" % (out, len(CASES), counts))


if __name__ == "__main__":
    main()
