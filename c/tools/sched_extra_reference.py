# -*- coding: utf-8 -*-
"""生成 WarmupLR / ReduceLROnPlateau 的跨语言对拍数据（JSON）

用法：python tools/sched_extra_reference.py [输出路径]
      默认写到 c/sched_extra_reference.json

数据来源是 AI.py 的 `class WarmupLR` 与 `class ReduceLROnPlateau`（它们只操作
`optimizer.lr`，所以这里用一个只有 lr 属性的 FakeOpt 驱动，不需要真的建模型）。
C 侧 tests/compare_sched_extra.c 读同一份 JSON 复跑同样的 epoch/metric 序列。

两个类都只做四则运算与比较，没有 pow/exp/log，所以所有用例 tol=0.0（逐位相同）。
base_scheduler 在 C 侧是函数指针，对照实现是下面的 HalveBaseScheduler：
每个 epoch 先把 lr 置成 base_lr，再连续乘 0.5 共 epoch 次（乘 0.5 是精确的）。
"""
import json
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

from AI import ReduceLROnPlateau, WarmupLR  # noqa: E402


class FakeOpt:
    """Python 侧调度器只用到 optimizer.lr"""

    def __init__(self, lr):
        self.lr = float(lr)


class HalveBaseScheduler:
    """对应 C 侧的 base_step 回调：lr = base_lr 连续乘 epoch 次 0.5"""

    def __init__(self, opt, base_lr):
        self.opt = opt
        self.base_lr = float(base_lr)
        self.epochs = []

    def step(self, epoch):
        self.epochs.append(int(epoch))
        v = self.base_lr
        for _ in range(int(epoch)):
            v *= 0.5
        self.opt.lr = v


def make_warmup_case(name, note, initial_lr, warmup_epochs, start_factor, epochs,
                     base_lr=None):
    opt = FakeOpt(initial_lr)
    base = HalveBaseScheduler(opt, base_lr) if base_lr is not None else None
    w = WarmupLR(opt, warmup_epochs=warmup_epochs, start_factor=start_factor,
                 base_scheduler=base)
    out = []
    for ep in epochs:
        w.step(ep)
        out.append(opt.lr)
    return {
        "name": name,
        "note": note,
        "tol": 0.0,
        "initial_lr": float(initial_lr),
        "warmup_epochs": int(warmup_epochs),
        "start_factor": float(start_factor),
        "epochs": [int(e) for e in epochs],
        "expect_lr": out,
        "base_lr": None if base_lr is None else float(base_lr),
        "expect_base_epochs": [] if base is None else list(base.epochs),
    }


def make_plateau_case(name, note, initial_lr, metrics, factor=0.5, patience=10,
                      min_lr=0.0, mode="min", threshold=1e-4, max_reductions=None):
    opt = FakeOpt(initial_lr)
    p = ReduceLROnPlateau(opt, factor=factor, patience=patience, min_lr=min_lr,
                          mode=mode, threshold=threshold, max_reductions=max_reductions)
    out = []
    for m in metrics:
        out.append(p.step(m))
    # JSON 里不能写裸 NaN（非法 JSON）：NaN 指标用 0.0 占位，索引记在 nan_at 里，
    # C 侧读到 nan_at 里的下标时自己构造 NAN。
    nan_at = [i for i, m in enumerate(metrics) if m != m]
    metrics_json = [0.0 if m != m else float(m) for m in metrics]
    return {
        "name": name,
        "note": note,
        "tol": 0.0,
        "initial_lr": float(initial_lr),
        "factor": float(factor),
        "patience": int(patience),
        "min_lr": float(min_lr),
        "mode": mode,
        "threshold": float(threshold),
        "max_reductions": -1 if max_reductions is None else int(max_reductions),
        "metrics": metrics_json,
        "nan_at": nan_at,
        "expect_lr": out,
        "expect_has_best": 1 if p.best is not None else 0,
        "expect_best": 0.0 if p.best is None else float(p.best),
        "expect_num_bad": int(p.num_bad),
        "expect_reductions": int(p.reductions),
    }


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(_here, "..", "sched_extra_reference.json"))

    warmup_cases = [
        make_warmup_case(
            "warmup5_start0", "start_factor=0：第 k 轮 lr = initial*k/5，之后保持 initial",
            initial_lr=1.0, warmup_epochs=5, start_factor=0.0,
            epochs=[1, 2, 3, 4, 5, 6, 7], base_lr=None),
        make_warmup_case(
            "start0.5_warmup3", "start_factor=0.5 的线性插值（含 1/3 这类非精确分数）",
            initial_lr=0.1, warmup_epochs=3, start_factor=0.5,
            epochs=[1, 2, 3, 4, 5], base_lr=None),
        make_warmup_case(
            "warmup4_with_base", "warmup 后把绝对 epoch 交给 base_scheduler（lr = base_lr*0.5^epoch）",
            initial_lr=2.0, warmup_epochs=4, start_factor=0.0,
            epochs=[1, 2, 3, 4, 5, 6, 8], base_lr=2.0),
        make_warmup_case(
            "clamp_epoch_to_1", "epoch<=0 时按 1 处理（Python: max(int(epoch),1)）",
            initial_lr=0.4, warmup_epochs=2, start_factor=0.25,
            epochs=[0, -3, 1, 2, 3], base_lr=None),
    ]

    plateau_cases = [
        make_plateau_case(
            "min_basic", "patience=2：连续 3 次没改善就乘 factor",
            initial_lr=1.0, metrics=[1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9],
            factor=0.5, patience=2),
        make_plateau_case(
            "max_mode", "mode=max：指标变大才叫改善",
            initial_lr=1.0, metrics=[0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7],
            factor=0.5, patience=1, mode="max"),
        make_plateau_case(
            "max_reductions_2", "max_reductions=2：降两次后即使继续停滞也不再降",
            initial_lr=1.0, metrics=[1.0] * 12, factor=0.5, patience=0,
            max_reductions=2),
        make_plateau_case(
            "min_lr_floor", "min_lr=0.3：lr 不会低于下限",
            initial_lr=1.0, metrics=[1.0] * 10, factor=0.5, patience=0, min_lr=0.3),
        make_plateau_case(
            "threshold_gate", "threshold=0.1：改善幅度不到 0.1 不算改善",
            initial_lr=1.0, metrics=[1.0, 0.95, 0.92, 0.9, 0.85, 0.85],
            factor=0.5, patience=1, threshold=0.1),
        make_plateau_case(
            "nan_ignored", "NaN 指标直接返回当前 lr，不改 best/num_bad（NaN 记为 nan_at 下标）",
            initial_lr=1.0, metrics=[1.0, float("nan"), 1.0, 1.0, 1.0],
            factor=0.5, patience=1),
    ]

    data = {
        "meta": {
            "generated_by": "tools/sched_extra_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": ("WarmupLR / ReduceLROnPlateau 由 AI.py 真实类驱动；"
                        "两者只用四则运算，故所有用例 tol=0 且要求逐位相同"),
        },
        "warmup_cases": warmup_cases,
        "plateau_cases": plateau_cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入 %s：%d 个 warmup 用例 + %d 个 plateau 用例"
          % (os.path.abspath(out_path), len(warmup_cases), len(plateau_cases)))
    for c in warmup_cases:
        print("  warmup %-20s epochs=%d tol=%.0e" % (c["name"], len(c["epochs"]), c["tol"]))
    for c in plateau_cases:
        print("  plateau %-19s metrics=%d tol=%.0e" % (c["name"], len(c["metrics"]), c["tol"]))


if __name__ == "__main__":
    main()
