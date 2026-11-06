# -*- coding: utf-8 -*-
"""生成 AdamW 的跨语言对拍数据（JSON）

用法：python tools/adamw_reference.py [输出路径]
      默认写到 c/adamw_reference.json

数据来源是 AI.py 里真正的 `class AdamW`（不是重新推导一遍公式）：本脚本造一个小
MLP、把权重/偏置换成固定值，用 random.Random(seed) 逐 step 造梯度，跑完后导出
最终权重、偏置以及 m/v 状态。C 侧 tests/compare_adamw.c 读同一份 JSON 复跑，
默认要求逐位相同（差值 == 0）。

关于容差：每个用例带 "tol" 与 "note"。tol=0.0 表示要求逐位相同；tol=1e-12 表示
该用例已知会受 libm 差异影响（本机 CPython 3.x 用 UCRT、MinGW gcc 链接 msvcrt.dll，
两者的 pow 在个别指数上差 1 ULP，实测 pow(0.9,k) 从 k=4 起、pow(0.999,k) 从 k=6 起
不再逐位相同）。这不是"为了让测试变绿"而放宽——JSON 与 compare 工具都会把容差和
最大差异打印出来。
"""
import json
import math
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main")
)
if not os.path.isdir(AI_PY_DIR):
    sys.exit("找不到 Python 参考实现目录: %s（可用环境变量 AI_PY_DIR 指定）" % AI_PY_DIR)
sys.path.insert(0, AI_PY_DIR)

from AI import MLP, AdamW  # noqa: E402


def flat(rows):
    """嵌套 list -> 行主序扁平 list"""
    return [x for row in rows for x in row]


def build_model(shapes, rnd):
    """按 [(fan_out, fan_in), ...] 造 MLP，并把权重/偏置换成固定值"""
    sizes = [shapes[0][1]] + [fo for fo, _ in shapes]
    model = MLP(sizes, ["linear"] * len(shapes))
    for layer, (fo, fi) in zip(model.layers, shapes):
        layer.weights = [[rnd.uniform(-0.5, 0.5) for _ in range(fi)] for _ in range(fo)]
        layer.biases = [rnd.uniform(-0.1, 0.1) for _ in range(fo)]
    return model


def make_case(name, shapes, seed, steps, lr, beta1, beta2, eps, weight_decay, tol, note):
    rnd = random.Random(seed)
    model = build_model(shapes, rnd)
    # 初始参数要在 step 之前抓下来（后面 layers 里的 w/b 是给 C 侧的输入，
    # expect 里的是跑完之后的期望值）
    w0 = [flat(layer.weights) for layer in model.layers]
    b0 = [list(layer.biases) for layer in model.layers]
    opt = AdamW(model, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)

    # 每层每步的梯度都先记下来：C 侧要复现同一串梯度
    gw_steps = [[] for _ in shapes]
    gb_steps = [[] for _ in shapes]
    for _ in range(steps):
        grads = []
        for k, (fo, fi) in enumerate(shapes):
            gw = [[rnd.uniform(-1.0, 1.0) for _ in range(fi)] for _ in range(fo)]
            gb = [rnd.uniform(-1.0, 1.0) for _ in range(fo)]
            gw_steps[k].append(flat(gw))
            gb_steps[k].append(gb)
            grads.append((gw, gb))
        opt.step(grads)

    layers = []
    for k, (fo, fi) in enumerate(shapes):
        layer = model.layers[k]
        layers.append({
            "fan_out": fo,
            "fan_in": fi,
            "w": w0[k],
            "b": b0[k],
            "gw_steps": gw_steps[k],
            "gb_steps": gb_steps[k],
            "expect": {
                "w": flat(layer.weights),
                "b": list(layer.biases),
                "m_w": flat(layer.m_w),
                "v_w": flat(layer.v_w),
                "m_b": list(layer.m_b),
                "v_b": list(layer.v_b),
            },
        })

    return {
        "name": name,
        "note": note,
        "tol": tol,
        "seed": seed,
        "steps": steps,
        "step_count": opt.step_count,
        "hyper": {
            "lr": lr, "beta1": beta1, "beta2": beta2,
            "eps": eps, "weight_decay": weight_decay,
        },
        "layers": layers,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "..", "adamw_reference.json")

    shapes = [(4, 3), (2, 4)]
    cases = [
        make_case(
            "dyadic_12steps", shapes, seed=101, steps=12,
            lr=0.05, beta1=0.5, beta2=0.75, eps=1e-8, weight_decay=0.01,
            tol=0.0,
            note="beta1/beta2 取二进制精确值（0.5/0.75），pow 结果精确 -> 要求逐位相同",
        ),
        make_case(
            "default_betas_3steps", shapes, seed=202, steps=3,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
            tol=0.0,
            note="真实超参；pow(0.9,k)/pow(0.999,k) 在 k<=3 时 msvcrt 与 UCRT 逐位一致",
        ),
        make_case(
            "default_betas_10steps", shapes, seed=202, steps=10,
            lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
            tol=1e-12,
            note="真实超参 10 步：pow(0.9,k) 自 k=4、pow(0.999,k) 自 k=6 起差 1 ULP（msvcrt vs UCRT），容差 1e-12",
        ),
        make_case(
            "no_decay_8steps", shapes, seed=303, steps=8,
            lr=0.1, beta1=0.5, beta2=0.75, eps=1e-8, weight_decay=0.0,
            tol=0.0,
            note="wd=0 -> decay==1.0；覆盖 *= decay 的精确路径",
        ),
    ]

    data = {
        "meta": {
            "generated_by": "tools/adamw_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": "期望值由 AI.py 的 class AdamW 真实跑出；tol=0 表示要求逐位相同",
        },
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入 %s：%d 个用例" % (os.path.abspath(out_path), len(cases)))
    for c in cases:
        print("  %-22s steps=%-3d tol=%.0e  %s" % (c["name"], c["steps"], c["tol"],
                                                   "逐位" if c["tol"] == 0.0 else "1e-12"))


if __name__ == "__main__":
    main()
