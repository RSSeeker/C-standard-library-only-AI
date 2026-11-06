# -*- coding: utf-8 -*-
"""生成 Adafactor 的跨语言对拍数据（JSON）

用法：python tools/adafactor_reference.py [输出路径]
      默认写到 c/adafactor_reference.json

数据来源是 AI.py 里真正的 `class Adafactor`（AI.py:17921）：造一个小 MLP、把
权重/偏置换成固定值，用 random.Random(seed) 逐 step 造梯度，跑完后导出最终权重、
偏置与行/列二阶矩 R/C、偏置二阶矩 V_b。C 侧 tests/compare_adafactor.c 读同一份
JSON 复跑，要求逐位相同。

容差：step 里唯一的 libm 调用是 pow(t, -0.8)（还有 sqrt，硬件指令逐位一致）。
实测本机 msvcrt 与 UCRT 在 t=1..1000 只有 t=278 一处差 1 ULP，所以步数
控制在 12 以内的用例是 tol=0.0 的严格逐位比较。

另外几个用例专门打分支：
  - grad_scale=1e-4：rms(g) < eps2 -> lr_t = lr*eps2 的分支
  - eps2=10.0：恒走 max(eps2, rms) 的 eps2 一侧
  - 全零梯度：R/C 全 0 -> mean_R<=0 -> 取 1.0 的分支
  - wd=0.5：检验"权重衰减作用在更新前参数上"的单表达式顺序
"""
import json
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

from AI import MLP, Adafactor  # noqa: E402


def flat(rows):
    """嵌套 list -> 行主序扁平 list"""
    return [x for row in rows for x in row]


def build_model(shapes, rnd):
    sizes = [shapes[0][1]] + [fo for fo, _ in shapes]
    model = MLP(sizes, ["linear"] * len(shapes))
    for layer, (fo, fi) in zip(model.layers, shapes):
        layer.weights = [[rnd.uniform(-0.5, 0.5) for _ in range(fi)] for _ in range(fo)]
        layer.biases = [rnd.uniform(-0.1, 0.1) for _ in range(fo)]
    return model


def make_case(name, shapes, seed, steps, lr, eps, eps2, beta2, weight_decay, tol, note,
              grad_scale=1.0, zero_grads=False):
    rnd = random.Random(seed)
    model = build_model(shapes, rnd)
    w0 = [flat(layer.weights) for layer in model.layers]
    b0 = [list(layer.biases) for layer in model.layers]
    opt = Adafactor(model, lr=lr, eps=eps, eps2=eps2, beta2=beta2,
                    weight_decay=weight_decay)

    gw_steps = [[] for _ in shapes]
    gb_steps = [[] for _ in shapes]
    for _ in range(steps):
        grads = []
        for k, (fo, fi) in enumerate(shapes):
            if zero_grads:
                gw = [[0.0] * fi for _ in range(fo)]
                gb = [0.0] * fo
            else:
                gw = [[rnd.uniform(-1.0, 1.0) * grad_scale for _ in range(fi)]
                      for _ in range(fo)]
                gb = [rnd.uniform(-1.0, 1.0) * grad_scale for _ in range(fo)]
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
                "R": list(opt.R[k]),
                "C": list(opt.C[k]),
                "V_b": list(opt.V_b[k]),
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
            "lr": lr, "eps": eps, "eps2": eps2,
            "beta2": beta2, "weight_decay": weight_decay,
        },
        "layers": layers,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "..", "adafactor_reference.json")

    shapes = [(4, 3), (2, 4)]
    cases = [
        make_case(
            "default_12steps", shapes, seed=501, steps=12,
            lr=1e-2, eps=1e-30, eps2=1e-3, beta2=0.999, weight_decay=0.01,
            tol=0.0,
            note="默认超参 12 步；只有 pow(t,-0.8) 与 sqrt，t<=12 实测逐位一致",
        ),
        make_case(
            "tiny_grads_10steps", shapes, seed=502, steps=10,
            lr=0.5, eps=1e-30, eps2=1e-3, beta2=0.999, weight_decay=0.0,
            tol=0.0,
            note="梯度缩放 1e-4 -> rms(g) 远小于 eps2，走 lr_t = lr*eps2 的分支",
            grad_scale=1e-4,
        ),
        make_case(
            "huge_eps2_6steps", shapes, seed=503, steps=6,
            lr=1e-2, eps=1e-30, eps2=10.0, beta2=0.9, weight_decay=0.01,
            tol=0.0,
            note="eps2=10 恒大于 rms(g) -> max() 恒取 eps2 一侧（lr_t = lr*10）",
        ),
        make_case(
            "wd_heavy_8steps", shapes, seed=504, steps=8,
            lr=0.1, eps=1e-30, eps2=1e-3, beta2=0.999, weight_decay=0.5,
            tol=0.0,
            note="wd=0.5：检验单表达式里 wd 乘的是更新前的 W（顺序错会差一个 lr*wd 项）",
        ),
        make_case(
            "zero_grads_5steps", shapes, seed=505, steps=5,
            lr=0.1, eps=1e-30, eps2=1e-3, beta2=0.999, weight_decay=0.1,
            tol=0.0,
            note="梯度恒为 0：R/C 保持 0 -> mean_R<=0 取 1.0 的分支；参数只受 wd 影响",
            zero_grads=True,
        ),
    ]

    data = {
        "meta": {
            "generated_by": "tools/adafactor_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": "期望值由 AI.py 的 class Adafactor 真实跑出；tol=0 表示要求逐位相同",
        },
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入 %s：%d 个用例" % (os.path.abspath(out_path), len(cases)))
    for c in cases:
        print("  %-22s steps=%-3d tol=%.0e" % (c["name"], c["steps"], c["tol"]))


if __name__ == "__main__":
    main()
