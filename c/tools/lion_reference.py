# -*- coding: utf-8 -*-
"""生成 Lion 的跨语言对拍数据（JSON）

用法：python tools/lion_reference.py [输出路径]
      默认写到 c/lion_reference.json

数据来源是 AI.py 里真正的 `class Lion`（AI.py:17303）：本脚本造一个小 MLP、把
权重/偏置换成固定值，用 random.Random(seed) 逐 step 造梯度，跑完后导出最终权重、
偏置与一阶矩状态。C 侧 tests/compare_lion.c 读同一份 JSON 复跑，要求逐位相同。

注意：Lion 的动量存在优化器自己身上（opt.m_w / opt.m_b，AI.py:17327-17329），
不是 Layer.m_w（那是 Adam/AdamW 用的），所以导出状态要从 opt 上取。

Lion 的 step 只有四则运算与三分支符号函数，没有 pow/exp 之类的 libm 调用，
所以这里的用例全部 tol=0.0（真正的逐位比较，不是"容差很小"）。
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

from AI import MLP, Lion  # noqa: E402


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


def make_case(name, shapes, seed, steps, lr, beta1, beta2, weight_decay, tol, note,
              zero_grads=False):
    rnd = random.Random(seed)
    model = build_model(shapes, rnd)
    # 初始参数要在 step 之前抓下来（layers 里的 w/b 是给 C 侧的输入）
    w0 = [flat(layer.weights) for layer in model.layers]
    b0 = [list(layer.biases) for layer in model.layers]
    opt = Lion(model, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)

    gw_steps = [[] for _ in shapes]
    gb_steps = [[] for _ in shapes]
    for s in range(steps):
        grads = []
        for k, (fo, fi) in enumerate(shapes):
            if zero_grads:
                # 每 4 个元素留一个精确的 0.0：覆盖 sign(0)==0 的分支
                gw = [[0.0 if (s + j * fi + i) % 4 == 0 else rnd.uniform(-1.0, 1.0)
                       for i in range(fi)] for j in range(fo)]
                gb = [0.0 if (s + j) % 4 == 0 else rnd.uniform(-1.0, 1.0)
                      for j in range(fo)]
            else:
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
                "m_w": flat(opt.m_w[k]),
                "m_b": list(opt.m_b[k]),
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
            "weight_decay": weight_decay,
        },
        "layers": layers,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_here, "..", "lion_reference.json")

    shapes = [(4, 3), (2, 4)]
    cases = [
        make_case(
            "default_20steps", shapes, seed=401, steps=20,
            lr=1e-3, beta1=0.9, beta2=0.99, weight_decay=0.01,
            tol=0.0,
            note="真实超参 20 步；Lion 无 libm 调用 -> 全逐位",
        ),
        make_case(
            "zero_grads_12steps", shapes, seed=402, steps=12,
            lr=0.01, beta1=0.9, beta2=0.99, weight_decay=0.01,
            tol=0.0,
            note="梯度按固定图案插入精确 0.0，覆盖 sign(0)==0（更新量里只剩 wd*w）的分支",
            zero_grads=True,
        ),
        make_case(
            "no_decay_15steps", shapes, seed=403, steps=15,
            lr=0.05, beta1=0.5, beta2=0.5, weight_decay=0.0,
            tol=0.0,
            note="wd=0 -> upd 只有 +-1；同时覆盖 dyadic beta 的精确路径",
        ),
        make_case(
            "wd_heavy_10steps", shapes, seed=404, steps=10,
            lr=0.1, beta1=0.0, beta2=0.9, weight_decay=0.5,
            tol=0.0,
            note="beta1=0 -> c 就等于当前梯度；wd=0.5 让衰减项主导，放大顺序错误的后果",
        ),
    ]

    data = {
        "meta": {
            "generated_by": "tools/lion_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": "期望值由 AI.py 的 class Lion 真实跑出；tol=0 表示要求逐位相同",
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
