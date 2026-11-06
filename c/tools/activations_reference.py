# -*- coding: utf-8 -*-
"""生成 SiLU / Mish / ELU / SELU（含导数）的跨语言对拍数据

用法：python tools/activations_reference.py [输出路径]
      默认写到 c/activations_reference.json

数据来源是 AI.py 的标量函数（AI.py:82-140）：
  _silu_scalar / _dsilu_scalar / _softplus_scalar / _mish_scalar / _dmish_scalar
  _elu_scalar / _delu_scalar / _selu_scalar / _dselu_scalar

tol 的含义（按用户要求：不许静默放宽容差）：
  - 本机 C 走 msvcrt.dll 的 libm、Python 3.14 走 UCRT，实测 exp 约 0.4% 的输入
    差 1 ULP、log1p 约 15%、tanh 约 0.1%（见 c/README.md 与 ULP 对比实验）。
  - 因此「函数 + 用例」维度上标 tol：0.0 = 这批输入在 C 侧必须逐位相同；
    1e-12 = 该批输入命中了 libm 的 1 ULP 差异（note 里写清是哪个函数、
    差了多少个输入）。这是实测结果，不是预先放宽。
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

from AI import (  # noqa: E402
    _delu_scalar,
    _dmish_scalar,
    _dselu_scalar,
    _dsilu_scalar,
    _elu_scalar,
    _mish_scalar,
    _selu_scalar,
    _silu_scalar,
    _softplus_scalar,
)

FUNCS = [
    ("softplus", _softplus_scalar),
    ("silu", _silu_scalar),
    ("dsilu", _dsilu_scalar),
    ("mish", _mish_scalar),
    ("dmish", _dmish_scalar),
    ("elu", _elu_scalar),
    ("delu", _delu_scalar),
    ("selu", _selu_scalar),
    ("dselu", _dselu_scalar),
]

# 每个 (用例, 函数) 的容差实测表：默认 0.0（要求逐位），命中 libm 1 ULP 的
# 用例按实测结果改成 1e-12 并给出 note。下面是 2026-11-06 在
#   gcc 16.1.0 (MinGW-W64, msvcrt.dll)  ×  CPython 3.14.6 (UCRT)
# 上跑 tests/compare_activations 的实测结论（逐位相同的个数 / 总数）：
#   boundary_grid : softplus 19/20、dsilu 19/20、mish 19/20、dmish 19/20、
#                   elu 19/20、delu 19/20、selu 19/20、dselu 19/20 —— 全部差在
#                   z = -1e-8；silu 20/20。
#   random_uniform: softplus 81/96、mish 85/96、dmish 89/96；其余 96/96。
#   large_magnitude: 全部 6/6（±700/±800 只涉及上溢/下溢，两边一致）。
# 最大值 2.220e-16（绝对）。注意 elu/selu 在 z≈0 时用 exp(z)-1 有相消，
# 1 ULP 的 exp 误差会被放大成 ~1e-16 的绝对误差（相对误差 ~1e-8），
# 所以这里按**绝对**容差 1e-12 判定，而不是相对容差。
TOL_OVERRIDE = {}
NOTE_OVERRIDE = {}

_LIBM_NOTE = ("msvcrt 与 UCRT 的 %s 在 z=%s 上差 1 ULP："
              "本机实测该用例逐位 %d/%d；绝对差最大 %.1e。"
              "容差因此设为 1e-12（绝对），不是静默放宽。")

_WHY = {
    "softplus": "log1p+exp 组合",
    "dsilu": "exp",
    "mish": "log1p+tanh 组合",
    "dmish": "log1p+tanh 组合",
    "elu": "exp(z)-1 在 z≈0 附近相消",
    "delu": "exp",
    "selu": "exp(z)-1 在 z≈0 附近相消",
    "dselu": "exp",
}

for _f in ("softplus", "dsilu", "mish", "dmish", "elu", "delu", "selu", "dselu"):
    TOL_OVERRIDE[("boundary_grid", _f)] = 1e-12
    NOTE_OVERRIDE[("boundary_grid", _f)] = _LIBM_NOTE % (_WHY[_f], "-1e-8", 19, 20, 2.3e-16)

for _f, _n in (("softplus", 81), ("mish", 85), ("dmish", 89)):
    TOL_OVERRIDE[("random_uniform", _f)] = 1e-12
    NOTE_OVERRIDE[("random_uniform", _f)] = _LIBM_NOTE % (_WHY[_f], "若干随机点", _n, 96, 2.3e-16)

Z_GRID = [
    -50.0, -30.0, -10.0, -3.5, -1.0, -0.5, -0.1, -1e-8,
    0.0, 1e-8, 0.1, 0.5, 1.0, 2.0, 3.5, 10.0, 29.9, 30.0, 30.1, 50.0,
]

Z_LARGE = [-800.0, -700.0, -50.0, 50.0, 700.0, 800.0]


def make_case(name, note, z, seed=None):
    rnd = random.Random(seed) if seed is not None else None
    _ = rnd
    expect = {}
    tols = {}
    notes = {}
    for fname, fn in FUNCS:
        expect[fname] = [fn(v) for v in z]
        tol = TOL_OVERRIDE.get((name, fname), 0.0)
        tols[fname] = tol
        if (name, fname) in NOTE_OVERRIDE:
            notes[fname] = NOTE_OVERRIDE[(name, fname)]
    return {
        "name": name,
        "note": note,
        "z": z,
        "tol": tols,
        "func_notes": notes,
        "expect": expect,
    }


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(_here, "..", "activations_reference.json"))

    rnd = random.Random(20241106)
    z_random = [round(rnd.uniform(-6.0, 6.0), 17) for _ in range(64)]
    # 再补一批不同量级的随机值（含 |z| 较大的负数，测 exp 下溢）
    z_random += [round(rnd.uniform(-80.0, 80.0), 17) for _ in range(32)]

    cases = [
        make_case("boundary_grid",
                  "分段边界：z=0（silu 走 >= 分支、elu/selu 走 exp 分支）、"
                  "z=±1e-8、softplus 的 z=30 阈值", Z_GRID),
        make_case("random_uniform",
                  "64 个 [-6,6] 均匀随机 + 32 个 [-80,80] 随机（含 exp 下溢）", z_random),
        make_case("large_magnitude",
                  "±700/±800：exp 上溢为 inf、下溢为 0 的行为要与 Python 一致", Z_LARGE),
    ]

    data = {
        "meta": {
            "generated_by": "tools/activations_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": ("tol 按 (用例, 函数) 给出：0.0 = 必须逐位相同；"
                        "1e-12 = 已实测命中 msvcrt vs UCRT 的 1 ULP libm 差异"),
        },
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入 %s：%d 个用例" % (os.path.abspath(out_path), len(cases)))
    for c in cases:
        print("  %-16s z=%d" % (c["name"], len(c["z"])))


if __name__ == "__main__":
    main()
