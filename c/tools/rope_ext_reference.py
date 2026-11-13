# -*- coding: utf-8 -*-
"""导出 RoPE 扩展（ABF / LongRoPE / 2D / 3D）的参考数据，供 C 实现逐位比对。

用法：python tools/rope_ext_reference.py [输出路径]
未给输出路径时写到 c/rope_ext_reference.json。

用例全部由固定种子的 random.Random 实例生成（不触碰全局 random），
因此脚本输出是确定性的：连跑两次字节一致。
"""
import json
import math
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _py_path  # noqa: E402  定位 Python 主仓库并加入 sys.path
import AI  # noqa: E402

TOL = 0.0  # 全部要求逐位相同；若某用例出现 ≤1 ULP 的 libm 差异再单独放宽

CASES = []


def add(**kw):
    CASES.append(kw)


def make_rows(rnd, n_rows, width):
    """固定种子随机行（保留 6 位小数，保证同一 JSON 文件即输入）。"""
    return [[round(rnd.uniform(-1.0, 1.0), 6) for _ in range(width)]
            for _ in range(n_rows)]


def add_error(name, op, args, call):
    """确认 Python 侧确实 raise，再记录一个 C 侧应返回 -1 的错误用例。"""
    try:
        call()
    except ValueError:
        pass
    except Exception as exc:  # pragma: no cover - 防御性检查
        raise SystemExit(f"{name}: 期望 ValueError，实际 {type(exc).__name__}: {exc}")
    else:
        raise SystemExit(f"{name}: Python 侧没有 raise，错误用例无效")
    add(kind="error", name=name, op=op, args=args)


def build_inv_freq_cases():
    for d_k in (2, 4, 8, 16, 64):
        for base in (10000.0, 500.0):
            for factor in (1.0, 4.0, 8.0):
                add(kind="abf", name=f"abf_d{d_k}_b{base:g}_f{factor:g}",
                    d_k=d_k, base=base, factor=factor,
                    out=AI.abf_inv_freqs(d_k, base=base, factor=factor), tol=TOL)
    longrope = [
        (2, 10000.0, 1.0, 1, 0.25),
        (8, 10000.0, 4.0, 512, 0.25),
        (8, 10000.0, 2.0, 512, 0.5),
        (8, 500.0, 8.0, 2048, 0.1),
        (16, 10000.0, 4.0, 512, 0.25),
        (16, 10000.0, 4.0, 2048, 0.5),
        (16, 10000.0, 1.0, 512, 0.25),
        (32, 10000.0, 4.0, 4096, 0.25),
        (32, 500.0, 2.0, 512, 0.75),
        (64, 10000.0, 8.0, 512, 0.25),
        (64, 10000.0, 4.0, 2048, 0.5),
        (6, 10000.0, 2.0, 512, 0.34),
    ]
    for d_k, base, factor, original_max, low_fraction in longrope:
        add(kind="longrope",
            name=f"longrope_d{d_k}_b{base:g}_f{factor:g}_m{original_max}_l{low_fraction:g}",
            d_k=d_k, base=base, factor=factor,
            original_max=original_max, low_fraction=low_fraction,
            out=AI.longrope_inv_freqs(d_k, base=base, factor=factor,
                                      original_max=original_max,
                                      low_fraction=low_fraction),
            tol=TOL)


def build_2d_cases():
    rnd = random.Random(20241113)
    shapes = [
        (1, 1, 4, 10000.0),
        (1, 4, 4, 10000.0),
        (4, 1, 4, 10000.0),
        (2, 3, 4, 10000.0),
        (3, 2, 8, 500.0),
        (2, 2, 8, 10000.0),
        (2, 3, 12, 10000.0),
        (1, 2, 16, 500.0),
        (3, 3, 4, 10000.0),
    ]
    for grid_h, grid_w, width, base in shapes:
        rows = make_rows(rnd, grid_h * grid_w, width)
        tag = f"2d_g{grid_h}x{grid_w}_d{width}_b{base:g}"
        add(kind="rope_2d", name=tag, rows=rows,
            grid_h=grid_h, grid_w=grid_w, base=base,
            out=AI.rope_2d(rows, grid_h, grid_w, base=base), tol=TOL)
        add(kind="rope_2d_inverse", name="inv_" + tag, rows=rows,
            grid_h=grid_h, grid_w=grid_w, base=base,
            out=AI.rope_2d_inverse(rows, grid_h, grid_w, base=base), tol=TOL)


def build_3d_cases():
    rnd = random.Random(20241114)
    shapes = [
        (1, 1, 1, 6, 10000.0),
        (2, 1, 1, 6, 10000.0),
        (1, 2, 2, 6, 10000.0),
        (2, 2, 2, 6, 10000.0),
        (3, 2, 2, 12, 500.0),
        (2, 2, 2, 18, 10000.0),
        (1, 2, 3, 24, 10000.0),
        (2, 2, 2, 6, 500.0),
    ]
    for grid_t, grid_h, grid_w, width, base in shapes:
        rows = make_rows(rnd, grid_t * grid_h * grid_w, width)
        tag = f"3d_g{grid_t}x{grid_h}x{grid_w}_d{width}_b{base:g}"
        add(kind="rope_3d", name=tag, rows=rows,
            grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, base=base,
            out=AI.rope_3d(rows, grid_t, grid_h, grid_w, base=base), tol=TOL)
        add(kind="rope_3d_inverse", name="inv_" + tag, rows=rows,
            grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, base=base,
            out=AI.rope_3d_inverse(rows, grid_t, grid_h, grid_w, base=base),
            tol=TOL)


def build_error_cases():
    rows4 = [[1.0, 2.0, 3.0, 4.0]]
    rows6 = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    # ABF
    add_error("err_abf_dk1", "abf", {"d_k": 1, "base": 10000.0, "factor": 8.0},
              lambda: AI.abf_inv_freqs(1, base=10000.0, factor=8.0))
    add_error("err_abf_dk0", "abf", {"d_k": 0, "base": 10000.0, "factor": 8.0},
              lambda: AI.abf_inv_freqs(0, base=10000.0, factor=8.0))
    add_error("err_abf_base0", "abf", {"d_k": 8, "base": 0.0, "factor": 8.0},
              lambda: AI.abf_inv_freqs(8, base=0.0, factor=8.0))
    add_error("err_abf_base_neg", "abf", {"d_k": 8, "base": -100.0, "factor": 8.0},
              lambda: AI.abf_inv_freqs(8, base=-100.0, factor=8.0))
    add_error("err_abf_factor_lt1", "abf", {"d_k": 8, "base": 10000.0, "factor": 0.5},
              lambda: AI.abf_inv_freqs(8, base=10000.0, factor=0.5))
    # LongRoPE
    add_error("err_lr_dk1", "longrope",
              {"d_k": 1, "base": 10000.0, "factor": 4.0,
               "original_max": 512, "low_fraction": 0.25},
              lambda: AI.longrope_inv_freqs(1, base=10000.0, factor=4.0,
                                            original_max=512, low_fraction=0.25))
    add_error("err_lr_max0", "longrope",
              {"d_k": 8, "base": 10000.0, "factor": 4.0,
               "original_max": 0, "low_fraction": 0.25},
              lambda: AI.longrope_inv_freqs(8, base=10000.0, factor=4.0,
                                            original_max=0, low_fraction=0.25))
    add_error("err_lr_frac0", "longrope",
              {"d_k": 8, "base": 10000.0, "factor": 4.0,
               "original_max": 512, "low_fraction": 0.0},
              lambda: AI.longrope_inv_freqs(8, base=10000.0, factor=4.0,
                                            original_max=512, low_fraction=0.0))
    add_error("err_lr_frac1", "longrope",
              {"d_k": 8, "base": 10000.0, "factor": 4.0,
               "original_max": 512, "low_fraction": 1.0},
              lambda: AI.longrope_inv_freqs(8, base=10000.0, factor=4.0,
                                            original_max=512, low_fraction=1.0))
    add_error("err_lr_frac_gt1", "longrope",
              {"d_k": 8, "base": 10000.0, "factor": 4.0,
               "original_max": 512, "low_fraction": 1.5},
              lambda: AI.longrope_inv_freqs(8, base=10000.0, factor=4.0,
                                            original_max=512, low_fraction=1.5))
    # 2D
    add_error("err_2d_rows_mismatch", "rope_2d",
              {"rows": [[1.0, 2.0, 3.0, 4.0]] * 3, "grid_h": 2, "grid_w": 2,
               "base": 10000.0},
              lambda: AI.rope_2d([[1.0, 2.0, 3.0, 4.0]] * 3, 2, 2, base=10000.0))
    add_error("err_2d_width6", "rope_2d",
              {"rows": rows6, "grid_h": 1, "grid_w": 1, "base": 10000.0},
              lambda: AI.rope_2d(rows6, 1, 1, base=10000.0))
    add_error("err_2d_gridh0", "rope_2d",
              {"rows": rows4, "grid_h": 0, "grid_w": 1, "base": 10000.0},
              lambda: AI.rope_2d(rows4, 0, 1, base=10000.0))
    add_error("err_2d_gridw0", "rope_2d",
              {"rows": rows4, "grid_h": 1, "grid_w": 0, "base": 10000.0},
              lambda: AI.rope_2d(rows4, 1, 0, base=10000.0))
    add_error("err_2d_base0", "rope_2d",
              {"rows": rows4, "grid_h": 1, "grid_w": 1, "base": 0.0},
              lambda: AI.rope_2d(rows4, 1, 1, base=0.0))
    add_error("err_2d_inv_rows_mismatch", "rope_2d_inverse",
              {"rows": rows4, "grid_h": 2, "grid_w": 2, "base": 10000.0},
              lambda: AI.rope_2d_inverse(rows4, 2, 2, base=10000.0))
    # 3D
    add_error("err_3d_rows_mismatch", "rope_3d",
              {"rows": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 2,
               "grid_t": 1, "grid_h": 1, "grid_w": 1, "base": 10000.0},
              lambda: AI.rope_3d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 2, 1, 1, 1,
                                 base=10000.0))
    add_error("err_3d_width8", "rope_3d",
              {"rows": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
               "grid_t": 1, "grid_h": 1, "grid_w": 1, "base": 10000.0},
              lambda: AI.rope_3d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
                                 1, 1, 1, base=10000.0))
    add_error("err_3d_gridt0", "rope_3d",
              {"rows": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
               "grid_t": 0, "grid_h": 1, "grid_w": 1, "base": 10000.0},
              lambda: AI.rope_3d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], 0, 1, 1,
                                 base=10000.0))
    add_error("err_3d_inv_rows_mismatch", "rope_3d_inverse",
              {"rows": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
               "grid_t": 1, "grid_h": 2, "grid_w": 2, "base": 10000.0},
              lambda: AI.rope_3d_inverse([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                                         1, 2, 2, base=10000.0))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rope_ext_reference.json")
    # 进程级固定种子：即便引用的模型/函数里用到全局 random 也可复现。
    random.seed(20261113)
    build_inv_freq_cases()
    build_2d_cases()
    build_3d_cases()
    build_error_cases()
    payload = {"cases": CASES}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_error = sum(1 for c in CASES if c["kind"] == "error")
    print(f"wrote {out_path} ({len(CASES)} cases, {n_error} error cases)")


if __name__ == "__main__":
    main()
