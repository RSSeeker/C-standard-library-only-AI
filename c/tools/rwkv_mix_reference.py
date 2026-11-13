# -*- coding: utf-8 -*-
"""导出 RWKV-5/6/7 time-mixing 的参考数据，供 C 实现逐位比对。

覆盖 AI.py 的：
  rwkv5_time_mixing / rwkv6_time_mixing / rwkv7_time_mixing

用法：python tools/rwkv_mix_reference.py [输出路径]
未给输出路径时写到 c/rwkv_mix_reference.json。

用例全部由固定种子的 random.Random 实例生成（不触碰全局 random），
因此脚本输出是确定性的：连跑两次字节一致。
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _py_path  # noqa: E402  定位 Python 主仓库并加入 sys.path
import AI  # noqa: E402

TOL = 0.0  # rwkv5 只含乘加，要求逐位相同
# RWKV-6/7 的 alpha = exp(-exp(w)) 走 libm；MinGW libm 与 CPython UCRT 的
# exp 存在 ≤1 ULP 差异，故这两组用例按 1e-12 容差对拍（其余仍逐位相同）。
EXP_TOL = 1e-12
EXP_NOTE = ("alpha=exp(-exp(w)) 走 libm；MinGW libm 与 CPython UCRT exp "
            "存在 ≤1 ULP 差异，故容差 1e-12")
CASES = []
RND = random.Random(20261113)

# 显式声明每个错误用例：C 侧对应函数应返回 -1。
ERRORS = []


def add(**kw):
    CASES.append(kw)


def make_rows(n_rows, width):
    """固定种子随机行（保留 6 位小数，保证 JSON 即输入）。"""
    return [[round(RND.uniform(-1.0, 1.0), 6) for _ in range(width)]
            for _ in range(n_rows)]


def make_seq(n, low=-1.0, high=1.0):
    return [round(RND.uniform(low, high), 6) for _ in range(n)]


def make_w_logits(t_len, dk, low=-2.0, high=2.0):
    return [[round(RND.uniform(low, high), 6) for _ in range(dk)]
            for _ in range(t_len)]


def zero_some_rows(rows, picks):
    for r, c in picks:
        rows[r][c] = 0.0
    return rows


def add_error(name, op, args, call):
    """确认 Python 侧确实 raise ValueError，再记录 C 侧应返回 -1 的用例。"""
    try:
        call()
    except ValueError:
        pass
    except Exception as exc:  # pragma: no cover - 防御性检查
        raise SystemExit(f"{name}: 期望 ValueError，实际 {type(exc).__name__}: {exc}")
    else:
        raise SystemExit(f"{name}: Python 侧没有 raise，错误用例无效")
    ERRORS.append({"kind": "error", "name": name, "op": op, "args": args})


def build_rwkv5_cases():
    specs = [
        ("rwkv5_basic", 5, 3, 2, 0.8),
        ("rwkv5_decay_zero", 4, 3, 2, 0.0),
        ("rwkv5_decay_one", 4, 2, 3, 1.0),
        ("rwkv5_t1", 1, 3, 4, 0.5),
        ("rwkv5_dv1", 5, 4, 1, 0.7),
        ("rwkv5_dk1", 4, 1, 3, 0.9),
        ("rwkv5_t10", 10, 4, 3, 0.95),
        ("rwkv5_wide", 7, 5, 2, 0.6),
    ]
    for name, t_len, dk, dv, decay in specs:
        r = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        add(kind="rwkv5", name=name, tol=TOL, r=r, k=k, v=v, decay=decay,
            out=AI.rwkv5_time_mixing(r, k, v, decay))

    # 逐 channel 衰减向量（Python 的 decay 为 list 分支）。
    r = make_rows(5, 3)
    k = make_rows(5, 3)
    v = make_rows(5, 2)
    decay_vec = [0.0, 0.5, 1.0]
    add(kind="rwkv5", name="rwkv5_decay_vec", tol=TOL, r=r, k=k, v=v,
        decay_vec=decay_vec, out=AI.rwkv5_time_mixing(r, k, v, decay_vec))

    # r 的部分分量为 0：验证 Python 的 `if ri == 0.0: continue`。
    r = zero_some_rows(make_rows(5, 3), [(1, 0), (3, 2), (4, 1)])
    k = make_rows(5, 3)
    v = make_rows(5, 2)
    add(kind="rwkv5", name="rwkv5_r_zero", tol=TOL, r=r, k=k, v=v, decay=0.5,
        out=AI.rwkv5_time_mixing(r, k, v, 0.5))


def build_rwkv6_cases():
    specs = [
        ("rwkv6_basic", 4, 3, 2),
        ("rwkv6_t1", 1, 3, 4),
        ("rwkv6_dv1", 5, 4, 1),
        ("rwkv6_dk1", 4, 1, 3),
        ("rwkv6_t10", 10, 4, 3),
        ("rwkv6_wide", 6, 5, 2),
    ]
    for name, t_len, dk, dv in specs:
        r = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        w = make_w_logits(t_len, dk)
        add(kind="rwkv6", name=name, tol=EXP_TOL, note=EXP_NOTE, r=r, k=k, v=v, w_logits=w,
            out=AI.rwkv6_time_mixing(r, k, v, w))

    # 大幅/负 w：alpha = exp(-exp(w)) 的两端。
    r = make_rows(5, 2)
    k = make_rows(5, 2)
    v = make_rows(5, 3)
    w = [[-6.0, 6.0], [-2.5, 2.5], [0.0, 1.0], [-1.0, 5.0], [4.0, -4.0]]
    add(kind="rwkv6", name="rwkv6_w_wide", tol=EXP_TOL, note=EXP_NOTE, r=r, k=k, v=v, w_logits=w,
        out=AI.rwkv6_time_mixing(r, k, v, w))

    # r 部分分量为 0。
    r = zero_some_rows(make_rows(5, 3), [(0, 1), (2, 0), (4, 2)])
    k = make_rows(5, 3)
    v = make_rows(5, 2)
    w = make_w_logits(5, 3)
    add(kind="rwkv6", name="rwkv6_r_zero", tol=EXP_TOL, note=EXP_NOTE, r=r, k=k, v=v, w_logits=w,
        out=AI.rwkv6_time_mixing(r, k, v, w))


def build_rwkv7_cases():
    specs = [
        ("rwkv7_basic", 4, 3, 2, 1.0),
        ("rwkv7_delta_zero", 4, 3, 2, 0.0),
        ("rwkv7_delta_half", 5, 3, 2, 0.5),
        ("rwkv7_delta_two", 4, 2, 3, 2.0),
        ("rwkv7_t1", 1, 3, 4, 1.0),
        ("rwkv7_dv1", 5, 4, 1, 0.7),
        ("rwkv7_dk1", 4, 1, 3, 1.0),
        ("rwkv7_t10", 10, 4, 3, 0.9),
    ]
    for name, t_len, dk, dv, delta in specs:
        r = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        w = make_w_logits(t_len, dk)
        add(kind="rwkv7", name=name, tol=EXP_TOL, note=EXP_NOTE, r=r, k=k, v=v, w_logits=w,
            delta=delta, out=AI.rwkv7_time_mixing(r, k, v, w, delta=delta))

    # k 的部分分量为 0：验证 sk 与状态更新的 `if ki == 0.0: continue`。
    r = make_rows(5, 3)
    k = zero_some_rows(make_rows(5, 3), [(1, 0), (2, 2), (4, 1)])
    v = make_rows(5, 2)
    w = make_w_logits(5, 3)
    add(kind="rwkv7", name="rwkv7_k_zero", tol=EXP_TOL, note=EXP_NOTE, r=r, k=k, v=v, w_logits=w,
        delta=1.0, out=AI.rwkv7_time_mixing(r, k, v, w, delta=1.0))


def build_error_cases():
    r, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_rwkv5_decay_neg", "rwkv5",
              {"r": r, "k": k, "v": v, "decay": -0.1},
              lambda: AI.rwkv5_time_mixing(r, k, v, -0.1))
    add_error("err_rwkv5_decay_gt1", "rwkv5",
              {"r": r, "k": k, "v": v, "decay": 1.5},
              lambda: AI.rwkv5_time_mixing(r, k, v, 1.5))
    add_error("err_rwkv5_decay_vec", "rwkv5",
              {"r": r, "k": k, "v": v, "decay_vec": [0.5, 1.5]},
              lambda: AI.rwkv5_time_mixing(r, k, v, [0.5, 1.5]))
    add_error("err_rwkv5_empty", "rwkv5",
              {"r": [], "k": [], "v": [], "decay": 0.5},
              lambda: AI.rwkv5_time_mixing([], [], [], 0.5))

    add_error("err_rwkv6_empty", "rwkv6",
              {"r": [], "k": [], "v": [], "w_logits": []},
              lambda: AI.rwkv6_time_mixing([], [], [], []))

    add_error("err_rwkv7_delta_neg", "rwkv7",
              {"r": r, "k": k, "v": v, "w_logits": make_w_logits(2, 2),
               "delta": -1.0},
              lambda: AI.rwkv7_time_mixing(
                  r, k, v, make_w_logits(2, 2), delta=-1.0))
    add_error("err_rwkv7_empty", "rwkv7",
              {"r": [], "k": [], "v": [], "w_logits": [], "delta": 1.0},
              lambda: AI.rwkv7_time_mixing([], [], [], [], delta=1.0))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "rwkv_mix_reference.json")
    # 进程级固定种子：即便引用的函数里用到全局 random 也可复现。
    random.seed(20261113)
    build_rwkv5_cases()
    build_rwkv6_cases()
    build_rwkv7_cases()
    build_error_cases()
    payload = {"cases": CASES + ERRORS}
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_err = sum(1 for c in CASES + ERRORS if c["kind"] == "error")
    print(f"wrote {out_path} ({len(CASES) + len(ERRORS)} cases, "
          f"{n_err} error cases)")
    # 检查参考数值不含 NaN/Inf（JSON 无法表示，C 侧也只对拍有限值）。
    for case in CASES:
        for key in ("out", "r", "k", "v", "w_logits"):
            val = case.get(key)
            if isinstance(val, list):
                for row in (val if val and isinstance(val[0], list) else [val]):
                    for x in row:
                        if isinstance(x, float) and (
                                x != x or x in (float("inf"), float("-inf"))):
                            raise SystemExit(f"{case['name']}.{key} 含非有限值")


if __name__ == "__main__":
    main()
