# -*- coding: utf-8 -*-
"""导出线性注意力 / 长卷积（linear_attn）的参考数据，供 C 实现逐位比对。

覆盖 AI.py 的：
  linear_attention_naive / lightning_attention / gla_forward /
  deltanet_forward / hyena_long_conv / hyena_forward

用法：python tools/linear_attn_reference.py [输出路径]
未给输出路径时写到 c/linear_attn_reference.json。

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

TOL = 0.0  # 全部要求逐位相同；naive 的 pow 用例按 ≤2 ULP 放宽到 1e-12
NAIVE_TOL = 1e-12
POW_NOTE = ("decay**(t-j) 走 libm pow；MinGW libm 与 CPython UCRT pow "
            "存在 ≤2 ULP 差异，故容差 1e-12")

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


def zero_some_rows(rows, picks):
    for r, c in picks:
        rows[r][c] = 0.0
    return rows


def build_naive_cases():
    specs = [
        ("naive_basic", 4, 3, 2, 1.0),
        ("naive_decay", 4, 3, 2, 0.5),
        ("naive_decay_zero", 3, 2, 2, 0.0),
        ("naive_t1", 1, 3, 4, 0.9),
        ("naive_wide", 6, 2, 5, 1.0),
        ("naive_dv1", 5, 4, 1, 0.7),
        ("naive_dk1", 5, 1, 3, 0.8),
        ("naive_t10", 10, 4, 3, 0.95),
    ]
    for name, t_len, dk, dv, decay in specs:
        q = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        add(kind="naive", name=name, tol=NAIVE_TOL, note=POW_NOTE, q=q, k=k,
            v=v, decay=decay,
            out=AI.linear_attention_naive(q, k, v, decay=decay))


def build_lightning_cases():
    specs = [
        ("lightning_basic", 4, 3, 2, 1.0),
        ("lightning_decay", 5, 3, 2, 0.5),
        ("lightning_decay_zero", 4, 2, 3, 0.0),
        ("lightning_t1", 1, 3, 2, 1.0),
        ("lightning_wide", 6, 2, 4, 0.9),
        ("lightning_dv1", 5, 4, 1, 0.7),
        ("lightning_dk1", 4, 1, 3, 0.8),
    ]
    for name, t_len, dk, dv, decay in specs:
        q = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        add(kind="lightning", name=name, tol=TOL, q=q, k=k, v=v, decay=decay,
            out=AI.lightning_attention(q, k, v, decay=decay))

    # q 的部分分量为 0：验证 Python 的 `if qi == 0.0: continue`。
    q = zero_some_rows(make_rows(5, 3), [(1, 0), (3, 2), (4, 1)])
    k = make_rows(5, 3)
    v = make_rows(5, 2)
    add(kind="lightning", name="lightning_q_zero", tol=TOL, q=q, k=k, v=v,
        decay=0.5, out=AI.lightning_attention(q, k, v, decay=0.5))


def build_gla_cases():
    specs = [
        ("gla_basic", 4, 3, 2),
        ("gla_t1", 1, 3, 4),
        ("gla_dv1", 5, 4, 1),
        ("gla_dk1", 4, 1, 3),
    ]
    for name, t_len, dk, dv in specs:
        q = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        gates = make_seq(t_len, -2.0, 2.0)
        add(kind="gla", name=name, tol=TOL, q=q, k=k, v=v, gates=gates,
            out=AI.gla_forward(q, k, v, gates))

    # 负门控 / 大幅门控：alpha = exp(-exp(g)) 的取值范围。
    q = make_rows(5, 2)
    k = make_rows(5, 2)
    v = make_rows(5, 3)
    gates = [round(RND.uniform(-8.0, 8.0), 6) for _ in range(5)]
    add(kind="gla", name="gla_gates_wide", tol=TOL, q=q, k=k, v=v, gates=gates,
        out=AI.gla_forward(q, k, v, gates))

    # q 部分分量为 0。
    q = zero_some_rows(make_rows(5, 3), [(0, 1), (2, 0), (4, 2)])
    k = make_rows(5, 3)
    v = make_rows(5, 2)
    gates = make_seq(5, -1.0, 1.0)
    add(kind="gla", name="gla_q_zero", tol=TOL, q=q, k=k, v=v, gates=gates,
        out=AI.gla_forward(q, k, v, gates))


def build_deltanet_cases():
    specs = [
        ("deltanet_basic", 3, 2, 2),
        ("deltanet_t1", 1, 3, 4),
        ("deltanet_dv1", 5, 4, 1),
        ("deltanet_dk1", 4, 1, 3),
    ]
    for name, t_len, dk, dv in specs:
        q = make_rows(t_len, dk)
        k = make_rows(t_len, dk)
        v = make_rows(t_len, dv)
        betas = [round(RND.uniform(0.0, 1.0), 6) for _ in range(t_len)]
        add(kind="deltanet", name=name, tol=TOL, q=q, k=k, v=v, betas=betas,
            out=AI.deltanet_forward(q, k, v, betas))

    q = make_rows(4, 3)
    k = make_rows(4, 3)
    v = make_rows(4, 2)
    betas = [0.0, 1.0, 0.25, 0.75]
    add(kind="deltanet", name="deltanet_beta_bounds", tol=TOL, q=q, k=k, v=v,
        betas=betas, out=AI.deltanet_forward(q, k, v, betas))

    # k 的部分分量为 0：验证 Python 的 `if ki == 0.0: continue`。
    q = make_rows(5, 3)
    k = zero_some_rows(make_rows(5, 3), [(1, 0), (2, 2), (4, 1)])
    v = make_rows(5, 2)
    betas = [0.2, 0.4, 0.6, 0.8, 0.5]
    add(kind="deltanet", name="deltanet_k_zero", tol=TOL, q=q, k=k, v=v,
        betas=betas, out=AI.deltanet_forward(q, k, v, betas))


def build_hyena_cases():
    specs = [
        ("hyena_conv_basic", 5, 3, 3),
        ("hyena_conv_t1", 1, 3, 3),
        ("hyena_conv_k_gt_t", 4, 2, 7),
        ("hyena_conv_k1", 5, 3, 1),
        ("hyena_conv_t10", 10, 4, 4),
    ]
    for name, t_len, d, k_len in specs:
        u = make_rows(t_len, d)
        kernel = make_seq(k_len)
        add(kind="hyena_conv", name=name, tol=TOL, u=u, kernel=kernel,
            out=AI.hyena_long_conv(u, kernel))

    u = make_rows(4, 3)
    kernel = [0.0, 0.0, 0.0]
    add(kind="hyena_conv", name="hyena_conv_zero_kernel", tol=TOL, u=u,
        kernel=kernel, out=AI.hyena_long_conv(u, kernel))

    specs = [
        ("hyena_basic", 4, 3, 3),
        ("hyena_t1", 1, 3, 3),
        ("hyena_k_gt_t", 3, 2, 5),
        ("hyena_k1", 5, 3, 1),
        ("hyena_t6", 6, 4, 3),
    ]
    for name, t_len, d, k_len in specs:
        u = make_rows(t_len, d)
        v = make_rows(t_len, d)
        kernel = make_seq(k_len)
        add(kind="hyena", name=name, tol=TOL, u=u, v=v, kernel=kernel,
            out=AI.hyena_forward(u, v, kernel))

    u = make_rows(4, 3)
    v = make_rows(4, 3)
    kernel = [0.0, 0.0, 0.0]
    add(kind="hyena", name="hyena_zero_kernel", tol=TOL, u=u, v=v,
        kernel=kernel, out=AI.hyena_forward(u, v, kernel))


def build_error_cases():
    # 空序列（t_len = 0）与 decay 越界。
    add_error("err_naive_empty", "naive",
              {"q": [], "k": [], "v": [], "decay": 0.5},
              lambda: AI.linear_attention_naive([], [], [], 0.5))
    q, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_naive_decay_neg", "naive",
              {"q": q, "k": k, "v": v, "decay": -0.1},
              lambda: AI.linear_attention_naive(q, k, v, -0.1))
    q, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_naive_decay_gt1", "naive",
              {"q": q, "k": k, "v": v, "decay": 1.5},
              lambda: AI.linear_attention_naive(q, k, v, 1.5))

    add_error("err_lightning_empty", "lightning",
              {"q": [], "k": [], "v": [], "decay": 0.5},
              lambda: AI.lightning_attention([], [], [], 0.5))
    q, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_lightning_decay_neg", "lightning",
              {"q": q, "k": k, "v": v, "decay": -0.25},
              lambda: AI.lightning_attention(q, k, v, -0.25))

    add_error("err_gla_empty", "gla",
              {"q": [], "k": [], "v": [], "gates": []},
              lambda: AI.gla_forward([], [], [], []))

    add_error("err_deltanet_empty", "deltanet",
              {"q": [], "k": [], "v": [], "betas": []},
              lambda: AI.deltanet_forward([], [], [], []))
    q, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_deltanet_beta_gt1", "deltanet",
              {"q": q, "k": k, "v": v, "betas": [0.5, 1.5]},
              lambda: AI.deltanet_forward(q, k, v, [0.5, 1.5]))
    q, k, v = make_rows(2, 2), make_rows(2, 2), make_rows(2, 2)
    add_error("err_deltanet_beta_neg", "deltanet",
              {"q": q, "k": k, "v": v, "betas": [-0.1, 0.5]},
              lambda: AI.deltanet_forward(q, k, v, [-0.1, 0.5]))

    add_error("err_hyena_conv_empty", "hyena_conv",
              {"u": [], "kernel": [0.5]},
              lambda: AI.hyena_long_conv([], [0.5]))
    u = make_rows(2, 2)
    add_error("err_hyena_conv_kernel_empty", "hyena_conv",
              {"u": u, "kernel": []},
              lambda: AI.hyena_long_conv(u, []))

    add_error("err_hyena_empty", "hyena",
              {"u": [], "v": [], "kernel": [0.5]},
              lambda: AI.hyena_forward([], [], [0.5]))
    u, v = make_rows(2, 2), make_rows(2, 2)
    add_error("err_hyena_kernel_empty", "hyena",
              {"u": u, "v": v, "kernel": []},
              lambda: AI.hyena_forward(u, v, []))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "linear_attn_reference.json")
    # 进程级固定种子：即便引用的函数里用到全局 random 也可复现。
    random.seed(20261113)
    build_naive_cases()
    build_lightning_cases()
    build_gla_cases()
    build_deltanet_cases()
    build_hyena_cases()
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
        for key in ("out", "q", "k", "v", "u", "gates", "betas"):
            val = case.get(key)
            if isinstance(val, list):
                for row in (val if val and isinstance(val[0], list) else [val]):
                    for x in row:
                        if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
                            raise SystemExit(f"{case['name']}.{key} 含非有限值")


if __name__ == "__main__":
    main()
