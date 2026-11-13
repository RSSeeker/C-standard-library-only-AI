# -*- coding: utf-8 -*-
"""导出模型合并（model soup / SLERP / TIES / DARE）的参考数据，供 C 实现对比。

覆盖 AI.py 的：
  model_soup / slerp_merge / ties_merge / dare_merge

C 侧用「n_tensors 个等长扁平张量 + n_elem」表达；参考脚本记录
shape（[rows, cols] 或 [n]）与扁平化的输入/输出，便于逐位对拍。
Python 侧按真实形状（2D 嵌套 / 1D 列表）调用，输出再展平。

用法：python tools/model_merge_reference.py [输出路径]
未给输出路径时写到 c/model_merge_reference.json。

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

TOL = 0.0  # model_soup / ties / dare 只有四则运算，要求逐位相同
# slerp 的 sqrt/acos/sin 走 libm；MinGW libm 与 CPython UCRT 存在 <=1 ULP 差异，
# 故容差 1e-12（model_soup / ties / dare 仍要求逐位相同）。
SLERP_TOL = 1e-12
SLERP_NOTE = ("slerp 的 sqrt/acos/sin 走 libm；MinGW libm 与 CPython UCRT "
              "存在 <=1 ULP 差异，故容差 1e-12")
CASES = []
RND = random.Random(20261113)

# 显式声明每个错误用例：C 侧对应函数应返回 -1。
ERRORS = []


def add(**kw):
    CASES.append(kw)


def make_tensor(shape):
    """按 shape（[n] 或 [rows, cols]）生成固定种子随机张量（保留 6 位小数）。"""
    if len(shape) == 1:
        return [round(RND.uniform(-1.0, 1.0), 6) for _ in range(shape[0])]
    rows, cols = shape
    return [[round(RND.uniform(-1.0, 1.0), 6) for _ in range(cols)]
            for _ in range(rows)]


def flatten(tensor):
    if tensor and isinstance(tensor[0], list):
        return [v for row in tensor for v in row]
    return list(tensor)


def nested(flat, shape):
    if len(shape) == 1:
        return list(flat)
    rows, cols = shape
    return [list(flat[i * cols:(i + 1) * cols]) for i in range(rows)]


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


def build_soup_cases():
    specs = [
        ("soup_two_2x3", [2, 3], 2),
        ("soup_three_2x3", [2, 3], 3),
        ("soup_four_1d", [5], 4),
        ("soup_single", [2, 2], 1),
        ("soup_two_1d", [3], 2),
        ("soup_three_1x4", [1, 4], 3),
    ]
    for name, shape, n_tensors in specs:
        tensors = [make_tensor(shape) for _ in range(n_tensors)]
        out = AI.model_soup(tensors)
        add(kind="model_soup", name=name, tol=TOL, shape=shape,
            weights=[flatten(t) for t in tensors], out=flatten(out))


def build_slerp_cases():
    specs = [
        ("slerp_basic", [2, 2], 0.3, 1e-8),
        ("slerp_t0", [2, 2], 0.0, 1e-8),
        ("slerp_t1", [2, 2], 1.0, 1e-8),
        ("slerp_half", [2, 3], 0.5, 1e-8),
        ("slerp_eps_large", [3], 0.4, 0.5),
        ("slerp_1d", [6], 0.25, 1e-8),
    ]
    for name, shape, t, eps in specs:
        a = make_tensor(shape)
        b = make_tensor(shape)
        out = AI.slerp_merge(a, b, t=t, eps=eps)
        add(kind="slerp", name=name, tol=SLERP_TOL, note=SLERP_NOTE, shape=shape, a=flatten(a),
            b=flatten(b), t=t, eps=eps, out=flatten(out))

    # 零范数分支：a 全零 → 退化为线性插值。
    shape = [2, 2]
    a = [[0.0, 0.0], [0.0, 0.0]]
    b = make_tensor(shape)
    out = AI.slerp_merge(a, b, t=0.3, eps=1e-8)
    add(kind="slerp", name="slerp_zero_a", tol=SLERP_TOL, note=SLERP_NOTE, shape=shape, a=flatten(a),
        b=flatten(b), t=0.3, eps=1e-8, out=flatten(out))

    # 近平行分支：cos 接近 1 → 退化为线性插值。
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.0, 2.0], [3.0, 4.0000001]]
    out = AI.slerp_merge(a, b, t=0.5, eps=1e-8)
    add(kind="slerp", name="slerp_near_parallel", tol=SLERP_TOL, note=SLERP_NOTE, shape=shape,
        a=flatten(a), b=flatten(b), t=0.5, eps=1e-8, out=flatten(out))

    # 正交方向：真正的球面插值。
    a = [[1.0, 0.0], [0.0, 1.0]]
    b = [[0.0, 1.0], [1.0, 0.0]]
    out = AI.slerp_merge(a, b, t=0.5, eps=1e-8)
    add(kind="slerp", name="slerp_orthogonal", tol=SLERP_TOL, note=SLERP_NOTE, shape=shape,
        a=flatten(a), b=flatten(b), t=0.5, eps=1e-8, out=flatten(out))


def build_ties_cases():
    shape = [2, 3]
    specs = [
        ("ties_basic", 3, 0.2, 1.0, True),
        ("ties_no_base", 3, 0.2, 1.0, False),
        ("ties_trim0", 2, 0.0, 1.0, True),
        ("ties_trim_half", 4, 0.5, 1.0, True),
        ("ties_density_half", 3, 0.2, 0.5, True),
        ("ties_density_full", 3, 0.0, 0.5, True),
    ]
    for name, n_tensors, trim, density, use_base in specs:
        weights = [make_tensor(shape) for _ in range(n_tensors)]
        base = make_tensor(shape) if use_base else None
        out = AI.ties_merge(weights, base=base, trim_fraction=trim,
                            density=density)
        case = dict(kind="ties", name=name, tol=TOL, shape=shape,
                    weights=[flatten(t) for t in weights],
                    trim_fraction=trim, density=density, out=flatten(out))
        if base is not None:
            case["base"] = flatten(base)
        add(**case)

    # 符号冲突：两个张量的 task vector 完全相反 → 符号和 0，合并为 0。
    base = [[0.5, -0.5, 1.0], [0.0, 2.0, -1.0]]
    w1 = [[1.5, 0.5, 2.0], [1.0, 3.0, 0.0]]
    w2 = [[-0.5, -1.5, 0.0], [-1.0, 1.0, -2.0]]
    out = AI.ties_merge([w1, w2], base=base, trim_fraction=0.0)
    add(kind="ties", name="ties_sign_conflict", tol=TOL, shape=shape,
        weights=[flatten(w1), flatten(w2)], base=flatten(base),
        trim_fraction=0.0, density=1.0, out=flatten(out))

    # 零元素：部分 task vector 为 0（阈值 0 时被排除）。
    base = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    w1 = [[1.0, 0.0, -1.0], [0.0, 2.0, 0.0]]
    w2 = [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]
    out = AI.ties_merge([w1, w2], base=base, trim_fraction=0.0)
    add(kind="ties", name="ties_zeros", tol=TOL, shape=shape,
        weights=[flatten(w1), flatten(w2)], base=flatten(base),
        trim_fraction=0.0, density=1.0, out=flatten(out))


def build_dare_cases():
    shape = [2, 3]
    specs = [
        ("dare_basic", 2, 0.5, True, 42, False),
        ("dare_no_rescale", 2, 0.5, False, 42, False),
        ("dare_drop_zero", 3, 0.0, True, 0, False),
        ("dare_drop_high", 2, 0.9, True, 7, True),
        ("dare_three", 3, 0.3, True, 0, False),
        ("dare_with_base", 2, 0.5, True, 12345, True),
        ("dare_single", 1, 0.5, True, 1, False),
    ]
    for name, n_tensors, drop, rescale, seed, use_base in specs:
        weights = [make_tensor(shape) for _ in range(n_tensors)]
        base = make_tensor(shape) if use_base else None
        out = AI.dare_merge(weights, base=base, drop_rate=drop,
                            rescale=rescale, seed=seed)
        case = dict(kind="dare", name=name, tol=TOL, shape=shape,
                    weights=[flatten(t) for t in weights], drop_rate=drop,
                    rescale=1 if rescale else 0, seed=seed,
                    out=flatten(out))
        if base is not None:
            case["base"] = flatten(base)
        add(**case)


def build_error_cases():
    shape = [2, 2]
    w = [make_tensor(shape) for _ in range(2)]
    add_error("err_soup_empty", "model_soup", {"weights": []},
              lambda: AI.model_soup([]))

    add_error("err_slerp_t_neg", "slerp",
              {"a": flatten(w[0]), "b": flatten(w[1]), "t": -0.1,
               "eps": 1e-8},
              lambda: AI.slerp_merge(w[0], w[1], t=-0.1))
    add_error("err_slerp_t_gt1", "slerp",
              {"a": flatten(w[0]), "b": flatten(w[1]), "t": 1.5,
               "eps": 1e-8},
              lambda: AI.slerp_merge(w[0], w[1], t=1.5))
    add_error("err_slerp_eps_neg", "slerp",
              {"a": flatten(w[0]), "b": flatten(w[1]), "t": 0.5,
               "eps": -1.0},
              lambda: AI.slerp_merge(w[0], w[1], t=0.5, eps=-1.0))

    add_error("err_ties_trim_one", "ties",
              {"weights": [flatten(t) for t in w], "trim_fraction": 1.0,
               "density": 1.0},
              lambda: AI.ties_merge(w, trim_fraction=1.0))
    add_error("err_ties_trim_neg", "ties",
              {"weights": [flatten(t) for t in w], "trim_fraction": -0.05,
               "density": 1.0},
              lambda: AI.ties_merge(w, trim_fraction=-0.05))
    add_error("err_ties_density_zero", "ties",
              {"weights": [flatten(t) for t in w], "trim_fraction": 0.2,
               "density": 0.0},
              lambda: AI.ties_merge(w, density=0.0))
    add_error("err_ties_density_gt1", "ties",
              {"weights": [flatten(t) for t in w], "trim_fraction": 0.2,
               "density": 1.5},
              lambda: AI.ties_merge(w, density=1.5))

    add_error("err_dare_drop_one", "dare",
              {"weights": [flatten(t) for t in w], "drop_rate": 1.0,
               "rescale": 1, "seed": 1},
              lambda: AI.dare_merge(w, drop_rate=1.0, seed=1))
    add_error("err_dare_drop_neg", "dare",
              {"weights": [flatten(t) for t in w], "drop_rate": -0.1,
               "rescale": 1, "seed": 1},
              lambda: AI.dare_merge(w, drop_rate=-0.1, seed=1))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_merge_reference.json")
    # 进程级固定种子：即便引用的函数里用到全局 random 也可复现。
    random.seed(20261113)
    build_soup_cases()
    build_slerp_cases()
    build_ties_cases()
    build_dare_cases()
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
        for key in ("weights", "a", "b", "base", "out"):
            val = case.get(key)
            if isinstance(val, list):
                seqs = val if val and isinstance(val[0], list) else [val]
                for seq in seqs:
                    for x in seq:
                        if isinstance(x, float) and (
                                x != x or x in (float("inf"), float("-inf"))):
                            raise SystemExit(f"{case['name']}.{key} 含非有限值")


if __name__ == "__main__":
    main()
