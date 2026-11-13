# -*- coding: utf-8 -*-
"""从 AI.py 导出 QK-Norm + DynamicTanh 参考值 -> c/qk_norm_reference.json

覆盖：
  qk_norm / qk_norm_backward / apply_qk_norm
  class DynamicTanh（forward / backward / sgd_step / param_count）

浮点用例都带 tol=1e-12；C 侧先按位比较（memcmp），不一致再比绝对差。

错误用例写成 {"kind": "error", "op": ..., "args": {...}}，导出时先确认
Python 侧确实 raise。注意 JSON 不能表示 NaN/Infinity，且 C 侧二维入参是
「扁平数组 + width」，所以 Python 里这两类错误无法在对拍中表达：
  - 矩阵元素为 NaN/inf（json.dump 会写出非法 JSON）
  - 向量长度与 width 不一致 / 矩阵各行不等长（C 调用方已经给定 width）
这些边界只在 README 里说明，不列入参考用例。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(C_DIR)
PY_REPO = os.path.join(os.path.dirname(REPO), "Python-standard-library-only-AI-main")
sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

CASES = []
TOL = 1e-12


def add(**kw):
    CASES.append(kw)


def add_qk(name, rows, gamma, dout=None, eps=1e-6):
    res = AI.qk_norm(rows, gamma, eps)
    case = {
        "kind": "qk_norm", "name": name, "tol": TOL,
        "rows": rows, "gamma": gamma, "eps": eps,
        "out": res["out"], "rms": res["rms"], "xhat": res["xhat"],
    }
    if dout is not None:
        grad = AI.qk_norm_backward(dout, res)
        case["backward"] = {
            "dout": dout,
            "gamma_grad": grad["gamma_grad"],
            "input_grad": grad["input_grad"],
        }
    add(**case)


def add_apply(name, q_rows, k_rows, q_gamma, k_gamma, eps=1e-6):
    res = AI.apply_qk_norm(q_rows, k_rows, q_gamma, k_gamma, eps)
    add(kind="apply_qk_norm", name=name, tol=TOL, eps=eps,
        q_rows=q_rows, k_rows=k_rows, q_gamma=q_gamma, k_gamma=k_gamma,
        q=res["q"], k=res["k"])


def add_dyt(name, d_model, alpha, forwards, dout=None, lr=None):
    model = AI.DynamicTanh(d_model, alpha=alpha)
    outs = [model.forward(rows) for rows in forwards]
    case = {
        "kind": "dyt", "name": name, "tol": TOL,
        "d_model": d_model, "alpha": alpha,
        "forwards": [{"rows": rows, "out": out}
                     for rows, out in zip(forwards, outs)],
        "param_count": model.param_count(),
    }
    if dout is not None:
        input_grad = model.backward(dout)
        case["backward"] = {
            "dout": dout, "input_grad": input_grad,
            "grad_alpha": model.grad_alpha,
            "grad_gamma": model.grad_gamma,
            "grad_beta": model.grad_beta,
        }
        if lr is not None:
            model.sgd_step(lr)
            case["sgd"] = {
                "lr": lr, "alpha": model.alpha, "gamma": model.gamma,
                "beta": model.beta, "grad_alpha": model.grad_alpha,
                "grad_gamma": model.grad_gamma, "grad_beta": model.grad_beta,
            }
    add(**case)


def add_error(name, op, args, call, exc=(ValueError,)):
    try:
        call()
    except exc:
        pass
    else:
        raise SystemExit("错误用例 %s 在 Python 侧没有 raise" % name)
    add(kind="error", name=name, op=op, args=args)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "qk_norm_reference.json")

    # ---------------- qk_norm（含可选 backward） ----------------
    add_qk("qk_basic", [[1.0, 2.0, 3.0, 4.0]], [1.0, 1.0, 1.0, 1.0],
           dout=[[0.1, 0.2, 0.3, 0.4]])
    add_qk("qk_gamma", [[0.5, -1.5, 2.0]], [1.2, -0.3, 2.5],
           dout=[[-1.0, 0.5, 2.0]])
    add_qk("qk_multi",
           [[1.0, -2.0, 3.0, 0.5, -4.0],
            [2.5, 0.0, -1.5, 4.0, 1.0],
            [-0.5, 0.25, 8.0, -3.0, 2.0]],
           [1.0, 0.5, -2.0, 1.5, 0.75],
           dout=[[0.1, -0.2, 0.3, -0.4, 0.5],
                 [1.0, 2.0, -1.0, 0.0, 0.5],
                 [-0.3, 0.6, -0.9, 1.2, -1.5]])
    add_qk("qk_eps", [[2.0, -4.0, 6.0], [0.1, 0.2, 0.3]], [1.0, -1.0, 0.5],
           dout=[[1.0, 0.0, -1.0], [-2.0, 3.0, 0.5]], eps=1e-3)
    add_qk("qk_width1", [[3.0], [-0.5], [0.0]], [2.0],
           dout=[[1.0], [-2.0], [0.5]])
    add_qk("qk_zero_row", [[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]],
           [1.0, 0.5, -2.0],
           dout=[[1.0, 1.0, 1.0], [0.2, -0.3, 0.4]])
    add_qk("qk_large", [[1000000.0, -2000000.0, 3000000.0],
                        [0.000001, 0.000002, 0.0]],
           [1.0, 1.5, -1.0],
           dout=[[1e-6, 2e-6, -3e-6], [0.5, -0.5, 0.25]])
    add_qk("qk_int", [[1, 2], [3, 4]], [1, 1], dout=[[1, -1], [2, -2]])

    # ---------------- apply_qk_norm ----------------
    add_apply("apply_basic",
              [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.0, -1.0]],
              [[2.0, 1.0, 0.0, -1.0], [-2.0, 3.0, 1.5, 0.5]],
              [1.0, 1.0, 1.0, 1.0], [0.5, 2.0, -1.0, 1.5])
    add_apply("apply_different",
              [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]],
              [[0.5, -0.5, 1.0, 2.0, -2.0]],
              [1.0, 0.5, -1.0], [2.0, 1.0, -1.0, 0.5, 1.5])
    add_apply("apply_eps", [[3.0, 4.0]], [[1.0, 1.0]],
              [1.0, 1.0], [1.0, 1.0], eps=1e-3)

    # ---------------- DynamicTanh ----------------
    add_dyt("dyt_basic", 3, 0.5,
            [[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]],
            dout=[[0.1, 0.2, -0.3], [1.0, -2.0, 0.5]], lr=0.1)
    add_dyt("dyt_alpha_neg", 2, -0.7,
            [[[1.0, -2.0], [0.25, 0.75]]],
            dout=[[1.0, 0.5], [-1.0, 2.0]], lr=0.01)
    add_dyt("dyt_d1", 1, 1.0,
            [[[3.0], [-4.0]]],
            dout=[[2.0], [-0.5]], lr=0.5)
    add_dyt("dyt_two_forwards", 4, 0.25,
            [[[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 2.0, 0.5]],
             [[-2.0, 1.0, 0.5, 3.0]]],
            dout=[[0.5, -0.5, 1.0, -1.0]], lr=0.02)
    add_dyt("dyt_no_sgd", 3, 0.5,
            [[[1.0, 0.0, -1.0]]],
            dout=[[0.3, -0.3, 0.9]])
    add_dyt("dyt_alpha_1", 2, 1.0,
            [[[1.0, 2.0], [-3.0, 4.0]]],
            dout=[[0.5, 0.5], [-0.5, 0.5]], lr=0.05)

    # ---------------- 错误用例 ----------------
    add_error("err_qk_rows_empty", "qk_norm",
              {"rows": [], "gamma": [1.0], "eps": 1e-6},
              lambda: AI.qk_norm([], [1.0]))
    add_error("err_qk_row_empty", "qk_norm",
              {"rows": [[]], "gamma": [], "eps": 1e-6},
              lambda: AI.qk_norm([[]], []))
    add_error("err_qk_eps0", "qk_norm",
              {"rows": [[1.0, 2.0]], "gamma": [1.0, 1.0], "eps": 0.0},
              lambda: AI.qk_norm([[1.0, 2.0]], [1.0, 1.0], eps=0.0))
    add_error("err_qk_eps_neg", "qk_norm",
              {"rows": [[1.0, 2.0]], "gamma": [1.0, 1.0], "eps": -1e-6},
              lambda: AI.qk_norm([[1.0, 2.0]], [1.0, 1.0], eps=-1e-6))
    add_error("err_bwd_row_count", "qk_norm_backward",
              {"rows": [[1.0, 2.0], [3.0, 4.0]], "gamma": [1.0, 1.0],
               "eps": 1e-6, "dout": [[1.0, 1.0]]},
              lambda: AI.qk_norm_backward(
                  [[1.0, 1.0]],
                  AI.qk_norm([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])))
    add_error("err_bwd_width", "qk_norm_backward",
              {"rows": [[1.0, 2.0, 3.0]], "gamma": [1.0, 1.0, 1.0],
               "eps": 1e-6, "dout": [[1.0, 1.0]]},
              lambda: AI.qk_norm_backward(
                  [[1.0, 1.0]],
                  AI.qk_norm([[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0])))
    add_error("err_apply_q_empty", "apply_qk_norm",
              {"q_rows": [], "k_rows": [[1.0, 2.0]],
               "q_gamma": [1.0], "k_gamma": [1.0, 1.0], "eps": 1e-6},
              lambda: AI.apply_qk_norm([], [[1.0, 2.0]], [1.0], [1.0, 1.0]))
    add_error("err_apply_k_empty", "apply_qk_norm",
              {"q_rows": [[1.0, 2.0]], "k_rows": [],
               "q_gamma": [1.0, 1.0], "k_gamma": [1.0], "eps": 1e-6},
              lambda: AI.apply_qk_norm([[1.0, 2.0]], [], [1.0, 1.0], [1.0]))
    add_error("err_apply_eps0", "apply_qk_norm",
              {"q_rows": [[1.0, 2.0]], "k_rows": [[1.0, 2.0]],
               "q_gamma": [1.0, 1.0], "k_gamma": [1.0, 1.0], "eps": 0.0},
              lambda: AI.apply_qk_norm([[1.0, 2.0]], [[1.0, 2.0]],
                                       [1.0, 1.0], [1.0, 1.0], eps=0.0))
    add_error("err_dyt_d0", "dyt_create",
              {"d_model": 0, "alpha": 0.5},
              lambda: AI.DynamicTanh(0))
    add_error("err_dyt_d_neg", "dyt_create",
              {"d_model": -2, "alpha": 0.5},
              lambda: AI.DynamicTanh(-2))
    add_error("err_dyt_alpha0", "dyt_create",
              {"d_model": 3, "alpha": 0.0},
              lambda: AI.DynamicTanh(3, alpha=0.0))
    add_error("err_dyt_forward_width", "dyt_forward",
              {"d_model": 3, "alpha": 0.5, "rows": [[1.0, 2.0]]},
              lambda: AI.DynamicTanh(3).forward([[1.0, 2.0]]))
    add_error("err_dyt_forward_empty", "dyt_forward",
              {"d_model": 3, "alpha": 0.5, "rows": []},
              lambda: AI.DynamicTanh(3).forward([]))
    add_error("err_dyt_backward_before", "dyt_backward",
              {"d_model": 3, "alpha": 0.5, "forward_rows": None,
               "dout": [[0.1, 0.2, 0.3]]},
              lambda: AI.DynamicTanh(3).backward([[0.1, 0.2, 0.3]]),
              exc=(RuntimeError,))
    def dyt_trained(lr):
        model = AI.DynamicTanh(3, alpha=0.5)
        model.forward([[1.0, 2.0, 3.0]])
        model.backward([[0.1, 0.2, 0.3]])
        model.sgd_step(lr)

    def dyt_backward_rows():
        model = AI.DynamicTanh(3, alpha=0.5)
        model.forward([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model.backward([[0.1, 0.2, 0.3]])

    add_error("err_dyt_backward_rows", "dyt_backward",
              {"d_model": 3, "alpha": 0.5,
               "forward_rows": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
               "dout": [[0.1, 0.2, 0.3]]},
              dyt_backward_rows)
    add_error("err_dyt_sgd_lr0", "dyt_sgd_step",
              {"d_model": 3, "alpha": 0.5,
               "rows": [[1.0, 2.0, 3.0]], "dout": [[0.1, 0.2, 0.3]],
               "lr": 0.0},
              lambda: dyt_trained(0.0))
    add_error("err_dyt_sgd_lr_neg", "dyt_sgd_step",
              {"d_model": 3, "alpha": 0.5,
               "rows": [[1.0, 2.0, 3.0]], "dout": [[0.1, 0.2, 0.3]],
               "lr": -0.1},
              lambda: dyt_trained(-0.1))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    n_err = sum(1 for c in CASES if c["kind"] == "error")
    print("wrote %s (%d cases, %d error cases)" % (out_path, len(CASES), n_err))


if __name__ == "__main__":
    main()
