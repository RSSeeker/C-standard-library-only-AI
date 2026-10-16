# -*- coding: utf-8 -*-
"""导出 BatchNorm1d / LayerNorm / Dropout 的参考数据，供 C 实现逐位比对

用法：python tools/norm_reference.py [输出路径]
默认写到 c/norm_reference.json

数据全部由固定种子决定，可复现；C 侧 tests/compare_norm.c 读取同一份文件重算并比对。
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)
from AI import BatchNorm1d, LayerNorm, Dropout      # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.abspath(os.path.join(_here, "..")), "norm_reference.json")

    data = {}

    # ---------------- BatchNorm1d：训练路径 ----------------
    random.seed(20240701)
    d, n = 3, 5
    bn = BatchNorm1d(d, eps=1e-5, momentum=0.1)
    bn.gamma = [0.5 + 0.25 * j for j in range(d)]
    bn.beta = [-0.1 + 0.05 * j for j in range(d)]
    bn.running_mean = [0.01 * j for j in range(d)]
    bn.running_var = [1.0 + 0.1 * j for j in range(d)]

    x_batch = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    out = bn.forward_batch([list(r) for r in x_batch])
    dout = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    # 注意：BatchNorm1d.backward 返回 (dx, grad_gamma, grad_beta)
    dx, grad_gamma, grad_beta = bn.backward([list(r) for r in dout])

    data["bn"] = {
        "d": d, "n": n, "eps": bn.eps, "momentum": bn.momentum,
        "gamma": bn.gamma, "beta": bn.beta,
        "running_mean": [0.01 * j for j in range(d)],
        "running_var": [1.0 + 0.1 * j for j in range(d)],
        "x": x_batch,
        "out": out,
        "dout": dout,
        "dx": dx,
        "grad_gamma": grad_gamma,
        "grad_beta": grad_beta,
        "running_mean_after": bn.running_mean,
        "running_var_after": bn.running_var,
        "mean": bn.mean, "var": bn.var, "inv_std": bn.inv_std,
        "x_hat": bn.x_hat,
    }

    # ---------------- BatchNorm1d：推理路径 ----------------
    x_single = [random.uniform(-1, 1) for _ in range(d)]
    bn_inf = BatchNorm1d(d, eps=1e-5, momentum=0.1)
    bn_inf.gamma = list(bn.gamma)
    bn_inf.beta = list(bn.beta)
    bn_inf.running_mean = list(bn.running_mean)
    bn_inf.running_var = list(bn.running_var)
    bn_inf.training = False
    data["bn_infer"] = {
        "d": d, "eps": bn_inf.eps,
        "gamma": bn_inf.gamma, "beta": bn_inf.beta,
        "running_mean": bn_inf.running_mean, "running_var": bn_inf.running_var,
        "x": x_single,
        "out": bn_inf.forward(list(x_single)),
    }

    # ---------------- LayerNorm ----------------
    dl = 5
    ln = LayerNorm(dl, eps=1e-5)
    ln.gamma = [0.75 + 0.1 * j for j in range(dl)]
    ln.beta = [0.2 - 0.03 * j for j in range(dl)]
    x_ln = [random.uniform(-1, 1) for _ in range(dl)]
    out_ln = ln.forward(list(x_ln))
    dout_ln = [random.uniform(-1, 1) for _ in range(dl)]
    # 注意：LayerNorm.backward 返回 (grad_gamma, grad_beta, dx)，与 BatchNorm1d 的顺序相反
    grad_gamma_ln, grad_beta_ln, dx_ln = ln.backward(list(dout_ln))
    data["ln"] = {
        "d": dl, "eps": ln.eps,
        "gamma": ln.gamma, "beta": ln.beta,
        "x": x_ln,
        "out": out_ln,
        "dout": dout_ln,
        "dx": dx_ln,
        "grad_gamma": grad_gamma_ln,
        "grad_beta": grad_beta_ln,
        "mean": ln.mean, "var": ln.var, "inv_std": ln.inv_std, "x_hat": ln.x_hat,
    }

    # ---------------- Dropout（灌入确定 mask） ----------------
    random.seed(20240702)
    p = 0.25
    dp = Dropout(p)
    x_dp = [random.uniform(-1, 1) for _ in range(12)]
    random.seed(999)                      # mask 由这个种子决定
    out_dp = dp.forward(list(x_dp))
    mask = list(dp.mask)
    dout_dp = [random.uniform(-1, 1) for _ in range(12)]
    dx_dp = dp.backward(list(dout_dp))
    data["dropout"] = {
        "p": p, "n": 12,
        "x": x_dp, "mask": mask, "out": out_dp,
        "dout": dout_dp, "dx": dx_dp,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"参考数据已写入 {out_path}")
    print(f"  BatchNorm1d(训练): d={d} n={n}，BN(推理): d={d}，LayerNorm: d={dl}，Dropout: p={p} n=12")
    print(f"  BN 训练首行输出: {[round(v, 10) for v in out[0]]}")
    print(f"  LN 输出: {[round(v, 10) for v in out_ln]}")
    print(f"  Dropout mask: {mask}")


if __name__ == "__main__":
    main()
