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
from AI import (BatchNorm1d, BatchNorm2d, DropPath, Dropout, GroupNorm,  # noqa: E402
                InstanceNorm, LayerNorm)


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

    # ---------------- BatchNorm2d：训练路径 ----------------
    random.seed(20240703)
    c2, h2, w2 = 2, 3, 2
    rm2 = [0.02 * j for j in range(c2)]
    rv2 = [1.0 + 0.2 * j for j in range(c2)]
    bn2 = BatchNorm2d(c2, eps=1e-5, momentum=0.2)
    bn2.gamma = [0.6 + 0.2 * j for j in range(c2)]
    bn2.beta = [-0.15 + 0.1 * j for j in range(c2)]
    bn2.running_mean = list(rm2)
    bn2.running_var = list(rv2)
    x_bn2 = [[[random.uniform(-1.0, 1.0) for _ in range(w2)] for _ in range(h2)]
             for _ in range(c2)]
    out_bn2 = bn2.forward([[[v for v in row] for row in ch] for ch in x_bn2])
    dout_bn2 = [[[random.uniform(-1.0, 1.0) for _ in range(w2)] for _ in range(h2)]
                for _ in range(c2)]
    # 注意：BatchNorm2d.backward 返回 (dx, grad_gamma, grad_beta)
    dx_bn2, gg_bn2, gb_bn2 = bn2.backward(
        [[[v for v in row] for row in ch] for ch in dout_bn2])
    data["bn2"] = {
        "c": c2, "h": h2, "w": w2, "eps": bn2.eps, "momentum": bn2.momentum,
        "gamma": bn2.gamma, "beta": bn2.beta,
        "running_mean": rm2, "running_var": rv2,
        "x": x_bn2, "out": out_bn2, "dout": dout_bn2, "dx": dx_bn2,
        "grad_gamma": gg_bn2, "grad_beta": gb_bn2,
        "running_mean_after": bn2.running_mean, "running_var_after": bn2.running_var,
        "mean": [m for m, _ in bn2.stats], "inv": [v for _, v in bn2.stats],
    }

    # ---------------- BatchNorm2d：推理路径（用 running 统计量） ----------------
    bn2i = BatchNorm2d(c2, eps=1e-5, momentum=0.2)
    bn2i.gamma = list(bn2.gamma)
    bn2i.beta = list(bn2.beta)
    bn2i.running_mean = list(rm2)
    bn2i.running_var = list(rv2)
    bn2i.eval()
    out_bn2i = bn2i.forward([[[v for v in row] for row in ch] for ch in x_bn2])
    dx_bn2i, gg_bn2i, gb_bn2i = bn2i.backward(
        [[[v for v in row] for row in ch] for ch in dout_bn2])
    data["bn2_infer"] = {
        "c": c2, "h": h2, "w": w2, "eps": bn2i.eps,
        "gamma": bn2i.gamma, "beta": bn2i.beta,
        "running_mean": rm2, "running_var": rv2,
        "x": x_bn2, "out": out_bn2i, "dout": dout_bn2, "dx": dx_bn2i,
        "grad_gamma": gg_bn2i, "grad_beta": gb_bn2i,
        "mean": [m for m, _ in bn2i.stats], "inv": [v for _, v in bn2i.stats],
    }

    # ---------------- GroupNorm ----------------
    random.seed(20240704)
    gc, gh, gw = 4, 2, 3
    gn = GroupNorm(2, gc, eps=1e-5)
    gn.gamma = [0.7 + 0.1 * j for j in range(gc)]
    gn.beta = [-0.2 + 0.05 * j for j in range(gc)]
    x_gn = [[[random.uniform(-1.0, 1.0) for _ in range(gw)] for _ in range(gh)]
            for _ in range(gc)]
    out_gn = gn.forward([[[v for v in row] for row in ch] for ch in x_gn])
    dout_gn = [[[random.uniform(-1.0, 1.0) for _ in range(gw)] for _ in range(gh)]
               for _ in range(gc)]
    # 注意：GroupNorm.backward 返回 (grad_gamma, grad_beta, dx)，与 BatchNorm2d 相反
    gg_gn, gb_gn, dx_gn = gn.backward([[[v for v in row] for row in ch] for ch in dout_gn])
    data["gn"] = {
        "groups": 2, "channels": gc, "h": gh, "w": gw, "eps": gn.eps,
        "gamma": gn.gamma, "beta": gn.beta,
        "x": x_gn, "out": out_gn, "dout": dout_gn, "dx": dx_gn,
        "grad_gamma": gg_gn, "grad_beta": gb_gn,
        "mean": [m for m, _ in gn.stats], "inv": [v for _, v in gn.stats],
    }

    # ---------------- InstanceNorm（groups == channels 的特例） ----------------
    random.seed(20240706)
    ic, ih, iw = 3, 2, 2
    inn = InstanceNorm(ic, eps=1e-5)
    inn.gamma = [0.5 + 0.2 * j for j in range(ic)]
    inn.beta = [0.1 - 0.05 * j for j in range(ic)]
    x_in = [[[random.uniform(-1.0, 1.0) for _ in range(iw)] for _ in range(ih)]
            for _ in range(ic)]
    out_in = inn.forward([[[v for v in row] for row in ch] for ch in x_in])
    dout_in = [[[random.uniform(-1.0, 1.0) for _ in range(iw)] for _ in range(ih)]
               for _ in range(ic)]
    gg_in, gb_in, dx_in = inn.backward(
        [[[v for v in row] for row in ch] for ch in dout_in])
    data["inorm"] = {
        "groups": ic, "channels": ic, "h": ih, "w": iw, "eps": inn.eps,
        "gamma": inn.gamma, "beta": inn.beta,
        "x": x_in, "out": out_in, "dout": dout_in, "dx": dx_in,
        "grad_gamma": gg_in, "grad_beta": gb_in,
        "mean": [m for m, _ in inn.stats], "inv": [v for _, v in inn.stats],
    }

    # ---------------- DropPath（训练：丢弃 / 保留各一例，再一例推理） ----------------
    random.seed(20240705)
    p_dp = 0.3
    x_dp2 = [random.uniform(-1.0, 1.0) for _ in range(6)]
    dout_dp2 = [random.uniform(-1.0, 1.0) for _ in range(6)]
    got = {}
    seed = 1
    while len(got) < 2 and seed <= 50:
        dp2 = DropPath(p_dp, scale_by_keep=True)
        random.seed(seed)
        out_dp2 = dp2.forward(list(x_dp2))
        keep = 0 if dp2.factor == 0.0 else 1
        if keep not in got:
            got[keep] = {
                "keep": keep, "factor": dp2.factor, "out": list(out_dp2),
                "dx": list(dp2.backward(list(dout_dp2))),
            }
        seed += 1
    if len(got) != 2:
        sys.exit("DropPath：50 个种子里没凑齐「丢弃 / 保留」两种情形")
    dp_eval = DropPath(p_dp, scale_by_keep=True)
    dp_eval.eval()
    out_dp_eval = dp_eval.forward(list(x_dp2))
    data["droppath"] = {
        "p": p_dp, "scale_by_keep": 1, "x": x_dp2, "dout": dout_dp2,
        "drop": got[0], "keep": got[1],
        "eval": {"factor": dp_eval.factor, "out": list(out_dp_eval),
                 "dx": list(dp_eval.backward(list(dout_dp2)))},
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
