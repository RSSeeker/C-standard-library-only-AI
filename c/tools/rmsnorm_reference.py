# -*- coding: utf-8 -*-
"""生成 RMSNorm 的跨语言对拍数据（JSON）

用法：python tools/rmsnorm_reference.py [输出路径]
      默认写到 c/rmsnorm_reference.json

数据来源是 AI.py 的 `class RMSNorm`（Llama 系：y = gamma * x / sqrt(mean(x^2)+eps)，
没有 beta、不减均值）。每个用例给固定 gamma 与若干组固定 (x, dout)，
记录 forward 的 y、inv_rms，以及 backward 的 grad_gamma 与 dx。

注意 Python 的语义（C 侧必须照抄，不能"顺手改好"）：
  - grad_gamma 每次 backward 都是**覆盖赋值**，不是对 batch 累加；
  - m_gamma / v_gamma 建了但从未使用。
两者都只用 dot/加法/sqrt，没有 pow/exp/log，故全部 tol=0.0。
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

from AI import RMSNorm  # noqa: E402


def make_case(name, note, d, eps, seed, n_samples, x_scale=1.0, gamma_scale=1.0):
    rnd = random.Random(seed)
    layer = RMSNorm(d, eps=eps)
    gamma = [rnd.uniform(0.5, 1.5) * gamma_scale for _ in range(d)]
    layer.gamma = list(gamma)

    samples = []
    for _ in range(n_samples):
        x = [rnd.gauss(0.0, 1.0) * x_scale for _ in range(d)]
        dout = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
        y = layer.forward(x)
        grad_gamma, dx = layer.backward(dout)
        samples.append({
            "x": x,
            "dout": dout,
            "expect_y": y,
            "expect_inv_rms": layer.inv_rms,
            "expect_grad_gamma": grad_gamma,
            "expect_dx": dx,
        })

    return {
        "name": name,
        "note": note,
        "tol": 0.0,
        "d": int(d),
        "eps": float(eps),
        "gamma": gamma,
        "samples": samples,
    }


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(_here, "..", "rmsnorm_reference.json"))

    cases = [
        make_case("d4_default_eps", "d=4、eps=1e-6、3 组样本（覆盖 grad_gamma 覆盖赋值）",
                  d=4, eps=1e-6, seed=901, n_samples=3),
        make_case("d1_edge", "d=1 的退化情形：r = 1/sqrt(x^2+eps)",
                  d=1, eps=1e-6, seed=902, n_samples=3),
        make_case("d8_eps1e-3", "d=8、eps=1e-3（eps 相对 mean_sq 不可忽略）",
                  d=8, eps=1e-3, seed=903, n_samples=2),
        make_case("d5_large_x", "d=5、x 放大 100 倍（r 很小，检验 g*x*r^3/d 的量级）",
                  d=5, eps=1e-6, seed=904, n_samples=2, x_scale=100.0),
        make_case("d3_gamma_scale", "d=3、gamma 放大 10 倍",
                  d=3, eps=1e-8, seed=905, n_samples=2, gamma_scale=10.0),
    ]

    data = {
        "meta": {
            "generated_by": "tools/rmsnorm_reference.py",
            "python": sys.version.split()[0],
            "ai_py_dir": AI_PY_DIR,
            "comment": "RMSNorm 的 y/inv_rms/grad_gamma/dx，全部 tol=0（只有 sqrt）",
        },
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("写入 %s：%d 个用例" % (os.path.abspath(out_path), len(cases)))
    for c in cases:
        print("  %-18s d=%d eps=%g samples=%d" % (c["name"], c["d"], c["eps"], len(c["samples"])))


if __name__ == "__main__":
    main()
