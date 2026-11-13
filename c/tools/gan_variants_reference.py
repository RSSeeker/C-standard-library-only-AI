"""生成 GAN 变体辅助函数的 C 对拍参考（JSON）。

运行：
  PYTHONPATH=/d/vscode/Python-standard-library-only-AI-main \
    python tools/gan_variants_reference.py gan_variants_reference.json
"""

import json
import os
import sys

import _py_path  # noqa: F401  (把 Python 主仓库加入 sys.path)


def _img(c, h, w, base):
    out = []
    for k in range(c):
        ch = []
        for i in range(h):
            row = []
            for j in range(w):
                row.append(float(base) + 0.5 * k + 0.25 * i + 0.125 * j)
            ch.append(row)
        out.append(ch)
    return out


def _flat(img):
    if isinstance(img, (list, tuple)):
        out = []
        for item in img:
            out.extend(_flat(item))
        return out
    return [float(img)]


def main():
    if len(sys.argv) != 2:
        print("usage: gan_variants_reference.py OUT.json", file=sys.stderr)
        return 2
    out_path = sys.argv[1]
    import AI  # 依赖 PYTHONPATH 指向 Python 主仓库

    cases = []

    low = _img(2, 2, 3, 0.1)
    high = _img(2, 4, 6, 0.2)
    cases.append({
        "func": "fade_in",
        "c": 2, "lh": 2, "lw": 3, "alpha": 0.35,
        "low": _flat(low), "high": _flat(high),
        "expected": _flat(AI.fade_in(low, high, 0.35)),
    })

    for step, total in ((0, 10), (3, 10), (10, 10), (15, 10)):
        cases.append({
            "func": "progressive_alpha",
            "step": step, "total": total,
            "expected": float(AI.progressive_alpha(step, total)),
        })

    stages = AI.progressive_stages([4, 8, 16], fade_steps=3)
    cases.append({
        "func": "progressive_stages",
        "resolutions": [4, 8, 16], "fade_steps": 3,
        "expected": [[int(lo), int(hi), float(a)] for lo, hi, a in stages],
    })

    feats = _img(2, 2, 3, 0.7)
    style = [1.5, -0.5, 0.2, 0.8]
    out, stats = AI.adain_style(feats, style, eps=1e-5)
    cases.append({
        "func": "adain_style",
        "c": 2, "h": 2, "w": 3, "eps": 1e-5,
        "features": _flat(feats), "style": style,
        "expected": _flat(out),
        "stats_mean": [float(m) for m, _ in stats],
        "stats_std": [float(s) for _, s in stats],
    })

    w = [0.4, -1.2, 2.0]
    w_avg = [0.1, -0.2, 0.5]
    cases.append({
        "func": "truncation_trick",
        "w": w, "w_avg": w_avg, "psi": 0.7,
        "expected": [float(v) for v in AI.truncation_trick(w, w_avg, 0.7)],
    })

    a = _img(1, 2, 2, 0.3)
    a_rec = _img(1, 2, 2, 0.1)
    b = _img(1, 2, 2, 0.9)
    b_rec = _img(1, 2, 2, 1.1)
    loss, grads = AI.cycle_consistency_loss(a, a_rec, b, b_rec, lambda_cycle=10.0)
    cases.append({
        "func": "cycle_consistency_loss",
        "c": 1, "h": 2, "w": 2, "lambda_cycle": 10.0,
        "a": _flat(a), "a_rec": _flat(a_rec), "b": _flat(b), "b_rec": _flat(b_rec),
        "expected": float(loss),
        "grad_a": _flat(grads[0]),
        "grad_b": _flat(grads[1]),
    })

    g_ba = _img(1, 2, 2, 0.35)
    g_ab = _img(1, 2, 2, 0.95)
    cases.append({
        "func": "identity_mapping_loss",
        "c": 1, "h": 2, "w": 2, "lambda_identity": 5.0,
        "a": _flat(a), "g_ba": _flat(g_ba), "b": _flat(b), "g_ab": _flat(g_ab),
        "expected": float(AI.identity_mapping_loss(a, g_ba, b, g_ab, 5.0)),
    })

    for real in (True, False):
        cases.append({
            "func": "patchgan_targets",
            "h": 3, "w": 2, "real": bool(real),
            "expected": _flat(AI.patchgan_targets(3, 2, real)),
        })

    scores = [[0.3, -0.4], [1.2, 0.0]]
    fake = _img(1, 2, 2, 0.4)
    real_img = _img(1, 2, 2, 0.6)
    cases.append({
        "func": "pix2pix_g_loss",
        "ns": 4, "c": 1, "h": 2, "w": 2, "lambda_l1": 100.0,
        "scores": _flat(scores), "fake": _flat(fake), "real": _flat(real_img),
        "expected": float(AI.pix2pix_g_loss(scores, fake, real_img, 100.0)),
    })

    d_scores = [[0.5, -0.5, 1.0], [0.25, 0.0, -0.75]]
    cases.append({
        "func": "r1_penalty",
        "n_samples": 2, "grad_dim": 3, "lam": 10.0,
        "grads": _flat(d_scores),
        "expected": float(AI.r1_penalty(d_scores, 10.0)),
    })
    cases.append({
        "func": "r2_penalty",
        "n_samples": 2, "grad_dim": 3, "lam": 10.0,
        "grads": _flat(d_scores),
        "expected": float(AI.r2_penalty(d_scores, 10.0)),
    })

    payload = {
        "meta": {"source": "gan_variants_reference.py",
                 "tolerance": 1e-12},
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path, "cases", len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
