# -*- coding: utf-8 -*-
"""导出数据增广内核的参考数据（C 侧 compare_augment.c 使用）。

随机性由参考脚本「注入」：Python 侧用 `random.Random(seed)` 抽出的 lam / 裁剪框 /
洞位置 / 操作序列与强度，全部写进 JSON，C 侧按同样的参数执行，于是可以逐位对拍。
"""
import json
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit("找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: %s）" % AI_PY_DIR)
sys.path.insert(0, AI_PY_DIR)

import AI  # noqa: E402


def flat(img):
    return [float(v) for ch in img for row in ch for v in row]


def make_img(C, H, W, seed):
    r = random.Random(seed)
    return [[[round(r.random(), 6) for _ in range(W)] for _ in range(H)] for _ in range(C)]


def case_brightness():
    img = make_img(2, 3, 4, 101)
    for factor in (1.4, 0.55):
        out = AI._aug_brightness(img, factor)
        yield {
            "name": "brightness_%s" % str(factor).replace(".", "_"),
            "func": "brightness",
            "C": 2, "H": 3, "W": 4,
            "factor": factor,
            "x": flat(img),
            "out": flat(out),
        }


def case_contrast():
    img = make_img(2, 3, 4, 202)
    for factor in (0.5, 1.6):
        out = AI._aug_contrast(img, factor)
        yield {
            "name": "contrast_%s" % str(factor).replace(".", "_"),
            "func": "contrast",
            "C": 2, "H": 3, "W": 4,
            "factor": factor,
            "x": flat(img),
            "out": flat(out),
        }


def case_shift():
    img = make_img(2, 3, 4, 303)
    for dy, dx in ((1, 0), (0, -2), (-1, 3)):
        out = AI._aug_shift(img, dy, dx)
        yield {
            "name": "shift_%d_%d" % (dy, dx),
            "func": "shift",
            "C": 2, "H": 3, "W": 4,
            "dy": dy, "dx": dx,
            "x": flat(img),
            "out": flat(out),
        }


def case_rotate():
    img = make_img(2, 3, 4, 404)
    for turns in (0, 1, 2, 3, -1):
        out = AI._aug_rotate90(img, turns)
        h = len(out[0])
        w = len(out[0][0])
        yield {
            "name": "rotate_%d" % turns,
            "func": "rotate90",
            "C": 2, "H": 3, "W": 4,
            "turns": turns,
            "x": flat(img),
            "oh": h, "ow": w,
            "out": flat(out),
        }


def case_cutout():
    img = make_img(2, 5, 6, 505)
    side = max(1, int(round(min(5, 6) * 0.3)))
    rng = random.Random(606)
    xs, ys = [], []
    for _ in range(3):
        xs.append(int(rng.randrange(max(1, 6 - side + 1))))
        ys.append(int(rng.randrange(max(1, 5 - side + 1))))
    # 用同一个种子重放，拿到 Python 的输出
    out = AI.augment_cutout(img, num_holes=3, hole_size=0.3, fill=0.25, rng=random.Random(606))
    yield {
        "name": "cutout_3holes",
        "func": "cutout",
        "C": 2, "H": 5, "W": 6,
        "num_holes": 3, "hole_size": 0.3, "fill": 0.25,
        "xs": xs, "ys": ys,
        "x": flat(img),
        "out": flat(out),
    }


def case_cutmix():
    a = make_img(2, 4, 5, 707)
    b = make_img(2, 4, 5, 708)
    y1 = [1.0, 0.0, 0.0]
    y2 = [0.0, 0.0, 1.0]
    for alpha, seed in ((1.0, 808), (2.0, 809)):
        # 重放随机流：先 _beta_lam（内部 random/gauss），再 randrange(W)、randrange(H)
        probe = random.Random(seed)
        lam0 = AI._beta_lam(alpha, probe, "ref")
        cx = int(probe.randrange(5))
        cy = int(probe.randrange(4))
        img, lab = AI.augment_cutmix(a, y1, b, y2, alpha=alpha, rng=random.Random(seed))
        yield {
            "name": "cutmix_alpha_%s" % str(alpha).replace(".", "_"),
            "func": "cutmix",
            "C": 2, "H": 4, "W": 5,
            "lam0": lam0, "cx": cx, "cy": cy,
            "xa": flat(a), "xb": flat(b),
            "y1": y1, "y2": y2,
            "out": flat(img),
            "label": [float(v) for v in lab],
        }


def case_mixup():
    x1 = [0.1, 0.2, -0.3, 0.4]
    x2 = [0.9, -0.5, 0.25, 0.0]
    mixed, soft = AI.mixup(x1, 1, x2, 2, alpha=0.0)
    yield {
        "name": "mixup_alpha0",
        "func": "mixup",
        "n": 4, "y1": 1, "y2": 2, "n_cls": 3,
        "lam": 0.5,
        "x1": x1, "x2": x2,
        "out": [float(v) for v in mixed],
        "soft": [float(v) for v in soft],
    }
    for lam in (0.3, 0.75):
        mixed = [lam * float(p) + (1.0 - lam) * float(q) for p, q in zip(x1, x2)]
        soft = [0.0] * 3
        soft[1] += lam
        soft[2] += 1.0 - lam
        yield {
            "name": "mixup_lam_%s" % str(lam).replace(".", "_"),
            "func": "mixup",
            "n": 4, "y1": 1, "y2": 2, "n_cls": 3,
            "lam": lam,
            "x1": x1, "x2": x2,
            "out": mixed,
            "soft": [float(v) for v in soft],
        }


def case_randaugment():
    img = make_img(2, 4, 5, 909)
    n = 3
    magnitude = 6.0
    scale = magnitude / 10.0
    pool = ["brightness", "contrast", "translate", "rotate90", "identity"]
    rng = random.Random(1111)
    ops, params = [], []
    for _ in range(n):
        op = int(rng.randrange(len(pool)))
        ops.append(op)
        if op == 0 or op == 1:
            params.append(1.0 - 0.5 * scale + scale * rng.random())
            params.append(0.0)
        elif op == 2:
            lim = max(1, int(round(scale * 0.25 * min(4, 5))))
            params.append(float(rng.randint(-lim, lim)))
            params.append(float(rng.randint(-lim, lim)))
        elif op == 3:
            params.append(float(rng.randrange(4)))
            params.append(0.0)
        else:
            params.append(0.0)
            params.append(0.0)
    out, applied = AI.augment_randaugment(img, n=n, magnitude=magnitude, rng=random.Random(1111))
    yield {
        "name": "randaugment_n3",
        "func": "randaugment",
        "C": 2, "H": 4, "W": 5,
        "n": n,
        "ops": ops,
        "params": params,
        "x": flat(img),
        "oh": len(out[0]), "ow": len(out[0][0]),
        "out": flat(out),
        "applied": applied,
    }


def main():
    cases = []
    for fn in (case_brightness, case_contrast, case_shift, case_rotate,
               case_cutout, case_cutmix, case_mixup, case_randaugment):
        cases.extend(fn())
    data = {
        "meta": {
            "source": "AI.py",
            "note": "随机量（lam / 裁剪框 / 洞位置 / 操作与强度）由参考脚本注入，"
                    "C 侧按同样参数执行，故可逐位对拍",
            "cases": len(cases),
        },
        "cases": cases,
    }
    out = sys.argv[1] if len(sys.argv) > 1 else "augment_reference.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write("\n")
    print("wrote %s (%d cases)" % (out, len(cases)))


if __name__ == "__main__":
    main()
