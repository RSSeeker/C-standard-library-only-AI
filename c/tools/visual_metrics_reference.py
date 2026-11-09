"""从 AI.py 导出 LPIPS / FVD 轻量代理参考值 -> c/visual_metrics_reference.json

图像/帧都是确定性生成（固定种子）的嵌套列表，JSON 里以通道主序扁平数组
存储；C 侧只读数组、不重新生成。数值用 Python repr 写出，JSON 解析回来
是双精度精确往返。
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(C_DIR)
PY_REPO = os.path.join(os.path.dirname(REPO), "Python-standard-library-only-AI-main")
sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

CASES = []


def flat(img):
    c = len(img)
    h = len(img[0])
    w = len(img[0][0])
    return dict(channels=c, height=h, width=w,
                pixels=[img[k][r][col] for k in range(c)
                        for r in range(h) for col in range(w)])


def make_img(channels, h, w, seed, scale=0.4):
    rng = random.Random(seed)
    return [[[scale * rng.uniform(-1.0, 1.0) for _ in range(w)]
             for _ in range(h)] for _ in range(channels)]


def add_noise(img, seed, sigma):
    rng = random.Random(seed)
    return [[[v + sigma * rng.gauss(0.0, 1.0) for v in row] for row in ch]
            for ch in img]


def add(name, func, value, tol=1e-12, **kw):
    case = dict(name=name, func=func, tol=tol)
    case.update(kw)
    case["value"] = value
    CASES.append(case)


def lpips_case(name, a, b, scales=(1, 2, 4), weights=None):
    kw = dict(a=flat(a), b=flat(b), scales=list(scales))
    if weights is not None:
        kw["weights"] = list(weights)
    value = AI.lpips_lite(a, b, scales=scales, weights=weights)
    add(name, "lpips", value, **kw)


def fvd_case(name, fa, fb):
    value = AI.fvd_lite(fa, fb)
    add(name, "fvd", value, frames_a=[flat(f) for f in fa],
        frames_b=[flat(f) for f in fb])


A = make_img(1, 8, 8, seed=3)
A3 = make_img(3, 8, 8, seed=4)
lpips_case("lpips_identical", A, A)
lpips_case("lpips_noise_small", A, add_noise(A, seed=5, sigma=0.02))
lpips_case("lpips_noise_large", A, add_noise(A, seed=5, sigma=0.30))
lpips_case("lpips_rgb", A3, add_noise(A3, seed=6, sigma=0.10))
lpips_case("lpips_weights", A, add_noise(A, seed=7, sigma=0.10),
           scales=(1, 2), weights=(1.0, 0.0))
lpips_case("lpips_scale1", A, add_noise(A, seed=7, sigma=0.10), scales=(1,))

frames = [make_img(1, 8, 8, seed=10 + i) for i in range(4)]
frames_shift = [[[[v + 0.5 for v in row] for row in ch] for ch in fr] for fr in frames]
frames_noise = [add_noise(fr, seed=20 + i, sigma=0.10) for i, fr in enumerate(frames)]
frames_rgb = [make_img(3, 8, 8, seed=30 + i) for i in range(3)]
frames_rgb2 = [add_noise(fr, seed=40 + i, sigma=0.20) for i, fr in enumerate(frames_rgb)]

fvd_case("fvd_identical", frames, frames)
fvd_case("fvd_shift", frames, frames_shift)
fvd_case("fvd_noise", frames, frames_noise)
fvd_case("fvd_rgb", frames_rgb, frames_rgb2)

out = {
    "meta": {
        "source": "AI.py",
        "generated_by": "c/tools/visual_metrics_reference.py",
        "cases": len(CASES),
        "note": "图像以通道主序扁平数组存储；LPIPS/FVD 均无预训练权重，"
                "只做相对比较；Python 的 ** 2 走 CPython 的 UCRT pow，MinGW "
                "的 msvcrt pow 在少数输入有 ≤1 ulp 末位差（本次实测 1.4e-17），"
                "故统一容差 1e-12（Linux glibc 下通常逐位一致）",
    },
    "cases": CASES,
}

path = os.path.join(C_DIR, "visual_metrics_reference.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=False)
    f.write("\n")
print("wrote", path, "cases:", len(CASES))
