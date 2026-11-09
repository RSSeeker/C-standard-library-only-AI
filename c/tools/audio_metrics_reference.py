"""从 AI.py 导出 STOI / PESQ 参考值 -> c/audio_metrics_reference.json

信号是确定性的（固定随机种子 + 嵌入 JSON 的采样数组），C 侧只读取数组，
不重新生成，因此跨平台不受随机数实现影响。所有数值用 Python repr 写出，
JSON 解析回来是双精度精确往返。

每个用例都显式写出全部参数（含默认值），C 侧不需要猜测缺省。
"""
import json
import math
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

STOI_DEFAULTS = dict(sample_rate=16000, frame_length=256, hop_length=128,
                     n_bands=8, fmin=0.0, fmax=None, eps=1e-12)
PESQ_DEFAULTS = dict(sample_rate=16000, frame_length=256, hop_length=128,
                     n_bands=16, fmin=0.0, fmax=None, eps=1e-10)


def add(name, func, clean, degraded, defaults, tol=1e-12, **kw):
    params = dict(defaults)
    params.update(kw)
    value = AI.stoi_score(clean, degraded, **params) if func == "stoi" \
        else AI.pesq_score(clean, degraded, **params)
    case = dict(name=name, func=func, clean=clean, degraded=degraded,
                value=value, tol=tol)
    case.update(params)
    if case["fmax"] is None:
        case["fmax"] = -1.0  # C 接口里负数表示默认奈奎斯特频率
    CASES.append(case)


def white(seed, n, scale=1.0):
    rng = random.Random(seed)
    return [scale * rng.uniform(-1.0, 1.0) for _ in range(n)]


def add_noise(sig, seed, snr_db):
    rng = random.Random(seed)
    noise = [rng.gauss(0.0, 1.0) for _ in sig]
    e_sig = sum(v * v for v in sig) / len(sig)
    e_noise = sum(v * v for v in noise) / len(noise)
    factor = math.sqrt(e_sig / max(e_noise, 1e-30) / (10.0 ** (snr_db / 10.0)))
    return [sig[i] + factor * noise[i] for i in range(len(sig))]


SAMPLES = 512
BASE = white(seed=11, n=SAMPLES, scale=0.35)
SILENCE = [0.0] * 600

# ---------- STOI ----------
add("stoi_identical", "stoi", BASE, BASE, STOI_DEFAULTS)
add("stoi_same_silence", "stoi", SILENCE, SILENCE, STOI_DEFAULTS)
for snr in (20, 10, 0):
    noisy = add_noise(BASE, seed=100 + snr, snr_db=snr)
    add("stoi_snr%d" % snr, "stoi", BASE, noisy, STOI_DEFAULTS)
add("stoi_invert", "stoi", BASE, [-v for v in BASE], STOI_DEFAULTS)
add("stoi_trim", "stoi", BASE, BASE[:400], STOI_DEFAULTS)
noisy7 = add_noise(BASE, seed=7, snr_db=10)
add("stoi_fmin_fmax", "stoi", BASE, noisy7, STOI_DEFAULTS,
    frame_length=128, hop_length=64, n_bands=6, fmin=100.0, fmax=6000.0)

# ---------- PESQ ----------
# PESQ 用自然对数 log；Windows 上 MinGW 的 msvcrt 与 CPython 的 UCRT 两个
# libm 对 log 有最多 1 ulp 的末位差，会让最终分数差到 ~2e-15。这不是算法
# 差异（STOI 用 log10/pow/cos/sin，在本机逐位一致），所以 PESQ 用例放
# 1e-12 容差并在 meta 里写明；STOI 仍要求逐位一致。
PESQ_TOL = 1e-12
add("pesq_identical", "pesq", BASE, BASE, PESQ_DEFAULTS, tol=PESQ_TOL)
add("pesq_same_silence", "pesq", SILENCE, SILENCE, PESQ_DEFAULTS, tol=PESQ_TOL)
for snr in (20, 10, 0):
    noisy = add_noise(BASE, seed=200 + snr, snr_db=snr)
    add("pesq_snr%d" % snr, "pesq", BASE, noisy, PESQ_DEFAULTS, tol=PESQ_TOL)
noisy31 = add_noise(BASE, seed=31, snr_db=8)
add("pesq_hop64", "pesq", BASE, noisy31, PESQ_DEFAULTS,
    frame_length=128, hop_length=64, n_bands=12, tol=PESQ_TOL)
add("pesq_polarity", "pesq", BASE, [-v for v in BASE], PESQ_DEFAULTS, tol=PESQ_TOL)

out = {
    "meta": {
        "source": "AI.py",
        "generated_by": "c/tools/audio_metrics_reference.py",
        "cases": len(CASES),
        "note": "信号数组确定；STOI/PESQ 含 cos/sin/log，Windows 上 MinGW 的 msvcrt "
                "与 CPython 的 UCRT libm 在部分输入有 ≤1 ulp 末位差（实测 ≤5e-16），"
                "故统一容差 1e-12（Linux glibc 下通常逐位一致）；"
                "fmax=-1 表示默认奈奎斯特频率",
    },
    "cases": CASES,
}

path = os.path.join(C_DIR, "audio_metrics_reference.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=False)
    f.write("\n")
print("wrote", path, "cases:", len(CASES))
