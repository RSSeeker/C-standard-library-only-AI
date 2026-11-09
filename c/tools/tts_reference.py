"""生成 C 侧 TTS 模块（griffin_lim / mel_to_linear / TextToMel）的参考数据。

用法：python tools/tts_reference.py tts_reference.json
"""
import json
import math
import random
import sys

import AI


def flatten2(rows):
    out = []
    for row in rows:
        out.extend(float(v) for v in row)
    return out


def flatten3(x):
    out = []
    for a in x:
        for row in a:
            out.extend(float(v) for v in row)
    return out


def wave_a(n, n_fft, hop, iterations, seed):
    wave = [0.3 * math.sin(2.0 * math.pi * 2.0 * i / n)
            + 0.15 * math.cos(2.0 * math.pi * 3.0 * i / n + 0.7)
            for i in range(n)]
    win = AI._tts_hann(n_fft)
    mags, _phases = AI._tts_mag_frames(wave, n_fft, hop, win)
    return mags, win


def build_rng_cases():
    cases = []
    for seed, count in ((0, 4), (7, 3), (123456789, 5)):
        rng = random.Random(seed)
        cases.append({
            "name": f"rng_{seed}",
            "seed": seed,
            "expected": [rng.random() for _ in range(count)],
        })
    return cases


def build_griffin_cases():
    cases = []
    for name, n, n_fft, hop, iterations, seed in (
            ("gl_small", 16, 8, 4, 3, 0),
            ("gl_mid", 20, 8, 2, 8, 7),
    ):
        mags, _win = wave_a(n, n_fft, hop, iterations, seed)
        y = AI.griffin_lim(mags, n_fft=n_fft, hop_length=hop,
                           iterations=iterations, seed=seed)
        cases.append({
            "name": name,
            "magnitude": flatten2(mags),
            "n_frames": len(mags),
            "n_bins": len(mags[0]),
            "n_fft": n_fft,
            "hop": hop,
            "iterations": iterations,
            "seed": seed,
            "expected": [float(v) for v in y],
            "tol": 1e-12,
        })
    return cases


def build_griffin_errors():
    mags, _win = wave_a(16, 8, 4, 2, 0)
    flat = flatten2(mags)
    neg = list(flat)
    neg[0] = -0.5
    return [
        {"name": "gl_bad_nfft", "magnitude": flat, "n_frames": len(mags),
         "n_bins": len(mags[0]), "n_fft": 7, "hop": 4, "iterations": 2,
         "seed": 0},
        {"name": "gl_negative_mag", "magnitude": neg, "n_frames": len(mags),
         "n_bins": len(mags[0]), "n_fft": 8, "hop": 4, "iterations": 2,
         "seed": 0},
        {"name": "gl_bad_hop", "magnitude": flat, "n_frames": len(mags),
         "n_bins": len(mags[0]), "n_fft": 8, "hop": 0, "iterations": 2,
         "seed": 0},
    ]


def build_mel_cases():
    fb = [
        [0.5, 0.3, 0.1, 0.0, 0.0],
        [0.0, 0.4, 0.3, 0.2, 0.0],
        [0.0, 0.0, 0.2, 0.4, 0.3],
    ]
    mel = [[0.2, 0.3, 0.1], [0.5, 0.1, 0.4], [0.05, 0.25, 0.35]]
    lin = AI.mel_to_linear(mel, fb)
    return [{
        "name": "mel_to_linear_small",
        "mel": flatten2(mel),
        "fb": flatten2(fb),
        "n_frames": len(mel),
        "n_mels": len(fb),
        "n_bins": len(fb[0]),
        "eps": 1e-10,
        "expected": flatten2(lin),
        "tol": 1e-12,
    }], [{
        "name": "mel_negative_eps",
        "mel": flatten2(mel),
        "fb": flatten2(fb),
        "n_frames": len(mel),
        "n_mels": len(fb),
        "n_bins": len(fb[0]),
        "eps": -1.0,
    }]


def build_text_to_mel():
    m = AI.TextToMel(vocab_size=6, d_model=4, n_mels=3,
                     kernel_size=3, seed=1)
    cases = []
    for name, ids, durations in (
            ("ttm_mixed", [1, 3, 2], [2, 1, 3]),
            ("ttm_single", [0], [4]),
    ):
        mel = m.forward(ids, durations)
        cases.append({
            "name": name,
            "ids": ids,
            "durations": durations,
            "expected": flatten2(mel),
        })
    return {
        "vocab_size": 6,
        "d_model": 4,
        "n_mels": 3,
        "kernel_size": 3,
        "param_count": m.param_count(),
        "emb": flatten2(m.embedding.weight),
        "c1w": flatten3(m.conv1.weights),
        "c1b": [float(v) for v in m.conv1.biases],
        "c2w": flatten3(m.conv2.weights),
        "c2b": [float(v) for v in m.conv2.biases],
        "ow": flatten2(m.out.weights),
        "ob": [float(v) for v in m.out.biases],
        "cases": cases,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tts_reference.json"
    doc = {
        "rng_cases": build_rng_cases(),
        "griffin_cases": build_griffin_cases(),
        "griffin_errors": build_griffin_errors(),
        "mel_cases": build_mel_cases()[0],
        "mel_errors": build_mel_cases()[1],
        "text_to_mel": build_text_to_mel(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    print("wrote", out_path, "griffin=", len(doc["griffin_cases"]),
          "mel=", len(doc["mel_cases"]),
          "ttm_cases=", len(doc["text_to_mel"]["cases"]))


if __name__ == "__main__":
    main()
