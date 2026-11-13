#!/usr/bin/env python3
"""生成 CycleGAN 一步训练的 C 对拍参考数据。

用法: python tools/cyclegan_reference.py cyclegan_reference.json
"""
import json
import os
import sys

import _py_path  # noqa: F401  (把 Python 主仓库加入 sys.path)

import AI  # noqa: E402


def flat(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(flat(v))
    return out


def dump_conv(c):
    return {"w": flat(c.weights), "b": flat(c.biases)}


def image(rows, cols, f):
    return [[[f(r, c) for c in range(cols)] for r in range(rows)]]


def dump_gen(g):
    return {
        "enc1": dump_conv(g.enc1),
        "enc2": dump_conv(g.enc2),
        "dec1": dump_conv(g.dec1),
        "dec2": dump_conv(g.dec2),
    }


def dump_disc(d):
    return {
        "conv1": dump_conv(d.conv1),
        "conv2": dump_conv(d.conv2),
        "conv3": dump_conv(d.conv3),
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "cyclegan_reference.json"

    g_ab = AI.Pix2PixGenerator(in_channels=1, out_channels=1, base=4, seed=101)
    g_ba = AI.Pix2PixGenerator(in_channels=1, out_channels=1, base=4, seed=303)
    d_a = AI.PatchDiscriminator(in_channels=1, base=4, seed=202)
    d_b = AI.PatchDiscriminator(in_channels=1, base=4, seed=404)

    size = 8
    real_a = image(size, size, lambda r, c: 0.15 + 0.05 * r - 0.02 * c)
    real_b = image(size, size, lambda r, c: 0.85 - 0.04 * r + 0.03 * c)

    before = {
        "g_ab": dump_gen(g_ab),
        "g_ba": dump_gen(g_ba),
        "d_a": dump_disc(d_a),
        "d_b": dump_disc(d_b),
    }

    result = AI.train_cyclegan_step(
        g_ab, g_ba, d_a, d_b, real_a, real_b,
        lambda_cycle=10.0, lambda_identity=5.0, lr_g=0.01, lr_d=0.01,
    )

    after = {
        "g_ab": dump_gen(g_ab),
        "g_ba": dump_gen(g_ba),
        "d_a": dump_disc(d_a),
        "d_b": dump_disc(d_b),
    }
    fake_b = g_ab.forward(real_a)
    fake_a = g_ba.forward(real_b)

    payload = {
        "meta": {"source": "cyclegan_reference.py", "tolerance": 1e-9},
        "model": {"channels": 1, "height": size, "width": size, "base": 4},
        "real": {"a": flat(real_a), "b": flat(real_b)},
        "before": before,
        "losses": result,
        "after": after,
        "fake_after": {"fake_b": flat(fake_b), "fake_a": flat(fake_a)},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path, "d_loss", result["d_loss"], "g_loss", result["g_loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
