#!/usr/bin/env python3
"""生成 Pix2Pix 的 C 对拍参考数据。

用法: python tools/pix2pix_reference.py pix2pix_reference.json
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


def dump_convt(c):
    return {"w": flat(c.weights), "b": flat(c.biases)}


def image(rows, cols, f):
    return [[[f(r, c) for c in range(cols)] for r in range(rows)]]


def dump_gen(g):
    return {
        "enc1": dump_conv(g.enc1),
        "enc2": dump_conv(g.enc2),
        "dec1": dump_convt(g.dec1),
        "dec2": dump_convt(g.dec2),
    }


def dump_disc(d):
    return {
        "conv1": dump_conv(d.conv1),
        "conv2": dump_conv(d.conv2),
        "conv3": dump_conv(d.conv3),
    }


def dump_grid(grid):
    return {"rows": len(grid), "cols": len(grid[0]), "data": flat(grid)}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "pix2pix_reference.json"

    gen = AI.Pix2PixGenerator(in_channels=1, out_channels=1, base=4, seed=101)
    disc = AI.PatchDiscriminator(in_channels=1, base=4, seed=202)

    size = 8
    source = image(size, size, lambda r, c: 0.10 + 0.06 * r - 0.03 * c)
    target = image(size, size, lambda r, c: 0.90 - 0.05 * r + 0.02 * c)

    fake0 = gen.forward(source)
    scores_real = disc.forward(target)
    scores_fake0 = disc.forward(fake0)

    d_loss, g_loss = None, None
    gen_before = dump_gen(gen)
    disc_before = dump_disc(disc)

    d_loss, g_loss = None, None
    result = AI.train_pix2pix_step(
        gen, disc, source, target,
        lambda_l1=10.0, lr_g=0.01, lr_d=0.01,
    )
    d_loss = result["d_loss"]
    g_loss = result["g_loss"]
    gen_after = dump_gen(gen)
    disc_after = dump_disc(disc)
    fake_after = gen.forward(source)

    payload = {
        "param": {
            "gen": gen.param_count(),
            "disc": disc.param_count(),
        },
        "input": {"size": size, "source": flat(source), "target": flat(target)},
        "init": {"gen": gen_before, "disc": disc_before},
        "forward": {
            "fake": flat(fake0),
            "real_scores": dump_grid(scores_real),
            "fake_scores": dump_grid(scores_fake0),
        },
        "step": {
            "lambda_l1": 10.0,
            "lr_g": 0.01,
            "lr_d": 0.01,
            "d_loss": d_loss,
            "g_loss": g_loss,
            "gen_after": gen_after,
            "disc_after": disc_after,
            "fake_after": flat(fake_after),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote", out_path)
    print("param gen/disc =", payload["param"]["gen"], payload["param"]["disc"])
    print("d_loss/g_loss = %.12f / %.12f" % (d_loss, g_loss))


if __name__ == "__main__":
    main()
