"""生成 ProgressiveGenerator 前向的 C 对拍参考（JSON）。

运行：
  PYTHONPATH=/d/vscode/Python-standard-library-only-AI-main \
    python tools/progressive_reference.py progressive_reference.json
"""

import json
import sys


def flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(flat(v))
        return out
    return [float(x)]


def scalar(v):
    return int(v[0]) if isinstance(v, (list, tuple)) else int(v)


def dump_layer(layer):
    return {
        "fan_in": int(layer.fan_in),
        "fan_out": int(layer.fan_out),
        "w": flat(layer.weights),
        "b": flat(layer.biases),
    }


def dump_convt(c):
    return {
        "in_channels": int(c.in_channels),
        "out_channels": int(c.out_channels),
        "k": scalar(c.kernel_size),
        "stride": scalar(c.stride),
        "padding": scalar(c.padding),
        "output_padding": scalar(c.output_padding),
        "groups": int(c.groups),
        "w": flat(c.weights),
        "b": flat(c.biases),
    }


def dump_conv(c):
    return {
        "in_channels": int(c.in_channels),
        "out_channels": int(c.out_channels),
        "k": scalar(c.kernel_size),
        "stride": scalar(c.stride),
        "padding": scalar(c.padding),
        "w": flat(c.weights),
        "b": flat(c.biases),
    }


def main():
    if len(sys.argv) != 2:
        print("usage: progressive_reference.py OUT.json", file=sys.stderr)
        return 2
    out_path = sys.argv[1]
    import AI

    gen = AI.ProgressiveGenerator(latent_dim=8, channels=(8, 4), base=2,
                                  out_channels=1, seed=5)
    z = [0.1 * i - 0.3 for i in range(8)]
    y_base_1 = gen.forward(z, alpha=1.0)
    y_base_half = gen.forward(z, alpha=0.5)
    pc_base = int(gen.param_count())
    out_shape_base = [int(v) for v in gen.out_shape()]

    fc = [dump_layer(gen.fc)]
    convs = [dump_convt(c) for c in gen.convs]
    heads = [dump_conv(c) for c in gen.rgb_heads]

    gen.grow()
    pc_grown = int(gen.param_count())
    y_grown = gen.forward(z, alpha=0.6)
    out_shape_grown = [int(v) for v in gen.out_shape()]

    payload = {
        "meta": {"source": "progressive_reference.py", "tolerance": 1e-12},
        "model": {
            "latent_dim": 8,
            "base": 2,
            "out_channels": 1,
            "channels": [8, 4],
            "param_count_base": pc_base,
            "out_shape_base": out_shape_base,
        },
        "base": {
            "fc": fc,
            "convs": convs,
            "rgb_heads": heads,
        },
        "base_forward": {
            "z": z,
            "alpha1": flat(y_base_1),
            "alpha05": flat(y_base_half),
        },
        "grown": {
            "channels": [8, 4, 4],
            "param_count": pc_grown,
            "out_shape": out_shape_grown,
            "convs": [dump_convt(gen.convs[-1])],
            "rgb_heads": [dump_conv(gen.rgb_heads[-1])],
            "alpha": 0.6,
            "y": flat(y_grown),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path, "param_base", pc_base, "param_grown", pc_grown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
