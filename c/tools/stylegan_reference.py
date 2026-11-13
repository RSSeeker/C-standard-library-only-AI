"""生成 StyleGAN 生成器前向的 C 对拍参考（JSON）。

运行：
  PYTHONPATH=/d/vscode/Python-standard-library-only-AI-main \
    python tools/stylegan_reference.py stylegan_reference.json
"""

import json
import sys

import _py_path  # noqa: F401  (把 Python 主仓库加入 sys.path)


def flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(flat(v))
        return out
    return [float(x)]


def dump_layer(layer):
    return {
        "fan_in": int(layer.fan_in),
        "fan_out": int(layer.fan_out),
        "w": flat(layer.weights),
        "b": flat(layer.biases),
    }


def dump_mlp(m):
    return [dump_layer(layer) for layer in m.layers]


def _scalar(v):
    return int(v[0]) if isinstance(v, (list, tuple)) else int(v)


def dump_conv(c):
    return {
        "in_channels": int(c.in_channels),
        "out_channels": int(c.out_channels),
        "kh": _scalar(c.kh),
        "kw": _scalar(c.kw),
        "stride": _scalar(c.stride),
        "padding": _scalar(c.padding),
        "w": flat(c.weights),
        "b": flat(c.biases),
    }


def main():
    if len(sys.argv) != 2:
        print("usage: stylegan_reference.py OUT.json", file=sys.stderr)
        return 2
    out_path = sys.argv[1]
    import AI

    gen = AI.StyleGANGenerator(latent_dim=8, d_model=8, base=4, out_channels=1, seed=3)
    z = [0.1 * i - 0.2 for i in range(8)]
    y = gen.forward(z)
    zs = [[0.05 * i for i in range(8)],
          [-0.1 + 0.03 * i for i in range(8)]]

    payload = {
        "meta": {"source": "stylegan_reference.py", "tolerance": 1e-12},
        "model": {
            "latent_dim": 8,
            "d_model": 8,
            "base": 4,
            "out_channels": 1,
            "param_count": int(gen.param_count()),
        },
        "const": flat(gen.const),
        "mapping": dump_mlp(gen.mapping),
        "style1": [dump_layer(gen.style1)],
        "style2": [dump_layer(gen.style2)],
        "conv1": dump_conv(gen.conv1),
        "conv2": dump_conv(gen.conv2),
        "to_rgb": dump_conv(gen.to_rgb),
        "forward": {"z": z, "y": flat(y)},
        "average_w": {"zs": zs,
                      "w": [float(v) for v in gen.average_w(gen, zs)]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path, "param_count", gen.param_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
