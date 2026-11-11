# -*- coding: utf-8 -*-
"""从 AI.py 导出 KAN（Kolmogorov-Arnold Network）参考值 -> c/kan_reference.json

导出：
  1) 单层 KANLayer：注入确定参数后的前向输出、反向梯度、SGD 一步后参数；
  2) 多层 KAN：注入参数后的前向输出、MSE 损失、输入梯度、各层梯度、
     SGD 一步后参数。
参数由公式生成，不依赖随机数，便于 C 端逐位对拍（见
c/tests/compare_kan.c）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(C_DIR)
PY_REPO = os.path.join(os.path.dirname(REPO), "Python-standard-library-only-AI-main")
sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

CASES = []


def flat3(matrix):
    out = []
    for plane in matrix:
        for row in plane:
            out.extend(float(v) for v in row)
    return out


def flat_rows(rows):
    out = []
    for row in rows:
        out.extend(float(v) for v in row)
    return out


def layer_params(layer, offset):
    in_dim, out_dim, nb = layer.in_dim, layer.out_dim, layer.num_basis
    coeff = []
    for i in range(out_dim):
        plane = []
        for j in range(in_dim):
            row = []
            for b in range(nb):
                val = (0.05 * (i + 1) + 0.02 * (j + 1) - 0.01 * b +
                       0.001 * ((i + 1) * 7 + (j + 1) * 3 + b * 5 + offset))
                row.append(round(val, 12))
            plane.append(row)
        coeff.append(plane)
    biases = [round(0.1 * (i - 1) + 0.01 * offset, 12) for i in range(out_dim)]
    layer.coeff = coeff
    layer.biases = biases
    layer.grad_coeff = None
    layer.grad_biases = None
    layer._cache = None
    return {
        "coeff": flat3(coeff),
        "biases": [float(v) for v in biases],
        "in_dim": in_dim,
        "out_dim": out_dim,
        "num_grid": layer.num_grid,
        "spline_order": layer.spline_order,
        "grid_range": layer.grid_range,
        "num_basis": nb,
    }


def add_layer_case():
    num_grid, order, grid_range = 4, 2, 1.0
    layer = AI.KANLayer(2, 3, num_grid=num_grid, spline_order=order,
                        grid_range=grid_range, seed=None)
    params = layer_params(layer, 0)
    xs = [[0.0, -0.5], [0.25, 0.75], [-1.0, 1.0], [0.1, -0.4]]
    dout = [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5],
            [-0.75, 0.5, 1.0], [0.0, -2.0, 0.5]]
    out = layer.forward(xs)
    layer.zero_grad()
    dx = layer.backward(dout)
    grads = {
        "grad_coeff": flat3(layer.grad_coeff),
        "grad_biases": [float(v) for v in layer.grad_biases],
    }
    layer.sgd_step(0.05)
    post = {
        "coeff": flat3(layer.coeff),
        "biases": [float(v) for v in layer.biases],
    }
    CASES.append({
        "kind": "layer",
        "name": "kan_layer_2x3",
        "tol": 1e-15,
        "params": params,
        "xs": flat_rows(xs),
        "n": len(xs),
        "dout": flat_rows(dout),
        "out": flat_rows(out),
        "dx": flat_rows(dx),
        "grads": grads,
        "post": post,
        "param_count": layer.param_count(),
    })


def add_model_case():
    sizes = [2, 3, 2]
    num_grid, order, grid_range = 4, 2, 1.0
    kan = AI.KAN(sizes, num_grid=num_grid, spline_order=order,
                 grid_range=grid_range, seed=None)
    layers_params = []
    for idx, layer in enumerate(kan.layers):
        layers_params.append(layer_params(layer, idx * 3))
    xs = [[0.2, -0.3], [0.6, 0.1], [-0.8, 0.4]]
    ys = [[0.3, -0.2], [-0.5, 0.7], [0.9, 0.1]]
    pred = kan.forward_batch(xs)
    kan.zero_grad()
    loss = kan.loss_and_backward(xs, ys)
    grads = []
    for layer in kan.layers:
        grads.append({
            "grad_coeff": flat3(layer.grad_coeff),
            "grad_biases": [float(v) for v in layer.grad_biases],
        })
    n = len(xs)
    out_dim = sizes[-1]
    dout = [[2.0 * (pred[r][k] - ys[r][k]) / n for k in range(out_dim)]
            for r in range(n)]
    pred2 = kan.forward_batch(xs)
    assert pred2 == pred
    kan.zero_grad()
    dx = kan.backward_batch(dout)
    kan.sgd_step(0.02)
    post = []
    for layer in kan.layers:
        post.append({
            "coeff": flat3(layer.coeff),
            "biases": [float(v) for v in layer.biases],
        })
    CASES.append({
        "kind": "model",
        "name": "kan_2x3x2",
        "tol": 1e-15,
        "sizes": sizes,
        "num_grid": num_grid,
        "spline_order": order,
        "grid_range": grid_range,
        "layers_params": layers_params,
        "xs": flat_rows(xs),
        "ys": flat_rows(ys),
        "n": n,
        "pred": flat_rows(pred),
        "loss": loss,
        "dx": flat_rows(dx),
        "grads": grads,
        "post": post,
        "param_count": kan.param_count(),
    })


def main():
    add_layer_case()
    add_model_case()
    out_path = sys.argv[1] if len(sys.argv) > 1 else "kan_reference.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, ensure_ascii=False)
    print("[kan_reference] wrote %d cases -> %s" % (len(CASES), out_path))


if __name__ == "__main__":
    main()
