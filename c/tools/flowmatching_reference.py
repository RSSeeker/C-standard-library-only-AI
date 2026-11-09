"""C 侧 Flow Matching 对拍参考：从 AI.py 生成确定性用例。

产物 c/flowmatching_reference.json，供 tests/compare_flowmatching.c 使用：
  - 三层 MLP 初始权重（data_dim+t_dim → hidden → hidden → data_dim）
  - 速度场/插值/目标速度
  - 训练一步的损失与梯度
  - 一次 sgd_step 后的权重与速度场
  - Euler / Heun 采样（给定初值，避免跨语言 RNG 差异）

用法：python tools/flowmatching_reference.py flowmatching_reference.json
"""
import json
import os
import random
import sys

PY_REPO = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_REPO not in sys.path:
    sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

DATA_DIM, HIDDEN, T_DIM = 3, 5, 4


class _FixedRng:
    """让 sample() 的初始噪声固定为给定向量。"""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def random(self):
        return 0.5

    def gauss(self, mu, sigma):
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return mu + sigma * v


def dump_layers(model):
    layers = []
    for layer in model.layers:
        layers.append({
            "w": [[float(v) for v in row] for row in layer.weights],
            "b": [float(v) for v in layer.biases],
        })
    return layers


def dump_grads(fm):
    grads = []
    for item in fm._g:
        if item is None:
            grads.append(None)
            continue
        gw, gb = item
        grads.append({
            "w": [[float(v) for v in row] for row in gw],
            "b": [float(v) for v in gb],
        })
    return grads


random.seed(42)
fm = AI.FlowMatching(DATA_DIM, HIDDEN, T_DIM)
layers = dump_layers(fm)

cases = []

x_t = [0.3, -0.2, 0.5]
cases.append({"name": "velocity", "func": "velocity", "x_t": x_t, "t": 0.25,
              "expected": [float(v) for v in fm.velocity(list(x_t), 0.25)]})

x0 = [0.1, -0.4, 0.9]
noise = [1.0, 0.5, -0.8]
cases.append({"name": "interpolate", "func": "interpolate", "x0": x0,
              "noise": noise, "t": 0.3,
              "expected": [float(v) for v in fm.interpolate(x0, noise, 0.3)]})
cases.append({"name": "target", "func": "target", "x0": x0, "noise": noise,
              "expected": [float(v) for v in fm.flow_target(x0, noise)]})

fm.zero_grad()
loss = fm.train_step(list(x0), t=0.4, noise=list(noise))
cases.append({"name": "train_step", "func": "train_step", "x0": x0,
              "noise": noise, "t": 0.4, "expected_loss": float(loss),
              "expected_grads": dump_grads(fm)})

lr = 0.01
fm.sgd_step(lr)
cases.append({"name": "sgd_after", "func": "sgd", "lr": lr,
              "expected_layers": dump_layers(fm),
              "expected_velocity": [float(v) for v in fm.velocity(list(x_t), 0.25)]})

x_init = [0.2, -0.1, 0.4]
cases.append({"name": "sample_euler", "func": "sample", "x_init": x_init,
              "steps": 5, "solver": "euler",
              "expected": [float(v) for v in
                           fm.sample(n=1, steps=5, solver="euler",
                                     rng=_FixedRng(x_init))[0]]})
cases.append({"name": "sample_heun", "func": "sample", "x_init": x_init,
              "steps": 4, "solver": "heun",
              "expected": [float(v) for v in
                           fm.sample(n=1, steps=4, solver="heun",
                                     rng=_FixedRng(x_init))[0]]})

doc = {
    "tol": 1e-12,
    "model": {"data_dim": DATA_DIM, "hidden_dim": HIDDEN, "t_dim": T_DIM,
              "param_count": 0, "layers": layers},
    "cases": cases,
}
doc["model"]["param_count"] = sum(
    layer.fan_in * layer.fan_out + layer.fan_out for layer in fm.layers)

out_path = sys.argv[1] if len(sys.argv) > 1 else "flowmatching_reference.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"wrote {out_path}: {len(cases)} cases")
