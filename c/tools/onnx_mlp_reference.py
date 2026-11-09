# -*- coding: utf-8 -*-
"""从 AI.py 导出 MLP→ONNX 的参考字节与执行结果 -> c/onnx_mlp_reference.json

每个用例给出：层结构、权重/偏置、输入行、Python 导出的 ONNX 字节（hex）、
Python 解析出的结构摘要、以及 Python 执行器算出的输出。C 侧对拍：
  - export 用例：C 导出的字节必须与 Python 字节逐字节相同；
  - 所有用例：C 解析 Python 字节后执行，输出按容差比较。
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


def fixed_weights(layer_sizes, seed):
    """确定性权重：用简单的整数/小数表，不依赖随机数。"""
    layers = []
    for k in range(len(layer_sizes) - 1):
        fan_out, fan_in = layer_sizes[k + 1], layer_sizes[k]
        w = []
        for r in range(fan_out):
            row = []
            for c in range(fan_in):
                row.append(round(((r * 7 + c * 3 + seed + k * 5) % 11 - 5) * 0.1, 6))
            w.append(row)
        b = [round(((r * 5 + seed + k) % 7 - 3) * 0.05, 6) for r in range(fan_out)]
        layers.append({"w": w, "b": b})
    return layers


def add_case(name, layer_sizes, activations, seed, rows, export=True,
             tolerance=0.0):
    model = AI.MLP(list(layer_sizes), list(activations))
    weights = fixed_weights(layer_sizes, seed)
    for k, layer in enumerate(model.layers):
        layer.weights = [list(row) for row in weights[k]["w"]]
        layer.biases = list(weights[k]["b"])
        layer.fan_in = layer_sizes[k]
        layer.fan_out = layer_sizes[k + 1]

    data = AI.export_onnx_mlp(model)
    parsed = AI.onnx_mlp_to_dict(data)
    forward = AI.onnx_mlp_forward(data, rows)
    graph = parsed["graph"]
    CASES.append({
        "name": name,
        "layer_sizes": list(layer_sizes),
        "activations": list(activations),
        "weights": weights,
        "rows": rows,
        "export": export,
        "tolerance": tolerance,
        "export_hex": data.hex(),
        "ir_version": parsed["ir_version"],
        "opset": parsed["opset"],
        "producer": parsed["producer"],
        "graph_name": graph["name"],
        "node_op_types": [n["op_type"] for n in graph["nodes"]],
        "node_names": [n["name"] for n in graph["nodes"]],
        "n_tensors": len(graph["initializers"]),
        "tensors": [[nm, t["dims"]] for nm, t in graph["initializers"].items()],
        "input_name": graph["inputs"][0]["name"],
        "input_dims": graph["inputs"][0]["dims"],
        "output_name": graph["outputs"][0]["name"],
        "output_dims": graph["outputs"][0]["dims"],
        "forward": forward,
    })


# 纯 relu / linear：数值路径只含乘加与 max，应可逐位一致
add_case("mlp_relu_linear", [3, 4, 2], ["relu", "linear"], 1,
         [[0.2, -0.4, 0.7], [-0.1, 0.3, 0.05]])
# leaky_relu + softmax：softmax 含 exp，容差 1e-12
add_case("mlp_leaky_softmax", [2, 3, 3], ["leaky_relu", "softmax"], 2,
         [[0.5, -0.2], [0.1, 0.4], [-0.3, 0.25]], tolerance=1e-12)
# 三层 relu / linear
add_case("mlp_three_layers", [2, 3, 2, 1], ["relu", "relu", "linear"], 3,
         [[0.4, 0.1], [-0.6, 0.2], [0.05, -0.9]])
# sigmoid 只在 Python 侧导出（C 的 Activation 枚举没有 sigmoid），C 侧只解析/执行
add_case("mlp_sigmoid_parse_only", [2, 3, 1], ["sigmoid", "linear"], 4,
         [[0.3, -0.5], [0.8, 0.1]], export=False, tolerance=1e-12)
# 纯 linear：Identity 链
add_case("mlp_linear_only", [3, 2], ["linear"], 5,
         [[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]], export=True, tolerance=0.0)

OUT = os.path.join(C_DIR, "onnx_mlp_reference.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES}, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT}: {len(CASES)} cases")
