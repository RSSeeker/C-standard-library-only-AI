"""C 侧 E(3) EGNN 对拍参考：从 AI.py 生成确定性用例。

产物 c/egnn_reference.json，供 tests/compare_egnn.c 使用：
  - 模型结构（d_model / d_ff / num_layers）
  - 每层三张 MLP（phi_e / phi_x / phi_h）与 readout 的初始权重
  - 一张确定性图（h、x、target）
  - 前向输出（h'、x'、scalar）、loss、一次 sgd_step(0.01) 后的输出
  - param_count

用法：python tools/egnn_reference.py egnn_reference.json
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_REPO = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_REPO not in sys.path:
    sys.path.insert(0, PY_REPO)

import AI  # noqa: E402


def dump_grads(grads):
    """把 [(gw, gb), ...] 梯度列表压平成 {w: [...], b: [...]}。"""
    out = []
    for gw, gb in grads:
        out.append({
            "w": [float(v) for row in gw for v in row],
            "b": [float(v) for v in gb],
        })
    return out


def dump_mlp(model):
    """把 MLP（或 _e3 风格的 MLP 容器）导出为 {w: [[...]], b: [...]} 层列表。"""
    layers = []
    for layer in model.layers:
        w = [[float(v) for v in row] for row in layer.weights]
        b = [float(v) for v in layer.biases]
        layers.append({"w": w, "b": b})
    return layers


def main():
    if len(sys.argv) != 2:
        print("用法：python tools/egnn_reference.py egnn_reference.json")
        return 1
    out_path = sys.argv[1]

    d_model, d_ff, num_layers = 4, 8, 2
    model = AI.EGNNModel(d_model=d_model, num_layers=num_layers, d_ff=d_ff, seed=21)

    h = [
        [0.10, -0.20, 0.30, 0.40],
        [0.25, 0.15, -0.05, 0.35],
        [-0.30, 0.20, 0.10, -0.15],
    ]
    x = [
        [0.20, 0.10, 0.00],
        [1.00, 0.30, 0.20],
        [0.40, 1.20, 0.10],
    ]
    target = 1.5
    lr = 0.01

    h_out, x_out, scalar = model.forward(h, x)
    # 训练前的权重：对拍必须用初始权重载入 C 侧
    layers_dump = [
        {
            "phi_e": dump_mlp(layer.phi_e),
            "phi_x": dump_mlp(layer.phi_x),
            "phi_h": dump_mlp(layer.phi_h),
        }
        for layer in model.layers
    ]
    readout_dump = dump_mlp(model.readout)
    loss = model.loss_and_backward(h, x, target)
    grads_dump = {
        "layers": [
            {
                "phi_e": dump_grads(layer.grad_phi_e),
                "phi_x": dump_grads(layer.grad_phi_x),
                "phi_h": dump_grads(layer.grad_phi_h),
            }
            for layer in model.layers
        ],
        "readout": dump_grads(model.grad_readout),
    }
    model.sgd_step(lr)
    h_out2, x_out2, scalar2 = model.forward(h, x)

    data = {
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "h": h,
        "x": x,
        "target": target,
        "lr": lr,
        "layers": layers_dump,
        "readout": readout_dump,
        "forward": {"h_out": h_out, "x_out": x_out, "scalar": scalar},
        "loss": loss,
        "grads": grads_dump,
        "after_step": {"h_out": h_out2, "x_out": x_out2, "scalar": scalar2},
        "param_count": model.param_count(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("wrote", out_path, "param_count", data["param_count"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
