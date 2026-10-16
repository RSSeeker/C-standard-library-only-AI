# -*- coding: utf-8 -*-
"""导出 Python 参考数据，供 C 实现做数值等价性比对

用法：python tools/c_reference.py [输出路径]
默认写到 c/reference.txt

Python 版 AI.py 的位置按以下顺序确定：
  1. 环境变量 AI_PY_DIR
  2. 默认的 ../Python-standard-library-only-AI-main（与 AI_C 同级）
"""
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)
from AI import MLP, Model          # noqa: E402
from AI import _mlp_backward       # noqa: E402


def fmt(v):
    return repr(float(v))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "c", "reference.txt")

    # ---------- 固定模型与数据（全部由种子决定，可复现） ----------
    random.seed(20240607)
    sizes = [4, 5, 3]
    acts = ["leaky_relu", "linear"]
    mlp = MLP(sizes, activations=acts)
    model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.05, task="regression")

    n_samples = 6
    X = [[random.uniform(-1, 1) for _ in range(sizes[0])] for _ in range(n_samples)]
    Y = [[random.uniform(-1, 1) for _ in range(sizes[-1])] for _ in range(n_samples)]

    K = 5          # 全批量 Adam 步数
    lr = 0.05

    # ---------- 参考前向 ----------
    fwd = [model.forward(x) for x in X]

    # ---------- 参考梯度：N 个样本的累加（未除以 N） ----------
    acc = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)] for layer in mlp.layers]
    accb = [[0.0] * layer.fan_out for layer in mlp.layers]
    for i in range(n_samples):
        pred = model.forward(X[i])
        _, grad = model._loss(pred, Y[i])
        grads = model.backward(grad)          # [(gw, gb), ...]
        for li, (gw, gb) in enumerate(grads):
            for j in range(len(gw)):
                row = acc[li][j]
                g_row = gw[j]
                for k in range(len(g_row)):
                    row[k] += g_row[k]
                accb[li][j] += gb[j]

    # ---------- 参考训练：K 轮全批量 Adam ----------
    model2 = Model(MLP(sizes, activations=acts), loss_fn="mse", optimizer="adam",
                   lr=lr, task="regression")
    # 用同一份初始权重
    for li, layer in enumerate(model2.model.layers):
        layer.weights = [list(r) for r in mlp.layers[li].weights]
        layer.biases = list(mlp.layers[li].biases)
    model2.optimizer = type(model2.optimizer)(model2.model, lr=lr)
    model2.fit(X, Y, epochs=K, batch_size=n_samples, shuffle=False,
               val_split=0.0, patience=0, verbose=False)

    lines = []
    lines.append("AI_REF 1")
    lines.append(f"n_layers {len(mlp.layers)}")
    lines.append("sizes " + " ".join(str(s) for s in sizes))
    lines.append("acts " + " ".join(str({"leaky_relu": 0, "relu": 1,
                                          "softmax": 2, "linear": 3}[mn.activation_name])
                                    for mn in mlp.layers))
    lines.append("W " + " ".join(fmt(w) for layer in mlp.layers for row in layer.weights for w in row))
    lines.append("b " + " ".join(fmt(v) for layer in mlp.layers for v in layer.biases))
    lines.append(f"n_samples {n_samples}")
    lines.append(f"in_dim {sizes[0]}")
    lines.append(f"out_dim {sizes[-1]}")
    lines.append(f"steps {K}")
    lines.append(f"lr {lr!r}")
    lines.append("X " + " ".join(fmt(v) for row in X for v in row))
    lines.append("Y " + " ".join(fmt(v) for row in Y for v in row))
    lines.append("out " + " ".join(fmt(v) for row in fwd for v in row))
    lines.append("gradW " + " ".join(fmt(v) for li, _ in enumerate(mlp.layers)
                                     for row in acc[li] for v in row))
    lines.append("gradb " + " ".join(fmt(v) for li, _ in enumerate(mlp.layers) for v in accb[li]))
    lines.append("W_after " + " ".join(fmt(w) for layer in model2.model.layers
                                       for row in layer.weights for w in row))
    lines.append("b_after " + " ".join(fmt(v) for layer in model2.model.layers
                                       for v in layer.biases))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"参考数据已写入 {out_path}")
    print(f"  模型 {sizes} / {acts}，{n_samples} 个样本，{K} 步全批量 Adam(lr={lr})")
    print(f"  前向首样本输出: {[round(v, 8) for v in fwd[0]]}")


if __name__ == "__main__":
    main()
