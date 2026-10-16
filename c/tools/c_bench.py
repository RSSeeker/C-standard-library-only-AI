# -*- coding: utf-8 -*-
"""与 C 版对等的性能基准（工作负载必须与 c/bench.c 一致）

用法：python tools/c_bench.py [epochs] [samples]
Python 版位置见 c_reference.py 的说明（支持 AI_PY_DIR 环境变量）
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)
from AI import MLP, Model              # noqa: E402


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    in_dim, out_dim = 64, 8

    x = [[(i * 7 + k * 13) % 100 / 50.0 - 1.0 for k in range(in_dim)] for i in range(n)]
    y = []
    for i in range(n):
        row = []
        for k in range(out_dim):
            s = sum(x[i][(k * 8 + t) % in_dim] for t in range(8))
            row.append(s / 8.0)
        y.append(row)

    mlp = MLP([64, 256, 128, 8], activations=["leaky_relu", "leaky_relu", "linear"])
    model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.001, task="regression")
    params = sum(l.fan_in * l.fan_out + l.fan_out for l in mlp.layers)

    print(f"Python 版: MLP[64,256,128,8] 参数 {params}, {n} 样本, {epochs} 轮, batch=32, Adam(0.001)")
    t0 = time.perf_counter()
    model.fit(x, y, epochs=epochs, batch_size=32, shuffle=True, val_split=0.0,
              patience=0, verbose=False)
    dt = time.perf_counter() - t0
    r2 = model.score(x, y)
    print(f"  耗时 {dt:.3f} s，最终 R²(训练集) = {r2:.6f}")
    print(f"  每轮 {dt * 1000.0 / epochs:.1f} ms，每样本每轮 {dt * 1e6 / (epochs * n):.1f} us")


if __name__ == "__main__":
    main()
