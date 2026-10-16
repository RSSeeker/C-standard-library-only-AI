# -*- coding: utf-8 -*-
"""JSON 存档互通测试的 Python 侧

与 c/tests/test_json.c 配合，验证两个方向：
  C 写 -> Python 读   （check 模式）
  Python 写 -> C 读   （write 模式）

用法：
  python tools/json_interop.py check   # 读 C 写的 c_model.json / c_forward.txt
  python tools/json_interop.py write   # 写 py_model.json / py_forward.txt 给 C 读
"""
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)
from AI import MLP, load_model, save_model      # noqa: E402

C_DIR = os.path.abspath(os.path.join(_here, ".."))
TEST_X = [0.25, -0.5, 0.75, 1.0]


def read_forward(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split()
    vals = [float(v) for v in lines]
    return vals[:4], vals[4:7]


def mode_check():
    """读 C 写的模型，验证权重与 Python 一致，且前向输出一致"""
    model_path = os.path.join(C_DIR, "c_model.json")
    fwd_path = os.path.join(C_DIR, "c_forward.txt")
    if not os.path.isfile(model_path):
        sys.exit(f"找不到 {model_path}（先跑 test_json write）")

    mlp = load_model(model_path)
    x_ref, out_ref = read_forward(fwd_path)
    out = mlp.forward(TEST_X)

    worst = max(abs(a - b) for a, b in zip(out, out_ref))
    print(f"[check] C 写的 JSON 被 Python 读取: {os.path.basename(model_path)}")
    print(f"  层结构 = {mlp.layer_sizes}")
    print(f"  激活   = {[l.activation_name for l in mlp.layers]}")
    print(f"  Python 前向 = {[round(v, 12) for v in out]}")
    print(f"  C 前向      = {[round(v, 12) for v in out_ref]}")
    print(f"  最大差异 = {worst:.3e}")
    print(f"  -> {'PASS' if worst <= 1e-12 else 'FAIL'}")
    return 0 if worst <= 1e-12 else 1


def mode_write():
    """用 Python 造一个模型存成 JSON，并写出参考前向输出"""
    mlp = MLP([4, 5, 3], activations=["relu", "softmax"])
    # 与 C 侧不同的确定性权重（含 softmax 输出层，覆盖激活名映射）
    for li, layer in enumerate(mlp.layers):
        for j in range(layer.fan_out):
            for i in range(layer.fan_in):
                layer.weights[j][i] = 0.007 * ((li + 2) * 1000 + j * 100 + i) - 0.3
            layer.biases[j] = 0.002 * ((li + 2) * 100 + j) - 0.07

    model_path = os.path.join(C_DIR, "py_model.json")
    save_model(mlp, model_path)
    out = mlp.forward(TEST_X)

    fwd_path = os.path.join(C_DIR, "py_forward.txt")
    with open(fwd_path, "w", encoding="utf-8") as f:
        f.write(" ".join(repr(v) for v in TEST_X) + "\n")
        f.write(" ".join(repr(v) for v in out) + "\n")
    print(f"[write] 已写出 {os.path.basename(model_path)} / {os.path.basename(fwd_path)}")
    print(f"  前向 = {out}")
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return mode_check()
    if mode == "write":
        return mode_write()
    sys.exit(f"未知模式: {mode}（可用 check/write）")


if __name__ == "__main__":
    sys.exit(main())
