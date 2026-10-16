# -*- coding: utf-8 -*-
"""对比 C 侧 tests/test_text.exe dump 的输出与 Python 版 TextProcessor / _softmax_with_temp"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
sys.path.insert(0, AI_PY_DIR)
from AI import TextProcessor, _softmax_with_temp      # noqa: E402

TEXT = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。"
C_DIR = os.path.abspath(os.path.join(_here, ".."))

# ---- C 侧输出 ----
exe = os.path.join(C_DIR, "tests", "test_text.exe")
r = subprocess.run([exe, "dump"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=C_DIR)
if r.returncode != 0:
    sys.exit(f"C dump 失败: {r.stderr}")
c_out = {}
for line in r.stdout.splitlines():
    if not line.strip():
        continue
    key, _, rest = line.partition(" ")
    c_out[key] = rest

# ---- Python 侧同样计算 ----
tp = TextProcessor(TEXT)
p_text_len = len(TEXT)
p_vocab = tp.vocab_size
p_chars = [ord(c) for c in tp.chars]
p_onehot = [tp.char_to_idx[tp.chars[i]] for i in range(min(8, tp.vocab_size))]
p_indices = tp.encode_indices(TEXT)[:20]

fails = []


def check(name, a, b):
    ok = a == b
    if not ok:
        fails.append(name)
    print(f"  {name}: {'一致' if ok else '不一致'}")
    if not ok:
        print(f"    C  = {a}")
        print(f"    Py = {b}")


print("[对比] C 版 text.c vs Python TextProcessor")
check("文本长度(码点)", int(c_out["TEXT_LEN"]), p_text_len)
check("词表大小", int(c_out["VOCAB"]), p_vocab)
check("词表码点序列", [int(v) for v in c_out["CHARS"].split()], p_chars)
check("前 8 个 one-hot 位置", [int(v) for v in c_out["ONEHOT_POS"].split()], p_onehot)
check("索引序列前 20", [int(v) for v in c_out["INDICES"].split()], p_indices)

# softmax_with_temp 逐位比对
logits = [0.5, -1.25, 2.0, 0.0, 0.75]
for T in (1.0, 0.5, 1.0):
    pass
for line in r.stdout.splitlines():
    if line.startswith("SOFTMAX"):
        parts = line.split()
        T = float(parts[1].split("=")[1])
        c_probs = [float(v) for v in parts[2:]]
        p_probs = _softmax_with_temp(logits, T)
        worst = max(abs(a - b) for a, b in zip(c_probs, p_probs))
        print(f"  softmax(T={T}) 最大差异: {worst:.3e}")
        if worst != 0.0:
            fails.append(f"softmax T={T}")

print(f"\n结论: {'全部一致' if not fails else '存在不一致: ' + str(fails)}")
sys.exit(0 if not fails else 1)
