# -*- coding: utf-8 -*-
"""对比 C 侧 tests/test_sched.exe dump 与 Python 版调度器/裁剪函数"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
sys.path.insert(0, AI_PY_DIR)
from AI import StepLR, ExponentialLR, CosineAnnealingLR, clip_grad_by_norm, clip_grad_by_value  # noqa: E402

C_DIR = os.path.abspath(os.path.join(_here, ".."))
exe = os.path.join(C_DIR, "tests", "test_sched.exe")
r = subprocess.run([exe, "dump"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", cwd=C_DIR)
if r.returncode != 0:
    sys.exit(f"C dump 失败: {r.stderr}")

c = {}
for line in r.stdout.splitlines():
    if not line.strip():
        continue
    parts = line.split()
    c[parts[0]] = parts[1:]

fails = []


class FakeOpt:
    def __init__(self, lr):
        self.lr = lr


def cmp_seq(name, c_list, p_list):
    worst = max(abs(a - b) for a, b in zip(c_list, p_list))
    ok = (len(c_list) == len(p_list)) and worst == 0.0
    print(f"  {name}: {'逐位一致' if ok else f'最大差异 {worst:.3e}'}")
    if not ok:
        fails.append(name)
        print(f"    C  = {c_list}")
        print(f"    Py = {p_list}")


print("[对比] C 版 sched.c vs Python 调度器/裁剪")

# StepLR
opt = FakeOpt(0.1)
s = StepLR(opt, step_size=3, gamma=0.5)
py = []
for e in range(1, 11):
    s.step(e)
    py.append(opt.lr)
cmp_seq("StepLR(step=3, gamma=0.5) 10 轮", [float(v) for v in c["STEP"]], py)

# ExponentialLR
opt = FakeOpt(0.1)
s = ExponentialLR(opt, gamma=0.9)
py = []
for e in range(1, 9):
    s.step(e)
    py.append(opt.lr)
cmp_seq("ExponentialLR(gamma=0.9) 8 轮", [float(v) for v in c["EXP"]], py)

# CosineAnnealingLR
opt = FakeOpt(0.1)
s = CosineAnnealingLR(opt, T_max=5, eta_min=0.01)
py = []
for e in range(1, 13):
    s.step(e)
    py.append(opt.lr)
cmp_seq("CosineAnnealingLR(T_max=5, eta_min=0.01) 12 轮", [float(v) for v in c["COSINE"]], py)

# 裁剪：C 侧用 (gw 6 个 + gb 2 个) 的 3->2 线性层
gw_vals = [3.0, -4.0, 0.5, 0.0, -0.25, 2.0]
gb_vals = [-1.0, 0.75]
gw = [[gw_vals[0], gw_vals[1], gw_vals[2]], [gw_vals[3], gw_vals[4], gw_vals[5]]]
gb = [gb_vals[0], gb_vals[1]]

norm = clip_grad_by_norm([(gw, gb)], 1.0)
c_norm = float(c["CLIP_NORM"][0].split("=")[1])
c_vals = [float(v) for v in c["CLIP_NORM"][1:]]
p_vals = [v for row in gw for v in row] + gb
print(f"  clip_grad_by_norm 前范数: C={c_norm!r} Py={norm!r} {'一致' if c_norm == norm else '不一致'}")
if c_norm != norm:
    fails.append("norm")
cmp_seq("clip_grad_by_norm(max=1.0)", c_vals, p_vals)

gw2 = [[gw_vals[0], gw_vals[1], gw_vals[2]], [gw_vals[3], gw_vals[4], gw_vals[5]]]
gb2 = [-1.0, 0.75]
clip_grad_by_value([(gw2, gb2)], 1.5)
cmp_seq("clip_grad_by_value(clip=1.5)", [float(v) for v in c["CLIP_VALUE"]],
        [v for row in gw2 for v in row] + gb2)

print(f"\n结论: {'全部逐位一致' if not fails else '存在不一致: ' + str(fails)}")
sys.exit(0 if not fails else 1)
