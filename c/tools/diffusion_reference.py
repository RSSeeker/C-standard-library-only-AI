"""C 侧扩散采样器对拍参考：从 AI.py 生成确定性用例。

产物 c/diffusion_reference.json，供 tests/compare_diffusion.c 使用：
  - diffusion_sigmas 调度
  - edm_euler_sampler / edm_heun_sampler / dpm_solver_sampler
  - consistency_training_loss / consistency_sampler
  - 参数非法时返回错误码的用例

去噪函数固定为 ``D(x, sigma) = scale/(1+sigma) * x + bias``，C 侧回调同构。

用法：python tools/diffusion_reference.py diffusion_reference.json
"""
import json
import os
import sys

PY_REPO = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_REPO not in sys.path:
    sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

BIAS = [0.1, -0.2, 0.05]


def make_denoise(scale):
    def fn(x, sigma):
        k = scale / (1.0 + sigma)
        return [k * x[i] + BIAS[i] for i in range(len(x))]
    return fn


CASES = []


def add_sigmas(name, steps, sigma_min, sigma_max, rho):
    CASES.append({
        "name": name, "func": "sigmas",
        "steps": steps, "sigma_min": sigma_min, "sigma_max": sigma_max,
        "rho": rho,
        "expected": [float(v) for v in AI._diffusion_sigmas(
            steps, sigma_min, sigma_max, rho)],
    })


def add_sampler(name, func, x, steps, sigma_min, sigma_max, rho, scale):
    fn = make_denoise(scale)
    if func == "edm_euler_sampler":
        out = AI.edm_euler_sampler(fn, list(x), steps, sigma_min, sigma_max, rho)
    elif func == "edm_heun_sampler":
        out = AI.edm_heun_sampler(fn, list(x), steps, sigma_min, sigma_max, rho)
    elif func == "dpm_solver_sampler":
        out = AI.dpm_solver_sampler(fn, list(x), steps, sigma_min, sigma_max, rho)
    elif func == "consistency_sampler":
        out = AI.consistency_sampler(fn, list(x), steps, sigma_min, sigma_max, rho)
    else:
        raise AssertionError(func)
    CASES.append({
        "name": name, "func": func, "x": [float(v) for v in x],
        "steps": steps, "sigma_min": sigma_min, "sigma_max": sigma_max, "rho": rho,
        "scale": scale, "bias": BIAS,
        "expected": [float(v) for v in out],
    })


def add_cd_loss(name, x, sigma, sigma_next, student_scale, teacher_scale):
    loss = AI.consistency_training_loss(
        make_denoise(student_scale), make_denoise(teacher_scale),
        list(x), sigma, sigma_next)
    CASES.append({
        "name": name, "func": "consistency_training_loss",
        "x": [float(v) for v in x], "sigma": sigma, "sigma_next": sigma_next,
        "student_scale": student_scale, "teacher_scale": teacher_scale,
        "student_bias": BIAS, "teacher_bias": BIAS, "bias": BIAS,
        "expected": [float(loss)],
    })


def add_error(name, func, **kwargs):
    CASES.append({"name": name, "func": "error", "target": func, **kwargs})


X = [0.5, -1.0, 0.25]

add_sigmas("sigmas_basic", 5, 0.01, 10.0, 3.0)
add_sigmas("sigmas_one_step", 2, 0.002, 80.0, 7.0)
add_sigmas("sigmas_high_rho", 7, 0.1, 50.0, 12.5)

add_sampler("euler_basic", "edm_euler_sampler", X, 4, 0.01, 5.0, 2.0, 0.85)
add_sampler("euler_short", "edm_euler_sampler", [1.0, 0.0], 2, 0.05, 20.0, 3.0, 1.0)
add_sampler("heun_basic", "edm_heun_sampler", X, 5, 0.01, 5.0, 2.0, 0.85)
add_sampler("heun_zero_bias", "edm_heun_sampler", [0.3, -0.7], 3, 0.02, 8.0, 4.0, 0.6)
add_sampler("dpm_basic", "dpm_solver_sampler", X, 6, 0.01, 5.0, 2.0, 0.85)
add_sampler("dpm_short", "dpm_solver_sampler", [0.9], 2, 0.1, 10.0, 5.0, 0.5)

add_cd_loss("cd_loss_positive_next", X, 2.0, 0.5, 0.9, 0.7)
add_cd_loss("cd_loss_zero_next", X, 1.5, 0.0, 0.6, 0.8)
add_cd_loss("cd_loss_small_gap", [0.2, 0.4], 0.8, 0.79, 1.0, 0.9)

add_sampler("consistency_basic", "consistency_sampler", X, 4, 0.02, 40.0, 7.0, 0.75)

add_error("error_sigmas_steps", "sigmas", steps=1)
add_error("error_sigmas_order", "sigmas", steps=5, sigma_min=10.0, sigma_max=0.01)
add_error("error_euler_params", "edm_euler_sampler", steps=4, sigma_min=-1.0,
          sigma_max=5.0, scale=0.8)
add_error("error_cd_next", "consistency_training_loss", sigma=1.0, sigma_next=1.0)

doc = {"tol": 1e-12, "cases": CASES}
out_path = sys.argv[1] if len(sys.argv) > 1 else "diffusion_reference.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"wrote {out_path}: {len(CASES)} cases")
