/* diffusion.h —— EDM / DPM-Solver / Consistency Model 采样器（VE 参数化，纯数值）
 *
 * 与 Python AI.py 的 edm_euler_sampler / edm_heun_sampler / dpm_solver_sampler /
 * consistency_training_loss / consistency_sampler 对应。去噪函数由调用方以
 * DiffDenoiseFn 回调提供；本模块不依赖任何模型结构。
 */
#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <stddef.h>

/* 去噪回调：x 指向 n 维含噪样本，sigma 为当前噪声强度，把去噪结果写入 out。 */
typedef void (*DiffDenoiseFn)(const double *x, int n, double sigma, double *out,
                              void *ctx);

/* EDM sigma 调度：对数间隔 + rho 弯曲，返回 [sigma_max, ..., sigma_min, 0]。
 * 写入 out（容量 out_cap），返回元素个数；参数非法返回 -1。 */
int diffusion_sigmas(int steps, double sigma_min, double sigma_max, double rho,
                     double *out, int out_cap);

/* 一阶 Euler 采样器：dx/dsigma = (x - D(x, sigma)) / sigma。 */
int edm_euler_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                      const double *x, int n, int steps,
                      double sigma_min, double sigma_max, double rho,
                      double *out);

/* 二阶 Heun 采样器：Euler 预测 + 两端平均校正。 */
int edm_heun_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                     const double *x, int n, int steps,
                     double sigma_min, double sigma_max, double rho,
                     double *out);

/* 一阶指数积分器（x0 在区间内恒定的精确解）：x' = x0 + (s_next/s)*(x - x0)。 */
int dpm_solver_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                       const double *x, int n, int steps,
                       double sigma_min, double sigma_max, double rho,
                       double *out);

/* 一致性蒸馏逐样本损失：student(x,sigma) 对齐 teacher 的两步 Euler 目标。 */
int consistency_training_loss(DiffDenoiseFn student_fn, void *student_ctx,
                              DiffDenoiseFn teacher_fn, void *teacher_ctx,
                              const double *x, int n, double sigma,
                              double sigma_next, double *loss);

/* 一致性采样：按 sigma 调度逐级预测 x0 并重新加噪。 */
int consistency_sampler(DiffDenoiseFn student_fn, void *ctx,
                        const double *x, int n, int steps,
                        double sigma_min, double sigma_max, double rho,
                        double *out);

#endif /* DIFFUSION_H */
