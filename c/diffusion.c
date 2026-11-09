/* diffusion.c —— EDM / DPM-Solver / Consistency Model 采样器（VE 参数化） */
#include "diffusion.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int diff_check_sampler(int n, int steps, double sigma_min,
                              double sigma_max, double rho) {
    if (n <= 0 || steps < 2) return -1;
    if (!isfinite(sigma_min) || !isfinite(sigma_max) || !isfinite(rho)) return -1;
    if (sigma_min <= 0.0 || sigma_max <= 0.0 || rho <= 0.0) return -1;
    if (sigma_min >= sigma_max) return -1;
    return 0;
}

int diffusion_sigmas(int steps, double sigma_min, double sigma_max, double rho,
                     double *out, int out_cap) {
    if (out == NULL || out_cap < steps + 1) return -1;
    if (steps < 2 || !isfinite(sigma_min) || !isfinite(sigma_max) || !isfinite(rho)) {
        return -1;
    }
    if (sigma_min <= 0.0 || sigma_max <= 0.0 || rho <= 0.0) return -1;
    if (sigma_min >= sigma_max) return -1;
    double lo = pow(sigma_min, 1.0 / rho);
    double hi = pow(sigma_max, 1.0 / rho);
    for (int i = 0; i < steps; i++) {
        double s = hi + (lo - hi) * (double)i / (double)(steps - 1);
        out[i] = pow(s, rho);
    }
    out[steps] = 0.0;
    return steps + 1;
}

/* 单步 Euler：sigma<=0 时直接取 x0。 */
static void diff_edm_step(const double *x, const double *x0, int n, double sigma,
                          double sigma_next, double *out) {
    if (sigma <= 0.0) {
        memcpy(out, x0, (size_t)n * sizeof(double));
        return;
    }
    for (int i = 0; i < n; i++) {
        out[i] = x[i] + (sigma_next - sigma) * (x[i] - x0[i]) / sigma;
    }
}

int edm_euler_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                      const double *x, int n, int steps,
                      double sigma_min, double sigma_max, double rho,
                      double *out) {
    if (denoise_fn == NULL || x == NULL || out == NULL) return -1;
    if (diff_check_sampler(n, steps, sigma_min, sigma_max, rho) != 0) return -1;
    double *sigmas = (double *)malloc((size_t)(steps + 1) * sizeof(double));
    double *cur = (double *)malloc((size_t)n * sizeof(double));
    double *x0 = (double *)malloc((size_t)n * sizeof(double));
    double *tmp = (double *)malloc((size_t)n * sizeof(double));
    if (!sigmas || !cur || !x0 || !tmp) {
        free(sigmas); free(cur); free(x0); free(tmp);
        return -1;
    }
    if (diffusion_sigmas(steps, sigma_min, sigma_max, rho, sigmas, steps + 1) < 0) {
        free(sigmas); free(cur); free(x0); free(tmp);
        return -1;
    }
    memcpy(cur, x, (size_t)n * sizeof(double));
    for (int i = 0; i < steps; i++) {
        denoise_fn(cur, n, sigmas[i], x0, ctx);
        diff_edm_step(cur, x0, n, sigmas[i], sigmas[i + 1], tmp);
        memcpy(cur, tmp, (size_t)n * sizeof(double));
    }
    memcpy(out, cur, (size_t)n * sizeof(double));
    free(sigmas); free(cur); free(x0); free(tmp);
    return 0;
}

int edm_heun_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                     const double *x, int n, int steps,
                     double sigma_min, double sigma_max, double rho,
                     double *out) {
    if (denoise_fn == NULL || x == NULL || out == NULL) return -1;
    if (diff_check_sampler(n, steps, sigma_min, sigma_max, rho) != 0) return -1;
    double *sigmas = (double *)malloc((size_t)(steps + 1) * sizeof(double));
    double *cur = (double *)malloc((size_t)n * sizeof(double));
    double *x0 = (double *)malloc((size_t)n * sizeof(double));
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    double *x0_next = (double *)malloc((size_t)n * sizeof(double));
    if (!sigmas || !cur || !x0 || !pred || !x0_next) {
        free(sigmas); free(cur); free(x0); free(pred); free(x0_next);
        return -1;
    }
    if (diffusion_sigmas(steps, sigma_min, sigma_max, rho, sigmas, steps + 1) < 0) {
        free(sigmas); free(cur); free(x0); free(pred); free(x0_next);
        return -1;
    }
    memcpy(cur, x, (size_t)n * sizeof(double));
    for (int i = 0; i < steps; i++) {
        double sigma = sigmas[i];
        double sigma_next = sigmas[i + 1];
        denoise_fn(cur, n, sigma, x0, ctx);
        diff_edm_step(cur, x0, n, sigma, sigma_next, pred);
        if (sigma_next > 0.0) {
            denoise_fn(pred, n, sigma_next, x0_next, ctx);
            for (int j = 0; j < n; j++) {
                double a = (cur[j] - x0[j]) / sigma;
                double b = (pred[j] - x0_next[j]) / sigma_next;
                cur[j] = cur[j] + (sigma_next - sigma) * 0.5 * (a + b);
            }
        } else {
            memcpy(cur, pred, (size_t)n * sizeof(double));
        }
    }
    memcpy(out, cur, (size_t)n * sizeof(double));
    free(sigmas); free(cur); free(x0); free(pred); free(x0_next);
    return 0;
}

int dpm_solver_sampler(DiffDenoiseFn denoise_fn, void *ctx,
                       const double *x, int n, int steps,
                       double sigma_min, double sigma_max, double rho,
                       double *out) {
    if (denoise_fn == NULL || x == NULL || out == NULL) return -1;
    if (diff_check_sampler(n, steps, sigma_min, sigma_max, rho) != 0) return -1;
    double *sigmas = (double *)malloc((size_t)(steps + 1) * sizeof(double));
    double *cur = (double *)malloc((size_t)n * sizeof(double));
    double *x0 = (double *)malloc((size_t)n * sizeof(double));
    if (!sigmas || !cur || !x0) {
        free(sigmas); free(cur); free(x0);
        return -1;
    }
    if (diffusion_sigmas(steps, sigma_min, sigma_max, rho, sigmas, steps + 1) < 0) {
        free(sigmas); free(cur); free(x0);
        return -1;
    }
    memcpy(cur, x, (size_t)n * sizeof(double));
    for (int i = 0; i < steps; i++) {
        double sigma = sigmas[i];
        double sigma_next = sigmas[i + 1];
        denoise_fn(cur, n, sigma, x0, ctx);
        if (sigma_next <= 0.0) {
            memcpy(cur, x0, (size_t)n * sizeof(double));
        } else {
            double ratio = sigma_next / sigma;
            for (int j = 0; j < n; j++) {
                cur[j] = x0[j] + ratio * (cur[j] - x0[j]);
            }
        }
    }
    memcpy(out, cur, (size_t)n * sizeof(double));
    free(sigmas); free(cur); free(x0);
    return 0;
}

int consistency_training_loss(DiffDenoiseFn student_fn, void *student_ctx,
                              DiffDenoiseFn teacher_fn, void *teacher_ctx,
                              const double *x, int n, double sigma,
                              double sigma_next, double *loss) {
    if (student_fn == NULL || teacher_fn == NULL || x == NULL || loss == NULL) return -1;
    if (n <= 0) return -1;
    if (!isfinite(sigma) || sigma <= 0.0) return -1;
    if (!isfinite(sigma_next) || sigma_next < 0.0 || sigma_next >= sigma) return -1;
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    double *teacher_x0 = (double *)malloc((size_t)n * sizeof(double));
    double *target = (double *)malloc((size_t)n * sizeof(double));
    double *x_next = (double *)malloc((size_t)n * sizeof(double));
    if (!pred || !teacher_x0 || !target || !x_next) {
        free(pred); free(teacher_x0); free(target); free(x_next);
        return -1;
    }
    student_fn(x, n, sigma, pred, student_ctx);
    teacher_fn(x, n, sigma, teacher_x0, teacher_ctx);
    if (sigma_next > 0.0) {
        double ratio = sigma_next / sigma;
        for (int j = 0; j < n; j++) {
            x_next[j] = teacher_x0[j] + ratio * (x[j] - teacher_x0[j]);
        }
        teacher_fn(x_next, n, sigma_next, target, teacher_ctx);
    } else {
        memcpy(target, teacher_x0, (size_t)n * sizeof(double));
    }
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
        double d = pred[j] - target[j];
        sum += d * d;
    }
    *loss = sum / (double)n;
    free(pred); free(teacher_x0); free(target); free(x_next);
    return 0;
}

int consistency_sampler(DiffDenoiseFn student_fn, void *ctx,
                        const double *x, int n, int steps,
                        double sigma_min, double sigma_max, double rho,
                        double *out) {
    if (student_fn == NULL || x == NULL || out == NULL) return -1;
    if (diff_check_sampler(n, steps, sigma_min, sigma_max, rho) != 0) return -1;
    double *sigmas = (double *)malloc((size_t)(steps + 1) * sizeof(double));
    double *cur = (double *)malloc((size_t)n * sizeof(double));
    double *x0 = (double *)malloc((size_t)n * sizeof(double));
    if (!sigmas || !cur || !x0) {
        free(sigmas); free(cur); free(x0);
        return -1;
    }
    if (diffusion_sigmas(steps, sigma_min, sigma_max, rho, sigmas, steps + 1) < 0) {
        free(sigmas); free(cur); free(x0);
        return -1;
    }
    memcpy(cur, x, (size_t)n * sizeof(double));
    for (int i = 0; i < steps; i++) {
        double sigma = sigmas[i];
        double sigma_next = sigmas[i + 1];
        student_fn(cur, n, sigma, x0, ctx);
        if (sigma_next <= 0.0) {
            memcpy(cur, x0, (size_t)n * sizeof(double));
        } else {
            double ratio = sigma_next / sigma;
            for (int j = 0; j < n; j++) {
                cur[j] = x0[j] + ratio * (cur[j] - x0[j]);
            }
        }
    }
    memcpy(out, cur, (size_t)n * sizeof(double));
    free(sigmas); free(cur); free(x0);
    return 0;
}
