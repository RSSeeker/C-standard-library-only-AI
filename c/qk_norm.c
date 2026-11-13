/*
 * qk_norm.c —— QK-Norm（逐行 RMSNorm）与 DynamicTanh（DyT）的 C 实现
 *
 * 见 qk_norm.h 的模块说明。与 Python 的对应关系：
 *   qk_norm          -> ai_qk_norm
 *   qk_norm_backward -> ai_qk_norm_backward
 *   apply_qk_norm    -> ai_apply_qk_norm
 *   DynamicTanh      -> AiDynamicTanh / ai_dyt_*
 */
#include "qk_norm.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "numeric.h"

static int qk_all_finite(const double *v, size_t n)
{
    size_t i;
    if (!v) return 0;
    for (i = 0; i < n; ++i) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

/* ================================ QK-Norm ================================ */

int ai_qk_norm(const double *rows, int n_rows, int width, const double *gamma,
               double eps, AiQkNormResult *out)
{
    size_t total;
    int r, i;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!rows || !gamma) return -1;
    if (n_rows < 1 || width < 1) return -1;
    if (n_rows > INT_MAX / width) return -1;
    if (!(eps > 0.0) || !isfinite(eps)) return -1;
    total = (size_t)n_rows * (size_t)width;
    if (!qk_all_finite(rows, total)) return -1;
    if (!qk_all_finite(gamma, (size_t)width)) return -1;

    out->out = (double *)malloc(sizeof(double) * total);
    out->xhat = (double *)malloc(sizeof(double) * total);
    out->rms = (double *)malloc(sizeof(double) * (size_t)n_rows);
    out->gamma = (double *)malloc(sizeof(double) * (size_t)width);
    if (!out->out || !out->xhat || !out->rms || !out->gamma) {
        ai_qk_norm_free(out);
        return -1;
    }
    out->n_rows = n_rows;
    out->width = width;
    memcpy(out->gamma, gamma, sizeof(double) * (size_t)width);

    for (r = 0; r < n_rows; ++r) {
        const double *row = rows + (size_t)r * (size_t)width;
        /* Python: mean_sq = sum(v*v for v in row) / width */
        const double mean_sq = dot_neumaier(row, row, width) / (double)width;
        const double rms = sqrt(mean_sq + eps);
        out->rms[r] = rms;
        for (i = 0; i < width; ++i) {
            const double xhat = row[i] / rms;
            out->xhat[(size_t)r * (size_t)width + (size_t)i] = xhat;
            out->out[(size_t)r * (size_t)width + (size_t)i] = gamma[i] * xhat;
        }
    }
    return 0;
}

void ai_qk_norm_free(AiQkNormResult *res)
{
    if (!res) return;
    free(res->out);
    free(res->rms);
    free(res->xhat);
    free(res->gamma);
    memset(res, 0, sizeof(*res));
}

int ai_qk_norm_backward(const double *dout_rows, int n_rows, int width,
                        const AiQkNormResult *cache, double *grad_gamma,
                        double *input_grad)
{
    double *dxhat;
    int t, i;

    if (!cache || !dout_rows || !grad_gamma || !input_grad) return -1;
    if (cache->n_rows < 1 || cache->width < 1) return -1;
    if (!cache->xhat || !cache->rms || !cache->gamma) return -1;
    if (n_rows != cache->n_rows || width != cache->width) return -1;
    if (!qk_all_finite(dout_rows,
                       (size_t)n_rows * (size_t)width)) return -1;

    dxhat = (double *)malloc(sizeof(double) * (size_t)width);
    if (!dxhat) return -1;
    for (i = 0; i < width; ++i) grad_gamma[i] = 0.0;
    for (t = 0; t < n_rows; ++t) {
        const size_t off = (size_t)t * (size_t)width;
        const double *xhat = cache->xhat + off;
        const double rms = cache->rms[t];
        const double *drow = dout_rows + off;
        double dot;
        for (i = 0; i < width; ++i) {
            /* Python: dxhat[i] = drow[i] * gamma[i]；grad_gamma[i] += drow[i]*xhat[i] */
            dxhat[i] = drow[i] * cache->gamma[i];
            grad_gamma[i] += drow[i] * xhat[i];
        }
        /* Python: dot = sum(dxhat[i]*xhat[i]) / width */
        dot = dot_neumaier(dxhat, xhat, width) / (double)width;
        for (i = 0; i < width; ++i) {
            input_grad[off + (size_t)i] = (dxhat[i] - xhat[i] * dot) / rms;
        }
    }
    free(dxhat);
    return 0;
}

int ai_apply_qk_norm(const double *q_rows, int q_n_rows, int q_width,
                     const double *q_gamma, const double *k_rows,
                     int k_n_rows, int k_width, const double *k_gamma,
                     double eps, AiQkNormResult *q_out,
                     AiQkNormResult *k_out)
{
    if (!q_out || !k_out) return -1;
    memset(q_out, 0, sizeof(*q_out));
    memset(k_out, 0, sizeof(*k_out));
    if (ai_qk_norm(q_rows, q_n_rows, q_width, q_gamma, eps, q_out) != 0)
        return -1;
    if (ai_qk_norm(k_rows, k_n_rows, k_width, k_gamma, eps, k_out) != 0) {
        ai_qk_norm_free(q_out);
        return -1;
    }
    return 0;
}

/* ============================== DynamicTanh ============================== */

int ai_dyt_create(AiDynamicTanh *self, int d_model, double alpha)
{
    int i;

    if (!self) return -1;
    memset(self, 0, sizeof(*self));
    if (d_model < 1) return -1;
    if (!isfinite(alpha) || alpha == 0.0) return -1;
    self->d_model = d_model;
    self->alpha = alpha;
    self->gamma = (double *)malloc(sizeof(double) * (size_t)d_model);
    self->beta = (double *)malloc(sizeof(double) * (size_t)d_model);
    self->grad_gamma = (double *)malloc(sizeof(double) * (size_t)d_model);
    self->grad_beta = (double *)malloc(sizeof(double) * (size_t)d_model);
    if (!self->gamma || !self->beta || !self->grad_gamma
        || !self->grad_beta) {
        ai_dyt_free(self);
        return -1;
    }
    for (i = 0; i < d_model; ++i) {
        self->gamma[i] = 1.0;
        self->beta[i] = 0.0;
        self->grad_gamma[i] = 0.0;
        self->grad_beta[i] = 0.0;
    }
    return 0;
}

void ai_dyt_free(AiDynamicTanh *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->beta);
    free(self->grad_gamma);
    free(self->grad_beta);
    free(self->cache_x);
    free(self->cache_tanh);
    memset(self, 0, sizeof(*self));
}

int ai_dyt_forward(AiDynamicTanh *self, const double *rows, int n_rows,
                   int width, double *out)
{
    double *new_x;
    double *new_tanh;
    int t, i;

    if (!self || !rows || !out) return -1;
    if (self->d_model < 1 || width != self->d_model) return -1;
    if (n_rows < 1) return -1;
    if (n_rows > INT_MAX / width) return -1;
    if (!qk_all_finite(rows, (size_t)n_rows * (size_t)width)) return -1;

    if (self->cache_cap < n_rows) {
        /* 缓存扩容：先保证两个指针都成功，失败时保留旧缓存 */
        new_x = (double *)realloc(self->cache_x,
                                  sizeof(double) * (size_t)n_rows
                                      * (size_t)width);
        if (!new_x) return -1;
        self->cache_x = new_x;
        new_tanh = (double *)realloc(self->cache_tanh,
                                     sizeof(double) * (size_t)n_rows
                                         * (size_t)width);
        if (!new_tanh) return -1;
        self->cache_tanh = new_tanh;
        self->cache_cap = n_rows;
    }
    for (t = 0; t < n_rows; ++t) {
        for (i = 0; i < width; ++i) {
            const size_t off = (size_t)t * (size_t)width + (size_t)i;
            const double th = tanh(self->alpha * rows[off]);
            self->cache_x[off] = rows[off];
            self->cache_tanh[off] = th;
            out[off] = self->gamma[i] * th + self->beta[i];
        }
    }
    self->cache_n = n_rows;
    return 0;
}

int ai_dyt_backward(AiDynamicTanh *self, const double *dout_rows, int n_rows,
                    int width, double *input_grad)
{
    int t, i;

    if (!self || !dout_rows || !input_grad) return -1;
    if (self->d_model < 1 || width != self->d_model) return -1;
    if (self->cache_n < 1 || n_rows != self->cache_n) return -1;
    if (!self->cache_x || !self->cache_tanh) return -1;

    /* 对齐 Python 的累加顺序：外层行、内层列；普通 += */
    for (t = 0; t < n_rows; ++t) {
        for (i = 0; i < width; ++i) {
            const size_t off = (size_t)t * (size_t)width + (size_t)i;
            const double d = dout_rows[off];
            const double x = self->cache_x[off];
            const double th = self->cache_tanh[off];
            const double sech2 = 1.0 - th * th;
            self->grad_gamma[i] += d * th;
            self->grad_beta[i] += d;
            self->grad_alpha += d * self->gamma[i] * sech2 * x;
            input_grad[off] = d * self->gamma[i] * sech2 * self->alpha;
        }
    }
    return 0;
}

int ai_dyt_zero_grad(AiDynamicTanh *self)
{
    int i;
    if (!self || self->d_model < 1) return -1;
    if (!self->grad_gamma || !self->grad_beta) return -1;
    self->grad_alpha = 0.0;
    for (i = 0; i < self->d_model; ++i) {
        self->grad_gamma[i] = 0.0;
        self->grad_beta[i] = 0.0;
    }
    return 0;
}

int ai_dyt_sgd_step(AiDynamicTanh *self, double lr)
{
    int i;
    if (!self || self->d_model < 1) return -1;
    if (!self->gamma || !self->beta || !self->grad_gamma || !self->grad_beta)
        return -1;
    if (!(lr > 0.0) || !isfinite(lr)) return -1;
    self->alpha -= lr * self->grad_alpha;
    for (i = 0; i < self->d_model; ++i) {
        self->gamma[i] -= lr * self->grad_gamma[i];
        self->beta[i] -= lr * self->grad_beta[i];
    }
    return ai_dyt_zero_grad(self);
}

int ai_dyt_param_count(const AiDynamicTanh *self)
{
    if (!self || self->d_model < 1) return -1;
    return 1 + 2 * self->d_model;
}
