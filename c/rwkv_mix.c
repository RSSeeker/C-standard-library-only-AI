/*
 * rwkv_mix.c —— RWKV-5/6/7 time-mixing 前向实现，见 rwkv_mix.h。
 *
 * 运算顺序严格照搬 AI.py：
 *   - 状态更新 row[c] = alpha * row[c] + k_t[i] * v_t[c]（朴素乘加）；
 *   - 输出 y[c] += r_t[i] * row[c]，r_t[i] == 0.0 时跳过该 channel；
 *   - RWKV-6/7 的 alpha = exp(-exp(w[t][i]))；
 *   - RWKV-7 在每步先用旧状态算 sk = S_{t-1}^T k_t，再做修正写入。
 */
#include "rwkv_mix.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int rk_matrix_ok(const double *m, int rows, int cols)
{
    int n;
    if (m == NULL || rows < 1 || cols < 1) return 0;
    n = rows * cols;
    for (int i = 0; i < n; i++) {
        if (!isfinite(m[i])) return 0;
    }
    return 1;
}

static int rk_decay_ok(double w)
{
    return isfinite(w) && w >= 0.0 && w <= 1.0;
}

/* 校验 r/k/v + out + 维数，返回 1 表示通过 */
static int rk_common_ok(const double *r, const double *k, const double *v,
                        int t_len, int dk, int dv, const double *out)
{
    if (out == NULL || t_len < 1 || dk < 1 || dv < 1) return 0;
    return rk_matrix_ok(r, t_len, dk) && rk_matrix_ok(k, t_len, dk)
        && rk_matrix_ok(v, t_len, dv);
}

/* y_t = r_t @ S_t，复用状态读出的公共循环 */
static void rk_read_out(const double *r, const double *state, int dk, int dv,
                        int t, double *y)
{
    for (int c = 0; c < dv; c++) y[c] = 0.0;
    for (int i = 0; i < dk; i++) {
        double ri = r[(size_t)t * dk + i];
        const double *row;
        if (ri == 0.0) continue;
        row = state + (size_t)i * dv;
        for (int c = 0; c < dv; c++) y[c] += ri * row[c];
    }
}

int ai_rwkv5_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *decay_vec, double decay, double *out)
{
    double *state;
    if (!rk_common_ok(r, k, v, t_len, dk, dv, out)) return -1;
    if (decay_vec != NULL) {
        for (int i = 0; i < dk; i++) {
            if (!rk_decay_ok(decay_vec[i])) return -1;
        }
    } else if (!rk_decay_ok(decay)) {
        return -1;
    }

    state = (double *)calloc((size_t)dk * (size_t)dv, sizeof(double));
    if (state == NULL) return -1;

    for (int t = 0; t < t_len; t++) {
        for (int i = 0; i < dk; i++) {
            double *row = state + (size_t)i * dv;
            double kt = k[(size_t)t * dk + i];
            double wi = decay_vec != NULL ? decay_vec[i] : decay;
            for (int c = 0; c < dv; c++) {
                row[c] = wi * row[c] + kt * v[(size_t)t * dv + c];
            }
        }
        rk_read_out(r, state, dk, dv, t, out + (size_t)t * dv);
    }
    free(state);
    return 0;
}

int ai_rwkv6_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *w_logits, double *out)
{
    double *state;
    if (!rk_common_ok(r, k, v, t_len, dk, dv, out)) return -1;
    if (!rk_matrix_ok(w_logits, t_len, dk)) return -1;

    state = (double *)calloc((size_t)dk * (size_t)dv, sizeof(double));
    if (state == NULL) return -1;

    for (int t = 0; t < t_len; t++) {
        for (int i = 0; i < dk; i++) {
            double *row = state + (size_t)i * dv;
            double alpha = exp(-exp(w_logits[(size_t)t * dk + i]));
            double kt = k[(size_t)t * dk + i];
            for (int c = 0; c < dv; c++) {
                row[c] = alpha * row[c] + kt * v[(size_t)t * dv + c];
            }
        }
        rk_read_out(r, state, dk, dv, t, out + (size_t)t * dv);
    }
    free(state);
    return 0;
}

int ai_rwkv7_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *w_logits, double delta, double *out)
{
    double *state;
    double *old;
    double *sk;
    if (!rk_common_ok(r, k, v, t_len, dk, dv, out)) return -1;
    if (!rk_matrix_ok(w_logits, t_len, dk)) return -1;
    if (!isfinite(delta) || delta < 0.0) return -1;

    state = (double *)calloc((size_t)dk * (size_t)dv, sizeof(double));
    old = (double *)calloc((size_t)dk * (size_t)dv, sizeof(double));
    sk = (double *)calloc((size_t)dv, sizeof(double));
    if (state == NULL || old == NULL || sk == NULL) {
        free(state);
        free(old);
        free(sk);
        return -1;
    }

    for (int t = 0; t < t_len; t++) {
        memcpy(old, state, (size_t)dk * (size_t)dv * sizeof(double));
        for (int c = 0; c < dv; c++) sk[c] = 0.0;
        for (int i = 0; i < dk; i++) {
            double ki = k[(size_t)t * dk + i];
            if (ki == 0.0) continue;
            for (int c = 0; c < dv; c++) {
                sk[c] += ki * old[(size_t)i * dv + c];
            }
        }
        for (int i = 0; i < dk; i++) {
            double alpha = exp(-exp(w_logits[(size_t)t * dk + i]));
            double ki = k[(size_t)t * dk + i];
            double *row = state + (size_t)i * dv;
            const double *orow = old + (size_t)i * dv;
            for (int c = 0; c < dv; c++) {
                row[c] = alpha * orow[c]
                       + ki * (v[(size_t)t * dv + c] - delta * sk[c]);
            }
        }
        rk_read_out(r, state, dk, dv, t, out + (size_t)t * dv);
    }
    free(state);
    free(old);
    free(sk);
    return 0;
}
