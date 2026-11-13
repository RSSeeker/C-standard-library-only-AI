/*
 * linear_attn.c —— 线性注意力 / 长卷积实现（对齐 AI.py 的运算顺序）
 *
 * 与 Python 版逐行对应：
 *   - linear_attention_naive：内积用 Neumaier 补偿求和（CPython 3.12+ sum()），
 *     外层 y[c] += w*dot*v[j][c] 为普通浮点累加；
 *   - lightning / gla / deltanet：state[dk][dv] 逐元素递推，q 分量为 0 时跳过；
 *   - hyena：因果长卷积后逐元素门控。
 */
#include "linear_attn.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "numeric.h"

/* 有限浮点矩阵校验：非 NULL、行列均为正、元素全部有限。 */
static int la_finite_matrix(const double *m, int rows, int width)
{
    if (!m || rows < 1 || width < 1)
        return 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < width; c++)
            if (!isfinite(m[(size_t)r * (size_t)width + (size_t)c]))
                return 0;
    return 1;
}

static int la_finite_seq(const double *s, int n)
{
    if (!s || n < 1)
        return 0;
    for (int i = 0; i < n; i++)
        if (!isfinite(s[i]))
            return 0;
    return 1;
}

/* 形状与元素校验：返回 0 合法 / -1 非法。 */
static int la_check_qkv(const double *q, const double *k, const double *v,
                        int t_len, int dk, int dv, const double *out)
{
    if (!out || t_len < 1 || dk < 1 || dv < 1)
        return -1;
    if (t_len > INT_MAX / dk || dk > INT_MAX / dv)
        return -1;
    if (!la_finite_matrix(q, t_len, dk))
        return -1;
    if (!la_finite_matrix(k, t_len, dk))
        return -1;
    if (!la_finite_matrix(v, t_len, dv))
        return -1;
    return 0;
}

static int la_check_decay(double decay)
{
    return isfinite(decay) && decay >= 0.0 && decay <= 1.0;
}

/* 申请并清零 state[dk][dv]；失败返回 NULL。 */
static double *la_state_alloc(int dk, int dv)
{
    size_t n = (size_t)dk * (size_t)dv;
    return (double *)calloc(n, sizeof(double));
}

int ai_linear_attention_naive(const double *q, const double *k, const double *v,
                              int t_len, int dk, int dv, double decay,
                              double *out)
{
    if (la_check_qkv(q, k, v, t_len, dk, dv, out) != 0)
        return -1;
    if (!la_check_decay(decay))
        return -1;

    for (int t = 0; t < t_len; t++) {
        double *y = out + (size_t)t * (size_t)dv;
        for (int c = 0; c < dv; c++)
            y[c] = 0.0;
        for (int j = 0; j <= t; j++) {
            /* Python: w = decay ** (t - j) */
            double w = pow(decay, (double)(t - j));
            /* Python: sum(q[t][i] * k[j][i] for i ...) —— 内建 sum 用 Neumaier */
            double dot = dot_neumaier(q + (size_t)t * (size_t)dk,
                                      k + (size_t)j * (size_t)dk, dk);
            const double *vj = v + (size_t)j * (size_t)dv;
            for (int c = 0; c < dv; c++)
                y[c] += w * dot * vj[c];
        }
    }
    return 0;
}

int ai_lightning_attention(const double *q, const double *k, const double *v,
                           int t_len, int dk, int dv, double decay,
                           double *out)
{
    double *state;

    if (la_check_qkv(q, k, v, t_len, dk, dv, out) != 0)
        return -1;
    if (!la_check_decay(decay))
        return -1;
    state = la_state_alloc(dk, dv);
    if (!state)
        return -1;

    for (int t = 0; t < t_len; t++) {
        const double *kt = k + (size_t)t * (size_t)dk;
        const double *vt = v + (size_t)t * (size_t)dv;
        for (int i = 0; i < dk; i++) {
            double *row = state + (size_t)i * (size_t)dv;
            double kti = kt[i];
            for (int c = 0; c < dv; c++)
                row[c] = decay * row[c] + kti * vt[c];
        }
        {
            double *y = out + (size_t)t * (size_t)dv;
            const double *qt = q + (size_t)t * (size_t)dk;
            for (int c = 0; c < dv; c++)
                y[c] = 0.0;
            for (int i = 0; i < dk; i++) {
                double qi = qt[i];
                if (qi == 0.0)
                    continue;
                {
                    const double *row = state + (size_t)i * (size_t)dv;
                    for (int c = 0; c < dv; c++)
                        y[c] += qi * row[c];
                }
            }
        }
    }
    free(state);
    return 0;
}

int ai_gla_forward(const double *q, const double *k, const double *v,
                   const double *gates, int t_len, int dk, int dv,
                   double *out)
{
    double *state;

    if (la_check_qkv(q, k, v, t_len, dk, dv, out) != 0)
        return -1;
    if (!la_finite_seq(gates, t_len))
        return -1;
    state = la_state_alloc(dk, dv);
    if (!state)
        return -1;

    for (int t = 0; t < t_len; t++) {
        /* Python: alpha = math.exp(-math.exp(gates[t])) */
        double alpha = exp(-exp(gates[t]));
        const double *kt = k + (size_t)t * (size_t)dk;
        const double *vt = v + (size_t)t * (size_t)dv;
        for (int i = 0; i < dk; i++) {
            double *row = state + (size_t)i * (size_t)dv;
            double kti = kt[i];
            for (int c = 0; c < dv; c++)
                row[c] = alpha * row[c] + kti * vt[c];
        }
        {
            double *y = out + (size_t)t * (size_t)dv;
            const double *qt = q + (size_t)t * (size_t)dk;
            for (int c = 0; c < dv; c++)
                y[c] = 0.0;
            for (int i = 0; i < dk; i++) {
                double qi = qt[i];
                if (qi == 0.0)
                    continue;
                {
                    const double *row = state + (size_t)i * (size_t)dv;
                    for (int c = 0; c < dv; c++)
                        y[c] += qi * row[c];
                }
            }
        }
    }
    free(state);
    return 0;
}

int ai_deltanet_forward(const double *q, const double *k, const double *v,
                        const double *betas, int t_len, int dk, int dv,
                        double *out)
{
    double *state;
    double *sk;

    if (la_check_qkv(q, k, v, t_len, dk, dv, out) != 0)
        return -1;
    if (!la_finite_seq(betas, t_len))
        return -1;
    for (int t = 0; t < t_len; t++)
        if (betas[t] < 0.0 || betas[t] > 1.0)
            return -1;

    state = la_state_alloc(dk, dv);
    sk = (double *)calloc((size_t)dv, sizeof(double));
    if (!state || !sk) {
        free(state);
        free(sk);
        return -1;
    }

    for (int t = 0; t < t_len; t++) {
        double beta = betas[t];
        const double *kt = k + (size_t)t * (size_t)dk;
        const double *vt = v + (size_t)t * (size_t)dv;
        for (int c = 0; c < dv; c++)
            sk[c] = 0.0;
        /* sk = sum_i k[t][i] * state[i]（k 分量为 0 时跳过） */
        for (int i = 0; i < dk; i++) {
            double ki = kt[i];
            if (ki == 0.0)
                continue;
            {
                const double *row = state + (size_t)i * (size_t)dv;
                for (int c = 0; c < dv; c++)
                    sk[c] += ki * row[c];
            }
        }
        /* state[i] -= beta * k[t][i] * sk */
        for (int i = 0; i < dk; i++) {
            double ki = kt[i];
            if (ki == 0.0)
                continue;
            {
                double *row = state + (size_t)i * (size_t)dv;
                for (int c = 0; c < dv; c++)
                    row[c] -= beta * ki * sk[c];
            }
        }
        /* state[i] += beta * k[t][i] * v[t] */
        for (int i = 0; i < dk; i++) {
            double ki = kt[i];
            if (ki == 0.0)
                continue;
            {
                double *row = state + (size_t)i * (size_t)dv;
                for (int c = 0; c < dv; c++)
                    row[c] += beta * ki * vt[c];
            }
        }
        /* y = q[t] @ state */
        {
            double *y = out + (size_t)t * (size_t)dv;
            const double *qt = q + (size_t)t * (size_t)dk;
            for (int c = 0; c < dv; c++)
                y[c] = 0.0;
            for (int i = 0; i < dk; i++) {
                double qi = qt[i];
                if (qi == 0.0)
                    continue;
                {
                    const double *row = state + (size_t)i * (size_t)dv;
                    for (int c = 0; c < dv; c++)
                        y[c] += qi * row[c];
                }
            }
        }
    }
    free(state);
    free(sk);
    return 0;
}

int ai_hyena_long_conv(const double *u, int t_len, int d, const double *kernel,
                       int k_len, double *out)
{
    if (!out || t_len < 1 || d < 1)
        return -1;
    if (t_len > INT_MAX / d)
        return -1;
    if (!la_finite_matrix(u, t_len, d))
        return -1;
    if (k_len < 1 || !la_finite_seq(kernel, k_len))
        return -1;

    for (int t = 0; t < t_len; t++) {
        double *y = out + (size_t)t * (size_t)d;
        int j_max = (t + 1 < k_len) ? (t + 1) : k_len;
        for (int c = 0; c < d; c++)
            y[c] = 0.0;
        for (int j = 0; j < j_max; j++) {
            double w = kernel[j];
            const double *src = u + (size_t)(t - j) * (size_t)d;
            for (int c = 0; c < d; c++)
                y[c] += w * src[c];
        }
    }
    return 0;
}

int ai_hyena_forward(const double *u, const double *v, int t_len, int d,
                     const double *kernel, int k_len, double *out)
{
    double *conv;
    int rc;

    if (!out || t_len < 1 || d < 1)
        return -1;
    if (t_len > INT_MAX / d)
        return -1;
    if (!la_finite_matrix(u, t_len, d) || !la_finite_matrix(v, t_len, d))
        return -1;
    if (k_len < 1 || !la_finite_seq(kernel, k_len))
        return -1;

    conv = (double *)calloc((size_t)t_len * (size_t)d, sizeof(double));
    if (!conv)
        return -1;
    rc = ai_hyena_long_conv(u, t_len, d, kernel, k_len, conv);
    if (rc == 0) {
        for (size_t i = 0; i < (size_t)t_len * (size_t)d; i++)
            out[i] = conv[i] * v[i];
    }
    free(conv);
    return rc;
}
