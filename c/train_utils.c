/*
 * train_utils.c —— 训练辅助工具的 C 实现
 *
 * 与 Python 版 AI.py 的对应用例见 tools/train_utils_reference.py，
 * 逐位/容差对拍见 tests/compare_train_utils.c。
 */
#include "train_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const double ai_nf4_levels[AI_NF4_LEVELS_COUNT] = {
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0
};

AiNf4Quant ai_nf4_quantize(const double *values, int n, int block_size)
{
    AiNf4Quant q;
    memset(&q, 0, sizeof q);
    if (!values || n <= 0 || block_size < 1) return q;

    const int n_blocks = (n + block_size - 1) / block_size;
    q.codes = (int *)malloc(sizeof(int) * (size_t)n);
    q.scales = (double *)malloc(sizeof(double) * (size_t)n_blocks);
    if (!q.codes || !q.scales) {
        free(q.codes);
        free(q.scales);
        memset(&q, 0, sizeof q);
        return q;
    }
    q.block_size = block_size;
    q.length = n;

    for (int b = 0; b < n_blocks; b++) {
        const int start = b * block_size;
        const int stop = (start + block_size < n) ? start + block_size : n;
        double absmax = 0.0;
        for (int i = start; i < stop; i++) {
            const double a = fabs(values[i]);
            if (a > absmax) absmax = a;
        }
        if (absmax <= 0.0) {
            q.scales[b] = 0.0;
            for (int i = start; i < stop; i++) q.codes[i] = 7;
            continue;
        }
        q.scales[b] = absmax;
        for (int i = start; i < stop; i++) {
            const double x = values[i] / absmax;
            int best = 0;
            double best_err = fabs(x - ai_nf4_levels[0]);
            for (int k = 1; k < AI_NF4_LEVELS_COUNT; k++) {
                const double err = fabs(x - ai_nf4_levels[k]);
                if (err < best_err) {
                    best = k;
                    best_err = err;
                }
            }
            q.codes[i] = best;
        }
    }
    return q;
}

void ai_nf4_quant_free(AiNf4Quant *q)
{
    if (!q) return;
    free(q->codes);
    free(q->scales);
    memset(q, 0, sizeof *q);
}

int ai_nf4_dequantize(const AiNf4Quant *q, double *out)
{
    if (!q || !out || q->length < 0 || q->block_size < 1) return -1;
    if (q->length > 0 && (!q->codes || !q->scales)) return -1;
    const int n_blocks = (q->length + q->block_size - 1) / q->block_size;
    (void)n_blocks;
    for (int i = 0; i < q->length; i++) {
        const int code = q->codes[i];
        if (code < 0 || code >= AI_NF4_LEVELS_COUNT) return -1;
        out[i] = ai_nf4_levels[code] * q->scales[i / q->block_size];
    }
    return 0;
}

int ai_nf4_fake_quantize(const double *values, int n, int block_size, double *out)
{
    if (!values || !out || n <= 0 || block_size < 1) return -1;
    AiNf4Quant q = ai_nf4_quantize(values, n, block_size);
    if (q.length <= 0) return -1;
    const int rc = ai_nf4_dequantize(&q, out);
    ai_nf4_quant_free(&q);
    return rc;
}

static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int ai_double_quantize(const double *scales, int n, int block_size,
                       int *codes, double *outers)
{
    if (!scales || !codes || !outers || n <= 0 || block_size < 1) return -1;
    const int n_blocks = (n + block_size - 1) / block_size;
    for (int b = 0; b < n_blocks; b++) {
        const int start = b * block_size;
        const int stop = (start + block_size < n) ? start + block_size : n;
        double outer = 0.0;
        for (int i = start; i < stop; i++) {
            if (scales[i] < 0.0) return -1;
            if (scales[i] > outer) outer = scales[i];
        }
        if (outer <= 0.0) {
            outers[b] = 0.0;
            for (int i = start; i < stop; i++) codes[i] = 0;
            continue;
        }
        outers[b] = outer;
        for (int i = start; i < stop; i++) {
            /* Python 的 round() 是四舍六入五成双，nearbyint 默认同样规则。 */
            const double scaled = 255.0 * scales[i] / outer;
            codes[i] = clamp_int((int)nearbyint(scaled), 0, 255);
        }
    }
    return 0;
}

int ai_double_dequantize(const int *codes, const double *outers, int n,
                         int block_size, double *out)
{
    if (!codes || !outers || !out || n <= 0 || block_size < 1) return -1;
    const int n_blocks = (n + block_size - 1) / block_size;
    (void)n_blocks;
    for (int i = 0; i < n; i++) {
        const int code = codes[i];
        if (code < 0 || code > 255) return -1;
        out[i] = (double)code * outers[i / block_size] / 255.0;
    }
    return 0;
}

int ai_nf4_fake_quantize_double(const double *values, int n, int block_size,
                                int outer_block_size, double *out)
{
    if (!values || !out || n <= 0 || block_size < 1 || outer_block_size < 1) return -1;
    AiNf4Quant q = ai_nf4_quantize(values, n, block_size);
    if (q.length <= 0) return -1;

    const int n_blocks = (n + block_size - 1) / block_size;
    int *codes2 = (int *)malloc(sizeof(int) * (size_t)n_blocks);
    double *outers = (double *)malloc(sizeof(double) * (size_t)n_blocks);
    double *approx = (double *)malloc(sizeof(double) * (size_t)n_blocks);
    if (!codes2 || !outers || !approx) {
        free(codes2);
        free(outers);
        free(approx);
        ai_nf4_quant_free(&q);
        return -1;
    }
    int rc = ai_double_quantize(q.scales, n_blocks, outer_block_size, codes2, outers);
    if (rc == 0) rc = ai_double_dequantize(codes2, outers, n_blocks, outer_block_size, approx);
    if (rc == 0) {
        for (int i = 0; i < n; i++) {
            out[i] = ai_nf4_levels[q.codes[i]] * approx[i / block_size];
        }
    }
    free(codes2);
    free(outers);
    free(approx);
    ai_nf4_quant_free(&q);
    return rc;
}

/* ---------- Weight Standardization ---------- */

void ai_ws_effective(const double *w, int fan_out, int fan_in, double eps,
                     double *ws, double *mu, double *sigma)
{
    if (!w || !ws || !mu || !sigma || fan_out < 1 || fan_in < 1) return;
    const double eps2 = eps * eps;
    for (int i = 0; i < fan_out; i++) {
        const double *row = w + (size_t)i * (size_t)fan_in;
        double m = 0.0;
        for (int j = 0; j < fan_in; j++) m += row[j];
        m /= (double)fan_in;
        double var = 0.0;
        for (int j = 0; j < fan_in; j++) {
            const double d = row[j] - m;
            var += d * d;
        }
        var /= (double)fan_in;
        const double sd = (var > eps2) ? sqrt(var) : eps;
        mu[i] = m;
        sigma[i] = sd;
        double *out_row = ws + (size_t)i * (size_t)fan_in;
        for (int j = 0; j < fan_in; j++) out_row[j] = (row[j] - m) / sd;
    }
}

void ai_ws_forward(const double *w, const double *b, const double *x,
                   int fan_out, int fan_in, int n, double eps, double *y,
                   double *ws, double *mu, double *sigma)
{
    if (!w || !b || !x || !y) return;
    ai_ws_effective(w, fan_out, fan_in, eps, ws, mu, sigma);
    for (int r = 0; r < n; r++) {
        const double *xr = x + (size_t)r * (size_t)fan_in;
        double *yr = y + (size_t)r * (size_t)fan_out;
        for (int i = 0; i < fan_out; i++) {
            const double *wrow = ws + (size_t)i * (size_t)fan_in;
            double acc = 0.0;
            for (int j = 0; j < fan_in; j++) acc += wrow[j] * xr[j];
            yr[i] = acc + b[i];
        }
    }
}

void ai_ws_backward(const double *w, const double *x, const double *dy,
                    int fan_out, int fan_in, int n, double eps,
                    double *dw, double *db, double *dx)
{
    if (!w || !x || !dy || !dw || !db || !dx) return;

    /* 重新算标准化权重（前向与反向不共享状态）。 */
    double *ws = (double *)malloc(sizeof(double) * (size_t)fan_out * (size_t)fan_in);
    double *mu = (double *)malloc(sizeof(double) * (size_t)fan_out);
    double *sigma = (double *)malloc(sizeof(double) * (size_t)fan_out);
    if (!ws || !mu || !sigma) {
        free(ws);
        free(mu);
        free(sigma);
        return;
    }
    ai_ws_effective(w, fan_out, fan_in, eps, ws, mu, sigma);

    /* db = Σ_r dY[r][i] */
    for (int i = 0; i < fan_out; i++) db[i] = 0.0;
    for (int r = 0; r < n; r++) {
        const double *dyr = dy + (size_t)r * (size_t)fan_out;
        for (int i = 0; i < fan_out; i++) db[i] += dyr[i];
    }

    /* dX[r][j] = Σ_i dY[r][i] * ws[i][j] */
    for (int r = 0; r < n; r++) {
        const double *dyr = dy + (size_t)r * (size_t)fan_out;
        double *dxr = dx + (size_t)r * (size_t)fan_in;
        for (int j = 0; j < fan_in; j++) {
            double acc = 0.0;
            for (int i = 0; i < fan_out; i++) acc += dyr[i] * ws[(size_t)i * (size_t)fan_in + j];
            dxr[j] = acc;
        }
    }

    /* 逐样本算 G_r = dY[r] ⊗ x[r] 并立即套反向公式再累加：
     * Python 参考实现是逐样本 backward 后相加，这里的运算顺序与它一致。 */
    for (int i = 0; i < fan_out; i++) {
        double *dwr = dw + (size_t)i * (size_t)fan_in;
        for (int j = 0; j < fan_in; j++) dwr[j] = 0.0;
    }
    for (int r = 0; r < n; r++) {
        const double *dyr = dy + (size_t)r * (size_t)fan_out;
        const double *xr = x + (size_t)r * (size_t)fan_in;
        for (int i = 0; i < fan_out; i++) {
            const double d = dyr[i];
            double *dwr = dw + (size_t)i * (size_t)fan_in;
            const double *wsr = ws + (size_t)i * (size_t)fan_in;
            double mean_g = 0.0;
            double mean_gv = 0.0;
            for (int j = 0; j < fan_in; j++) {
                const double g = d * xr[j];
                mean_g += g;
                mean_gv += g * wsr[j];
            }
            mean_g /= (double)fan_in;
            mean_gv /= (double)fan_in;
            const double sd = sigma[i];
            for (int j = 0; j < fan_in; j++) {
                const double g = d * xr[j];
                dwr[j] += (g - mean_g - wsr[j] * mean_gv) / sd;
            }
        }
    }
    free(ws);
    free(mu);
    free(sigma);
}

/* ---------- GradScaler ---------- */

void ai_grad_scaler_init(AiGradScaler *s, double init_scale,
                         double growth_factor, double backoff_factor,
                         int growth_interval)
{
    if (!s) return;
    memset(s, 0, sizeof *s);
    s->scale = init_scale;
    s->growth_factor = growth_factor;
    s->backoff_factor = backoff_factor;
    s->growth_interval = growth_interval;
    s->found_inf = 0;
}

double ai_grad_scaler_scale_loss(AiGradScaler *s, double loss)
{
    if (!s) return loss;
    return loss * s->scale;
}

int ai_grad_scaler_unscale(AiGradScaler *s, double *grads, int n)
{
    if (!s || !grads) return 0;
    int seen_inf = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(grads[i])) {
            seen_inf = 1;
        } else {
            grads[i] /= s->scale;
        }
    }
    s->found_inf = seen_inf;
    return seen_inf;
}

int ai_grad_scaler_update(AiGradScaler *s, int found_inf_override)
{
    if (!s) return 1;
    const int overflow = (found_inf_override < 0) ? s->found_inf : (found_inf_override != 0);
    s->total_steps++;
    s->found_inf = 0;
    if (overflow) {
        s->scale *= s->backoff_factor;
        s->growth_tracker = 0;
        s->steps_skipped++;
        return 0;
    }
    s->growth_tracker++;
    if (s->growth_tracker >= s->growth_interval) {
        s->scale *= s->growth_factor;
        s->growth_tracker = 0;
    }
    s->steps_taken++;
    return 1;
}

/* ---------- MoE 负载均衡损失 ---------- */

double ai_moe_load_balancing_loss(const int *topk_idx, int tokens, int top_k,
                                  const double *gate_probs, int experts,
                                  double coefficient)
{
    if (!topk_idx || !gate_probs || tokens < 1 || top_k < 1 || experts < 1) return NAN;
    int *counts = (int *)calloc((size_t)experts, sizeof(int));
    if (!counts) return NAN;
    for (int t = 0; t < tokens; t++) {
        for (int k = 0; k < top_k; k++) {
            const int e = topk_idx[(size_t)t * (size_t)top_k + (size_t)k];
            if (e < 0 || e >= experts) {
                free(counts);
                return NAN;
            }
            counts[e]++;
        }
    }
    const double denom = (double)tokens * (double)top_k;
    double sum_fp = 0.0;
    for (int e = 0; e < experts; e++) {
        const double f = (double)counts[e] / denom;
        double p = 0.0;
        for (int t = 0; t < tokens; t++) {
            p += gate_probs[(size_t)t * (size_t)experts + (size_t)e];
        }
        p /= (double)tokens;
        sum_fp += f * p;
    }
    free(counts);
    return coefficient * (double)experts * sum_fp;
}
