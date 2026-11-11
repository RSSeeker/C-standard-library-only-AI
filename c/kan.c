/*
 * kan.c —— Kolmogorov-Arnold Network（B 样条边函数）C 实现
 *
 * 节点向量与 Cox-de Boor 递推、前向/反向的累加顺序均与
 * Python 版 AI.py 的 _kan_knots / _kan_basis / KANLayer 一致，
 * 以便逐位对拍（见 tests/compare_kan.c）。
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kan.h"

static int kan_check_pos_int(int v) { return v > 0; }

static int kan_check_positive_number(double v)
{
    return isfinite(v) && v > 0.0;
}

/* 均匀扩展节点向量，长度 num_grid + 2*order + 1。 */
static void kan_knots(int num_grid, int order, double range, double *knots)
{
    const double lo = -range;
    const double hi = range;
    const double step = (hi - lo) / (double)num_grid;
    const int n = num_grid + 2 * order + 1;
    for (int i = 0; i < n; i++)
        knots[i] = lo + (double)(i - order) * step;
}

/* 返回 (基函数值, 基函数导数)，长度均为 num_grid + order。 */
static int kan_basis(double x, int num_grid, int order, double range,
                     double *vals, double *ders)
{
    const int nk = num_grid + 2 * order + 1;
    const int n0 = nk - 1;
    double *knots = (double *)malloc(sizeof(double) * (size_t)nk);
    double *cur = (double *)malloc(sizeof(double) * (size_t)n0);
    double *curd = (double *)malloc(sizeof(double) * (size_t)n0);
    double *nv = (double *)malloc(sizeof(double) * (size_t)n0);
    double *nd = (double *)malloc(sizeof(double) * (size_t)n0);
    if (!knots || !cur || !curd || !nv || !nd) {
        free(knots);
        free(cur);
        free(curd);
        free(nv);
        free(nd);
        return -1;
    }
    kan_knots(num_grid, order, range, knots);
    for (int i = 0; i < n0; i++) {
        cur[i] = (knots[i] <= x && x < knots[i + 1]) ? 1.0 : 0.0;
        curd[i] = 0.0;
    }
    for (int p = 1; p <= order; p++) {
        const int m = n0 - p;
        for (int i = 0; i < m; i++) {
            const double d1 = knots[i + p] - knots[i];
            const double d2 = knots[i + p + 1] - knots[i + 1];
            const double a = d1 > 0.0 ? (x - knots[i]) / d1 : 0.0;
            const double b = d2 > 0.0 ? (knots[i + p + 1] - x) / d2 : 0.0;
            nv[i] = a * cur[i] + b * cur[i + 1];
            {
                double dv = 0.0;
                if (d1 > 0.0) dv += (double)p / d1 * cur[i];
                if (d2 > 0.0) dv -= (double)p / d2 * cur[i + 1];
                nd[i] = dv;
            }
        }
        for (int i = 0; i < m; i++) {
            cur[i] = nv[i];
            curd[i] = nd[i];
        }
    }
    for (int i = 0; i < num_grid + order; i++) {
        vals[i] = cur[i];
        ders[i] = curd[i];
    }
    free(knots);
    free(cur);
    free(curd);
    free(nv);
    free(nd);
    return 0;
}

int ai_kan_layer_create(AiKANLayer **out, int in_dim, int out_dim,
                        int num_grid, int spline_order, double grid_range)
{
    AiKANLayer *layer;
    size_t n_coeff;
    if (!out || !kan_check_pos_int(in_dim) || !kan_check_pos_int(out_dim) ||
        !kan_check_pos_int(num_grid) || spline_order <= 0 || spline_order > 5 ||
        !kan_check_positive_number(grid_range))
        return -1;
    layer = (AiKANLayer *)calloc(1, sizeof(*layer));
    if (!layer) return -1;
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->num_grid = num_grid;
    layer->spline_order = spline_order;
    layer->grid_range = grid_range;
    layer->num_basis = num_grid + spline_order;
    n_coeff = (size_t)out_dim * (size_t)in_dim * (size_t)layer->num_basis;
    layer->coeff = (double *)calloc(n_coeff, sizeof(double));
    layer->biases = (double *)calloc((size_t)out_dim, sizeof(double));
    if (!layer->coeff || !layer->biases) {
        ai_kan_layer_free(layer);
        return -1;
    }
    *out = layer;
    return 0;
}

void ai_kan_layer_free(AiKANLayer *layer)
{
    if (!layer) return;
    free(layer->coeff);
    free(layer->biases);
    free(layer->grad_coeff);
    free(layer->grad_biases);
    free(layer->cache_basis);
    free(layer->cache_ders);
    free(layer);
}

int ai_kan_layer_param_count(const AiKANLayer *layer)
{
    if (!layer) return 0;
    return layer->out_dim * (layer->in_dim * layer->num_basis + 1);
}

static int kan_layer_ensure_cache(AiKANLayer *layer, int n)
{
    const size_t size = (size_t)n * (size_t)layer->in_dim *
                        (size_t)layer->num_basis;
    if (layer->cache_basis && layer->cache_n == n) return 0;
    free(layer->cache_basis);
    free(layer->cache_ders);
    layer->cache_basis = (double *)malloc(sizeof(double) * (size ? size : 1));
    layer->cache_ders = (double *)malloc(sizeof(double) * (size ? size : 1));
    if (!layer->cache_basis || !layer->cache_ders) {
        free(layer->cache_basis);
        free(layer->cache_ders);
        layer->cache_basis = NULL;
        layer->cache_ders = NULL;
        layer->cache_n = 0;
        return -1;
    }
    layer->cache_n = n;
    return 0;
}

int ai_kan_layer_forward(AiKANLayer *layer, const double *xs, int n,
                         double *out)
{
    const int in = layer ? layer->in_dim : 0;
    const int outd = layer ? layer->out_dim : 0;
    const int nb = layer ? layer->num_basis : 0;
    if (!layer || !xs || !out || n <= 0) return -1;
    if (kan_layer_ensure_cache(layer, n) != 0) return -1;
    for (int r = 0; r < n; r++) {
        for (int j = 0; j < in; j++) {
            const double x = xs[(size_t)r * in + j];
            double *v = layer->cache_basis + ((size_t)r * in + j) * nb;
            double *d = layer->cache_ders + ((size_t)r * in + j) * nb;
            if (kan_basis(x, layer->num_grid, layer->spline_order,
                          layer->grid_range, v, d) != 0)
                return -1;
        }
    }
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < outd; i++) {
            double total = layer->biases[i];
            for (int j = 0; j < in; j++) {
                const double *cj = layer->coeff + ((size_t)i * in + j) * nb;
                const double *bj =
                    layer->cache_basis + ((size_t)r * in + j) * nb;
                for (int b = 0; b < nb; b++) total += cj[b] * bj[b];
            }
            out[(size_t)r * outd + i] = total;
        }
    }
    return 0;
}

int ai_kan_layer_backward(AiKANLayer *layer, const double *dout, int n,
                          double *dx)
{
    const int in = layer ? layer->in_dim : 0;
    const int outd = layer ? layer->out_dim : 0;
    const int nb = layer ? layer->num_basis : 0;
    const size_t n_coeff = (size_t)(outd * in * nb);
    if (!layer || !dout || !dx || n <= 0 || layer->cache_n != n) return -1;
    if (!layer->grad_coeff) {
        layer->grad_coeff = (double *)calloc(n_coeff ? n_coeff : 1,
                                             sizeof(double));
        layer->grad_biases = (double *)calloc((size_t)outd, sizeof(double));
        if (!layer->grad_coeff || !layer->grad_biases) {
            free(layer->grad_coeff);
            free(layer->grad_biases);
            layer->grad_coeff = NULL;
            layer->grad_biases = NULL;
            return -1;
        }
    }
    for (int r = 0; r < n; r++) {
        for (int i = 0; i < outd; i++) {
            const double d = dout[(size_t)r * outd + i];
            layer->grad_biases[i] += d;
            if (d == 0.0) continue;
            for (int j = 0; j < in; j++) {
                const double *bj =
                    layer->cache_basis + ((size_t)r * in + j) * nb;
                double *gc = layer->grad_coeff + ((size_t)i * in + j) * nb;
                for (int b = 0; b < nb; b++) gc[b] += d * bj[b];
            }
        }
        for (int j = 0; j < in; j++) {
            const double *dbj =
                layer->cache_ders + ((size_t)r * in + j) * nb;
            double total = 0.0;
            for (int i = 0; i < outd; i++) {
                const double d = dout[(size_t)r * outd + i];
                if (d == 0.0) continue;
                {
                    const double *cj = layer->coeff + ((size_t)i * in + j) * nb;
                    double inner = 0.0;
                    for (int b = 0; b < nb; b++) inner += cj[b] * dbj[b];
                    total += d * inner;
                }
            }
            dx[(size_t)r * in + j] = total;
        }
    }
    return 0;
}

void ai_kan_layer_zero_grad(AiKANLayer *layer)
{
    if (!layer) return;
    free(layer->grad_coeff);
    free(layer->grad_biases);
    layer->grad_coeff = NULL;
    layer->grad_biases = NULL;
}

int ai_kan_layer_sgd_step(AiKANLayer *layer, double lr)
{
    if (!layer || !layer->grad_coeff || !kan_check_positive_number(lr))
        return -1;
    {
        const int in = layer->in_dim;
        const int outd = layer->out_dim;
        const int nb = layer->num_basis;
        for (int i = 0; i < outd; i++) {
            for (int j = 0; j < in; j++) {
                double *cj = layer->coeff + ((size_t)i * in + j) * nb;
                const double *gj =
                    layer->grad_coeff + ((size_t)i * in + j) * nb;
                for (int b = 0; b < nb; b++) cj[b] -= lr * gj[b];
            }
            layer->biases[i] -= lr * layer->grad_biases[i];
        }
    }
    ai_kan_layer_zero_grad(layer);
    return 0;
}

int ai_kan_create(AiKAN **out, const int *sizes, int n_sizes,
                  int num_grid, int spline_order, double grid_range)
{
    AiKAN *kan;
    if (!out || !sizes || n_sizes < 2 || !kan_check_pos_int(num_grid) ||
        spline_order <= 0 || spline_order > 5 ||
        !kan_check_positive_number(grid_range))
        return -1;
    for (int i = 0; i < n_sizes; i++)
        if (!kan_check_pos_int(sizes[i])) return -1;
    kan = (AiKAN *)calloc(1, sizeof(*kan));
    if (!kan) return -1;
    kan->n_layers = n_sizes - 1;
    kan->sizes = (int *)malloc(sizeof(int) * (size_t)n_sizes);
    kan->layers = (AiKANLayer **)calloc((size_t)kan->n_layers,
                                        sizeof(AiKANLayer *));
    if (!kan->sizes || !kan->layers) {
        ai_kan_free(kan);
        return -1;
    }
    for (int i = 0; i < n_sizes; i++) kan->sizes[i] = sizes[i];
    for (int i = 0; i < kan->n_layers; i++) {
        if (ai_kan_layer_create(&kan->layers[i], sizes[i], sizes[i + 1],
                                num_grid, spline_order, grid_range) != 0) {
            ai_kan_free(kan);
            return -1;
        }
    }
    *out = kan;
    return 0;
}

void ai_kan_free(AiKAN *kan)
{
    if (!kan) return;
    if (kan->layers) {
        for (int i = 0; i < kan->n_layers; i++)
            ai_kan_layer_free(kan->layers[i]);
        free(kan->layers);
    }
    free(kan->sizes);
    free(kan);
}

int ai_kan_param_count(const AiKAN *kan)
{
    int total = 0;
    if (!kan) return 0;
    for (int i = 0; i < kan->n_layers; i++)
        total += ai_kan_layer_param_count(kan->layers[i]);
    return total;
}

int ai_kan_forward_batch(AiKAN *kan, const double *xs, int n, double *out)
{
    int prev;
    double *buf;
    if (!kan || !xs || !out || n <= 0) return -1;
    prev = kan->sizes[0];
    buf = (double *)malloc(sizeof(double) * (size_t)n * (size_t)prev);
    if (!buf) return -1;
    memcpy(buf, xs, sizeof(double) * (size_t)n * (size_t)prev);
    for (int idx = 0; idx < kan->n_layers; idx++) {
        const int cur = kan->sizes[idx + 1];
        double *next = (double *)malloc(sizeof(double) * (size_t)n *
                                        (size_t)cur);
        if (!next) {
            free(buf);
            return -1;
        }
        if (ai_kan_layer_forward(kan->layers[idx], buf, n, next) != 0) {
            free(buf);
            free(next);
            return -1;
        }
        free(buf);
        buf = next;
        prev = cur;
    }
    memcpy(out, buf, sizeof(double) * (size_t)n * (size_t)prev);
    free(buf);
    return 0;
}

int ai_kan_backward_batch(AiKAN *kan, const double *dout, int n, double *dx)
{
    int cur;
    double *buf;
    if (!kan || !dout || !dx || n <= 0) return -1;
    cur = kan->sizes[kan->n_layers];
    buf = (double *)malloc(sizeof(double) * (size_t)n * (size_t)cur);
    if (!buf) return -1;
    memcpy(buf, dout, sizeof(double) * (size_t)n * (size_t)cur);
    for (int idx = kan->n_layers - 1; idx >= 0; idx--) {
        const int prev = kan->sizes[idx];
        double *next = (double *)malloc(sizeof(double) * (size_t)n *
                                        (size_t)prev);
        if (!next) {
            free(buf);
            return -1;
        }
        if (ai_kan_layer_backward(kan->layers[idx], buf, n, next) != 0) {
            free(buf);
            free(next);
            return -1;
        }
        free(buf);
        buf = next;
        cur = prev;
    }
    memcpy(dx, buf, sizeof(double) * (size_t)n * (size_t)cur);
    free(buf);
    return 0;
}

int ai_kan_loss_and_backward(AiKAN *kan, const double *xs, const double *ys,
                             int n, double *out_loss, double *out_dx)
{
    int out_dim;
    double *pred;
    double *dout;
    double total = 0.0;
    if (!kan || !xs || !ys || n <= 0) return -1;
    out_dim = kan->sizes[kan->n_layers];
    pred = (double *)malloc(sizeof(double) * (size_t)n * (size_t)out_dim);
    dout = (double *)malloc(sizeof(double) * (size_t)n * (size_t)out_dim);
    if (!pred || !dout) {
        free(pred);
        free(dout);
        return -1;
    }
    if (ai_kan_forward_batch(kan, xs, n, pred) != 0) {
        free(pred);
        free(dout);
        return -1;
    }
    for (int r = 0; r < n; r++) {
        for (int k = 0; k < out_dim; k++) {
            const double diff =
                pred[(size_t)r * out_dim + k] - ys[(size_t)r * out_dim + k];
            total += diff * diff;
            dout[(size_t)r * out_dim + k] = 2.0 * diff / (double)n;
        }
    }
    if (out_dx) {
        if (ai_kan_backward_batch(kan, dout, n, out_dx) != 0) {
            free(pred);
            free(dout);
            return -1;
        }
    } else {
        double *scratch = (double *)malloc(sizeof(double) * (size_t)n *
                                           (size_t)kan->sizes[0]);
        if (!scratch) {
            free(pred);
            free(dout);
            return -1;
        }
        if (ai_kan_backward_batch(kan, dout, n, scratch) != 0) {
            free(scratch);
            free(pred);
            free(dout);
            return -1;
        }
        free(scratch);
    }
    free(pred);
    free(dout);
    if (out_loss) *out_loss = total / (double)n;
    return 0;
}

void ai_kan_zero_grad(AiKAN *kan)
{
    if (!kan) return;
    for (int i = 0; i < kan->n_layers; i++)
        ai_kan_layer_zero_grad(kan->layers[i]);
}

int ai_kan_sgd_step(AiKAN *kan, double lr)
{
    if (!kan || !kan_check_positive_number(lr)) return -1;
    for (int i = 0; i < kan->n_layers; i++)
        if (ai_kan_layer_sgd_step(kan->layers[i], lr) != 0) return -1;
    return 0;
}
