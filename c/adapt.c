/*
 * adapt.c — LoRA / EMA / SWA / Lookahead 的 C17 实现
 *
 * 每一处求和都按 AI.py 的写法选累加方式：
 *   sum(map(mul, row, x))      -> sum_neumaier（CPython 3.12+ 的补偿求和）
 *   显式 for 循环 acc += ...    -> 朴素累加
 */
#include "adapt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numeric.h"

static void *ad_malloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "adapt: malloc(%zu) 失败\n", n);
        exit(1);
    }
    return p;
}

static void *ad_calloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "adapt: calloc(%zu, %zu) 失败\n", count, size);
        exit(1);
    }
    return p;
}

static void ad_die(const char *where, const char *msg)
{
    fprintf(stderr, "%s: %s\n", where, msg);
    exit(1);
}

static void ad_check(int cond, const char *where, const char *msg)
{
    if (!cond) ad_die(where, msg);
}

/* ---------- LoRA ---------- */

LoRA *lora_create(int fan_in, int fan_out, int rank, double alpha)
{
    ad_check(fan_in > 0, "lora_create", "fan_in 必须是正整数");
    ad_check(fan_out > 0, "lora_create", "fan_out 必须是正整数");
    ad_check(rank > 0, "lora_create", "rank 必须是正整数");

    LoRA *l = (LoRA *)ad_calloc(1, sizeof(LoRA));
    l->fan_in = fan_in;
    l->fan_out = fan_out;
    l->rank = rank;
    l->alpha = (alpha > 0.0) ? alpha : (double)rank;
    l->scaling = l->alpha / (double)rank;
    l->merged = 0;

    l->A = (double *)ad_calloc((size_t)rank * fan_in, sizeof(double));
    l->gA = (double *)ad_calloc((size_t)rank * fan_in, sizeof(double));
    l->B = (double *)ad_calloc((size_t)fan_out * rank, sizeof(double));
    l->gB = (double *)ad_calloc((size_t)fan_out * rank, sizeof(double));
    l->x = (double *)ad_malloc((size_t)fan_in * sizeof(double));
    l->h = (double *)ad_malloc((size_t)rank * sizeof(double));
    return l;
}

void lora_free(LoRA *l)
{
    if (!l) return;
    free(l->A);
    free(l->gA);
    free(l->B);
    free(l->gB);
    free(l->x);
    free(l->h);
    free(l);
}

void lora_zero_grad(LoRA *l)
{
    ad_check(l != NULL, "lora_zero_grad", "l 不能为空");
    memset(l->gA, 0, (size_t)l->rank * l->fan_in * sizeof(double));
    memset(l->gB, 0, (size_t)l->fan_out * l->rank * sizeof(double));
}

void lora_set_params(LoRA *l, const double *A, const double *B)
{
    ad_check(l != NULL, "lora_set_params", "l 不能为空");
    if (A) memcpy(l->A, A, (size_t)l->rank * l->fan_in * sizeof(double));
    if (B) memcpy(l->B, B, (size_t)l->fan_out * l->rank * sizeof(double));
}

int lora_trainable_params(const LoRA *l)
{
    ad_check(l != NULL, "lora_trainable_params", "l 不能为空");
    return l->rank * l->fan_in + l->fan_out * l->rank;
}

void lora_forward(LoRA *l, const double *W, const double *b,
                  const double *x, double *y)
{
    ad_check(l != NULL, "lora_forward", "l 不能为空");
    ad_check(W != NULL && x != NULL && y != NULL, "lora_forward", "W/x/y 不能为空");

    const int fi = l->fan_in;
    const int fo = l->fan_out;
    const int r = l->rank;

    if (l->merged) {
        for (int j = 0; j < fo; j++) {
            const double *row = W + (size_t)j * fi;
            double z = dot_neumaier(row, x, fi);
            y[j] = z + (b ? b[j] : 0.0);
        }
        return;
    }

    memcpy(l->x, x, (size_t)fi * sizeof(double));
    for (int k = 0; k < r; k++) {
        l->h[k] = dot_neumaier(l->A + (size_t)k * fi, x, fi);
    }
    for (int j = 0; j < fo; j++) {
        double base = dot_neumaier(W + (size_t)j * fi, x, fi);
        double delta = dot_neumaier(l->B + (size_t)j * r, l->h, r);
        y[j] = base + (b ? b[j] : 0.0) + l->scaling * delta;
    }
}

void lora_backward(LoRA *l, const double *W, const double *dout, double *dx)
{
    ad_check(l != NULL, "lora_backward", "l 不能为空");
    ad_check(W != NULL && dout != NULL && dx != NULL, "lora_backward", "W/dout/dx 不能为空");

    const int fi = l->fan_in;
    const int fo = l->fan_out;
    const int r = l->rank;

    /* dx_base：原层反向（原层参数冻结，只取 dx） */
    double *col = (double *)ad_malloc((size_t)fo * sizeof(double));
    for (int i = 0; i < fi; i++) {
        for (int j = 0; j < fo; j++) col[j] = W[(size_t)j * fi + i];
        double t = dot_neumaier(col, dout, fo);
        dx[i] = t;
    }
    free(col);

    if (l->merged) return;

    /* grad_B += scaling * dout ⊗ h */
    for (int j = 0; j < fo; j++) {
        const double g = l->scaling * dout[j];
        double *row = l->gB + (size_t)j * r;
        for (int k = 0; k < r; k++) row[k] += g * l->h[k];
    }

    /* dh = scaling * B^T dout（朴素累加） */
    double *dh = (double *)ad_malloc((size_t)r * sizeof(double));
    for (int k = 0; k < r; k++) {
        double acc = 0.0;
        for (int j = 0; j < fo; j++) acc += l->B[(size_t)j * r + k] * dout[j];
        dh[k] = l->scaling * acc;
    }

    /* grad_A += dh ⊗ x */
    for (int k = 0; k < r; k++) {
        const double g = dh[k];
        double *row = l->gA + (size_t)k * fi;
        for (int i = 0; i < fi; i++) row[i] += g * l->x[i];
    }

    /* dx += A^T dh（朴素累加） */
    for (int i = 0; i < fi; i++) {
        double acc = 0.0;
        for (int k = 0; k < r; k++) acc += l->A[(size_t)k * fi + i] * dh[k];
        dx[i] += acc;
    }
    free(dh);
}

void lora_merge(LoRA *l, double *W)
{
    ad_check(l != NULL, "lora_merge", "l 不能为空");
    ad_check(W != NULL, "lora_merge", "W 不能为空");
    if (l->merged) return;

    const int fi = l->fan_in;
    const int fo = l->fan_out;
    const int r = l->rank;
    for (int j = 0; j < fo; j++) {
        for (int i = 0; i < fi; i++) {
            double acc = 0.0;
            for (int k = 0; k < r; k++) {
                acc += l->B[(size_t)j * r + k] * l->A[(size_t)k * fi + i];
            }
            W[(size_t)j * fi + i] += l->scaling * acc;
        }
    }
    memset(l->B, 0, (size_t)fo * r * sizeof(double));
    l->merged = 1;
}

void lora_effective_weight(const LoRA *l, const double *W, double *out)
{
    ad_check(l != NULL, "lora_effective_weight", "l 不能为空");
    ad_check(W != NULL && out != NULL, "lora_effective_weight", "W/out 不能为空");
    const size_t n = (size_t)l->fan_out * l->fan_in;
    memcpy(out, W, n * sizeof(double));
    if (l->merged) return;
    for (int j = 0; j < l->fan_out; j++) {
        for (int i = 0; i < l->fan_in; i++) {
            double acc = 0.0;
            for (int k = 0; k < l->rank; k++) {
                acc += l->B[(size_t)j * l->rank + k] * l->A[(size_t)k * l->fan_in + i];
            }
            out[(size_t)j * l->fan_in + i] += l->scaling * acc;
        }
    }
}

/* ---------- 扁平快照 ---------- */

int mlp_flat_size(const MLP *m)
{
    ad_check(m != NULL, "mlp_flat_size", "m 不能为空");
    int n = 0;
    for (int i = 0; i < m->n_layers; i++) {
        n += m->layers[i].fan_out * m->layers[i].fan_in + m->layers[i].fan_out;
    }
    return n;
}

void mlp_flat_read(const MLP *m, double *out)
{
    ad_check(m != NULL && out != NULL, "mlp_flat_read", "m/out 不能为空");
    int off = 0;
    for (int i = 0; i < m->n_layers; i++) {
        const Layer *L = &m->layers[i];
        const int wn = L->fan_out * L->fan_in;
        memcpy(out + off, L->W, (size_t)wn * sizeof(double));
        off += wn;
        memcpy(out + off, L->b, (size_t)L->fan_out * sizeof(double));
        off += L->fan_out;
    }
}

void mlp_flat_write(MLP *m, const double *in)
{
    ad_check(m != NULL && in != NULL, "mlp_flat_write", "m/in 不能为空");
    int off = 0;
    for (int i = 0; i < m->n_layers; i++) {
        Layer *L = &m->layers[i];
        const int wn = L->fan_out * L->fan_in;
        memcpy(L->W, in + off, (size_t)wn * sizeof(double));
        off += wn;
        memcpy(L->b, in + off, (size_t)L->fan_out * sizeof(double));
        off += L->fan_out;
    }
}

/* ---------- EMA ---------- */

EMA *ema_create(const MLP *m, double decay, int warmup)
{
    ad_check(m != NULL, "ema_create", "m 不能为空");
    ad_check(decay >= 0.0 && decay < 1.0, "ema_create", "decay 必须在 [0, 1) 内");
    ad_check(warmup >= 0, "ema_create", "warmup 必须是非负整数");

    EMA *e = (EMA *)ad_calloc(1, sizeof(EMA));
    e->n = mlp_flat_size(m);
    e->shadow = (double *)ad_malloc((size_t)e->n * sizeof(double));
    e->decay = decay;
    e->warmup = warmup;
    e->step = 0;
    mlp_flat_read(m, e->shadow);
    return e;
}

void ema_free(EMA *e)
{
    if (!e) return;
    free(e->shadow);
    free(e);
}

double ema_effective_decay(const EMA *e)
{
    ad_check(e != NULL, "ema_effective_decay", "e 不能为空");
    if (e->warmup > 0 && e->step < e->warmup) {
        const double tf = (1.0 + (double)e->step) / (10.0 + (double)e->step);
        return (e->decay < tf) ? e->decay : tf;
    }
    return e->decay;
}

void ema_update(EMA *e, const MLP *m)
{
    ad_check(e != NULL && m != NULL, "ema_update", "e/m 不能为空");
    const double d = ema_effective_decay(e);
    const double od = 1.0 - d;
    e->step += 1;

    double *cur = (double *)ad_malloc((size_t)e->n * sizeof(double));
    mlp_flat_read(m, cur);
    for (int i = 0; i < e->n; i++) {
        e->shadow[i] = d * e->shadow[i] + od * cur[i];
    }
    free(cur);
}

void ema_copy_to(const EMA *e, MLP *m)
{
    ad_check(e != NULL && m != NULL, "ema_copy_to", "e/m 不能为空");
    ad_check(mlp_flat_size(m) == e->n, "ema_copy_to", "形状与模型不符");
    mlp_flat_write(m, e->shadow);
}

/* ---------- SWA ---------- */

SWA *swa_create(const MLP *m)
{
    ad_check(m != NULL, "swa_create", "m 不能为空");
    SWA *s = (SWA *)ad_calloc(1, sizeof(SWA));
    s->n = mlp_flat_size(m);
    s->avg = (double *)ad_malloc((size_t)s->n * sizeof(double));
    s->count = 0;
    mlp_flat_read(m, s->avg);
    return s;
}

void swa_free(SWA *s)
{
    if (!s) return;
    free(s->avg);
    free(s);
}

void swa_update(SWA *s, const MLP *m)
{
    ad_check(s != NULL && m != NULL, "swa_update", "s/m 不能为空");
    ad_check(mlp_flat_size(m) == s->n, "swa_update", "形状与模型不符");
    s->count += 1;
    double *cur = (double *)ad_malloc((size_t)s->n * sizeof(double));
    mlp_flat_read(m, cur);
    const double inv = 1.0 / (double)s->count;
    for (int i = 0; i < s->n; i++) {
        s->avg[i] = s->avg[i] + (cur[i] - s->avg[i]) * inv;
    }
    free(cur);
}

void swa_apply(const SWA *s, MLP *m)
{
    ad_check(s != NULL && m != NULL, "swa_apply", "s/m 不能为空");
    ad_check(mlp_flat_size(m) == s->n, "swa_apply", "形状与模型不符");
    mlp_flat_write(m, s->avg);
}

/* ---------- Lookahead ---------- */

Lookahead *lookahead_create(const MLP *m, int k, double alpha)
{
    ad_check(m != NULL, "lookahead_create", "m 不能为空");
    ad_check(k > 0, "lookahead_create", "k 必须是正整数");
    ad_check(alpha > 0.0 && alpha <= 1.0, "lookahead_create", "alpha 必须在 (0, 1] 内");

    Lookahead *l = (Lookahead *)ad_calloc(1, sizeof(Lookahead));
    l->n = mlp_flat_size(m);
    l->slow = (double *)ad_malloc((size_t)l->n * sizeof(double));
    l->k = k;
    l->alpha = alpha;
    l->step = 0;
    mlp_flat_read(m, l->slow);
    return l;
}

void lookahead_free(Lookahead *l)
{
    if (!l) return;
    free(l->slow);
    free(l);
}

int lookahead_step(Lookahead *l, MLP *m)
{
    ad_check(l != NULL && m != NULL, "lookahead_step", "l/m 不能为空");
    ad_check(mlp_flat_size(m) == l->n, "lookahead_step", "形状与模型不符");
    l->step += 1;
    if (l->step % l->k != 0) return 0;

    double *fast = (double *)ad_malloc((size_t)l->n * sizeof(double));
    mlp_flat_read(m, fast);
    for (int i = 0; i < l->n; i++) {
        l->slow[i] = l->slow[i] + l->alpha * (fast[i] - l->slow[i]);
    }
    mlp_flat_write(m, l->slow);
    free(fast);
    return 1;
}
