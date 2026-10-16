/*
 * layers_norm.c —— Dropout / BatchNorm1d / LayerNorm 的 C17 实现
 *
 * 移植自 AI.py:453-591，逐行对照 Python 的运算顺序。三处最容易写错的地方：
 *   1. BatchNorm1d.forward_batch 的列统计：Python 先 zip(*x_batch) 转置再 sum(col)，
 *      这里同样先转置到列主序缓冲，再用 sum_neumaier 求和（顺序完全一致）。
 *   2. BatchNorm1d.backward 的 s1/s2 只依赖 j（AI.py:548-551 修过的性能问题）：
 *      C 版按列一次算完，等价于"提到 i 循环外"的写法，且逐元素表达式顺序不变。
 *   3. 推理路径是 (x - mean) / sqrt(var + eps)，不是 (x - mean) * (1/sqrt(...))。
 */
#include "layers_norm.h"

#include "numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================== 内存 ============================== */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "layers_norm: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "layers_norm: out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

/* Module 包装：tensor.c 的 module_free() 会 free(m) 本身，所以这里必须用 malloc */
static Module *wrap_module(const ModuleVT *vt, void *self)
{
    Module *m = xmalloc(sizeof(Module));
    m->vt = vt;
    m->self = self;
    return m;
}

/* ============ 随机数（与 ai.c 同算法的 xorshift64*，不追求与 Python 的 MT 一致） ============ */

#define LAYERS_NORM_SEED 0x9E3779B97F4A7C15ULL

static unsigned long long rng_next(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(unsigned long long *s)
{
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

/* ============================== Dropout ==============================
 * Python:
 *   forward (training) : mask[i] = scale if random.random() > p else 0.0；out[i] = x[i]*mask[i]
 *   forward (eval)     : mask 全 1，直接返回 x
 *   backward           : delta[i] * mask[i]
 */
static void dropout_check_p(double p, const char *who)
{
    /* Python 在 forward 里算 1/(1-p)：p=1.0 抛 ZeroDivisionError，负数无意义。
     * 这里提前在构造/设置时校验，报错退出而不是产出 inf/nan。 */
    if (!(p >= 0.0 && p < 1.0)) {
        fprintf(stderr, "%s: p 必须在 [0, 1) 内, 实得 %g\n", who, p);
        exit(1);
    }
}

Dropout *dropout_create(double p)
{
    dropout_check_p(p, "dropout_create");
    Dropout *self = xcalloc(1, sizeof(Dropout));
    self->p = p;
    self->training = 1; /* 与 Python 的 self.training = True 一致 */
    self->rng = LAYERS_NORM_SEED;
    return self;
}

void dropout_free(Dropout *self)
{
    if (!self) return;
    free(self->mask);
    free(self);
}

void dropout_zero_grad(Dropout *self)
{
    (void)self; /* Dropout 没有可训练参数，空实现仅为统一接口 */
}

void dropout_set_training(Dropout *self, int training)
{
    self->training = training ? 1 : 0;
}

void dropout_set_p(Dropout *self, double p)
{
    dropout_check_p(p, "dropout_set_p");
    self->p = p;
}

void dropout_set_seed(Dropout *self, unsigned long long seed)
{
    /* xorshift 的 0 是吸收态，seed=0 会让 mask 恒定，这里退回默认种子 */
    self->rng = seed ? seed : LAYERS_NORM_SEED;
}

void dropout_reserve(Dropout *self, int n)
{
    if (n <= self->mask_cap) return;
    int cap = self->mask_cap > 0 ? self->mask_cap : 16;
    while (cap < n) cap *= 2;
    free(self->mask);
    self->mask = xmalloc((size_t)cap * sizeof(double));
    self->mask_cap = cap;
    self->mask_len = 0; /* 旧 mask 作废：backward 之前必须重新 forward */
}

void dropout_set_mask(Dropout *self, const double *mask)
{
    self->pending = mask; /* 不拷贝：下一次 forward 时才知道长度，届时按输入 size 拷进内部缓冲 */
    self->has_pending = (mask != NULL);
}

const double *dropout_mask(const Dropout *self)
{
    return self->mask;
}

int dropout_mask_len(const Dropout *self)
{
    return self->mask_len;
}

void dropout_forward(Dropout *self, const Tensor *x, Tensor *out)
{
    /* 形状任意，但 out 必须与 x 同形：不允许静默按较短的那个截断 */
    tensor_check_shape(out, x->ndim, x->shape, "dropout_forward(out)");
    const int n = x->size;
    if (n <= 0) {
        fprintf(stderr, "dropout_forward: 输入为空 (size=%d)\n", n);
        exit(1);
    }

    dropout_reserve(self, n); /* 只在输入变大时扩容一次 */
    self->mask_len = n;

    if (self->training) {
        if (self->has_pending) {
            /* 灌入的确定性 mask（一次性）：拷进内部缓冲，backward 仍走同一条路径 */
            memcpy(self->mask, self->pending, (size_t)n * sizeof(double));
            self->has_pending = 0;
            self->pending = NULL;
        } else {
            const double scale = 1.0 / (1.0 - self->p); /* inverted dropout */
            for (int i = 0; i < n; i++)
                self->mask[i] = (rng_uniform(&self->rng) > self->p) ? scale : 0.0;
        }
        for (int i = 0; i < n; i++) out->data[i] = x->data[i] * self->mask[i];
    } else {
        /* 推理：透传（Python 返回 x 本身并把 mask 置 1；这里写入 out 并同步 mask，
         * 这样 forward 之后紧接着 backward 也不会拿到过期的 mask） */
        for (int i = 0; i < n; i++) {
            self->mask[i] = 1.0;
            out->data[i] = x->data[i];
        }
    }
}

void dropout_backward(Dropout *self, const Tensor *dout, Tensor *dx)
{
    tensor_check_shape(dx, dout->ndim, dout->shape, "dropout_backward(dx)");
    if (self->mask_len <= 0) {
        fprintf(stderr, "dropout_backward: 请先调用 dropout_forward (反向依赖它缓存的 mask)\n");
        exit(1);
    }
    if (dout->size != self->mask_len) {
        fprintf(stderr, "dropout_backward: dout size=%d 与最近一次 forward 的输入 size=%d 不一致\n",
                dout->size, self->mask_len);
        exit(1);
    }
    for (int i = 0; i < dout->size; i++) dx->data[i] = dout->data[i] * self->mask[i];
}

int dropout_param_count(Dropout *self)
{
    (void)self;
    return 0;
}

/* ---------- Dropout 的 vtable ---------- */

static void dropout_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    dropout_forward((Dropout *)m->self, x, out);
}

static void dropout_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    dropout_backward((Dropout *)m->self, dout, dx);
}

static void dropout_vt_zero_grad(Module *m)
{
    dropout_zero_grad((Dropout *)m->self);
}

static void dropout_vt_free(Module *m)
{
    dropout_free((Dropout *)m->self);
}

static int dropout_vt_param_count(Module *m)
{
    return dropout_param_count((Dropout *)m->self);
}

static void dropout_vt_to_string(Module *m, char *buf, size_t n)
{
    const Dropout *d = (const Dropout *)m->self;
    snprintf(buf, n, "Dropout(p=%g, training=%d)", d->p, d->training);
}

static const ModuleVT DROPOUT_VT = {
    "Dropout",
    dropout_vt_forward,
    dropout_vt_backward,
    dropout_vt_zero_grad,
    dropout_vt_free,
    dropout_vt_param_count,
    dropout_vt_to_string
};

Module *dropout_as_module(Dropout *self)
{
    return wrap_module(&DROPOUT_VT, self);
}

/* ============================ BatchNorm1d ============================ */

BatchNorm1d *bn1d_create(int num_features, double eps, double momentum)
{
    if (num_features <= 0) {
        fprintf(stderr, "bn1d_create: num_features 必须为正, 实得 %d\n", num_features);
        exit(1);
    }
    BatchNorm1d *self = xcalloc(1, sizeof(BatchNorm1d));
    const int d = num_features;
    self->d = d;
    self->eps = eps;
    self->momentum = momentum;
    self->training = 1; /* 与 Python 的 self.training = True 一致 */

    self->gamma = xmalloc((size_t)d * sizeof(double));
    self->beta = xcalloc((size_t)d, sizeof(double));
    self->running_mean = xcalloc((size_t)d, sizeof(double));
    self->running_var = xmalloc((size_t)d * sizeof(double));
    self->grad_gamma = xcalloc((size_t)d, sizeof(double));
    self->grad_beta = xcalloc((size_t)d, sizeof(double));
    self->mean = xcalloc((size_t)d, sizeof(double));
    self->var = xcalloc((size_t)d, sizeof(double));
    self->inv_std = xcalloc((size_t)d, sizeof(double));
    for (int j = 0; j < d; j++) {
        self->gamma[j] = 1.0;
        self->running_var[j] = 1.0;
    }
    /* x_hat / col 依赖 batch 大小，首次 forward_batch 时按需分配（bn1d_reserve） */
    return self;
}

void bn1d_free(BatchNorm1d *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->beta);
    free(self->running_mean);
    free(self->running_var);
    free(self->grad_gamma);
    free(self->grad_beta);
    free(self->x_hat);
    free(self->col);
    free(self->mean);
    free(self->var);
    free(self->inv_std);
    free(self);
}

void bn1d_zero_grad(BatchNorm1d *self)
{
    memset(self->grad_gamma, 0, (size_t)self->d * sizeof(double));
    memset(self->grad_beta, 0, (size_t)self->d * sizeof(double));
}

void bn1d_set_training(BatchNorm1d *self, int training)
{
    self->training = training ? 1 : 0;
}

void bn1d_set_params(BatchNorm1d *self, const double *gamma, const double *beta)
{
    if (!gamma || !beta) {
        fprintf(stderr, "bn1d_set_params: gamma/beta 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->gamma, gamma, (size_t)self->d * sizeof(double));
    memcpy(self->beta, beta, (size_t)self->d * sizeof(double));
}

void bn1d_set_running(BatchNorm1d *self, const double *mean, const double *var)
{
    if (!mean || !var) {
        fprintf(stderr, "bn1d_set_running: mean/var 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->running_mean, mean, (size_t)self->d * sizeof(double));
    memcpy(self->running_var, var, (size_t)self->d * sizeof(double));
}

void bn1d_reserve(BatchNorm1d *self, int n)
{
    if (n <= self->cap) return;
    int cap = self->cap > 0 ? self->cap : 8;
    while (cap < n) cap *= 2;
    free(self->x_hat);
    free(self->col);
    self->x_hat = xmalloc((size_t)self->d * (size_t)cap * sizeof(double));
    self->col = xmalloc((size_t)self->d * (size_t)cap * sizeof(double));
    self->cap = cap;
    self->n = 0;           /* 旧缓存作废 */
    self->batch_ready = 0;
}

const double *bn1d_grad_gamma(const BatchNorm1d *self)
{
    return self->grad_gamma;
}

const double *bn1d_grad_beta(const BatchNorm1d *self)
{
    return self->grad_beta;
}

void bn1d_forward_batch(BatchNorm1d *self, const Tensor *x_batch, Tensor *out)
{
    const int d = self->d;
    /* (N, D)：ndim 不是 2 时也会在 tensor_check_shape 里报错退出 */
    int want_in[2] = { x_batch->shape[0], d };
    tensor_check_shape(x_batch, 2, want_in, "bn1d_forward_batch(x_batch)");
    const int n = x_batch->shape[0];
    if (n <= 0) {
        fprintf(stderr, "bn1d_forward_batch: batch 不能为空\n");
        exit(1);
    }
    int want_out[2] = { n, d };
    tensor_check_shape(out, 2, want_out, "bn1d_forward_batch(out)");

    bn1d_reserve(self, n); /* 只在 batch 变大时扩容一次，正常训练零分配 */
    self->n = n;

    double *col = self->col;

    /* 1) 输入转置成列主序：对应 Python 的 cols = list(zip(*x_batch))，
     *    此后每一列在内存里连续，可以直接交给 sum_neumaier */
    for (int i = 0; i < n; i++) {
        const double *row = x_batch->data + (size_t)i * d;
        for (int j = 0; j < d; j++) col[(size_t)j * n + i] = row[j];
    }

    /* 2) 列均值 / 列方差（Neumaier 求和，与 Python 的 sum(c) / sum((v-m)**2) 同序） */
    for (int j = 0; j < d; j++) {
        double *c = col + (size_t)j * n;
        const double m = sum_neumaier(c, n) / n; /* Python: sum(c) / n */
        self->mean[j] = m;
        for (int i = 0; i < n; i++) c[i] = c[i] - m;  /* Python: (v - m) */
        self->var[j] = dot_neumaier(c, c, n) / n;     /* Python: sum((v - m) ** 2) / n */
    }

    /* 3) running 统计量（朴素 +，与 Python 一致；n=1 时无偏估计无定义，用有偏估计 fallback） */
    for (int j = 0; j < d; j++) {
        self->running_mean[j] = (1.0 - self->momentum) * self->running_mean[j]
                              + self->momentum * self->mean[j];
        if (n > 1) {
            /* Python: (1-m)*rv + m*var*n/(n-1)，乘法从左到右 */
            self->running_var[j] = (1.0 - self->momentum) * self->running_var[j]
                                 + self->momentum * self->var[j] * (double)n / (double)(n - 1);
        } else {
            self->running_var[j] = (1.0 - self->momentum) * self->running_var[j]
                                 + self->momentum * self->var[j];
        }
    }

    /* 4) inv_std 与 x_hat（列主序 d×n：x_hat[j*n + i] 即 Python 的 x_hat[i][j]） */
    for (int j = 0; j < d; j++) {
        const double s = 1.0 / sqrt(self->var[j] + self->eps);
        self->inv_std[j] = s;
        const double *c = col + (size_t)j * n;
        double *xh = self->x_hat + (size_t)j * n;
        for (int i = 0; i < n; i++) xh[i] = c[i] * s; /* Python: (v - m) * inv_std */
    }

    /* 5) 输出：out[i][j] = gamma[j] * x_hat[i][j] + beta[j]（逐元素乘加） */
    for (int i = 0; i < n; i++) {
        double *row = out->data + (size_t)i * d;
        for (int j = 0; j < d; j++)
            row[j] = self->gamma[j] * self->x_hat[(size_t)j * n + i] + self->beta[j];
    }

    self->batch_ready = 1;
}

void bn1d_forward(BatchNorm1d *self, const Tensor *x, Tensor *out)
{
    int want[1] = { self->d };
    tensor_check_shape(x, 1, want, "bn1d_forward(x)");
    tensor_check_shape(out, 1, want, "bn1d_forward(out)");

    for (int j = 0; j < self->d; j++) {
        /* Python: xh = (x[j] - running_mean[j]) / math.sqrt(running_var[j] + eps)
         *         out[j] = gamma[j] * xh + beta[j]
         * 注意这里是"除以 sqrt"而不是"乘 1/sqrt"：两者最后一位不同 */
        const double xh = (x->data[j] - self->running_mean[j])
                        / sqrt(self->running_var[j] + self->eps);
        out->data[j] = self->gamma[j] * xh + self->beta[j];
    }

    /* 推理路径不产生 batch 缓存，之后调 backward 会直接报错而不是拿旧缓存算出错值 */
    self->batch_ready = 0;
}

void bn1d_backward(BatchNorm1d *self, const Tensor *dout, Tensor *dx)
{
    const int d = self->d;
    const int n = self->n;
    if (!self->batch_ready || n <= 0) {
        fprintf(stderr, "bn1d_backward: 请先调用 bn1d_forward_batch (反向依赖其缓存的 x_hat/inv_std)\n");
        exit(1);
    }
    int want[2] = { n, d };
    tensor_check_shape(dout, 2, want, "bn1d_backward(dout)");
    tensor_check_shape(dx, 2, want, "bn1d_backward(dx)");

    double *col = self->col; /* 复用列缓冲：反向时放 dout 的列 */

    for (int j = 0; j < d; j++) {
        /* 转置出第 j 列：col[i] = dout[i][j] */
        for (int i = 0; i < n; i++) col[i] = dout->data[(size_t)i * d + j];
        const double *xh = self->x_hat + (size_t)j * n;

        /* Python: grad_gamma[j] = sum(dout[i][j] * x_hat[i][j])；grad_beta[j] = sum(dout[i][j])
         * 按 Module 约定累加（Python 是覆盖赋值），调用方负责清零 / 按 batch 平均 */
        self->grad_gamma[j] += dot_neumaier(col, xh, n);
        self->grad_beta[j] += sum_neumaier(col, n);

        /* Python: dx_hat[i][j] = dout[i][j] * gamma[j]（原地缩放 col） */
        const double g = self->gamma[j];
        for (int i = 0; i < n; i++) col[i] = col[i] * g;

        /* s1[j] = sum(dx_hat[i][j])，s2[j] = sum(dx_hat[i][j] * x_hat[i][j])
         * 两者只依赖 j，Python 已把它们提到 i 循环外（AI.py:548-551 的性能修复），
         * 这里按列一次算完，结果与"提到循环外"逐位相同 */
        const double s1 = sum_neumaier(col, n);
        const double s2 = dot_neumaier(col, xh, n);

        /* Python: dx[i][j] = inv_std[j] * (n * dx_hat[i][j] - s1[j] - x_hat[i][j] * s2[j]) / n
         * 括号与除法的结合顺序必须照抄：写成 (inv_std/n) * (...) 会差最后几位 */
        const double inv_std = self->inv_std[j];
        const double nd = (double)n;
        for (int i = 0; i < n; i++)
            dx->data[(size_t)i * d + j] = inv_std * (nd * col[i] - s1 - xh[i] * s2) / nd;
    }
}

int bn1d_param_count(BatchNorm1d *self)
{
    return 2 * self->d; /* gamma + beta（running 统计量不是可训练参数） */
}

/* ---------- BatchNorm1d 的 vtable ---------- */

static void bn1d_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    /* 分派：训练走 batch 统计量 (N,D)，推理走 running 统计量 (D,)
     * —— 与 Python 的 forward_batch / forward 两条路径对应 */
    BatchNorm1d *bn = (BatchNorm1d *)m->self;
    if (bn->training) bn1d_forward_batch(bn, x, out);
    else              bn1d_forward(bn, x, out);
}

static void bn1d_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    bn1d_backward((BatchNorm1d *)m->self, dout, dx);
}

static void bn1d_vt_zero_grad(Module *m)
{
    bn1d_zero_grad((BatchNorm1d *)m->self);
}

static void bn1d_vt_free(Module *m)
{
    bn1d_free((BatchNorm1d *)m->self);
}

static int bn1d_vt_param_count(Module *m)
{
    return bn1d_param_count((BatchNorm1d *)m->self);
}

static void bn1d_vt_to_string(Module *m, char *buf, size_t n)
{
    const BatchNorm1d *bn = (const BatchNorm1d *)m->self;
    snprintf(buf, n, "BatchNorm1d(%d, eps=%g, momentum=%g, training=%d)",
             bn->d, bn->eps, bn->momentum, bn->training);
}

static const ModuleVT BN1D_VT = {
    "BatchNorm1d",
    bn1d_vt_forward,
    bn1d_vt_backward,
    bn1d_vt_zero_grad,
    bn1d_vt_free,
    bn1d_vt_param_count,
    bn1d_vt_to_string
};

Module *bn1d_as_module(BatchNorm1d *self)
{
    return wrap_module(&BN1D_VT, self);
}

/* ============================= LayerNorm =============================
 * Python:
 *   mean = sum(x) / d；var = sum((xi - mean)**2) / d；inv_std = 1/sqrt(var + eps)
 *   x_hat = (xi - mean) * inv_std；out = gamma[j] * x_hat[j] + beta[j]
 *   backward: grad_gamma[j] = dout[j]*x_hat[j]（逐元素，不对 batch 求和）
 *             grad_beta = dout；dx_hat = dout[j]*gamma[j]
 *             s1 = sum(dx_hat)；s2 = sum(dx_hat[j]*x_hat[j])
 *             dx[j] = inv_std * (d*dx_hat[j] - s1 - x_hat[j]*s2) / d
 */

LayerNorm *layernorm_create(int num_features, double eps)
{
    if (num_features <= 0) {
        fprintf(stderr, "layernorm_create: num_features 必须为正, 实得 %d\n", num_features);
        exit(1);
    }
    LayerNorm *self = xcalloc(1, sizeof(LayerNorm));
    const int d = num_features;
    self->d = d;
    self->eps = eps;

    self->gamma = xmalloc((size_t)d * sizeof(double));
    self->beta = xcalloc((size_t)d, sizeof(double));
    self->grad_gamma = xcalloc((size_t)d, sizeof(double));
    self->grad_beta = xcalloc((size_t)d, sizeof(double));
    self->x_hat = xcalloc((size_t)d, sizeof(double));
    self->dx_hat = xcalloc((size_t)d, sizeof(double));
    for (int j = 0; j < d; j++) self->gamma[j] = 1.0;
    return self;
}

void layernorm_free(LayerNorm *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->beta);
    free(self->grad_gamma);
    free(self->grad_beta);
    free(self->x_hat);
    free(self->dx_hat);
    free(self);
}

void layernorm_zero_grad(LayerNorm *self)
{
    memset(self->grad_gamma, 0, (size_t)self->d * sizeof(double));
    memset(self->grad_beta, 0, (size_t)self->d * sizeof(double));
}

void layernorm_set_params(LayerNorm *self, const double *gamma, const double *beta)
{
    if (!gamma || !beta) {
        fprintf(stderr, "layernorm_set_params: gamma/beta 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->gamma, gamma, (size_t)self->d * sizeof(double));
    memcpy(self->beta, beta, (size_t)self->d * sizeof(double));
}

const double *layernorm_grad_gamma(const LayerNorm *self)
{
    return self->grad_gamma;
}

const double *layernorm_grad_beta(const LayerNorm *self)
{
    return self->grad_beta;
}

void layernorm_forward(LayerNorm *self, const Tensor *x, Tensor *out)
{
    const int d = self->d;
    int want[1] = { d };
    tensor_check_shape(x, 1, want, "layernorm_forward(x)");
    tensor_check_shape(out, 1, want, "layernorm_forward(out)");

    const double mean = sum_neumaier(x->data, d) / d; /* Python: sum(x) / d */
    double *xh = self->x_hat;
    for (int j = 0; j < d; j++) xh[j] = x->data[j] - mean; /* Python: (xi - mean) */
    const double var = dot_neumaier(xh, xh, d) / d;       /* Python: sum((xi-mean)**2) / d */
    const double inv_std = 1.0 / sqrt(var + self->eps);
    for (int j = 0; j < d; j++) xh[j] = xh[j] * inv_std;   /* Python: (xi - mean) * inv_std */
    for (int j = 0; j < d; j++) out->data[j] = self->gamma[j] * xh[j] + self->beta[j];

    self->mean = mean;
    self->var = var;
    self->inv_std = inv_std;
    self->ready = 1;
}

void layernorm_backward(LayerNorm *self, const Tensor *dout, Tensor *dx)
{
    const int d = self->d;
    if (!self->ready) {
        fprintf(stderr, "layernorm_backward: 请先调用 layernorm_forward (反向依赖其缓存的 x_hat/inv_std)\n");
        exit(1);
    }
    int want[1] = { d };
    tensor_check_shape(dout, 1, want, "layernorm_backward(dout)");
    tensor_check_shape(dx, 1, want, "layernorm_backward(dx)");

    const double *xh = self->x_hat;
    double *dxh = self->dx_hat;
    for (int j = 0; j < d; j++) {
        /* Python: grad_gamma[j] = dout[j] * x_hat[j]；grad_beta[j] = dout[j]
         * 按 Module 约定累加（TransformerBlock 逐 token += 累加，AI.py:1582-1584） */
        self->grad_gamma[j] += dout->data[j] * xh[j];
        self->grad_beta[j] += dout->data[j];
        dxh[j] = dout->data[j] * self->gamma[j]; /* Python: dx_hat[j] = dout[j] * gamma[j] */
    }

    const double s1 = sum_neumaier(dxh, d);      /* Python: s1 = sum(dx_hat) */
    const double s2 = dot_neumaier(dxh, xh, d);  /* Python: s2 = sum(dx_hat[j] * x_hat[j]) */

    /* Python: dx[j] = inv_std * (d * dx_hat[j] - s1 - x_hat[j] * s2) / d */
    const double dd = (double)d;
    for (int j = 0; j < d; j++)
        dx->data[j] = self->inv_std * (dd * dxh[j] - s1 - xh[j] * s2) / dd;
}

int layernorm_param_count(LayerNorm *self)
{
    return 2 * self->d; /* gamma + beta */
}

/* ---------- LayerNorm 的 vtable ---------- */

static void layernorm_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    layernorm_forward((LayerNorm *)m->self, x, out);
}

static void layernorm_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    layernorm_backward((LayerNorm *)m->self, dout, dx);
}

static void layernorm_vt_zero_grad(Module *m)
{
    layernorm_zero_grad((LayerNorm *)m->self);
}

static void layernorm_vt_free(Module *m)
{
    layernorm_free((LayerNorm *)m->self);
}

static int layernorm_vt_param_count(Module *m)
{
    return layernorm_param_count((LayerNorm *)m->self);
}

static void layernorm_vt_to_string(Module *m, char *buf, size_t n)
{
    const LayerNorm *ln = (const LayerNorm *)m->self;
    snprintf(buf, n, "LayerNorm(%d, eps=%g)", ln->d, ln->eps);
}

static const ModuleVT LAYERNORM_VT = {
    "LayerNorm",
    layernorm_vt_forward,
    layernorm_vt_backward,
    layernorm_vt_zero_grad,
    layernorm_vt_free,
    layernorm_vt_param_count,
    layernorm_vt_to_string
};

Module *layernorm_as_module(LayerNorm *self)
{
    return wrap_module(&LAYERNORM_VT, self);
}
