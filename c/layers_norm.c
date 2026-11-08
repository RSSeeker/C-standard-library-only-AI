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

/* ============================== RMSNorm ==============================
 * Python:
 *   mean_sq = sum(xi * xi) / d；inv_rms = 1/sqrt(mean_sq + eps)
 *   out[j] = gamma[j] * x[j] * r
 *   backward: grad_gamma[j] = dout[j] * x[j] * r（覆盖赋值，不累加）
 *             g = sum(dout[i] * gamma[i] * x[i])；r3 = r*r*r
 *             dx[j] = gamma[j]*dout[j]*r - g*x[j]*r3/d
 */

RMSNorm *rmsnorm_create(int num_features, double eps)
{
    if (num_features <= 0) {
        fprintf(stderr, "rmsnorm_create: num_features 必须为正, 实得 %d\n", num_features);
        exit(1);
    }
    RMSNorm *self = xcalloc(1, sizeof(RMSNorm));
    const int d = num_features;
    self->d = d;
    self->eps = eps;

    self->gamma = xmalloc((size_t)d * sizeof(double));
    self->grad_gamma = xcalloc((size_t)d, sizeof(double));
    self->x_cache = xcalloc((size_t)d, sizeof(double));
    self->tmp = xcalloc((size_t)d, sizeof(double));
    for (int j = 0; j < d; j++) self->gamma[j] = 1.0;
    self->inv_rms = 0.0;
    self->ready = 0;
    return self;
}

void rmsnorm_free(RMSNorm *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->grad_gamma);
    free(self->x_cache);
    free(self->tmp);
    free(self);
}

void rmsnorm_zero_grad(RMSNorm *self)
{
    if (!self) return;
    /* 注意：Python 的 grad_gamma 每次 backward 都覆盖赋值，
     * 清零只是为了让调用方有统一接口，并不会让下次反向"从头累加"。 */
    memset(self->grad_gamma, 0, (size_t)self->d * sizeof(double));
}

void rmsnorm_set_params(RMSNorm *self, const double *gamma)
{
    if (!gamma) {
        fprintf(stderr, "rmsnorm_set_params: gamma 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->gamma, gamma, (size_t)self->d * sizeof(double));
}

const double *rmsnorm_gamma(const RMSNorm *self)
{
    return self ? self->gamma : NULL;
}

const double *rmsnorm_grad_gamma(const RMSNorm *self)
{
    return self ? self->grad_gamma : NULL;
}

double rmsnorm_inv_rms(const RMSNorm *self)
{
    return self ? self->inv_rms : 0.0;
}

void rmsnorm_forward(RMSNorm *self, const Tensor *x, Tensor *out)
{
    const int d = self->d;
    int want[1] = { d };
    tensor_check_shape(x, 1, want, "rmsnorm_forward(x)");
    tensor_check_shape(out, 1, want, "rmsnorm_forward(out)");

    memcpy(self->x_cache, x->data, (size_t)d * sizeof(double)); /* Python: self.x = x */
    const double mean_sq = dot_neumaier(x->data, x->data, d) / (double)d; /* sum(xi*xi)/d */
    const double r = 1.0 / sqrt(mean_sq + self->eps);
    for (int j = 0; j < d; j++) out->data[j] = self->gamma[j] * x->data[j] * r;

    self->inv_rms = r;
    self->ready = 1;
}

void rmsnorm_backward(RMSNorm *self, const Tensor *dout, Tensor *dx)
{
    const int d = self->d;
    if (!self->ready) {
        fprintf(stderr, "rmsnorm_backward: 请先调用 rmsnorm_forward (反向依赖其缓存的 x/inv_rms)\n");
        exit(1);
    }
    int want[1] = { d };
    tensor_check_shape(dout, 1, want, "rmsnorm_backward(dout)");
    tensor_check_shape(dx, 1, want, "rmsnorm_backward(dx)");

    const double r = self->inv_rms;
    const double *x = self->x_cache;
    /* Python: grad_gamma = [dout[j] * x[j] * r ...]（覆盖赋值，不是 +=） */
    for (int j = 0; j < d; j++) self->grad_gamma[j] = dout->data[j] * x[j] * r;

    /* Python: g = sum(dout[i] * gamma[i] * x[i])，生成器求和 -> Neumaier */
    for (int i = 0; i < d; i++) self->tmp[i] = dout->data[i] * self->gamma[i] * x[i];
    const double g = sum_neumaier(self->tmp, d);

    const double r3 = r * r * r;
    const double dd = (double)d;
    for (int j = 0; j < d; j++)
        dx->data[j] = self->gamma[j] * dout->data[j] * r - g * x[j] * r3 / dd;
}

int rmsnorm_param_count(RMSNorm *self)
{
    return self ? self->d : 0; /* 只有 gamma */
}

/* ---------- RMSNorm 的 vtable ---------- */

static void rmsnorm_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    rmsnorm_forward((RMSNorm *)m->self, x, out);
}

static void rmsnorm_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    rmsnorm_backward((RMSNorm *)m->self, dout, dx);
}

static void rmsnorm_vt_zero_grad(Module *m)
{
    rmsnorm_zero_grad((RMSNorm *)m->self);
}

static void rmsnorm_vt_free(Module *m)
{
    rmsnorm_free((RMSNorm *)m->self);
}

static int rmsnorm_vt_param_count(Module *m)
{
    return rmsnorm_param_count((RMSNorm *)m->self);
}

static void rmsnorm_vt_to_string(Module *m, char *buf, size_t n)
{
    const RMSNorm *rn = (const RMSNorm *)m->self;
    snprintf(buf, n, "RMSNorm(%d, eps=%g)", rn->d, rn->eps);
}

static const ModuleVT RMSNORM_VT = {
    "RMSNorm",
    rmsnorm_vt_forward,
    rmsnorm_vt_backward,
    rmsnorm_vt_zero_grad,
    rmsnorm_vt_free,
    rmsnorm_vt_param_count,
    rmsnorm_vt_to_string
};

Module *rmsnorm_as_module(RMSNorm *self)
{
    return wrap_module(&RMSNORM_VT, self);
}

/* ======================= 归一化家族：BatchNorm2d / GroupNorm / DropPath =======================
 * 移植自 AI.py：
 *   BatchNorm2d (AI.py:13161) -> bn2d_*
 *   GroupNorm   (AI.py:13332) -> gn_*
 *   InstanceNorm(AI.py:13441) -> inorm_create（groups == channels 的特例）
 *   DropPath    (AI.py:12766) -> droppath_*
 *
 * 逐位一致的关键是照抄 Python 的求和方式：
 *   - Python 里 `sum(列表)` 走 CPython 的 Neumaier 补偿求和 -> sum_neumaier；
 *   - Python 里显式的 `s += x` 循环（GroupNorm 的 mean / var / s1 / s2、
 *     grad_gamma、grad_beta）-> C 里也必须是朴素 +=，不能换成补偿求和。
 * 混用会让最后几位变掉，对拍立刻暴露（容差 0）。
 */

/* ---------------------------------------------------------------- 公共小助手 */

static void norm_ext_need3(const Tensor *t, int c, int h, int w, const char *who)
{
    int want[3] = { c, h, w };
    tensor_check_shape(t, 3, want, who);
}

static double *norm_ext_grow(double *buf, int *cap, int need)
{
    if (need <= *cap) return buf;
    double *n = xmalloc((size_t)need * sizeof(double));
    if (buf) {
        memcpy(n, buf, (size_t)(*cap) * sizeof(double));
        free(buf);
    }
    *cap = need;
    return n;
}

/* ================================ BatchNorm2d ================================ */

BatchNorm2d *bn2d_create(int num_features, double eps, double momentum)
{
    if (num_features < 1) {
        fprintf(stderr, "bn2d_create: num_features 必须是正整数，收到 %d\n", num_features);
        exit(1);
    }
    if (!(eps > 0.0)) {
        fprintf(stderr, "bn2d_create: eps 必须 > 0，收到 %g\n", eps);
        exit(1);
    }
    if (!(momentum >= 0.0 && momentum <= 1.0)) {
        fprintf(stderr, "bn2d_create: momentum 必须在 [0,1] 内，收到 %g\n", momentum);
        exit(1);
    }
    BatchNorm2d *self = xcalloc(1, sizeof(BatchNorm2d));
    const int c = num_features;
    self->c = c;
    self->eps = eps;
    self->momentum = momentum;
    self->training = 1;

    self->gamma = xmalloc((size_t)c * sizeof(double));
    self->beta = xmalloc((size_t)c * sizeof(double));
    self->running_mean = xmalloc((size_t)c * sizeof(double));
    self->running_var = xmalloc((size_t)c * sizeof(double));
    self->grad_gamma = xcalloc((size_t)c, sizeof(double));
    self->grad_beta = xcalloc((size_t)c, sizeof(double));
    self->mean = xcalloc((size_t)c, sizeof(double));
    self->inv = xcalloc((size_t)c, sizeof(double));
    for (int i = 0; i < c; i++) {
        self->gamma[i] = 1.0;
        self->beta[i] = 0.0;
        self->running_mean[i] = 0.0;
        self->running_var[i] = 1.0;
    }
    return self;
}

void bn2d_free(BatchNorm2d *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->beta);
    free(self->running_mean);
    free(self->running_var);
    free(self->grad_gamma);
    free(self->grad_beta);
    free(self->mean);
    free(self->inv);
    free(self->x_hat);
    free(self->scratch);
    free(self->scratch2);
    free(self);
}

void bn2d_zero_grad(BatchNorm2d *self)
{
    if (!self) return;
    memset(self->grad_gamma, 0, (size_t)self->c * sizeof(double));
    memset(self->grad_beta, 0, (size_t)self->c * sizeof(double));
}

void bn2d_set_training(BatchNorm2d *self, int training)
{
    if (self) self->training = training ? 1 : 0;
}

void bn2d_set_params(BatchNorm2d *self, const double *gamma, const double *beta)
{
    if (!self) return;
    if (!gamma || !beta) {
        fprintf(stderr, "bn2d_set_params: gamma / beta 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->gamma, gamma, (size_t)self->c * sizeof(double));
    memcpy(self->beta, beta, (size_t)self->c * sizeof(double));
}

void bn2d_set_running(BatchNorm2d *self, const double *mean, const double *var)
{
    if (!self) return;
    if (!mean || !var) {
        fprintf(stderr, "bn2d_set_running: mean / var 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->running_mean, mean, (size_t)self->c * sizeof(double));
    memcpy(self->running_var, var, (size_t)self->c * sizeof(double));
}

const double *bn2d_running_mean(const BatchNorm2d *self) { return self ? self->running_mean : NULL; }
const double *bn2d_running_var(const BatchNorm2d *self)  { return self ? self->running_var : NULL; }
const double *bn2d_grad_gamma(const BatchNorm2d *self)   { return self ? self->grad_gamma : NULL; }
const double *bn2d_grad_beta(const BatchNorm2d *self)    { return self ? self->grad_beta : NULL; }
int bn2d_cache_mode(const BatchNorm2d *self)             { return self ? self->cache_mode : 0; }

void bn2d_forward(BatchNorm2d *self, const Tensor *x, Tensor *out)
{
    if (!self || !x || !out) return;
    if (x->ndim != 3) {
        fprintf(stderr, "bn2d_forward: 输入必须是 (C,H,W)，实际 ndim = %d\n", x->ndim);
        exit(1);
    }
    const int C = x->shape[0], H = x->shape[1], W = x->shape[2];
    if (C != self->c) {
        fprintf(stderr, "bn2d_forward: 通道数 %d 与本层 %d 不一致\n", C, self->c);
        exit(1);
    }
    if (H < 1 || W < 1) {
        fprintf(stderr, "bn2d_forward: 输入不能为空（H=%d, W=%d）\n", H, W);
        exit(1);
    }
    norm_ext_need3(out, C, H, W, "bn2d_forward(out)");

    const int m = H * W;
    self->in_c = C;
    self->in_h = H;
    self->in_w = W;
    self->x_hat = norm_ext_grow(self->x_hat, &self->x_hat_cap, C * m);
    self->scratch = norm_ext_grow(self->scratch, &self->scratch_cap, m);
    self->scratch2 = norm_ext_grow(self->scratch2, &self->scratch2_cap, m);

    self->cache_mode = self->training ? 1 : 0;
    for (int c = 0; c < C; c++) {
        const double *flat = x->data + (size_t)c * (size_t)m;
        double mean, var;
        if (self->cache_mode) {
            mean = sum_neumaier(flat, m) / (double)m;
            for (int i = 0; i < m; i++) {
                const double d = flat[i] - mean;
                self->scratch[i] = d * d;
            }
            var = sum_neumaier(self->scratch, m) / (double)m;
            /* Python：先用有偏 var 归一化，running 里存无偏估计 */
            self->running_mean[c] = (1.0 - self->momentum) * self->running_mean[c]
                                    + self->momentum * mean;
            const double unb = (m > 1) ? var * (double)m / (double)(m - 1) : var;
            self->running_var[c] = (1.0 - self->momentum) * self->running_var[c]
                                   + self->momentum * unb;
        } else {
            mean = self->running_mean[c];
            var = self->running_var[c];
        }
        const double inv = 1.0 / sqrt(var + self->eps);
        self->mean[c] = mean;
        self->inv[c] = inv;

        double *xh = self->x_hat + (size_t)c * (size_t)m;
        for (int i = 0; i < m; i++) xh[i] = (flat[i] - mean) * inv;
        double *o = out->data + (size_t)c * (size_t)m;
        for (int i = 0; i < m; i++) o[i] = self->gamma[c] * xh[i] + self->beta[c];
    }
    self->ready = 1;
}

void bn2d_backward(BatchNorm2d *self, const Tensor *dout, Tensor *dx)
{
    if (!self || !dout || !dx) return;
    if (!self->ready) {
        fprintf(stderr, "bn2d_backward: 必须先调用 bn2d_forward()\n");
        exit(1);
    }
    const int C = self->in_c, H = self->in_h, W = self->in_w;
    norm_ext_need3(dout, C, H, W, "bn2d_backward(dout)");
    norm_ext_need3(dx, C, H, W, "bn2d_backward(dx)");

    const int m = H * W;
    const double dm = (double)m;
    for (int c = 0; c < C; c++) {
        const double *d = dout->data + (size_t)c * (size_t)m;
        const double *xh = self->x_hat + (size_t)c * (size_t)m;

        /* grad_gamma = sum(d * x_hat)，grad_beta = sum(d)：都是 CPython sum() */
        for (int i = 0; i < m; i++) self->scratch[i] = d[i] * xh[i];
        self->grad_gamma[c] += sum_neumaier(self->scratch, m);
        self->grad_beta[c] += sum_neumaier(d, m);

        /* dxh = d * gamma[c] */
        double *dxh = self->scratch;
        const double g = self->gamma[c];
        for (int i = 0; i < m; i++) dxh[i] = d[i] * g;

        double *o = dx->data + (size_t)c * (size_t)m;
        if (!self->cache_mode) {
            /* 推理：mean/var 与样本无关，dx = inv * dxh（没有 m 项） */
            const double inv = self->inv[c];
            for (int i = 0; i < m; i++) o[i] = inv * dxh[i];
        } else {
            const double s1 = sum_neumaier(dxh, m);
            for (int i = 0; i < m; i++) self->scratch2[i] = dxh[i] * xh[i];
            const double s2 = sum_neumaier(self->scratch2, m);
            const double inv = self->inv[c];
            for (int i = 0; i < m; i++) {
                o[i] = inv * (dm * dxh[i] - s1 - xh[i] * s2) / dm;
            }
        }
    }
}

int bn2d_param_count(BatchNorm2d *self)
{
    return self ? 2 * self->c : 0;   /* gamma + beta */
}

static void bn2d_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    bn2d_forward((BatchNorm2d *)m->self, x, out);
}

static void bn2d_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    bn2d_backward((BatchNorm2d *)m->self, dout, dx);
}

static void bn2d_vt_zero_grad(Module *m)
{
    bn2d_zero_grad((BatchNorm2d *)m->self);
}

static void bn2d_vt_free(Module *m)
{
    bn2d_free((BatchNorm2d *)m->self);
}

static int bn2d_vt_param_count(Module *m)
{
    return bn2d_param_count((BatchNorm2d *)m->self);
}

static void bn2d_vt_to_string(Module *m, char *buf, size_t n)
{
    const BatchNorm2d *b = (const BatchNorm2d *)m->self;
    snprintf(buf, n, "BatchNorm2d(%d, eps=%g, momentum=%g)", b->c, b->eps, b->momentum);
}

static const ModuleVT BN2D_VT = {
    "BatchNorm2d",
    bn2d_vt_forward,
    bn2d_vt_backward,
    bn2d_vt_zero_grad,
    bn2d_vt_free,
    bn2d_vt_param_count,
    bn2d_vt_to_string
};

Module *bn2d_as_module(BatchNorm2d *self)
{
    return wrap_module(&BN2D_VT, self);
}

/* ================================= GroupNorm ================================= */

static GroupNorm *gn_create_impl(int num_groups, int num_channels, double eps, int instance)
{
    if (num_groups < 1) {
        fprintf(stderr, "gn_create: num_groups 必须是正整数，收到 %d\n", num_groups);
        exit(1);
    }
    if (num_channels < 1) {
        fprintf(stderr, "gn_create: num_channels 必须是正整数，收到 %d\n", num_channels);
        exit(1);
    }
    if (num_channels % num_groups != 0) {
        fprintf(stderr, "gn_create: num_channels=%d 必须能被 num_groups=%d 整除\n",
                num_channels, num_groups);
        exit(1);
    }
    if (!(eps > 0.0)) {
        fprintf(stderr, "gn_create: eps 必须 > 0，收到 %g\n", eps);
        exit(1);
    }
    GroupNorm *self = xcalloc(1, sizeof(GroupNorm));
    self->groups = num_groups;
    self->channels = num_channels;
    self->eps = eps;
    self->instance = instance;

    const int c = num_channels;
    self->gamma = xmalloc((size_t)c * sizeof(double));
    self->beta = xmalloc((size_t)c * sizeof(double));
    self->grad_gamma = xcalloc((size_t)c, sizeof(double));
    self->grad_beta = xcalloc((size_t)c, sizeof(double));
    self->mean = xcalloc((size_t)num_groups, sizeof(double));
    self->inv = xcalloc((size_t)num_groups, sizeof(double));
    for (int i = 0; i < c; i++) {
        self->gamma[i] = 1.0;
        self->beta[i] = 0.0;
    }
    return self;
}

GroupNorm *gn_create(int num_groups, int num_channels, double eps)
{
    return gn_create_impl(num_groups, num_channels, eps, 0);
}

GroupNorm *inorm_create(int num_channels, double eps)
{
    return gn_create_impl(num_channels, num_channels, eps, 1);
}

void gn_free(GroupNorm *self)
{
    if (!self) return;
    free(self->gamma);
    free(self->beta);
    free(self->grad_gamma);
    free(self->grad_beta);
    free(self->mean);
    free(self->inv);
    free(self->x_hat);
    free(self);
}

void gn_zero_grad(GroupNorm *self)
{
    if (!self) return;
    memset(self->grad_gamma, 0, (size_t)self->channels * sizeof(double));
    memset(self->grad_beta, 0, (size_t)self->channels * sizeof(double));
}

void gn_set_params(GroupNorm *self, const double *gamma, const double *beta)
{
    if (!self) return;
    if (!gamma || !beta) {
        fprintf(stderr, "gn_set_params: gamma / beta 不能为 NULL\n");
        exit(1);
    }
    memcpy(self->gamma, gamma, (size_t)self->channels * sizeof(double));
    memcpy(self->beta, beta, (size_t)self->channels * sizeof(double));
}

const double *gn_grad_gamma(const GroupNorm *self) { return self ? self->grad_gamma : NULL; }
const double *gn_grad_beta(const GroupNorm *self)  { return self ? self->grad_beta : NULL; }

void gn_forward(GroupNorm *self, const Tensor *x, Tensor *out)
{
    if (!self || !x || !out) return;
    if (x->ndim != 3) {
        fprintf(stderr, "gn_forward: 输入必须是 (C,H,W)，实际 ndim = %d\n", x->ndim);
        exit(1);
    }
    const int C = x->shape[0], H = x->shape[1], W = x->shape[2];
    if (C != self->channels) {
        fprintf(stderr, "gn_forward: 通道数 %d 与本层 %d 不一致\n", C, self->channels);
        exit(1);
    }
    if (H < 1 || W < 1) {
        fprintf(stderr, "gn_forward: 输入不能为空（H=%d, W=%d）\n", H, W);
        exit(1);
    }
    norm_ext_need3(out, C, H, W, "gn_forward(out)");

    self->in_c = C;
    self->in_h = H;
    self->in_w = W;
    self->x_hat = norm_ext_grow(self->x_hat, &self->x_hat_cap, C * H * W);

    const int per = self->channels / self->groups;
    const double n = (double)(per * H * W);

    for (int g = 0; g < self->groups; g++) {
        const int c0 = g * per;
        /* mean：逐行 sum(row)（Neumaier）再朴素累加到 mean，最后 /= n */
        double mean = 0.0;
        for (int c = c0; c < c0 + per; c++) {
            for (int h = 0; h < H; h++) {
                mean += sum_neumaier(x->data + ((size_t)c * H + h) * W, W);
            }
        }
        mean /= n;
        /* var：显式朴素 += d*d（Python 就是这么写的） */
        double var = 0.0;
        for (int c = c0; c < c0 + per; c++) {
            for (int h = 0; h < H; h++) {
                const double *row = x->data + ((size_t)c * H + h) * W;
                for (int w = 0; w < W; w++) {
                    const double d = row[w] - mean;
                    var += d * d;
                }
            }
        }
        var /= n;
        const double inv = 1.0 / sqrt(var + self->eps);
        self->mean[g] = mean;
        self->inv[g] = inv;

        for (int c = c0; c < c0 + per; c++) {
            for (int h = 0; h < H; h++) {
                const double *row = x->data + ((size_t)c * H + h) * W;
                double *xh_row = self->x_hat + ((size_t)c * H + h) * W;
                double *o_row = out->data + ((size_t)c * H + h) * W;
                for (int w = 0; w < W; w++) {
                    xh_row[w] = (row[w] - mean) * inv;
                    o_row[w] = self->gamma[c] * xh_row[w] + self->beta[c];
                }
            }
        }
    }
    self->ready = 1;
}

void gn_backward(GroupNorm *self, const Tensor *dout, Tensor *dx)
{
    if (!self || !dout || !dx) return;
    if (!self->ready) {
        fprintf(stderr, "gn_backward: 必须先调用 gn_forward()\n");
        exit(1);
    }
    const int C = self->in_c, H = self->in_h, W = self->in_w;
    norm_ext_need3(dout, C, H, W, "gn_backward(dout)");
    norm_ext_need3(dx, C, H, W, "gn_backward(dx)");

    const int per = self->channels / self->groups;
    const double n = (double)(per * H * W);

    /* grad_beta[c] = Σ_h sum(dout[c][h])（逐行 Neumaier，跨行朴素累加）
     * grad_gamma[c] = Σ_{h,w} dout*x_hat（显式朴素累加） */
    for (int c = 0; c < C; c++) {
        double s = 0.0;
        for (int h = 0; h < H; h++) {
            s += sum_neumaier(dout->data + ((size_t)c * H + h) * W, W);
        }
        self->grad_beta[c] += s;

        double t = 0.0;
        for (int h = 0; h < H; h++) {
            const double *d_row = dout->data + ((size_t)c * H + h) * W;
            const double *xh_row = self->x_hat + ((size_t)c * H + h) * W;
            for (int w = 0; w < W; w++) t += d_row[w] * xh_row[w];
        }
        self->grad_gamma[c] += t;
    }

    for (int g = 0; g < self->groups; g++) {
        const int c0 = g * per;
        const double mean = self->mean[g];       /* 与 Python 一样不用，保留可读性 */
        const double inv = self->inv[g];
        (void)mean;

        double s1 = 0.0;   /* sum(dx_hat) */
        double s2 = 0.0;   /* sum(dx_hat * x_hat) */
        for (int c = c0; c < c0 + per; c++) {
            const double gam = self->gamma[c];
            for (int h = 0; h < H; h++) {
                const double *d_row = dout->data + ((size_t)c * H + h) * W;
                const double *xh_row = self->x_hat + ((size_t)c * H + h) * W;
                for (int w = 0; w < W; w++) {
                    const double d = d_row[w] * gam;
                    s1 += d;
                    s2 += d * xh_row[w];
                }
            }
        }
        for (int c = c0; c < c0 + per; c++) {
            const double gam = self->gamma[c];
            for (int h = 0; h < H; h++) {
                const double *d_row = dout->data + ((size_t)c * H + h) * W;
                const double *xh_row = self->x_hat + ((size_t)c * H + h) * W;
                double *o_row = dx->data + ((size_t)c * H + h) * W;
                for (int w = 0; w < W; w++) {
                    const double d = d_row[w] * gam;
                    o_row[w] = inv * (n * d - s1 - xh_row[w] * s2) / n;
                }
            }
        }
    }
}

int gn_param_count(GroupNorm *self)
{
    return self ? 2 * self->channels : 0;   /* gamma + beta */
}

static void gn_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    gn_forward((GroupNorm *)m->self, x, out);
}

static void gn_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    gn_backward((GroupNorm *)m->self, dout, dx);
}

static void gn_vt_zero_grad(Module *m)
{
    gn_zero_grad((GroupNorm *)m->self);
}

static void gn_vt_free(Module *m)
{
    gn_free((GroupNorm *)m->self);
}

static int gn_vt_param_count(Module *m)
{
    return gn_param_count((GroupNorm *)m->self);
}

static void gn_vt_to_string(Module *m, char *buf, size_t n)
{
    const GroupNorm *g = (const GroupNorm *)m->self;
    if (g->instance) {
        snprintf(buf, n, "InstanceNorm(%d, eps=%g)", g->channels, g->eps);
    } else {
        snprintf(buf, n, "GroupNorm(%d, %d, eps=%g)", g->groups, g->channels, g->eps);
    }
}

static const ModuleVT GN_VT = {
    "GroupNorm",
    gn_vt_forward,
    gn_vt_backward,
    gn_vt_zero_grad,
    gn_vt_free,
    gn_vt_param_count,
    gn_vt_to_string
};

static const ModuleVT INORM_VT = {
    "InstanceNorm",
    gn_vt_forward,
    gn_vt_backward,
    gn_vt_zero_grad,
    gn_vt_free,
    gn_vt_param_count,
    gn_vt_to_string
};

Module *gn_as_module(GroupNorm *self)
{
    return wrap_module(&GN_VT, self);
}

Module *inorm_as_module(GroupNorm *self)
{
    return wrap_module(&INORM_VT, self);
}

/* ================================== DropPath ================================== */

DropPath *droppath_create(double p, int scale_by_keep)
{
    if (!(p >= 0.0 && p < 1.0)) {
        fprintf(stderr, "droppath_create: p 必须在 [0,1) 内，收到 %g\n", p);
        exit(1);
    }
    DropPath *self = xcalloc(1, sizeof(DropPath));
    self->p = p;
    self->scale_by_keep = scale_by_keep ? 1 : 0;
    self->training = 1;
    self->factor = 1.0;
    self->rng = 0x9E3779B97F4A7C15ULL;
    return self;
}

void droppath_free(DropPath *self)
{
    free(self);
}

void droppath_zero_grad(DropPath *self)
{
    (void)self;   /* 无参数：空实现，仅为统一接口 */
}

void droppath_set_training(DropPath *self, int training)
{
    if (self) self->training = training ? 1 : 0;
}

void droppath_set_seed(DropPath *self, unsigned long long seed)
{
    if (!self) return;
    self->rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

void droppath_set_keep(DropPath *self, int keep)
{
    if (!self) return;
    self->has_pending = 1;
    self->pending_keep = keep ? 1 : 0;
}

double droppath_factor(const DropPath *self)
{
    return self ? self->factor : 1.0;
}

static int droppath_draw_keep(DropPath *self)
{
    /* xorshift64*：与 Dropout 一致；跨语言不追求与 Python 的 MT 逐位一致，
     * 等价性验证一律走 droppath_set_keep()。 */
    unsigned long long s = self->rng;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    self->rng = s;
    const unsigned long long r = s * 2685821657736338717ULL;
    const double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return (u >= self->p) ? 1 : 0;
}

void droppath_forward(DropPath *self, const Tensor *x, Tensor *out)
{
    if (!self || !x || !out) return;
    if (x->size != out->size) {
        fprintf(stderr, "droppath_forward: 输入 %d 个元素与输出 %d 个不一致\n", x->size, out->size);
        exit(1);
    }
    if (!self->training || self->p == 0.0) {
        self->factor = 1.0;
        memcpy(out->data, x->data, (size_t)x->size * sizeof(double));
        return;
    }
    int keep;
    if (self->has_pending) {
        keep = self->pending_keep;
        self->has_pending = 0;
    } else {
        keep = droppath_draw_keep(self);
    }
    if (!keep) {
        self->factor = 0.0;
    } else {
        self->factor = self->scale_by_keep ? 1.0 / (1.0 - self->p) : 1.0;
    }
    const double k = self->factor;
    for (int i = 0; i < x->size; i++) out->data[i] = x->data[i] * k;
}

void droppath_backward(DropPath *self, const Tensor *dout, Tensor *dx)
{
    if (!self || !dout || !dx) return;
    if (dout->size != dx->size) {
        fprintf(stderr, "droppath_backward: dout %d 个元素与 dx %d 个不一致\n",
                dout->size, dx->size);
        exit(1);
    }
    const double k = self->factor;
    for (int i = 0; i < dout->size; i++) dx->data[i] = dout->data[i] * k;
}

int droppath_param_count(DropPath *self)
{
    (void)self;
    return 0;
}

static void droppath_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    droppath_forward((DropPath *)m->self, x, out);
}

static void droppath_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    droppath_backward((DropPath *)m->self, dout, dx);
}

static void droppath_vt_zero_grad(Module *m)
{
    droppath_zero_grad((DropPath *)m->self);
}

static void droppath_vt_free(Module *m)
{
    droppath_free((DropPath *)m->self);
}

static int droppath_vt_param_count(Module *m)
{
    return droppath_param_count((DropPath *)m->self);
}

static void droppath_vt_to_string(Module *m, char *buf, size_t n)
{
    const DropPath *d = (const DropPath *)m->self;
    snprintf(buf, n, "DropPath(p=%g%s)", d->p, d->scale_by_keep ? "" : ", scale_by_keep=False");
}

static const ModuleVT DROPPATH_VT = {
    "DropPath",
    droppath_vt_forward,
    droppath_vt_backward,
    droppath_vt_zero_grad,
    droppath_vt_free,
    droppath_vt_param_count,
    droppath_vt_to_string
};

Module *droppath_as_module(DropPath *self)
{
    return wrap_module(&DROPPATH_VT, self);
}
