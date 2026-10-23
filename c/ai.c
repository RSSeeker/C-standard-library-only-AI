/*
 * ai.c —— Python 版神经网络工具箱的 C 实现（核心数值路径）
 *
 * 移植自 AI.py，数值上刻意对齐 Python 的运算顺序与求和方式：
 *   - 点积 / MSE 求和使用 Neumaier 补偿求和，对齐 CPython 3.12+ 的 sum()
 *   - Adam 的偏差修正、梯度累加顺序与 Model.fit 完全一致
 *   - softmax 输出层的反向沿用 softmax+CE 联合梯度约定
 */
#include "ai.h"
#include "sched.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 严格 C17 下 math.h 不保证提供 M_PI（那是 POSIX 扩展），自己定义 */
#define AI_PI 3.14159265358979323846

/* Safety check failed: abort (C has no exceptions; OOB writes corrupt memory) */
static void ai_die(const char *where, const char *msg)
{
    fprintf(stderr, "error [%s]: %s\n", where, msg);
    exit(1);
}

/* ============================ 内存 ============================ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

/* ================= 求和：对齐 CPython 3.12+ 的 sum() =================
 *
 * CPython 3.12 起，sum() 对浮点使用 Neumaier 补偿求和：
 *     t = s + x;
 *     if (fabs(s) >= fabs(x)) c += (s - t) + x; else c += (x - t) + s;
 *     s = t;
 *   最后返回 s + c;
 * 这里照搬同一算法，使得 C 与 Python 的点积/求和结果逐位一致。
 */
static double dot_neumaier(const double *a, const double *b, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a[i] * b[i];
        double t = s + x;
        if (fabs(s) >= fabs(x))
            c += (s - t) + x;
        else
            c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* ==================== 随机数（仅 C 独立 demo 使用） ====================
 * Python 用 Mersenne Twister + 极坐标法生成高斯，这里不追求与之逐位一致；
 * 需要跨语言对齐的场合（等价性测试）改为从 Python 导出权重后加载。
 */
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

static double rng_gauss(unsigned long long *s)
{
    /* Box-Muller */
    double u1 = rng_uniform(s);
    double u2 = rng_uniform(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * AI_PI * u2);
}

/* ============================ 激活函数 ============================ */

double act_apply(Activation act, const double *z, double *out, int n)
{
    switch (act) {
    case ACT_LEAKY_RELU:
        for (int i = 0; i < n; i++) out[i] = (z[i] > 0.0) ? z[i] : 0.01 * z[i];
        return 0.0;
    case ACT_RELU:
        for (int i = 0; i < n; i++) out[i] = (z[i] > 0.0) ? z[i] : 0.0;
        return 0.0;
    case ACT_LINEAR:
        memcpy(out, z, (size_t)n * sizeof(double));
        return 0.0;
    case ACT_SOFTMAX: {
        double mx = z[0];
        for (int i = 1; i < n; i++) if (z[i] > mx) mx = z[i];
        double s = 0.0;
        for (int i = 0; i < n; i++) { out[i] = exp(z[i] - mx); s += out[i]; }
        for (int i = 0; i < n; i++) out[i] /= s;
        return s;
    }
    }
    return 0.0;
}

void act_deriv(Activation act, const double *z, const double *a, double *out, int n)
{
    (void)a;
    switch (act) {
    case ACT_LEAKY_RELU:
        for (int i = 0; i < n; i++) out[i] = (z[i] > 0.0) ? 1.0 : 0.01;
        break;
    case ACT_RELU:
        for (int i = 0; i < n; i++) out[i] = (z[i] > 0.0) ? 1.0 : 0.0;
        break;
    case ACT_LINEAR:
        for (int i = 0; i < n; i++) out[i] = 1.0;
        break;
    case ACT_SOFTMAX:
        /* 与 Python 版一致：恒为 1，仅在与 cross_entropy 搭配时正确
         * （CE 已直接给出 softmax+CE 的联合梯度） */
        for (int i = 0; i < n; i++) out[i] = 1.0;
        break;
    }
}

/* ============================ Layer ============================ */

void he_init(Layer *l, unsigned long long *rng_state)
{
    double std = sqrt(2.0 / (double)l->fan_in);
    for (int j = 0; j < l->fan_out; j++) {
        for (int i = 0; i < l->fan_in; i++)
            l->W[j * l->fan_in + i] = rng_gauss(rng_state) * std;
        l->b[j] = 0.0;
    }
}

static void layer_alloc(Layer *l, int fan_in, int fan_out, Activation act)
{
    l->fan_in = fan_in;
    l->fan_out = fan_out;
    l->act = act;
    l->W  = xcalloc((size_t)fan_out * fan_in, sizeof(double));
    l->b  = xcalloc((size_t)fan_out, sizeof(double));
    l->gW = xcalloc((size_t)fan_out * fan_in, sizeof(double));
    l->gb = xcalloc((size_t)fan_out, sizeof(double));
    l->mW = xcalloc((size_t)fan_out * fan_in, sizeof(double));
    l->vW = xcalloc((size_t)fan_out * fan_in, sizeof(double));
    l->mb = xcalloc((size_t)fan_out, sizeof(double));
    l->vb = xcalloc((size_t)fan_out, sizeof(double));
    l->x  = xcalloc((size_t)fan_in, sizeof(double));
    l->z  = xcalloc((size_t)fan_out, sizeof(double));
    l->a  = xcalloc((size_t)fan_out, sizeof(double));
    l->d_act = xcalloc((size_t)fan_out, sizeof(double));
    l->tmpw  = xcalloc((size_t)fan_out, sizeof(double));
}

static void layer_free(Layer *l)
{
    free(l->W); free(l->b); free(l->gW); free(l->gb);
    free(l->mW); free(l->vW); free(l->mb); free(l->vb);
    free(l->x); free(l->z); free(l->a);
    free(l->d_act); free(l->tmpw);
    memset(l, 0, sizeof(*l));
}

static void layer_forward(Layer *l, const double *x)
{
    memcpy(l->x, x, (size_t)l->fan_in * sizeof(double));
    for (int j = 0; j < l->fan_out; j++)
        l->z[j] = dot_neumaier(l->W + (size_t)j * l->fan_in, x, l->fan_in) + l->b[j];
    act_apply(l->act, l->z, l->a, l->fan_out);
}

/* 反向：把 delta 往回传，梯度累加到 gW/gb（调用方负责清零/按 batch 求平均）
 * 临时缓冲用 Layer 上预分配的 d_act / tmpw，热路径不做任何分配 */
static void layer_backward(Layer *l, const double *delta, double *next_delta)
{
    double *d_act = l->d_act;
    double *w = l->tmpw;
    act_deriv(l->act, l->z, l->a, d_act, l->fan_out);

    /* delta * d_act 同时用于 grad_b 与回传，只算一次 */
    for (int j = 0; j < l->fan_out; j++) w[j] = delta[j] * d_act[j];

    for (int j = 0; j < l->fan_out; j++) {
        double s = w[j];
        double *grow = l->gW + (size_t)j * l->fan_in;
        for (int i = 0; i < l->fan_in; i++) grow[i] += s * l->x[i];
        l->gb[j] += s;
    }

    if (next_delta) {
        for (int i = 0; i < l->fan_in; i++) {
            double acc = 0.0, c = 0.0;      /* 逐列点积，同 Python 的 zip(*W) + sum(map) */
            for (int j = 0; j < l->fan_out; j++) {
                double x = l->W[(size_t)j * l->fan_in + i] * w[j];
                double t = acc + x;
                if (fabs(acc) >= fabs(x)) c += (acc - t) + x; else c += (x - t) + acc;
                acc = t;
            }
            next_delta[i] = acc + c;
        }
    }
}

/* ============================ MLP ============================ */

MLP *mlp_create(const int *sizes, const Activation *acts, int n_sizes)
{
    if (n_sizes < 2) return NULL;
    MLP *m = xmalloc(sizeof(MLP));
    m->n_layers = n_sizes - 1;
    m->layers = xcalloc((size_t)m->n_layers, sizeof(Layer));
    for (int i = 0; i < m->n_layers; i++)
        layer_alloc(&m->layers[i], sizes[i], sizes[i + 1], acts[i]);

    int max_dim = sizes[0];
    for (int i = 1; i <= m->n_layers; i++) if (sizes[i] > max_dim) max_dim = sizes[i];
    m->scratch_dim = max_dim;
    m->scratch_a = xcalloc((size_t)max_dim, sizeof(double));
    m->scratch_b = xcalloc((size_t)max_dim, sizeof(double));
    return m;
}

void mlp_free(MLP *m)
{
    if (!m) return;
    for (int i = 0; i < m->n_layers; i++) layer_free(&m->layers[i]);
    free(m->layers);
    free(m->scratch_a);
    free(m->scratch_b);
    free(m);
}

void mlp_zero_grad(MLP *m)
{
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        memset(l->gW, 0, (size_t)l->fan_out * l->fan_in * sizeof(double));
        memset(l->gb, 0, (size_t)l->fan_out * sizeof(double));
    }
}

void mlp_forward(MLP *m, const double *x, double *out)
{
    const double *cur = x;
    for (int i = 0; i < m->n_layers; i++) {
        layer_forward(&m->layers[i], cur);
        cur = m->layers[i].a;
    }
    memcpy(out, cur, (size_t)m->layers[m->n_layers - 1].fan_out * sizeof(double));
}

void mlp_backward(MLP *m, const double *dout)
{
    int last = m->n_layers - 1;
    int n = m->layers[last].fan_out;

    /* delta 在预分配的两个缓冲间乒乓，训练热路径零分配 */
    double *cur = m->scratch_a;
    double *nxt = m->scratch_b;
    memcpy(cur, dout, (size_t)n * sizeof(double));

    for (int i = last; i >= 0; i--) {
        Layer *l = &m->layers[i];
        double *next = (i > 0) ? nxt : NULL;
        layer_backward(l, cur, next);
        double *tmp = cur; cur = nxt; nxt = tmp;
    }
}

/* ============================ 损失 ============================ */

double mse_loss(const double *pred, const double *target, int n)
{
    /* 与 Python 一致：sum((p-t)**2 ...) / n，用补偿求和（内联，避免临时数组） */
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double e = pred[i] - target[i];
        double x = e * e;
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x; else c += (x - t) + s;
        s = t;
    }
    return (s + c) / (double)n;
}

void mse_grad(const double *pred, const double *target, int n, double *grad)
{
    for (int i = 0; i < n; i++)
        grad[i] = 2.0 * (pred[i] - target[i]) / (double)n;
}

double ce_loss_grad(const double *pred, int n, int target_idx, double *grad)
{
    /* target_idx 由调用方传入：负数会指向数组外，越界读写直接破坏内存，
       必须在这里拦住（Python 版对应 cross_entropy_loss 的索引校验） */
    if (target_idx < 0 || target_idx >= n)
        ai_die("ce_loss_grad", "target_idx 越界：负索引/超界会破坏内存");
    const double eps = 1e-15;
    double p = pred[target_idx] > eps ? pred[target_idx] : eps;
    double loss = -log(p);
    for (int i = 0; i < n; i++) grad[i] = pred[i];
    grad[target_idx] -= 1.0;
    return loss;
}

/* ============================ 优化器 ============================ */

void optimizer_init(Optimizer *o, OptKind kind, double lr)
{
    memset(o, 0, sizeof(*o));
    o->kind = kind;
    o->lr = lr;
    o->beta1 = 0.9;
    o->beta2 = 0.999;
    o->eps = 1e-8;
    o->momentum = 0.9;
    o->alpha = 0.99;
    o->weight_decay = 0.0;
    o->step_count = 0;
    if (kind == OPT_RMSPROP && o->eps == 1e-8) o->eps = 1e-8;
}

/* 与 Python 各优化器 step() 的运算顺序逐条对齐 */
void optimizer_step(MLP *m, Optimizer *o)
{
    o->step_count++;
    double lr = o->lr;

    /* Adam 的偏差修正只依赖 step_count：每步只算一次（原来每层都算一遍 pow） */
    double c1 = 0.0, c2 = 0.0, o1 = 0.0, o2 = 0.0;
    if (o->kind == OPT_ADAM) {
        c1 = 1.0 - pow(o->beta1, o->step_count);
        c2 = 1.0 - pow(o->beta2, o->step_count);
        o1 = 1.0 - o->beta1;
        o2 = 1.0 - o->beta2;
    }

    for (int li = 0; li < m->n_layers; li++) {
        Layer *l = &m->layers[li];
        int nw = l->fan_out * l->fan_in;

        switch (o->kind) {
        case OPT_ADAM: {
            double b1 = o->beta1, b2 = o->beta2, eps = o->eps;
            for (int k = 0; k < nw; k++) {
                double g = l->gW[k];
                double mm = b1 * l->mW[k] + o1 * g;
                double vv = b2 * l->vW[k] + o2 * g * g;
                l->mW[k] = mm;
                l->vW[k] = vv;
                l->W[k] -= lr * (mm / c1) / (sqrt(vv / c2) + eps);
            }
            for (int j = 0; j < l->fan_out; j++) {
                double g = l->gb[j];
                double mm = b1 * l->mb[j] + o1 * g;
                double vv = b2 * l->vb[j] + o2 * g * g;
                l->mb[j] = mm;
                l->vb[j] = vv;
                l->b[j] -= lr * (mm / c1) / (sqrt(vv / c2) + eps);
            }
            break;
        }
        case OPT_SGD: {
            double wd = o->weight_decay;
            for (int k = 0; k < nw; k++) {
                double g = l->gW[k];
                if (wd > 0.0) g += wd * l->W[k];
                l->W[k] -= lr * g;
            }
            for (int j = 0; j < l->fan_out; j++) {
                double g = l->gb[j];
                if (wd > 0.0) g += wd * l->b[j];
                l->b[j] -= lr * g;
            }
            break;
        }
        case OPT_SGD_MOMENTUM: {
            double mu = o->momentum;
            for (int k = 0; k < nw; k++) {
                double v = mu * l->mW[k] + lr * l->gW[k];
                l->mW[k] = v;
                l->W[k] -= v;
            }
            for (int j = 0; j < l->fan_out; j++) {
                double v = mu * l->mb[j] + lr * l->gb[j];
                l->mb[j] = v;
                l->b[j] -= v;
            }
            break;
        }
        case OPT_RMSPROP: {
            double al = o->alpha, eps = o->eps, om = 1.0 - al;
            for (int k = 0; k < nw; k++) {
                double g = l->gW[k];
                double s = al * l->vW[k] + om * g * g;
                l->vW[k] = s;
                l->W[k] -= lr * g / (sqrt(s) + eps);
            }
            for (int j = 0; j < l->fan_out; j++) {
                double g = l->gb[j];
                double s = al * l->vb[j] + om * g * g;
                l->vb[j] = s;
                l->b[j] -= lr * g / (sqrt(s) + eps);
            }
            break;
        }
        }
    }
}

/* ============================ 训练器 ============================ */

void trainer_init(Trainer *t, MLP *net, LossKind loss, OptKind kind, double lr)
{
    memset(t, 0, sizeof(*t));
    t->net = net;
    t->loss = loss;
    optimizer_init(&t->opt, kind, lr);
}

void trainer_release(Trainer *t)
{
    free(t->pred_buf);
    free(t->grad_buf);
    free(t->idx_buf);
    t->pred_buf = NULL;
    t->grad_buf = NULL;
    t->idx_buf = NULL;
    t->buf_dim = 0;
    t->idx_dim = 0;
}

/* 确保索引缓冲足够大；只在样本数变大时分配一次 */
static int *trainer_idx_buf(Trainer *t, int n)
{
    if (t->idx_dim < n) {
        free(t->idx_buf);
        t->idx_buf = xmalloc((size_t)n * sizeof(int));
        t->idx_dim = n;
    }
    return t->idx_buf;
}

/* 确保内部缓冲足够大；只在维度变化时分配一次 */
static void trainer_ensure_buf(Trainer *t, int out_dim)
{
    if (t->buf_dim >= out_dim) return;
    free(t->pred_buf);
    free(t->grad_buf);
    t->pred_buf = xmalloc((size_t)out_dim * sizeof(double));
    t->grad_buf = xmalloc((size_t)out_dim * sizeof(double));
    t->buf_dim = out_dim;
}

double trainer_accumulate(Trainer *t, const double *x, const double *y, int y_idx)
{
    int out_dim = t->net->layers[t->net->n_layers - 1].fan_out;
    trainer_ensure_buf(t, out_dim);
    double *pred = t->pred_buf;
    double *grad = t->grad_buf;

    mlp_forward(t->net, x, pred);
    double loss;
    if (t->loss == LOSS_MSE) {
        loss = mse_loss(pred, y, out_dim);
        mse_grad(pred, y, out_dim, grad);
    } else {
        loss = ce_loss_grad(pred, out_dim, y_idx, grad);
    }
    mlp_backward(t->net, grad);
    return loss;
}

/* 梯度裁剪（按全局范数），与 Python 的 clip_grad_by_norm 等价 */
static void clip_by_norm(MLP *m, double max_norm)
{
    /* 负阈值会让 scale 为负，等于把所有梯度翻转符号 */
    if (max_norm <= 0.0)
        ai_die("clip_by_norm", "max_norm 必须 > 0");
    double total = 0.0;
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++) total += l->gW[k] * l->gW[k];
        for (int j = 0; j < l->fan_out; j++) total += l->gb[j] * l->gb[j];
    }
    total = sqrt(total);
    if (total > max_norm && total > 0.0) {
        double scale = max_norm / total;   /* 与 Python 的 clip_grad_by_norm 完全一致（原先分母多加了 1e-6，两边结果不同） */
        for (int i = 0; i < m->n_layers; i++) {
            Layer *l = &m->layers[i];
            for (int k = 0; k < l->fan_out * l->fan_in; k++) l->gW[k] *= scale;
            for (int j = 0; j < l->fan_out; j++) l->gb[j] *= scale;
        }
    }
}

static void grads_div(MLP *m, int batch_size)
{
    double inv = 1.0 / (double)batch_size;
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++) l->gW[k] *= inv;
        for (int j = 0; j < l->fan_out; j++) l->gb[j] *= inv;
    }
}

int trainer_fit(Trainer *t, const Dataset *train, const Dataset *val,
                const FitConfig *cfg, double *history_train, double *history_val)
{
    int n = train->n;
    int in_dim = train->in_dim, out_dim = train->out_dim;
    int *idx = trainer_idx_buf(t, n);            /* 复用 trainer 的缓冲，热路径零分配 */
    for (int i = 0; i < n; i++) idx[i] = i;

    /* 验证集切分（与 Model.fit 一致：先打乱，再取前 n_val 个） */
    int n_train = n;
    int *val_idx = NULL;
    int n_val = 0;
    unsigned long long rng = (unsigned long long)cfg->seed * 6364136223846793005ULL + 1442695040888963407ULL;

    if (!val && cfg->val_split > 0.0) {
        n_val = (int)(n * cfg->val_split);
        if (n_val < 1) n_val = 1;
        /* 至少留 1 个训练样本：否则 n_train=0 -> batch_size=0 -> grads_div 除零 */
        if (n_val >= n) n_val = (n > 1) ? n - 1 : 0;
        for (int i = n - 1; i > 0; i--) {          /* Fisher-Yates */
            int j = (int)(rng_uniform(&rng) * (double)(i + 1));
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        val_idx = xmalloc((size_t)n_val * sizeof(int));
        memcpy(val_idx, idx, (size_t)n_val * sizeof(int));
        memmove(idx, idx + n_val, (size_t)(n - n_val) * sizeof(int));
        n_train = n - n_val;
    }

    int batch_size = cfg->batch_size > 0 ? cfg->batch_size : n_train;
    double best_val = 1e308;
    int no_improve = 0;
    double best_train = 1e308;
    int no_improve_train = 0;
    int epoch = 0;

    for (epoch = 1; epoch <= cfg->epochs; epoch++) {
        if (cfg->shuffle) {
            for (int i = n_train - 1; i > 0; i--) {
                int j = (int)(rng_uniform(&rng) * (double)(i + 1));
                int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }
        }
        double total_loss = 0.0;
        for (int b = 0; b < n_train; b += batch_size) {
            int bsize = (b + batch_size <= n_train) ? batch_size : n_train - b;
            mlp_zero_grad(t->net);
            for (int s = 0; s < bsize; s++) {
                int i = idx[b + s];
                const double *x = train->x + (size_t)i * in_dim;
                const double *y = train->y ? train->y + (size_t)i * out_dim : NULL;
                int y_idx = train->y_idx ? train->y_idx[i] : 0;
                total_loss += trainer_accumulate(t, x, y, y_idx);
            }
            grads_div(t->net, bsize);
            if (cfg->grad_clip > 0.0) clip_by_norm(t->net, cfg->grad_clip);
            optimizer_step(t->net, &t->opt);
        }
        total_loss /= (double)n_train;
        if (history_train) history_train[epoch - 1] = total_loss;

        /* 验证 */
        if (n_val > 0 || val) {
            double vloss = 0.0;
            int vn = val ? val->n : n_val;
            for (int s = 0; s < vn; s++) {
                int i = val ? s : val_idx[s];
                const double *x = val ? val->x + (size_t)i * val->in_dim
                                      : train->x + (size_t)i * in_dim;
                const double *y = val ? (val->y ? val->y + (size_t)i * val->out_dim : NULL)
                                      : (train->y ? train->y + (size_t)i * out_dim : NULL);
                int y_idx = val ? (val->y_idx ? val->y_idx[i] : 0)
                                : (train->y_idx ? train->y_idx[i] : 0);
                int od = val ? val->out_dim : out_dim;
                trainer_ensure_buf(t, od);
                double *pred = t->pred_buf;
                mlp_forward(t->net, x, pred);
                if (t->loss == LOSS_MSE) vloss += mse_loss(pred, y, od);
                else vloss += ce_loss_grad(pred, od, y_idx, pred);
            }
            vloss /= (double)vn;
            if (history_val) history_val[epoch - 1] = vloss;
            if (cfg->patience > 0) {
                if (vloss < best_val - 1e-12) { best_val = vloss; no_improve = 0; }
                else if (++no_improve >= cfg->patience) break;
            }
        }
        /* 用训练损失做早停判据（early_stop_on_train=1） */
        if (cfg->early_stop_on_train && cfg->patience > 0) {
            if (total_loss < best_train - 1e-12) { best_train = total_loss; no_improve_train = 0; }
            else if (++no_improve_train >= cfg->patience) break;
        }

        /* 学习率调度：位置与 Python Model.fit 一致（在验证/早停判断之后；
         * 早停 break 时不会走到这里） */
        if (cfg->lr_scheduler) lrsched_step(cfg->lr_scheduler, epoch);

        /* 每轮回调：外部可用它请求停止（GUI 的 Stop 按钮）或同步学习率。
         * 返回非 0 就提前结束训练并返回已完成的轮数。 */
        if (cfg->on_epoch) {
            double v = (n_val > 0 || val) ? history_val[epoch - 1] : -1.0;
            if (cfg->on_epoch(cfg->on_epoch_ctx, epoch, total_loss, v)) break;
        }

        if (cfg->verbose && epoch % 50 == 0)
            printf("  epoch %d | loss %.6f\n", epoch, total_loss);
    }

    free(val_idx);
    return epoch > cfg->epochs ? cfg->epochs : epoch;
}

void trainer_predict(Trainer *t, const Dataset *d, double *out)
{
    for (int i = 0; i < d->n; i++)
        mlp_forward(t->net, d->x + (size_t)i * d->in_dim, out + (size_t)i * d->out_dim);
}

double trainer_accuracy(Trainer *t, const Dataset *d)
{
    int out_dim = t->net->layers[t->net->n_layers - 1].fan_out;
    double *pred = xmalloc((size_t)out_dim * sizeof(double));
    int correct = 0;
    for (int i = 0; i < d->n; i++) {
        mlp_forward(t->net, d->x + (size_t)i * d->in_dim, pred);
        int best = 0;
        for (int j = 1; j < out_dim; j++) if (pred[j] > pred[best]) best = j;
        if (best == d->y_idx[i]) correct++;
    }
    free(pred);
    return (double)correct / (double)d->n;
}

double trainer_r2(Trainer *t, const Dataset *d)
{
    int out_dim = t->net->layers[t->net->n_layers - 1].fan_out;
    double *pred = xmalloc((size_t)d->n * out_dim * sizeof(double));
    trainer_predict(t, d, pred);

    double total = 0.0;
    for (int dim = 0; dim < out_dim; dim++) {
        double ss_res = 0.0, mean = 0.0;
        for (int i = 0; i < d->n; i++) mean += d->y[(size_t)i * out_dim + dim];
        mean /= (double)d->n;
        double ss_tot = 0.0;
        for (int i = 0; i < d->n; i++) {
            double diff = d->y[(size_t)i * out_dim + dim] - pred[(size_t)i * out_dim + dim];
            ss_res += diff * diff;
            double dm = d->y[(size_t)i * out_dim + dim] - mean;
            ss_tot += dm * dm;
        }
        /* 目标恒定（ss_tot==0）时 R² 无定义：完全预测对记 1，否则记 0。
           原先钳到 1e-15 会得到约 -1e15 的荒谬值（与 Python 侧同一处修复） */
        if (ss_tot == 0.0)
            total += (ss_res == 0.0) ? 1.0 : 0.0;
        else
            total += 1.0 - ss_res / ss_tot;
    }
    free(pred);
    return total / (double)out_dim;
}

/* ============================ 持久化（文本格式） ============================
 * 格式（全部以空白分隔）：
 *   AI_MODEL 1
 *   n_layers <L>
 *   sizes <s0> <s1> ... <sL>
 *   acts <a0> ... <a[L-1]>          激活用整数 0..3
 *   W <L 层权重依次、每层行主序 fan_out*fan_in 个>
 *   b <L 层偏置依次、每层 fan_out 个>
 * 说明：Python 版的 JSON 格式需要额外的 JSON 解析器，此处先用这个简单格式；
 * 与 Python 侧互转由 tools/c_reference.py 负责。
 */
int mlp_save(const MLP *m, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "AI_MODEL 1\n");
    fprintf(f, "n_layers %d\n", m->n_layers);
    fprintf(f, "sizes");
    fprintf(f, " %d", m->layers[0].fan_in);
    for (int i = 0; i < m->n_layers; i++) fprintf(f, " %d", m->layers[i].fan_out);
    fprintf(f, "\nacts");
    for (int i = 0; i < m->n_layers; i++) fprintf(f, " %d", (int)m->layers[i].act);
    fprintf(f, "\nW");
    for (int i = 0; i < m->n_layers; i++) {
        const Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++) fprintf(f, " %.17g", l->W[k]);
    }
    fprintf(f, "\nb");
    for (int i = 0; i < m->n_layers; i++) {
        const Layer *l = &m->layers[i];
        for (int j = 0; j < l->fan_out; j++) fprintf(f, " %.17g", l->b[j]);
    }
    fprintf(f, "\n");
    fclose(f);
    return 0;
}

static int read_tag(FILE *f, const char *tag)
{
    char buf[64];
    if (fscanf(f, "%63s", buf) != 1) return -1;
    return strcmp(buf, tag) == 0 ? 0 : -1;
}

MLP *mlp_load(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    int version = 0;
    if (read_tag(f, "AI_MODEL") != 0 || fscanf(f, "%d", &version) != 1) { fclose(f); return NULL; }
    if (read_tag(f, "n_layers") != 0) { fclose(f); return NULL; }
    int L = 0;
    if (fscanf(f, "%d", &L) != 1 || L < 1) { fclose(f); return NULL; }
    int *sizes = xmalloc((size_t)(L + 1) * sizeof(int));
    Activation *acts = xmalloc((size_t)L * sizeof(Activation));
    if (read_tag(f, "sizes") != 0) { fclose(f); free(sizes); free(acts); return NULL; }
    for (int i = 0; i <= L; i++) if (fscanf(f, "%d", &sizes[i]) != 1) { fclose(f); free(sizes); free(acts); return NULL; }
    if (read_tag(f, "acts") != 0) { fclose(f); free(sizes); free(acts); return NULL; }
    for (int i = 0; i < L; i++) { int a; if (fscanf(f, "%d", &a) != 1) { fclose(f); free(sizes); free(acts); return NULL; } acts[i] = (Activation)a; }

    MLP *m = mlp_create(sizes, acts, L + 1);
    if (read_tag(f, "W") != 0) { fclose(f); free(sizes); free(acts); mlp_free(m); return NULL; }
    for (int i = 0; i < L; i++) {
        Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++)
            if (fscanf(f, "%lf", &l->W[k]) != 1) { fclose(f); free(sizes); free(acts); mlp_free(m); return NULL; }
    }
    if (read_tag(f, "b") != 0) { fclose(f); free(sizes); free(acts); mlp_free(m); return NULL; }
    for (int i = 0; i < L; i++) {
        Layer *l = &m->layers[i];
        for (int j = 0; j < l->fan_out; j++)
            if (fscanf(f, "%lf", &l->b[j]) != 1) { fclose(f); free(sizes); free(acts); mlp_free(m); return NULL; }
    }
    fclose(f);
    free(sizes);
    free(acts);
    return m;
}
