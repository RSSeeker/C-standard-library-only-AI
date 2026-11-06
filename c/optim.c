/*
 * optim.c —— AdamW / Lion / Adafactor 的 C17 实现
 *
 * 逐行对照 AI.py（行号见 optim.h），三处最容易写错的地方：
 *   1. AdamW 的 `B[j] *= decay` 与 `B[j] -= lr*(m/c1)/(sqrt(v/c2)+eps)` 是两条语句，
 *      必须分开写；写成一条复合赋值会少一次舍入。
 *   2. 偏差修正里的 (m/c1) 与 (v/c2) 都是"先除后 sqrt/除"，不能改写成 m*sqrt(c2)/...
 *   3. 每层先更新偏置、再更新权重，与 Python 的循环嵌套顺序一致（互不依赖，但保持一致）。
 */
#include "optim.h"

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
        fprintf(stderr, "optim: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "optim: out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

/* 形状/指针校验：C 侧多出来的一层保护（Python 在 AttributeError/形状错时会崩，
 * 这里返回错误串，让调用方拿到 const char **err 而不是段错误） */
static const char *check_layers(const OptLayer *layers, int n_layers, const char *who)
{
    static char msg[160];
    if (!layers || n_layers < 1) {
        snprintf(msg, sizeof msg, "%s: layers 不能为空且 n_layers 必须 >= 1", who);
        return msg;
    }
    for (int k = 0; k < n_layers; k++) {
        const OptLayer *l = &layers[k];
        if (l->fan_out < 1 || l->fan_in < 1) {
            snprintf(msg, sizeof msg, "%s: 第 %d 层 fan_out/fan_in 必须 >= 1", who, k);
            return msg;
        }
        if (!l->w || !l->b || !l->gw || !l->gb) {
            snprintf(msg, sizeof msg, "%s: 第 %d 层的参数/梯度指针不能为 NULL", who, k);
            return msg;
        }
    }
    return NULL;
}

/* ============================== AdamW ============================== */

struct AdamW {
    int n_layers;
    OptLayer *layers;

    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    int step_count;

    /* 每层一套状态：m_w/v_w 长度 fan_out*fan_in，m_b/v_b 长度 fan_out */
    double **m_w;
    double **v_w;
    double **m_b;
    double **v_b;
};

AdamW *adamw_create(const OptLayer *layers, int n_layers,
                    double lr, double beta1, double beta2, double eps,
                    double weight_decay, const char **err)
{
    if (err) *err = NULL;

    const char *bad = check_layers(layers, n_layers, "AdamW");
    /* 与 Python 一致：只校验 weight_decay，beta 的范围 Python 也没查 */
    if (!bad && weight_decay < 0.0) bad = "AdamW: weight_decay 必须 >= 0";
    if (bad) {
        if (err) *err = bad;
        return NULL;
    }

    AdamW *self = xcalloc(1, sizeof(AdamW));
    self->n_layers = n_layers;
    self->layers = xmalloc((size_t)n_layers * sizeof(OptLayer));
    memcpy(self->layers, layers, (size_t)n_layers * sizeof(OptLayer));

    self->lr = lr;
    self->beta1 = beta1;
    self->beta2 = beta2;
    self->eps = eps;
    self->weight_decay = weight_decay;
    self->step_count = 0;

    self->m_w = xcalloc((size_t)n_layers, sizeof(double *));
    self->v_w = xcalloc((size_t)n_layers, sizeof(double *));
    self->m_b = xcalloc((size_t)n_layers, sizeof(double *));
    self->v_b = xcalloc((size_t)n_layers, sizeof(double *));
    for (int k = 0; k < n_layers; k++) {
        const size_t nw = (size_t)layers[k].fan_out * (size_t)layers[k].fan_in;
        const size_t nb = (size_t)layers[k].fan_out;
        self->m_w[k] = xcalloc(nw, sizeof(double));
        self->v_w[k] = xcalloc(nw, sizeof(double));
        self->m_b[k] = xcalloc(nb, sizeof(double));
        self->v_b[k] = xcalloc(nb, sizeof(double));
    }
    return self;
}

void adamw_free(AdamW *self)
{
    if (!self) return;
    for (int k = 0; k < self->n_layers; k++) {
        free(self->m_w[k]);
        free(self->v_w[k]);
        free(self->m_b[k]);
        free(self->v_b[k]);
    }
    free(self->m_w);
    free(self->v_w);
    free(self->m_b);
    free(self->v_b);
    free(self->layers);
    free(self);
}

int adamw_step(AdamW *self, const char **err)
{
    if (err) *err = NULL;
    if (!self) {
        if (err) *err = "AdamW: step 收到 NULL 优化器";
        return -1;
    }

    self->step_count += 1;
    const double b1 = self->beta1, b2 = self->beta2;
    const double lr = self->lr, eps = self->eps;
    /* Python: decay = 1.0 - lr * weight_decay */
    const double decay = 1.0 - lr * self->weight_decay;
    /* Python: c1 = 1 - b1 ** step_count（** 落地为 libm pow） */
    const double c1 = 1.0 - pow(b1, (double)self->step_count);
    const double c2 = 1.0 - pow(b2, (double)self->step_count);
    const double o1 = 1.0 - b1;
    const double o2 = 1.0 - b2;

    for (int k = 0; k < self->n_layers; k++) {
        OptLayer *l = &self->layers[k];
        double *m_w = self->m_w[k], *v_w = self->v_w[k];
        double *m_b = self->m_b[k], *v_b = self->v_b[k];
        const int fo = l->fan_out, fi = l->fan_in;

        for (int j = 0; j < fo; j++) {
            /* ---- 偏置：Python 里先更新 B[j]，再进 fan_in 内循环 ---- */
            double g = l->gb[j];
            double m = b1 * m_b[j] + o1 * g;
            double v = b2 * v_b[j] + o2 * g * g;
            m_b[j] = m;
            v_b[j] = v;
            l->b[j] *= decay;
            l->b[j] -= lr * (m / c1) / (sqrt(v / c2) + eps);

            /* ---- 权重第 j 行 ---- */
            const double *gwj = l->gw + (size_t)j * (size_t)fi;
            double *Wj = l->w + (size_t)j * (size_t)fi;
            double *mwj = m_w + (size_t)j * (size_t)fi;
            double *vwj = v_w + (size_t)j * (size_t)fi;
            for (int i = 0; i < fi; i++) {
                g = gwj[i];
                m = b1 * mwj[i] + o1 * g;
                v = b2 * vwj[i] + o2 * g * g;
                mwj[i] = m;
                vwj[i] = v;
                Wj[i] *= decay;
                Wj[i] -= lr * (m / c1) / (sqrt(v / c2) + eps);
            }
        }
    }
    return 0;
}

int adamw_step_count(const AdamW *self)
{
    return self ? self->step_count : 0;
}

int adamw_num_layers(const AdamW *self)
{
    return self ? self->n_layers : 0;
}

double adamw_lr(const AdamW *self)
{
    return self ? self->lr : 0.0;
}

void adamw_set_lr(AdamW *self, double lr)
{
    if (self) self->lr = lr;
}

double adamw_weight_decay(const AdamW *self)
{
    return self ? self->weight_decay : 0.0;
}

const double *adamw_m_w(const AdamW *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->m_w[k];
}

const double *adamw_v_w(const AdamW *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->v_w[k];
}

const double *adamw_m_b(const AdamW *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->m_b[k];
}

const double *adamw_v_b(const AdamW *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->v_b[k];
}

/* =============================== Lion ============================== */

struct Lion {
    int n_layers;
    OptLayer *layers;

    double lr;
    double beta1;
    double beta2;
    double weight_decay;
    int step_count;

    /* 只有一阶矩：m_w 长度 fan_out*fan_in，m_b 长度 fan_out */
    double **m_w;
    double **m_b;
};

Lion *lion_create(const OptLayer *layers, int n_layers,
                  double lr, double beta1, double beta2,
                  double weight_decay, const char **err)
{
    if (err) *err = NULL;

    const char *bad = check_layers(layers, n_layers, "Lion");
    /* 校验顺序与文案照抄 Python（beta1 -> beta2 -> weight_decay） */
    if (!bad && (beta1 < 0.0 || beta1 > 1.0)) bad = "Lion: beta1 必须在 [0,1]";
    if (!bad && (beta2 < 0.0 || beta2 > 1.0)) bad = "Lion: beta2 必须在 [0,1]";
    if (!bad && weight_decay < 0.0) bad = "Lion: weight_decay 必须 >= 0";
    if (bad) {
        if (err) *err = bad;
        return NULL;
    }

    Lion *self = xcalloc(1, sizeof(Lion));
    self->n_layers = n_layers;
    self->layers = xmalloc((size_t)n_layers * sizeof(OptLayer));
    memcpy(self->layers, layers, (size_t)n_layers * sizeof(OptLayer));

    self->lr = lr;
    self->beta1 = beta1;
    self->beta2 = beta2;
    self->weight_decay = weight_decay;
    self->step_count = 0;

    self->m_w = xcalloc((size_t)n_layers, sizeof(double *));
    self->m_b = xcalloc((size_t)n_layers, sizeof(double *));
    for (int k = 0; k < n_layers; k++) {
        const size_t nw = (size_t)layers[k].fan_out * (size_t)layers[k].fan_in;
        const size_t nb = (size_t)layers[k].fan_out;
        self->m_w[k] = xcalloc(nw, sizeof(double));
        self->m_b[k] = xcalloc(nb, sizeof(double));
    }
    return self;
}

void lion_free(Lion *self)
{
    if (!self) return;
    for (int k = 0; k < self->n_layers; k++) {
        free(self->m_w[k]);
        free(self->m_b[k]);
    }
    free(self->m_w);
    free(self->m_b);
    free(self->layers);
    free(self);
}

/* Python 的 sign 是三分支表达式：c>0 -> 1.0，c<0 -> -1.0，c==0 -> 0.0。
 * 注意不能写成 copysign(1.0, c)（c==0 时 copysign 给 +1），也不能写成 (c>0)-(c<0) 之后再乘，
 * 这里保持与 Python 完全相同的分支结构。 */
static double lion_sign(double c)
{
    if (c > 0.0) return 1.0;
    if (c < 0.0) return -1.0;
    return 0.0;
}

int lion_step(Lion *self, const char **err)
{
    if (err) *err = NULL;
    if (!self) {
        if (err) *err = "Lion: step 收到 NULL 优化器";
        return -1;
    }

    self->step_count += 1;
    const double b1 = self->beta1, b2 = self->beta2;
    const double lr = self->lr, wd = self->weight_decay;
    const double o1 = 1.0 - b1;
    const double o2 = 1.0 - b2;

    for (int k = 0; k < self->n_layers; k++) {
        OptLayer *l = &self->layers[k];
        double *m_w = self->m_w[k], *m_b = self->m_b[k];
        const int fo = l->fan_out, fi = l->fan_in;

        for (int j = 0; j < fo; j++) {
            /* ---- 偏置：c 用旧动量与当前梯度，upd 里的 wd 项用旧参数 ---- */
            double g = l->gb[j];
            double c = b1 * m_b[j] + o1 * g;
            double upd = lion_sign(c) + wd * l->b[j];
            l->b[j] -= lr * upd;
            m_b[j] = b2 * m_b[j] + o2 * g;

            /* ---- 权重第 j 行 ---- */
            const double *gwj = l->gw + (size_t)j * (size_t)fi;
            double *Wj = l->w + (size_t)j * (size_t)fi;
            double *mwj = m_w + (size_t)j * (size_t)fi;
            for (int i = 0; i < fi; i++) {
                g = gwj[i];
                c = b1 * mwj[i] + o1 * g;
                upd = lion_sign(c) + wd * Wj[i];
                Wj[i] -= lr * upd;
                mwj[i] = b2 * mwj[i] + o2 * g;
            }
        }
    }
    return 0;
}

int lion_step_count(const Lion *self)
{
    return self ? self->step_count : 0;
}

int lion_num_layers(const Lion *self)
{
    return self ? self->n_layers : 0;
}

double lion_lr(const Lion *self)
{
    return self ? self->lr : 0.0;
}

void lion_set_lr(Lion *self, double lr)
{
    if (self) self->lr = lr;
}

double lion_weight_decay(const Lion *self)
{
    return self ? self->weight_decay : 0.0;
}

double lion_beta1(const Lion *self)
{
    return self ? self->beta1 : 0.0;
}

double lion_beta2(const Lion *self)
{
    return self ? self->beta2 : 0.0;
}

const double *lion_m_w(const Lion *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->m_w[k];
}

const double *lion_m_b(const Lion *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->m_b[k];
}

/* ============================= Adafactor ============================= */

struct Adafactor {
    int n_layers;
    OptLayer *layers;

    double lr;
    double eps;
    double eps2;
    double beta2;       /* 只保存/校验，step 里不用（Python 也一样：没有一阶矩） */
    double weight_decay;
    int step_count;

    /* R 每层长度 fan_out，C 每层长度 fan_in，V_b 每层长度 fan_out */
    double **R;
    double **C;
    double **V_b;
};

Adafactor *adafactor_create(const OptLayer *layers, int n_layers,
                            double lr, double eps, double eps2, double beta2,
                            double weight_decay, const char **err)
{
    if (err) *err = NULL;

    const char *bad = check_layers(layers, n_layers, "Adafactor");
    /* 校验顺序与文案照抄 Python：eps/eps2 -> beta2 -> weight_decay */
    if (!bad && (eps <= 0.0 || eps2 <= 0.0)) bad = "Adafactor: eps / eps2 必须 > 0";
    if (!bad && !(beta2 >= 0.0 && beta2 < 1.0)) bad = "Adafactor: beta2 必须在 [0,1)";
    if (!bad && weight_decay < 0.0) bad = "Adafactor: weight_decay 必须 >= 0";
    if (bad) {
        if (err) *err = bad;
        return NULL;
    }

    Adafactor *self = xcalloc(1, sizeof(Adafactor));
    self->n_layers = n_layers;
    self->layers = xmalloc((size_t)n_layers * sizeof(OptLayer));
    memcpy(self->layers, layers, (size_t)n_layers * sizeof(OptLayer));

    self->lr = lr;
    self->eps = eps;
    self->eps2 = eps2;
    self->beta2 = beta2;
    self->weight_decay = weight_decay;
    self->step_count = 0;

    self->R = xcalloc((size_t)n_layers, sizeof(double *));
    self->C = xcalloc((size_t)n_layers, sizeof(double *));
    self->V_b = xcalloc((size_t)n_layers, sizeof(double *));
    for (int k = 0; k < n_layers; k++) {
        self->R[k] = xcalloc((size_t)layers[k].fan_out, sizeof(double));
        self->C[k] = xcalloc((size_t)layers[k].fan_in, sizeof(double));
        self->V_b[k] = xcalloc((size_t)layers[k].fan_out, sizeof(double));
    }
    return self;
}

void adafactor_free(Adafactor *self)
{
    if (!self) return;
    for (int k = 0; k < self->n_layers; k++) {
        free(self->R[k]);
        free(self->C[k]);
        free(self->V_b[k]);
    }
    free(self->R);
    free(self->C);
    free(self->V_b);
    free(self->layers);
    free(self);
}

int adafactor_step(Adafactor *self, const char **err)
{
    if (err) *err = NULL;
    if (!self) {
        if (err) *err = "Adafactor: step 收到 NULL 优化器";
        return -1;
    }

    self->step_count += 1;
    const int t = self->step_count;
    /* Python: decay = 1.0 - (t ** -0.8) */
    double decay = 1.0 - pow((double)t, -0.8);
    if (decay < 0.0) decay = 0.0;
    const double lr = self->lr, eps = self->eps, eps2 = self->eps2;
    const double wd = self->weight_decay;

    for (int k = 0; k < self->n_layers; k++) {
        OptLayer *l = &self->layers[k];
        double *R = self->R[k], *C = self->C[k], *Vb = self->V_b[k];
        const int fo = l->fan_out, fi = l->fan_in;
        const double om = 1.0 - decay;

        /* 1) 行均值：acc 与 sq 在同一次内循环里朴素累加（与 Python 一致） */
        double sq = 0.0;
        for (int j = 0; j < fo; j++) {
            const double *gwj = l->gw + (size_t)j * (size_t)fi;
            double acc = 0.0;
            for (int i = 0; i < fi; i++) {
                const double g = gwj[i];
                acc += g * g;
                sq += g * g;
            }
            R[j] = decay * R[j] + om * (acc / (double)fi);
        }

        /* 2) 列均值（跨行同一个 i 累加） */
        for (int i = 0; i < fi; i++) {
            double acc = 0.0;
            for (int j = 0; j < fo; j++) {
                const double g = l->gw[(size_t)j * (size_t)fi + (size_t)i];
                acc += g * g;
            }
            C[i] = decay * C[i] + om * (acc / (double)fo);
        }

        /* 3) mean_R = sum(R)/fo —— Python 用内置 sum()，C 侧必须走 Neumaier */
        double mean_R = sum_neumaier(R, fo) / (double)fo;
        if (mean_R <= 0.0) mean_R = 1.0;

        /* 4) 相对步长：lr_t = lr * max(eps2, sqrt(mean(g^2))) */
        const double lr_t = lr * fmax(eps2, sqrt(sq / ((double)fo * (double)fi)));

        /* 5) 更新：权重用单表达式（wd 乘的是更新前的 Wj[i]），偏置先更 Vb 再更 B */
        for (int j = 0; j < fo; j++) {
            const double *gwj = l->gw + (size_t)j * (size_t)fi;
            double *Wj = l->w + (size_t)j * (size_t)fi;
            for (int i = 0; i < fi; i++) {
                const double v = R[j] * C[i] / mean_R;
                Wj[i] = (Wj[i] - lr_t * gwj[i] / sqrt(v + eps)
                         - lr * wd * Wj[i]);
            }
            const double g = l->gb[j];
            Vb[j] = decay * Vb[j] + om * g * g;
            l->b[j] = l->b[j] - lr_t * g / sqrt(Vb[j] + eps) - lr * wd * l->b[j];
        }
    }
    return 0;
}

int adafactor_step_count(const Adafactor *self)
{
    return self ? self->step_count : 0;
}

int adafactor_num_layers(const Adafactor *self)
{
    return self ? self->n_layers : 0;
}

double adafactor_lr(const Adafactor *self)
{
    return self ? self->lr : 0.0;
}

void adafactor_set_lr(Adafactor *self, double lr)
{
    if (self) self->lr = lr;
}

double adafactor_eps(const Adafactor *self)
{
    return self ? self->eps : 0.0;
}

double adafactor_eps2(const Adafactor *self)
{
    return self ? self->eps2 : 0.0;
}

double adafactor_beta2(const Adafactor *self)
{
    return self ? self->beta2 : 0.0;
}

double adafactor_weight_decay(const Adafactor *self)
{
    return self ? self->weight_decay : 0.0;
}

const double *adafactor_row(const Adafactor *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->R[k];
}

const double *adafactor_col(const Adafactor *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->C[k];
}

const double *adafactor_v_b(const Adafactor *self, int k)
{
    if (!self || k < 0 || k >= self->n_layers) return NULL;
    return self->V_b[k];
}
