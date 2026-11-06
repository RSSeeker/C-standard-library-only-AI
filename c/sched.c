/*
 * sched.c —— 梯度裁剪与学习率调度的实现
 */
#include "sched.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>

/* 严格 C17 下 math.h 不保证提供 M_PI（POSIX 扩展），自己定义 */
#define AI_PI 3.14159265358979323846

double clip_grad_norm(MLP *m, double max_norm)
{
    double total_sq = 0.0;
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++) total_sq += l->gW[k] * l->gW[k];
        for (int j = 0; j < l->fan_out; j++) total_sq += l->gb[j] * l->gb[j];
    }
    double total_norm = sqrt(total_sq);
    if (total_norm > max_norm && total_norm > 0.0) {
        double scale = max_norm / total_norm;
        for (int i = 0; i < m->n_layers; i++) {
            Layer *l = &m->layers[i];
            for (int k = 0; k < l->fan_out * l->fan_in; k++) l->gW[k] *= scale;
            for (int j = 0; j < l->fan_out; j++) l->gb[j] *= scale;
        }
    }
    return total_norm;
}

void clip_grad_value(MLP *m, double clip_val)
{
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        for (int k = 0; k < l->fan_out * l->fan_in; k++) {
            double g = l->gW[k];
            l->gW[k] = (g < -clip_val) ? -clip_val : (g > clip_val ? clip_val : g);
        }
        for (int j = 0; j < l->fan_out; j++) {
            double g = l->gb[j];
            l->gb[j] = (g < -clip_val) ? -clip_val : (g > clip_val ? clip_val : g);
        }
    }
}

void lrsched_init_step(LRScheduler *s, Optimizer *opt, int step_size, double gamma)
{
    memset(s, 0, sizeof(*s));   /* 必须清零：custom_step/custom_ctx 否则是栈上垃圾，lrsched_step 会调用它 */
    s->kind = SCHED_STEP;
    s->opt = opt;
    s->initial_lr = opt->lr;
    s->step_size = step_size;
    s->gamma = gamma;
    s->T_max = 0;
    s->eta_min = 0.0;
}

void lrsched_init_exponential(LRScheduler *s, Optimizer *opt, double gamma)
{
    memset(s, 0, sizeof(*s));   /* 必须清零：custom_step/custom_ctx 否则是栈上垃圾，lrsched_step 会调用它 */
    s->kind = SCHED_EXPONENTIAL;
    s->opt = opt;
    s->initial_lr = opt->lr;
    s->step_size = 0;
    s->gamma = gamma;
    s->T_max = 0;
    s->eta_min = 0.0;
}

void lrsched_init_cosine(LRScheduler *s, Optimizer *opt, int T_max, double eta_min)
{
    memset(s, 0, sizeof(*s));   /* 必须清零：custom_step/custom_ctx 否则是栈上垃圾，lrsched_step 会调用它 */
    s->kind = SCHED_COSINE;
    s->opt = opt;
    s->initial_lr = opt->lr;
    s->step_size = 0;
    s->gamma = 1.0;
    s->T_max = T_max;
    s->eta_min = eta_min;
}

void lrsched_step(LRScheduler *s, int epoch)
{
    /* 外部自定义调度优先（C++ 侧的虚函数通过它接进来） */
    if (s->custom_step) {
        s->custom_step(s->custom_ctx, s->opt, epoch);
        return;
    }
    switch (s->kind) {
    case SCHED_STEP:
        if (epoch > 0 && s->step_size > 0 && epoch % s->step_size == 0)
            s->opt->lr = s->initial_lr * pow(s->gamma, (double)(epoch / s->step_size));
        break;
    case SCHED_EXPONENTIAL:
        s->opt->lr = s->initial_lr * pow(s->gamma, (double)epoch);
        break;
    case SCHED_COSINE: {
        if (s->T_max <= 0) break;
        /* T_max=1 时每一轮都是一个完整周期的末尾：直接落在 eta_min（Python 侧同样处理） */
        double progress = (s->T_max == 1) ? 1.0
                        : (double)((epoch - 1) % s->T_max) / (double)(s->T_max - 1);
        s->opt->lr = s->eta_min + 0.5 * (s->initial_lr - s->eta_min) *
                     (1 + cos(AI_PI * progress));
        break;
    }
    }
}

/* ==================== WarmupLR / ReduceLROnPlateau ==================== */

static void *sched_xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "sched: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

WarmupLR *warmuplr_create(double *lr, int warmup_epochs, double start_factor,
                          const char **err)
{
    if (err) *err = NULL;
    if (!lr) {
        if (err) *err = "WarmupLR: lr 指针不能为 NULL";
        return NULL;
    }
    /* 与 Python 一致：warmup_epochs >= 1，start_factor 在 [0,1] */
    if (warmup_epochs < 1) {
        if (err) *err = "WarmupLR: warmup_epochs 必须 >= 1";
        return NULL;
    }
    if (!(start_factor >= 0.0 && start_factor <= 1.0)) {
        if (err) *err = "WarmupLR: start_factor 必须在 [0,1]";
        return NULL;
    }
    WarmupLR *w = sched_xmalloc(sizeof(WarmupLR));
    memset(w, 0, sizeof(*w));
    w->lr = lr;
    w->initial_lr = *lr;
    w->warmup_epochs = warmup_epochs;
    w->start_factor = start_factor;
    return w;
}

void warmuplr_set_base(WarmupLR *w, void (*base_step)(void *, int), void *ctx)
{
    if (!w) return;
    w->base_step = base_step;
    w->base_ctx = ctx;
}

double warmuplr_step(WarmupLR *w, int epoch)
{
    if (!w) return 0.0;
    if (epoch < 1) epoch = 1;   /* Python: epoch = max(int(epoch), 1) */
    if (epoch <= w->warmup_epochs) {
        /* Python: frac = epoch / warmup_epochs；lr = initial*(start + (1-start)*frac) */
        double frac = (double)epoch / (double)w->warmup_epochs;
        *w->lr = w->initial_lr * (w->start_factor + (1.0 - w->start_factor) * frac);
        return *w->lr;
    }
    /* warmup 结束：先回到基准 lr，再让后续调度器按绝对 epoch 接管 */
    *w->lr = w->initial_lr;
    if (w->base_step) w->base_step(w->base_ctx, epoch);
    return *w->lr;
}

int warmuplr_warmup_epochs(const WarmupLR *w)
{
    return w ? w->warmup_epochs : 0;
}

double warmuplr_start_factor(const WarmupLR *w)
{
    return w ? w->start_factor : 0.0;
}

double warmuplr_initial_lr(const WarmupLR *w)
{
    return w ? w->initial_lr : 0.0;
}

void warmuplr_free(WarmupLR *w)
{
    free(w);
}

/* ------------------------- ReduceLROnPlateau ------------------------- */

struct ReduceLROnPlateau {
    double *lr;
    double initial_lr;
    double factor;
    int patience;
    double min_lr;
    PlateauMode mode;
    double threshold;
    int max_reductions;   /* -1 = None */
    int has_best;
    double best;
    int num_bad;
    int reductions;
};

ReduceLROnPlateau *plateau_create(double *lr, double factor, int patience,
                                  double min_lr, PlateauMode mode, double threshold,
                                  int max_reductions, const char **err)
{
    if (err) *err = NULL;
    if (!lr) {
        if (err) *err = "ReduceLROnPlateau: lr 指针不能为 NULL";
        return NULL;
    }
    /* 校验顺序与文案照抄 Python：factor -> mode -> patience -> min_lr -> max_reductions */
    if (!(factor > 0.0 && factor < 1.0)) {
        if (err) *err = "ReduceLROnPlateau: factor 必须在 (0,1)";
        return NULL;
    }
    if (mode != PLATEAU_MIN && mode != PLATEAU_MAX) {
        if (err) *err = "ReduceLROnPlateau: mode 只能是 min 或 max";
        return NULL;
    }
    if (patience < 0) {
        if (err) *err = "ReduceLROnPlateau: patience 必须 >= 0";
        return NULL;
    }
    if (min_lr < 0.0) {
        if (err) *err = "ReduceLROnPlateau: min_lr 必须 >= 0";
        return NULL;
    }
    if (max_reductions < -1) {
        if (err) *err = "ReduceLROnPlateau: max_reductions 必须是 -1（None）或 >= 0";
        return NULL;
    }
    ReduceLROnPlateau *p = sched_xmalloc(sizeof(*p));
    memset(p, 0, sizeof(*p));
    p->lr = lr;
    p->initial_lr = *lr;
    p->factor = factor;
    p->patience = patience;
    p->min_lr = min_lr;
    p->mode = mode;
    p->threshold = threshold;
    p->max_reductions = max_reductions;
    p->has_best = 0;
    p->best = 0.0;
    p->num_bad = 0;
    p->reductions = 0;
    return p;
}

double plateau_step(ReduceLROnPlateau *p, double metric)
{
    if (!p) return 0.0;
    const double m = metric;
    if (m != m) return *p->lr;          /* NaN 不参与比较 */
    if (!p->has_best) {                 /* Python: if self.best is None */
        p->best = m;
        p->has_best = 1;
        return *p->lr;
    }
    int better;
    if (p->mode == PLATEAU_MIN) better = m < p->best - p->threshold;
    else                        better = m > p->best + p->threshold;
    if (better) {
        p->best = m;
        p->num_bad = 0;
        return *p->lr;
    }
    p->num_bad += 1;
    if (p->num_bad > p->patience) {
        if (p->max_reductions >= 0 && p->reductions >= p->max_reductions) return *p->lr;
        const double scaled = *p->lr * p->factor;
        *p->lr = (p->min_lr > scaled) ? p->min_lr : scaled;   /* max(min_lr, lr*factor) */
        p->num_bad = 0;
        p->reductions += 1;
    }
    return *p->lr;
}

void plateau_free(ReduceLROnPlateau *p)
{
    free(p);
}

double plateau_lr(const ReduceLROnPlateau *p)
{
    return p ? *p->lr : 0.0;
}

double plateau_initial_lr(const ReduceLROnPlateau *p)
{
    return p ? p->initial_lr : 0.0;
}

double plateau_factor(const ReduceLROnPlateau *p)
{
    return p ? p->factor : 0.0;
}

int plateau_patience(const ReduceLROnPlateau *p)
{
    return p ? p->patience : 0;
}

double plateau_min_lr(const ReduceLROnPlateau *p)
{
    return p ? p->min_lr : 0.0;
}

PlateauMode plateau_mode(const ReduceLROnPlateau *p)
{
    return p ? p->mode : PLATEAU_MIN;
}

double plateau_threshold(const ReduceLROnPlateau *p)
{
    return p ? p->threshold : 0.0;
}

int plateau_max_reductions(const ReduceLROnPlateau *p)
{
    return p ? p->max_reductions : -1;
}

int plateau_has_best(const ReduceLROnPlateau *p)
{
    return p ? p->has_best : 0;
}

double plateau_best(const ReduceLROnPlateau *p)
{
    return (p && p->has_best) ? p->best : NAN;
}

int plateau_num_bad(const ReduceLROnPlateau *p)
{
    return p ? p->num_bad : 0;
}

int plateau_reductions(const ReduceLROnPlateau *p)
{
    return p ? p->reductions : 0;
}
