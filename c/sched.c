/*
 * sched.c —— 梯度裁剪与学习率调度的实现
 */
#include "sched.h"

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
