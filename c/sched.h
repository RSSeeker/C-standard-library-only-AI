/*
 * sched.h —— 梯度裁剪与学习率调度（对应 Python 的 clip_grad_by_norm /
 *            clip_grad_by_value / StepLR / ExponentialLR / CosineAnnealingLR）
 *
 * 数值与 Python 版逐条对齐，包括：
 *   - clip_grad_by_norm 里 `scale = max_norm / total_norm`（Python 无 1e-6 平滑项，
 *     注意 ai.c 里 Model.fit 的 grad_clip 走的是另一条路径，那里有 +1e-6）
 *   - CosineAnnealingLR 的 progress = ((epoch - 1) % T_max) / T_max（支持周期重启）
 *   - StepLR 只在 epoch > 0 且 epoch % step_size == 0 时更新
 */
#ifndef AI_SCHED_H
#define AI_SCHED_H

#include "ai.h"

/* ---------- 梯度裁剪 ---------- */
/* 返回裁剪前的总范数；超过 max_norm 时原地缩放 gW/gb */
double clip_grad_norm(MLP *m, double max_norm);
/* 按值裁剪（原地） */
void   clip_grad_value(MLP *m, double clip_val);

/* ---------- 学习率调度 ---------- */
typedef enum {
    SCHED_STEP = 0,
    SCHED_EXPONENTIAL = 1,
    SCHED_COSINE = 2
} SchedKind;

typedef struct LRScheduler {
    SchedKind kind;
    Optimizer *opt;        /* 被调度的优化器（只改它的 lr） */
    double initial_lr;
    int step_size;         /* StepLR */
    double gamma;          /* StepLR / ExponentialLR */
    int T_max;             /* CosineAnnealingLR */
    double eta_min;        /* CosineAnnealingLR */
    /* 外部自定义调度（可为 NULL）：一旦设置，lrsched_step 直接转调它。
     * 用于把 C++ 侧带虚函数的调度器接进 C 引擎（见 c_model.cpp）。 */
    void (*custom_step)(void *ctx, Optimizer *opt, int epoch);
    void *custom_ctx;
} LRScheduler;

void lrsched_init_step(LRScheduler *s, Optimizer *opt, int step_size, double gamma);
void lrsched_init_exponential(LRScheduler *s, Optimizer *opt, double gamma);
void lrsched_init_cosine(LRScheduler *s, Optimizer *opt, int T_max, double eta_min);
/* 按 Python 的语义在给定 epoch 上更新 opt->lr */
void lrsched_step(LRScheduler *s, int epoch);

#endif /* AI_SCHED_H */
