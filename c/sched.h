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

/* ---------- 线性 warmup（对应 Python 的 WarmupLR） ----------
 * 前 warmup_epochs 轮把 lr 从 initial_lr*start_factor 线性升到 initial_lr，
 * 之后回到 initial_lr，并（可选）把绝对 epoch 交给后续调度器。
 * 与 AI.py 的 WarmupLR 一致：epoch 从 1 开始，<=0 按 1 处理。
 *
 * 这里不绑定 Optimizer，只拿一个 double* 指针（通常传 &opt->lr），
 * 这样也能直接驱动 optim.h 里的 AdamW/Lion/Adafactor 的 lr 字段。
 */
typedef struct WarmupLR {
    double *lr;            /* 被调度的 lr（非拥有指针，调用方保证生命周期） */
    double initial_lr;     /* create 时抓下的 *lr */
    int warmup_epochs;
    double start_factor;
    /* 可选：warmup 结束后的后续调度器（对应 Python 的 base_scheduler） */
    void (*base_step)(void *ctx, int epoch);
    void *base_ctx;
} WarmupLR;

WarmupLR *warmuplr_create(double *lr, int warmup_epochs, double start_factor,
                          const char **err);
/* 设置后续调度器（传 NULL 取消）；只保存函数指针，不拥有 ctx */
void warmuplr_set_base(WarmupLR *w, void (*base_step)(void *, int), void *ctx);
/* step 返回更新后的 *lr（Python 返回 None，这里返回 lr 只是方便断言） */
double warmuplr_step(WarmupLR *w, int epoch);
int     warmuplr_warmup_epochs(const WarmupLR *w);
double  warmuplr_start_factor(const WarmupLR *w);
double  warmuplr_initial_lr(const WarmupLR *w);
void    warmuplr_free(WarmupLR *w);

/* ---------- 指标停滞降 lr（对应 Python 的 ReduceLROnPlateau） ----------
 * 由 step(metric) 驱动：指标在 patience 次内没有改善（相对 threshold）就把
 * lr 乘 factor；降 max_reductions 次后停在 min_lr。mode 决定"改善"的方向。
 * NaN 指标不参与比较（与 Python 的 `if m != m: return lr` 一致）。
 */
typedef enum {
    PLATEAU_MIN = 0,   /* 指标越小越好（loss） */
    PLATEAU_MAX = 1    /* 指标越大越好（accuracy） */
} PlateauMode;

typedef struct ReduceLROnPlateau ReduceLROnPlateau;

/* max_reductions: -1 表示 None（不限制次数）；其余必须 >= 0。
 * lr 指向被调度的学习率（通常 &opt->lr）。 */
ReduceLROnPlateau *plateau_create(double *lr, double factor, int patience,
                                  double min_lr, PlateauMode mode, double threshold,
                                  int max_reductions, const char **err);
/* 用这一轮的验证指标调用一次；返回当前 lr（与 Python 的返回值一致） */
double plateau_step(ReduceLROnPlateau *p, double metric);
void   plateau_free(ReduceLROnPlateau *p);

double plateau_lr(const ReduceLROnPlateau *p);
double plateau_initial_lr(const ReduceLROnPlateau *p);
double plateau_factor(const ReduceLROnPlateau *p);
int    plateau_patience(const ReduceLROnPlateau *p);
double plateau_min_lr(const ReduceLROnPlateau *p);
PlateauMode plateau_mode(const ReduceLROnPlateau *p);
double plateau_threshold(const ReduceLROnPlateau *p);
int    plateau_max_reductions(const ReduceLROnPlateau *p);  /* -1 = None */
/* best 未设置时返回 NAN，用 plateau_has_best 区分 */
int    plateau_has_best(const ReduceLROnPlateau *p);
double plateau_best(const ReduceLROnPlateau *p);
int    plateau_num_bad(const ReduceLROnPlateau *p);
int    plateau_reductions(const ReduceLROnPlateau *p);

#endif /* AI_SCHED_H */
