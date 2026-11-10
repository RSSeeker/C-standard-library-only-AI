/*
 * adapt.h — 低秩适配（LoRA）与权重平均（EMA / SWA / Lookahead）的 C17 实现
 *
 * 与 AI.py 的对应关系：
 *   LoRA       -> 21190 附近的 class LoRA
 *   EMA        -> class EMA
 *   SWA        -> class SWA
 *   Lookahead  -> class Lookahead
 *
 * 约定（与 C 侧其它模块一致）：
 *   - LoRA 的 A 是 rank x fan_in（行主序），B 是 fan_out x rank，scaling = alpha / rank；
 *   - 权重 W 是 fan_out x fan_in（行主序，与 ai.h 的 Layer.W 同布局）；
 *   - 前向/反向里凡是 Python 用 sum(...) 的地方用 Neumaier（numeric.h 的 sum_neumaier），
 *     凡是显式 += 循环的地方用朴素累加——两者不可互换，否则末位会差。
 */
#ifndef ADAPT_H
#define ADAPT_H

#include "ai.h"

/* ---------- LoRA ---------- */
typedef struct {
    int fan_in;
    int fan_out;
    int rank;
    int merged;

    double alpha;
    double scaling;

    double *A;   /* rank x fan_in */
    double *B;   /* fan_out x rank */
    double *gA;  /* rank x fan_in */
    double *gB;  /* fan_out x rank */

    double *x;   /* 前向缓存：输入 */
    double *h;   /* 前向缓存：A x（rank 维） */
} LoRA;

/* alpha <= 0 表示「alpha = rank」（与 Python 默认一致）。B 初始化为 0，A 初始化为 0，
 * 需要非零 A 时调用 lora_set_params。 */
LoRA *lora_create(int fan_in, int fan_out, int rank, double alpha);
void lora_free(LoRA *l);
void lora_zero_grad(LoRA *l);
void lora_set_params(LoRA *l, const double *A, const double *B);
int lora_trainable_params(const LoRA *l);

/* y = W x + b + scaling * B (A x)；merged 时退化成 y = W x + b */
void lora_forward(LoRA *l, const double *W, const double *b,
                  const double *x, double *y);

/* dx = W^T dout + A^T (scaling * B^T dout)，同时累加 gA/gB（原层梯度不累加：冻结） */
void lora_backward(LoRA *l, const double *W, const double *dout, double *dx);

/* 把 scaling * B A 折进 W，并把 B 清零、置 merged */
void lora_merge(LoRA *l, double *W);

/* W_eff = W + scaling * B A（不修改 W）；merged 时就是 W 的拷贝 */
void lora_effective_weight(const LoRA *l, const double *W, double *out);

/* ---------- 扁平快照工具 ---------- */
int mlp_flat_size(const MLP *m);
void mlp_flat_read(const MLP *m, double *out);
void mlp_flat_write(MLP *m, const double *in);

/* ---------- EMA ---------- */
typedef struct {
    int n;
    double *shadow;
    double decay;
    int warmup;
    int step;
} EMA;

EMA *ema_create(const MLP *m, double decay, int warmup);
void ema_free(EMA *e);
double ema_effective_decay(const EMA *e);
void ema_update(EMA *e, const MLP *m);
void ema_copy_to(const EMA *e, MLP *m);

/* ---------- SWA ---------- */
typedef struct {
    int n;
    double *avg;
    int count;
} SWA;

SWA *swa_create(const MLP *m);
void swa_free(SWA *s);
void swa_update(SWA *s, const MLP *m);
void swa_apply(const SWA *s, MLP *m);

/* ---------- Lookahead ---------- */
typedef struct {
    int n;
    double *slow;
    int k;
    double alpha;
    int step;
} Lookahead;

Lookahead *lookahead_create(const MLP *m, int k, double alpha);
void lookahead_free(Lookahead *l);

/* 在优化器已经更新完快权重之后调用；返回 1 表示这一步触发了同步 */
int lookahead_step(Lookahead *l, MLP *m);

#endif /* ADAPT_H */
