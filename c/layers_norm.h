/*
 * layers_norm.h —— Dropout / BatchNorm1d / LayerNorm 的 C17 移植
 *
 * 对应 AI.py：
 *   Dropout      (AI.py:457)  -> Dropout
 *   BatchNorm1d  (AI.py:480)  -> BatchNorm1d（训练 forward_batch / 推理 forward 两条路径都保留）
 *   LayerNorm    (AI.py:561)  -> LayerNorm
 *
 * 数值约定（与 Python 逐位对齐的关键，踩过的坑都记在这里）：
 *   1. Python 里所有 sum(...) 在 C 侧一律走 numeric.h 的 sum_neumaier / dot_neumaier
 *      （CPython 3.12+ 的 sum 是 Neumaier 补偿求和）；只有 Python 本身就是朴素 += 的
 *      地方（running 统计量更新、参数梯度累加）才用普通 +。
 *   2. BatchNorm1d 的列统计照抄 Python 的 `cols = list(zip(*x_batch))`：C 版先把 (N,D)
 *      输入转置进列主序缓冲再求和，这样"列"在内存里连续，可以直接喂给 sum_neumaier，
 *      而且求和顺序与 Python 的 sum(col) 完全一致。
 *   3. 运算顺序（含括号）与 Python 一一对应，不做任何"代数化简"：
 *      dx 写成 inv_std * (...) / n，而不是 (inv_std / n) * (...)；
 *      推理路径写成 (x - mean) / sqrt(var + eps)，而不是 (x - mean) * (1/sqrt(...))。
 *      改一处就会差最后几位。
 *   4. 参数梯度按 tensor.h 的 Module 约定"累加"进内部缓冲（调用方负责清零/按 batch 求平均）。
 *      这与 Python 里"每次 forward 覆盖赋值"的写法不同，但正是本项目的实际用法：
 *      TransformerBlock 逐 token 调 LayerNorm.backward 再手工 += 累加（AI.py:1582-1584）。
 *
 * 内存约定：
 *   - forward / backward 热路径不 malloc；所有形状固定的缓冲在 create 时分配。
 *   - BatchNorm1d 的 x_hat / col 与 Dropout 的 mask 长度取决于 batch（输入）大小，
 *     create 时无从得知：首次遇到更大的 batch 会扩容一次（容量按 2 倍增长并保留），
 *     之后同一量级的调用零分配。想彻底零分配，训练前调一次 bn1d_reserve / dropout_reserve。
 *   - 形状不匹配一律走 tensor_check_shape()，报错退出，不静默截断。
 */
#ifndef AI_LAYERS_NORM_H
#define AI_LAYERS_NORM_H

#include "tensor.h"

/* ============================== Dropout ==============================
 * Python: mask[i] = scale if random.random() > p else 0.0，scale = 1/(1-p)；
 *         backward 只把梯度回传给 mask 非零的位置。
 * C 版的不同点：RNG 换成自带 xorshift64*（不追求与 Python 的 MT 逐位一致），
 *         并提供 dropout_set_mask() 灌入确定 mask 做等价性验证。
 */
typedef struct {
    double p;                  /* 置零概率，构造时校验 0 <= p < 1 */
    int training;              /* 1 = 训练（丢弃 + inverted scaling），0 = 推理（透传） */
    unsigned long long rng;    /* xorshift64* 状态 */

    double *mask;              /* 长度 mask_cap：保留位置 = scale，丢弃位置 = 0.0；推理时全 1.0 */
    int mask_len;              /* 最近一次 forward 的输入元素数（backward 用它校验形状） */
    int mask_cap;              /* mask 缓冲容量 */

    const double *pending;     /* 灌入的确定性 mask（不持有所有权，一次性） */
    int has_pending;
} Dropout;

Dropout *dropout_create(double p);
void dropout_free(Dropout *self);
void dropout_zero_grad(Dropout *self);   /* Dropout 无参数：空实现，仅为统一接口 */
/* 输入形状任意，按扁平 size 处理；out 必须与 x 同形 */
void dropout_forward(Dropout *self, const Tensor *x, Tensor *out);
/* dx[i] = dout[i] * mask[i]；dx/dout 必须同形，且长度要与最近一次 forward 一致 */
void dropout_backward(Dropout *self, const Tensor *dout, Tensor *dx);
int  dropout_param_count(Dropout *self);

void dropout_set_training(Dropout *self, int training);
void dropout_set_p(Dropout *self, double p);
void dropout_set_seed(Dropout *self, unsigned long long seed);
/* 预分配 mask 缓冲（在 forward 之前调用），之后同量级输入零分配 */
void dropout_reserve(Dropout *self, int n);
/* 灌入确定性 mask：长度 = 下一次 forward 的输入 size（由调用方保证），
 * 该数组必须保持有效直到下一次 forward（forward 时才按输入 size 拷进内部缓冲）。
 * 只对下一次 forward 生效（一次性），之后恢复 RNG 生成。 */
void dropout_set_mask(Dropout *self, const double *mask);
const double *dropout_mask(const Dropout *self);  /* 内部 mask，便于测试比对 */
int dropout_mask_len(const Dropout *self);

Module *dropout_as_module(Dropout *self);

/* ============================ BatchNorm1d ============================
 * Python 两条路径在 C 里都保留：
 *   bn1d_forward_batch(bn, x_batch(N,D), out) —— 训练：用 batch 统计量，并更新 running_mean/var
 *   bn1d_forward(bn, x(D,), out)              —— 推理：用 running 统计量
 * Module vtable 的 forward 按 self->training 自动分派到上面两者（供 Sequential 使用）。
 *
 * 内部缓存：
 *   x_hat 为列主序 d×n（x_hat[j*n + i]），与 Python 的 self.x_hat[i][j] 一一对应；
 *   col   为列主序工作缓冲，前向放输入列、反向放 dout 列（复用，不额外分配）。
 */
typedef struct {
    int d;                     /* num_features */
    double eps;
    double momentum;
    int training;              /* 只为 Module vtable 分派与 bn1d_set_training 用 */

    double *gamma;             /* d */
    double *beta;              /* d */
    double *running_mean;      /* d，初值 0 */
    double *running_var;       /* d，初值 1 */

    double *grad_gamma;        /* d，累加 */
    double *grad_beta;         /* d，累加 */

    double *x_hat;             /* d × cap，列主序前向缓存（backward 依赖） */
    double *col;               /* d × cap，列主序工作缓冲 */
    double *mean;              /* d，最近一次 batch 的列均值 */
    double *var;               /* d，最近一次 batch 的有偏方差 */
    double *inv_std;           /* d，1/sqrt(var + eps) */
    int n;                     /* 最近一次 forward_batch 的 batch 大小 */
    int cap;                   /* x_hat / col 的容量（按列长 n 计） */
    int batch_ready;           /* 最近一次前向是 forward_batch（backward 可用） */
} BatchNorm1d;

BatchNorm1d *bn1d_create(int num_features, double eps, double momentum);
void bn1d_free(BatchNorm1d *self);
void bn1d_zero_grad(BatchNorm1d *self);
void bn1d_forward_batch(BatchNorm1d *self, const Tensor *x_batch, Tensor *out); /* (N,D) -> (N,D) */
void bn1d_forward(BatchNorm1d *self, const Tensor *x, Tensor *out);             /* (D,)  -> (D,)  */
void bn1d_backward(BatchNorm1d *self, const Tensor *dout, Tensor *dx);          /* (N,D) -> (N,D) */
int  bn1d_param_count(BatchNorm1d *self);

void bn1d_set_training(BatchNorm1d *self, int training);
/* 灌入 gamma/beta（跨语言等价性验证用），长度均为 d */
void bn1d_set_params(BatchNorm1d *self, const double *gamma, const double *beta);
/* 灌入 running_mean/running_var（推理路径比对用），长度均为 d */
void bn1d_set_running(BatchNorm1d *self, const double *mean, const double *var);
/* 预分配 batch 缓冲；必须在 forward_batch 之前调用（会作废已有的前向缓存） */
void bn1d_reserve(BatchNorm1d *self, int n);

const double *bn1d_grad_gamma(const BatchNorm1d *self);
const double *bn1d_grad_beta(const BatchNorm1d *self);

Module *bn1d_as_module(BatchNorm1d *self);

/* ============================= LayerNorm =============================
 * 单个样本的 (D,) -> (D,)，与 Python 一致：逐 token 调用，batch 维由调用方循环。
 * 注意 Python 的 LayerNorm.backward 里 grad_gamma 是逐元素 dout[j]*x_hat[j]（没有对
 * batch 求和），C 版按 Module 约定把它累加进内部缓冲。
 * 所有缓冲在 create 时按 d 分配，前向/反向零分配。
 */
typedef struct {
    int d;                     /* num_features */
    double eps;

    double *gamma;             /* d */
    double *beta;              /* d */
    double *grad_gamma;        /* d，累加 */
    double *grad_beta;         /* d，累加 */

    double *x_hat;             /* d，前向缓存 */
    double *dx_hat;            /* d，反向临时缓冲 */

    double mean;               /* 最近一次前向的统计量（与原版字段同名） */
    double var;
    double inv_std;
    int ready;                 /* 是否已 forward（backward 的前置条件） */
} LayerNorm;

LayerNorm *layernorm_create(int num_features, double eps);
void layernorm_free(LayerNorm *self);
void layernorm_zero_grad(LayerNorm *self);
void layernorm_forward(LayerNorm *self, const Tensor *x, Tensor *out);   /* (D,) -> (D,) */
void layernorm_backward(LayerNorm *self, const Tensor *dout, Tensor *dx);
int  layernorm_param_count(LayerNorm *self);

/* 灌入 gamma/beta（跨语言等价性验证用），长度均为 d */
void layernorm_set_params(LayerNorm *self, const double *gamma, const double *beta);

const double *layernorm_grad_gamma(const LayerNorm *self);
const double *layernorm_grad_beta(const LayerNorm *self);

Module *layernorm_as_module(LayerNorm *self);

/* ========================= Module 适配层说明 =========================
 * xxx_as_module() 会 malloc 一个 Module 包装（self 指向原对象），用于
 * module_forward / module_backward / module_zero_grad / module_param_count。
 * 释放时 module_free(m) 会把包装和原对象一起释放（等价于 xxx_free(self)），
 * 因此 module_free 与 xxx_free 二选一，不要重复释放。
 * BatchNorm1d 的 vtable forward 按 training 分派：1 -> forward_batch(N,D)，0 -> forward(D,)。
 */

#endif /* AI_LAYERS_NORM_H */
