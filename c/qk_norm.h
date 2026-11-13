/*
 * qk_norm.h —— QK-Norm（逐行 RMSNorm）与 DynamicTanh（DyT）的 C17 实现
 *
 * 对应 AI.py「注意力扩展：QK-Norm 与 DynamicTanh」一节：
 *   qk_norm            (AI.py:7876) -> ai_qk_norm
 *   qk_norm_backward   (AI.py:7894) -> ai_qk_norm_backward
 *   apply_qk_norm      (AI.py:7927) -> ai_apply_qk_norm
 *   class DynamicTanh  (AI.py:7937) -> AiDynamicTanh + ai_dyt_*
 *
 * 数值约定：
 *   - Python 的 mean_sq = sum(v*v for v in row) 与反向的
 *     dot = sum(dxhat[i]*xhat[i]) 都走 numeric.h 的 dot_neumaier
 *     （CPython 3.12+ 的 sum 是 Neumaier 补偿求和）。
 *   - grad_gamma / grad_beta / grad_alpha 的累加顺序与 Python 一致：
 *     先按行、再按列，普通 +=（不用补偿求和）。
 *   - tanh 直接用 libm；测试按 1e-12 容差对拍（不同 libm 可能差 1 ULP）。
 *
 * 入参形状：C 侧用「扁平数组 + n_rows/width」表达二维矩阵。Python 的
 * 「矩阵非空、各行等长、元素有限」在 C 侧由 ai_qk_norm / ai_dyt_forward
 * 校验（n_rows >= 1、width >= 1、元素有限）；「各行不等长」在扁平入参下
 * 不可表达（width 由调用方给定），因此不在 C 侧检查。
 *
 * 所有函数返回 0 成功、-1 参数非法（对齐 Python 侧抛 ValueError）。
 * AiQkNormResult 由 ai_qk_norm 分配，必须用 ai_qk_norm_free 释放。
 */
#ifndef AI_QK_NORM_H
#define AI_QK_NORM_H

#ifdef __cplusplus
extern "C" {
#endif

/* qk_norm 的完整结果（同时作为 backward 的 cache）：
 *   out   = gamma * xhat        n_rows*width
 *   rms   = sqrt(mean(x^2)+eps) n_rows
 *   xhat  = x / rms             n_rows*width
 *   gamma 前向使用的 gamma      width
 */
typedef struct {
    int n_rows;
    int width;
    double *out;
    double *rms;
    double *xhat;
    double *gamma;
} AiQkNormResult;

/* 逐行 RMSNorm：rows 为 n_rows*width 的扁平矩阵，gamma 长度隐含为 width。
 * Python 的 eps 默认 1e-6，这里必须显式传入。 */
int ai_qk_norm(const double *rows, int n_rows, int width, const double *gamma,
               double eps, AiQkNormResult *out);
void ai_qk_norm_free(AiQkNormResult *res);

/* 反向：grad_gamma 长度 width，input_grad 长度 n_rows*width（都是覆盖写）。
 * n_rows/width 必须与 cache 一致；dout 元素必须有限（Python 会校验）。
 * 返回 0/-1。 */
int ai_qk_norm_backward(const double *dout_rows, int n_rows, int width,
                        const AiQkNormResult *cache, double *grad_gamma,
                        double *input_grad);

/* 对注意力 Q/K 分别做 QK-Norm（Python 的 apply_qk_norm）。
 * q 与 k 的行数/宽度可以不同（Python 也是分别校验）。 */
int ai_apply_qk_norm(const double *q_rows, int q_n_rows, int q_width,
                     const double *q_gamma, const double *k_rows,
                     int k_n_rows, int k_width, const double *k_gamma,
                     double eps, AiQkNormResult *q_out,
                     AiQkNormResult *k_out);

/* ============================ DynamicTanh ============================ */

/* y = gamma * tanh(alpha*x) + beta。
 * Python 的构造参数 seed 当前未被使用（gamma/beta 是固定初始化），
 * C 侧因此省略；语义对齐见 AI.py:7940。 */
typedef struct {
    int d_model;
    double alpha;
    double *gamma;      /* d_model，初值 1.0 */
    double *beta;       /* d_model，初值 0.0 */
    double grad_alpha;  /* 累加梯度 */
    double *grad_gamma; /* d_model，累加梯度 */
    double *grad_beta;  /* d_model，累加梯度 */

    double *cache_x;    /* 最近一次 forward 的输入  n*d_model */
    double *cache_tanh; /* 最近一次 forward 的 tanh  n*d_model */
    int cache_n;        /* 最近一次 forward 的行数，0 = 还没 forward */
    int cache_cap;      /* cache_x/cache_tanh 的容量（行数） */
} AiDynamicTanh;

/* 创建：校验 d_model >= 1、alpha 非零且有限；gamma=1、beta=0、梯度清零。 */
int ai_dyt_create(AiDynamicTanh *self, int d_model, double alpha);
void ai_dyt_free(AiDynamicTanh *self);

/* 前向：rows 为 n_rows*width，width 必须等于 d_model；out 同形。
 * 成功返回 0 并把输入/tanh 写进 cache（供 backward 使用）。 */
int ai_dyt_forward(AiDynamicTanh *self, const double *rows, int n_rows,
                   int width, double *out);

/* 反向：必须已经 forward 过（Python 抛 RuntimeError，C 侧返回 -1）；
 * n_rows/width 必须与 cache 一致。累积 grad_alpha/grad_gamma/grad_beta，
 * 并把输入梯度写进 input_grad（n_rows*width）。
 * 注意：Python 的 DyT.backward 不校验 dout 元素有限，C 侧同样不校验。 */
int ai_dyt_backward(AiDynamicTanh *self, const double *dout_rows, int n_rows,
                    int width, double *input_grad);

/* 梯度清零（对齐 Python 的 zero_grad）。 */
int ai_dyt_zero_grad(AiDynamicTanh *self);

/* SGD 更新 alpha/gamma/beta 后清零梯度；lr 必须是正有限数。 */
int ai_dyt_sgd_step(AiDynamicTanh *self, double lr);

/* 可训练参数数 = 1 + 2*d_model（alpha + gamma + beta）。 */
int ai_dyt_param_count(const AiDynamicTanh *self);

#ifdef __cplusplus
}
#endif

#endif /* AI_QK_NORM_H */
