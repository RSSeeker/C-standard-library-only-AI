/*
 * linear_attn.h —— 线性注意力 / 长卷积的简化纯数值实现（对齐 AI.py）
 *
 * 对应 Python 版函数：
 *   linear_attention_naive / lightning_attention / gla_forward /
 *   deltanet_forward / hyena_long_conv / hyena_forward
 *
 * 矩阵按行主序扁平存放：
 *   q/k 为 t_len x dk，v 为 t_len x dv，out 为 t_len x dv；
 *   u 为 t_len x d，out 为 t_len x d。
 * 全部函数返回 0 成功 / -1 非法参数。
 *
 * 说明：
 *   - Python 中「q/k/v 序列长度不一致、q 与 k 特征维度不一致」等形状校验
 *     在 C 的扁平入参（共享 t_len/dk）下无法表达，由调用者保证，不做对拍；
 *   - NaN/Inf 输入在 Python 侧同样被拒绝，但参考 JSON 无法表示非有限值，
 *     相应错误路径只在 C 侧实现、不纳入对拍；
 *   - 退化情形（t_len/dk/dv < 1、decay ∉ [0,1]、betas ∉ [0,1]、
 *     kernel 为空）对应 Python 的 ValueError。
 */
#ifndef AI_C_LINEAR_ATTN_H
#define AI_C_LINEAR_ATTN_H

#ifdef __cplusplus
extern "C" {
#endif

/* y[t] = sum_{j<=t} decay^(t-j) * <q[t], k[j]> * v[j]（O(T^2) 展开） */
int ai_linear_attention_naive(const double *q, const double *k, const double *v,
                              int t_len, int dk, int dv, double decay,
                              double *out);

/* 标量衰减递推：state = decay*state + k^T v，y = q @ state */
int ai_lightning_attention(const double *q, const double *k, const double *v,
                           int t_len, int dk, int dv, double decay,
                           double *out);

/* GLA：alpha_t = exp(-exp(gates[t]))，state = alpha*state + k^T v */
int ai_gla_forward(const double *q, const double *k, const double *v,
                   const double *gates, int t_len, int dk, int dv,
                   double *out);

/* DeltaNet：beta 门控的 delta 规则（betas 需在 [0, 1] 内） */
int ai_deltanet_forward(const double *q, const double *k, const double *v,
                        const double *betas, int t_len, int dk, int dv,
                        double *out);

/* 因果长卷积：y[t] = sum_{j<min(t+1,k_len)} kernel[j] * u[t-j] */
int ai_hyena_long_conv(const double *u, int t_len, int d, const double *kernel,
                       int k_len, double *out);

/* Hyena 块：长卷积后与 v 逐元素相乘（v 与 u 同形状） */
int ai_hyena_forward(const double *u, const double *v, int t_len, int d,
                     const double *kernel, int k_len, double *out);

#ifdef __cplusplus
}
#endif
#endif
