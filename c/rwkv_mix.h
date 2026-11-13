/*
 * rwkv_mix.h —— RWKV-5 / RWKV-6 / RWKV-7 time-mixing 的纯前向 C 实现
 *
 * 对应 Python `AI.rwkv5_time_mixing` / `AI.rwkv6_time_mixing` /
 * `AI.rwkv7_time_mixing`（AI.py 中「RWKV-5/6/7 time-mixing」一段），
 * 只做逐 token 数值前向，无反向 / 训练 / 生成。
 *
 * 约定：
 *   - 矩阵按行主序扁平传入：r、k 形状为 (T, dk)，v 形状为 (T, dv)；
 *     out 由调用者分配，形状为 (T, dv)。
 *   - rwkv5 的 decay 可为标量（decay_vec == NULL）或长度 dk 的逐 channel
 *     向量（decay_vec != NULL，此时忽略标量 decay）。
 *   - 与 Python 一致：y[c] 的累加是朴素 +=；只有 Python 源码里用 sum(...)
 *     的位置（本组函数没有）才需要 Neumaier 补偿。
 *   - 状态衰减 alpha = exp(-exp(w)) 走 libm，与 Python math.exp 可能有
 *     ≤1 ULP 的 libm 差异，对拍时按需给 1e-12 容差。
 *
 * 结构错误（r/k/v 序列长度不一致、行长度不一致、非有限元素等）在 C 扁平
 * 入参下无法表达，因此不做校验；这些错误路径只在 Python 侧测试。
 * 返回 0 成功，-1 参数非法（空指针、维数 < 1、decay/delta 越界等）。
 */
#ifndef AI_RWKV_MIX_H
#define AI_RWKV_MIX_H

#ifdef __cplusplus
extern "C" {
#endif

/* S_t = diag(decay) S_{t-1} + k_t (x) v_t,  y_t = r_t @ S_t */
int ai_rwkv5_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *decay_vec, double decay, double *out);

/* alpha_t = exp(-exp(w_t));  S_t = diag(alpha_t) S_{t-1} + k_t (x) v_t */
int ai_rwkv6_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *w_logits, double *out);

/* RWKV-7 简化增量修正：先读出旧状态再按 delta 修正写入
 * S_t = diag(alpha) S_{t-1} + k_t (x) (v_t - delta * S_{t-1}^T k_t) */
int ai_rwkv7_time_mixing(const double *r, const double *k, const double *v,
                         int t_len, int dk, int dv,
                         const double *w_logits, double delta, double *out);

#ifdef __cplusplus
}
#endif

#endif /* AI_RWKV_MIX_H */
