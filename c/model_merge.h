/*
 * model_merge.h —— 模型合并（model soup / SLERP / TIES / DARE）的 C 实现
 *
 * 对应 Python AI.model_soup / slerp_merge / ties_merge / dare_merge
 * （AI.py 中「模型合并」一段）。Python 版接受任意形状的嵌套数字列表，
 * C 侧统一用「n_tensors 个等长扁平张量 + n_elem」表达：
 *   weights[j * n_elem + i] 是第 j 个参数张量的第 i 个元素；
 *   输出 out 与输入同形状（扁平、行主序）。
 * 因此 Python 侧的形状不一致错误在 C 扁平入参下不可表达，不做校验；
 * 非有限数字（NaN/Inf）仍会被拒。
 *
 * 数值约定：
 *   - Python 里用 sum(...) 的位置（model_soup 的逐元素均值、slerp 的范数与
 *     点积、ties 的同号均值）用 numeric.h 的 Neumaier 补偿求和；
 *   - 其余朴素 += 用普通加法（与 Python 一致）。
 *   - dare_merge 的随机丢弃使用与 Python `random.Random(seed)` 逐位一致的
 *     MT19937；C 侧要求显式 seed（Python 的 seed=None 走全局 random，
 *     不可复现，不支持）。
 *
 * 返回 0 成功，-1 参数非法。
 */
#ifndef AI_MODEL_MERGE_H
#define AI_MODEL_MERGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* 逐元素均值：out[i] = (Σ_j weights[j][i]) / n_tensors */
int ai_model_soup(const double *weights, int n_tensors, int n_elem,
                  double *out);

/* 球面线性插值：out = (1-t)*a + t*b（范数过小或近平行时退化为线性插值） */
int ai_slerp_merge(const double *a, const double *b, int n_elem,
                   double t, double eps, double *out);

/* TIES：裁剪小任务向量、符号选举、同号取均值，可选 density 稀疏化 */
int ai_ties_merge(const double *weights, int n_tensors, int n_elem,
                  const double *base, double trim_fraction, double density,
                  double *out);

/* DARE：按 drop_rate 随机丢弃任务向量元素，可选按 1/(1-p) 重缩放后取均值 */
int ai_dare_merge(const double *weights, int n_tensors, int n_elem,
                  const double *base, double drop_rate, int rescale,
                  unsigned long long seed, double *out);

#ifdef __cplusplus
}
#endif

#endif /* AI_MODEL_MERGE_H */
