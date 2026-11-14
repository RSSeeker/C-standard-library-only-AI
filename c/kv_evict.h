/*
 * kv_evict.h —— KV cache 淘汰策略（对应 Python 版 AI.py）
 *
 * 覆盖：
 *   - streaming_kv_keep（StreamingLLM：attention sink + 最近窗口）
 *   - h2o_kv_keep（H2O：按累计注意力选 top token）
 *   - snapkv_kv_keep（SnapKV：只用最后 obs 行 query 的分数）
 *   - kv_cache_evict（完整淘汰：返回保留下标与收缩后的 K/V）
 *
 * 纯计算、无全局状态；二维数组默认行主序扁平 double。
 * 输入合法性由调用方保证（与 Python 版校验规则一致，非法输入未定义）。
 */
#ifndef AI_KV_EVICT_H
#define AI_KV_EVICT_H

/* StreamingLLM：保留前 sink 个 + 最近 (budget-sink) 个。
 * out 容量 >= budget；返回保留个数（= budget），下标升序。 */
int ai_streaming_kv_keep(int seq_len, int budget, int sink, int *out);

/* H2O：scores 为 nq×seq_len 行主序，按列求和得到重要度；
 * 强制保留前 sink 个与最近 recent 个，其余按 (-score, index) 取 top。
 * out 容量 >= budget；返回保留个数，下标升序。 */
int ai_h2o_kv_keep(const double *scores, int nq, int seq_len, int budget,
                   int sink, int recent, int *out);

/* SnapKV：只用最后 obs 行 query 聚合；obs<=0 时取 max(1, nq/4)。
 * out 容量 >= budget；返回保留个数，下标升序。 */
int ai_snapkv_kv_keep(const double *scores, int nq, int seq_len, int budget,
                      int sink, int obs, int *out);

/* 完整淘汰：keys/values 为 seq_len×dim 行主序。
 * policy: 0=streaming, 1=h2o, 2=snapkv。
 * out_keep 容量 >= budget；out_keys/out_values 容量 >= budget*dim。
 * 返回保留个数。 */
int ai_kv_cache_evict(const double *keys, const double *values,
                      int seq_len, int dim, int budget,
                      const double *scores, int nq, int policy,
                      int sink, int recent, int obs,
                      int *out_keep, double *out_keys, double *out_values);

#endif
