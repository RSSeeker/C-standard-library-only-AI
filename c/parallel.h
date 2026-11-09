/*
 * parallel.h —— 分布式训练数值模拟：FSDP 分片 / ZeRO 更新 / 张量并行 / 序列并行
 *
 * 对应 Python AI.py 的 fsdp_shard_flat / fsdp_all_gather /
 * zero_optimizer_step / fsdp_reduce_scatter /
 * tensor_parallel_linear_forward / sequence_parallel_layer_norm。
 * 返回值：0 成功，-1 非法参数。
 */
#ifndef AI_PARALLEL_H
#define AI_PARALLEL_H

/* 把 n 个元素按 world_size 个 rank 切成连续分片：前 rem 个 rank 多 1 个。
 * starts/ends 长度 >= world_size。 */
int par_split_ranges(int n, int world_size, int *starts, int *ends);

/* FSDP 分片：把 flat[0..n) 切成 world_size 段；out 收各段数值（按 rank 顺序拼接，
 * 返回时与 flat 相同），sizes 收各段长度。out 长度 >= n。 */
int par_fsdp_shard_flat(const double *flat, int n, int world_size,
                        double *out, int *sizes);

/* FSDP all-gather：shards 为「按 rank 顺序的数值拼接」，starts/ends 为区间。
 * 成功时 out 长度 == 各区间长度之和。 */
int par_fsdp_all_gather(const double *shard_values, const int *starts,
                        const int *ends, int n_shards, double *out);

/* ZeRO-1/SGD：每个 rank 只更新自己的分片（out_params 初始为 params 副本）。
 * 返回更新过的区间 starts/ends（长度 world_size）。 */
int par_zero_step(const double *params, const double *grads, int n, double lr,
                  int world_size, double *out_params,
                  int *upd_starts, int *upd_ends);

/* FSDP reduce-scatter：world_size 个 rank 的完整梯度求平均，再按 rank 切分片。
 * rank_grads 为 world_size*n 个元素（rank 主序），out 长度 n（平均后的完整向量）。 */
int par_fsdp_reduce_scatter(const double *rank_grads, int world_size, int n,
                            double *out);

/* Megatron 式张量并行线性层前向（不反传）。
 * mode=0 column（按输出行切，各 rank 算一段输出后拼接）；
 * mode=1 row（按输入列切，各 rank 算部分和后相加，bias 只加一次）。
 * full_out 长度 rows；shard_out 长度 rows*world_size（各 rank 段按 rank 拼接）；
 * shard_sizes 长度 world_size（每段长度：column 为输出行数，row 均为 rows）。 */
int par_tensor_parallel(const double *W, int rows, int cols,
                        const double *bias, int has_bias,
                        const double *x, int world_size, int mode,
                        double *full_out, double *shard_out, int *shard_sizes);

/* 序列并行 LayerNorm：逐 token 归一化（token 维不分片，仅用于对照数值）。 */
int par_sequence_parallel_layer_norm(const double *x_seq, int n_tokens, int d,
                                     double eps, double *out);

#endif
