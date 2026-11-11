/*
 * tree_attn.h —— 树状注意力辅助（草稿树 / 可见性偏置与掩码）
 *
 * 对应 Python 版 AI.py：
 *   build_draft_tree(sequences) -> {parents, tokens, leaves, paths}
 *   tree_attention_bias(parents, neg_inf)
 *   tree_attention_mask(parents)
 *
 * 草稿树用「根为 0、parent=-1、根 token=-1」的扁平结构表示；
 * 相同前缀只建一次节点，便于一次前向验证多条 draft 分支。
 */
#ifndef AI_TREE_ATTN_H
#define AI_TREE_ATTN_H

/* 由多条草稿序列（flat_tokens 按 seq_lens 切分）构造前缀共享树。
 * 成功返回 0，并把 malloc 的数组写入 out_*：
 *   out_parents/out_tokens/out_leaves 由调用方 free；
 *   out_paths/out_path_lens 为每叶子的节点下标路径（根→叶，不含根）。
 * 参数非法返回 -1。 */
int ai_build_draft_tree(const int *flat_tokens, const int *seq_lens, int n_seqs,
                        int **out_parents, int **out_tokens,
                        int **out_leaves, int *out_n_nodes,
                        int *out_n_leaves, int **out_paths,
                        int **out_path_lens);

void ai_draft_tree_free(int *parents, int *tokens, int *leaves,
                        int *paths, int *path_lens);

/* 可见性偏置：仅「祖先 + 自己」为 0.0，其余 neg_inf。bias 长度 n*n。 */
int ai_tree_attention_bias(const int *parents, int n, double neg_inf,
                           double *bias);

/* 可见性布尔掩码：1 表示节点 j 对节点 i 可见。mask 长度 n*n。 */
int ai_tree_attention_mask(const int *parents, int n, int *mask);

#endif /* AI_TREE_ATTN_H */
