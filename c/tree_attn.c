/*
 * tree_attn.c —— 树状注意力辅助的 C 实现
 *
 * 前缀共享 trie 用扁平数组线性查孩子（节点数很小，O(n^2) 足够）。
 */
#include "tree_attn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int ai_build_draft_tree(const int *flat_tokens, const int *seq_lens, int n_seqs,
                        int **out_parents, int **out_tokens,
                        int **out_leaves, int *out_n_nodes,
                        int *out_n_leaves, int **out_paths,
                        int **out_path_lens)
{
    int cap = 1, n = 1; /* 根节点 */
    int *parents = NULL, *tokens = NULL;
    int total_tokens = 0;
    int s, i, j, t, cur;
    if (!flat_tokens || !seq_lens || n_seqs < 1 || !out_parents ||
        !out_tokens || !out_leaves || !out_n_nodes || !out_n_leaves ||
        !out_paths || !out_path_lens)
        return -1;
    for (s = 0; s < n_seqs; s++) {
        if (seq_lens[s] < 1) return -1;
        total_tokens += seq_lens[s];
    }
    cap = total_tokens + 1;
    parents = (int *)malloc(sizeof(int) * (size_t)cap);
    tokens = (int *)malloc(sizeof(int) * (size_t)cap);
    if (!parents || !tokens) {
        free(parents);
        free(tokens);
        return -1;
    }
    parents[0] = -1;
    tokens[0] = -1;
    {
        int pos = 0;
        for (s = 0; s < n_seqs; s++) {
            cur = 0;
            for (i = 0; i < seq_lens[s]; i++) {
                t = flat_tokens[pos++];
                if (t < 0) {
                    free(parents);
                    free(tokens);
                    return -1;
                }
                {
                    int nxt = -1;
                    for (j = 1; j < n; j++) {
                        if (parents[j] == cur && tokens[j] == t) {
                            nxt = j;
                            break;
                        }
                    }
                    if (nxt < 0) {
                        nxt = n;
                        parents[n] = cur;
                        tokens[n] = t;
                        n++;
                    }
                    cur = nxt;
                }
            }
        }
    }
    {
        int *has_child = (int *)calloc((size_t)n, sizeof(int));
        int *leaves = (int *)malloc(sizeof(int) * (size_t)n);
        int n_leaves = 0, k;
        int total_path = 0, off = 0;
        int *paths = NULL, *path_lens = NULL;
        if (!has_child || !leaves) {
            free(has_child);
            free(leaves);
            free(parents);
            free(tokens);
            return -1;
        }
        for (i = 1; i < n; i++) has_child[parents[i]] = 1;
        for (i = 1; i < n; i++) {
            if (!has_child[i]) {
                int depth = 0, c = i;
                while (c != 0) {
                    depth++;
                    c = parents[c];
                }
                leaves[n_leaves++] = i;
                total_path += depth;
            }
        }
        paths = (int *)malloc(sizeof(int) * (size_t)(total_path ? total_path : 1));
        path_lens = (int *)malloc(sizeof(int) * (size_t)(n_leaves ? n_leaves : 1));
        if (!paths || !path_lens) {
            free(paths);
            free(path_lens);
            free(has_child);
            free(leaves);
            free(parents);
            free(tokens);
            return -1;
        }
        for (k = 0; k < n_leaves; k++) {
            int c = leaves[k];
            int depth = 0;
            int tmp[4096];
            while (c != 0) {
                if (depth >= 4096) {
                    free(paths);
                    free(path_lens);
                    free(has_child);
                    free(leaves);
                    free(parents);
                    free(tokens);
                    return -1;
                }
                tmp[depth++] = c;
                c = parents[c];
            }
            path_lens[k] = depth;
            for (j = 0; j < depth; j++) paths[off++] = tmp[depth - 1 - j];
        }
        *out_parents = parents;
        *out_tokens = tokens;
        *out_leaves = leaves;
        *out_n_nodes = n;
        *out_n_leaves = n_leaves;
        *out_paths = paths;
        *out_path_lens = path_lens;
        free(has_child);
        return 0;
    }
}

void ai_draft_tree_free(int *parents, int *tokens, int *leaves,
                        int *paths, int *path_lens)
{
    free(parents);
    free(tokens);
    free(leaves);
    free(paths);
    free(path_lens);
}

/* 判断 j 是否为 i 的祖先或自身。 */
static int is_visible(const int *parents, int i, int j)
{
    int cur = i;
    while (cur != -1) {
        if (cur == j) return 1;
        cur = parents[cur];
    }
    return 0;
}

int ai_tree_attention_bias(const int *parents, int n, double neg_inf,
                           double *bias)
{
    int i, j;
    if (!parents || n < 1 || !bias || !(neg_inf < 0.0) || !isfinite(neg_inf))
        return -1;
    if (parents[0] != -1) return -1;
    for (i = 1; i < n; i++) {
        if (parents[i] < -1 || parents[i] >= i) return -1;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            bias[(size_t)i * (size_t)n + j] =
                is_visible(parents, i, j) ? 0.0 : neg_inf;
        }
    }
    return 0;
}

int ai_tree_attention_mask(const int *parents, int n, int *mask)
{
    int i, j;
    if (!parents || n < 1 || !mask) return -1;
    if (parents[0] != -1) return -1;
    for (i = 1; i < n; i++) {
        if (parents[i] < -1 || parents[i] >= i) return -1;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mask[(size_t)i * (size_t)n + j] = is_visible(parents, i, j);
        }
    }
    return 0;
}
