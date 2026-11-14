/*
 * kv_evict.c —— KV cache 淘汰策略的 C 实现（与 AI.py 逐位一致）
 *
 * 三个策略共享同一套「先标记强制保留、再按分数选 top」的逻辑；
 * 选择时按 (-score, index) 排序（分数相同取更靠前的下标），结果升序。
 */
#include "kv_evict.h"

#include <stdlib.h>
#include <string.h>

static void mark_forced(char *keep, int seq_len, int sink, int recent)
{
    for (int i = 0; i < sink && i < seq_len; i++) keep[i] = 1;
    for (int i = seq_len - recent; i < seq_len; i++)
        if (i >= 0) keep[i] = 1;
}

/* 从 agg 中按 (-score, index) 选 count 个未强制保留的位置。 */
static void select_top(const double *agg, int seq_len, char *keep, int count)
{
    int *cand = (int *)malloc(sizeof(int) * (size_t)(seq_len ? seq_len : 1));
    if (!cand) return;
    int m = 0;
    for (int j = 0; j < seq_len; j++)
        if (!keep[j]) cand[m++] = j;
    for (int i = 1; i < m; i++) { /* 插入排序：分数降序、下标升序 */
        int v = cand[i];
        int k = i - 1;
        while (k >= 0 &&
               (agg[cand[k]] < agg[v] ||
                (agg[cand[k]] == agg[v] && cand[k] > v))) {
            cand[k + 1] = cand[k];
            k--;
        }
        cand[k + 1] = v;
    }
    for (int i = 0; i < count && i < m; i++) keep[cand[i]] = 1;
    free(cand);
}

static double *column_sums(const double *scores, int nq, int seq_len)
{
    double *agg = (double *)calloc((size_t)(seq_len ? seq_len : 1),
                                   sizeof(double));
    if (!agg) return NULL;
    for (int r = 0; r < nq; r++)
        for (int j = 0; j < seq_len; j++)
            agg[j] += scores[(size_t)r * (size_t)seq_len + (size_t)j];
    return agg;
}

int ai_streaming_kv_keep(int seq_len, int budget, int sink, int *out)
{
    int recent = budget - sink;
    int n = 0;
    for (int i = 0; i < sink; i++) out[n++] = i;
    for (int i = seq_len - recent; i < seq_len; i++) out[n++] = i;
    return n;
}

int ai_h2o_kv_keep(const double *scores, int nq, int seq_len, int budget,
                   int sink, int recent, int *out)
{
    char *keep = (char *)calloc((size_t)(seq_len ? seq_len : 1), 1);
    if (!keep) return 0;
    mark_forced(keep, seq_len, sink, recent);
    double *agg = column_sums(scores, nq, seq_len);
    if (!agg) { free(keep); return 0; }
    int forced = 0;
    for (int j = 0; j < seq_len; j++) forced += keep[j] ? 1 : 0;
    select_top(agg, seq_len, keep, budget - forced);
    int n = 0;
    for (int j = 0; j < seq_len; j++)
        if (keep[j]) out[n++] = j;
    free(agg);
    free(keep);
    return n;
}

int ai_snapkv_kv_keep(const double *scores, int nq, int seq_len, int budget,
                      int sink, int obs, int *out)
{
    int start = 0;
    if (obs <= 0) {
        obs = nq / 4;
        if (obs < 1) obs = 1;
    }
    start = nq - obs;
    char *keep = (char *)calloc((size_t)(seq_len ? seq_len : 1), 1);
    if (!keep) return 0;
    mark_forced(keep, seq_len, sink, 0);
    double *agg = column_sums(scores + (size_t)start * (size_t)seq_len,
                              obs, seq_len);
    if (!agg) { free(keep); return 0; }
    int forced = 0;
    for (int j = 0; j < seq_len; j++) forced += keep[j] ? 1 : 0;
    select_top(agg, seq_len, keep, budget - forced);
    int n = 0;
    for (int j = 0; j < seq_len; j++)
        if (keep[j]) out[n++] = j;
    free(agg);
    free(keep);
    return n;
}

int ai_kv_cache_evict(const double *keys, const double *values,
                      int seq_len, int dim, int budget,
                      const double *scores, int nq, int policy,
                      int sink, int recent, int obs,
                      int *out_keep, double *out_keys, double *out_values)
{
    int *keep = (int *)malloc(sizeof(int) * (size_t)(budget ? budget : 1));
    if (!keep) return 0;
    int n = 0;
    if (budget == seq_len) {
        for (int j = 0; j < seq_len; j++) keep[n++] = j;
    } else if (policy == 0) {
        n = ai_streaming_kv_keep(seq_len, budget, sink, keep);
    } else if (policy == 1) {
        n = ai_h2o_kv_keep(scores, nq, seq_len, budget, sink, recent, keep);
    } else {
        n = ai_snapkv_kv_keep(scores, nq, seq_len, budget, sink, obs, keep);
    }
    for (int i = 0; i < n; i++) {
        int j = keep[i];
        out_keep[i] = j;
        for (int d = 0; d < dim; d++) {
            out_keys[(size_t)i * (size_t)dim + (size_t)d] =
                keys[(size_t)j * (size_t)dim + (size_t)d];
            out_values[(size_t)i * (size_t)dim + (size_t)d] =
                values[(size_t)j * (size_t)dim + (size_t)d];
        }
    }
    free(keep);
    return n;
}
