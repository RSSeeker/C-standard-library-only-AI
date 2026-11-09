/*
 * ctc_beam.c —— CTC 前缀 beam search（与 Python AI.py 对齐）
 *
 * 算法（Graves 等）：每个前缀维护两个以 log 概率表示的状态
 *   p_b  = 以 blank 结尾的概率和
 *   p_nb = 以非 blank 结尾的概率和
 * 逐帧扩展后按 log_sum_exp(p_b,p_nb) 排序，保留 beam_width 个前缀；
 * 最终用可选的字符级 LM 与长度惩罚重排取最优。
 */
#include "ctc_beam.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CTC_NEG (-INFINITY)

typedef struct {
    int n;
    int *tokens;      /* 长度 max_len 的容量 */
    double p_b;
    double p_nb;
} CBeam;

typedef struct {
    CBeam *items;
    int count;
    int cap;
    int max_len;
} CBeamList;

static double ctc_lse(double a, double b)
{
    if (a == CTC_NEG) return b;
    if (b == CTC_NEG) return a;
    {
        const double m = (a > b) ? a : b;
        return m + log(exp(a - m) + exp(b - m));
    }
}

static double ctc_total(const CBeam *b)
{
    return ctc_lse(b->p_b, b->p_nb);
}

static int beamlist_init(CBeamList *l, int cap, int max_len)
{
    if (!l) return -1;
    l->count = 0;
    l->cap = (cap > 0) ? cap : 1;
    l->max_len = (max_len > 0) ? max_len : 1;
    l->items = (CBeam *)calloc((size_t)l->cap, sizeof(CBeam));
    return l->items ? 0 : -1;
}

static void beam_free(CBeam *b)
{
    if (b) {
        free(b->tokens);
        b->tokens = NULL;
        b->n = 0;
    }
}

static void beamlist_free(CBeamList *l)
{
    if (!l || !l->items) return;
    for (int i = 0; i < l->count; i++) beam_free(&l->items[i]);
    free(l->items);
    l->items = NULL;
    l->count = 0;
    l->cap = 0;
}

static int beamlist_reserve(CBeamList *l, int need)
{
    if (need <= l->cap) return 0;
    {
        int cap = l->cap;
        CBeam *items;
        while (cap < need) cap *= 2;
        items = (CBeam *)realloc(l->items, (size_t)cap * sizeof(CBeam));
        if (!items) return -1;
        memset(items + l->cap, 0, (size_t)(cap - l->cap) * sizeof(CBeam));
        l->items = items;
        l->cap = cap;
    }
    return 0;
}

static int beamlist_find(const CBeamList *l, const int *tokens, int n)
{
    for (int i = 0; i < l->count; i++) {
        const CBeam *b = &l->items[i];
        if (b->n != n) continue;
        if (n == 0) return i;
        if (memcmp(b->tokens, tokens, (size_t)n * sizeof(int)) == 0) return i;
    }
    return -1;
}

/* 把 (tokens, n) 前缀的概率端点并入列表（不存在则新建）。 */
static int beamlist_merge(CBeamList *l, const int *tokens, int n,
                          double p_b, double p_nb)
{
    int idx = beamlist_find(l, tokens, n);
    if (idx >= 0) {
        CBeam *b = &l->items[idx];
        b->p_b = ctc_lse(b->p_b, p_b);
        b->p_nb = ctc_lse(b->p_nb, p_nb);
        return 0;
    }
    if (beamlist_reserve(l, l->count + 1) != 0) return -1;
    {
        CBeam *b = &l->items[l->count];
        if (n > l->max_len) return -1;
        b->tokens = (int *)malloc((size_t)l->max_len * sizeof(int));
        if (!b->tokens) return -1;
        if (n > 0) memcpy(b->tokens, tokens, (size_t)n * sizeof(int));
        b->n = n;
        b->p_b = p_b;
        b->p_nb = p_nb;
        l->count++;
    }
    return 0;
}

/* 稳定降序排序：分数相等时保持插入顺序（与 Python 的 sorted 稳定行为一致）。 */
static void beamlist_sort_desc(CBeamList *l)
{
    for (int i = 1; i < l->count; i++) {
        CBeam tmp = l->items[i];
        int j = i - 1;
        while (j >= 0 && ctc_total(&l->items[j]) < ctc_total(&tmp)) {
            l->items[j + 1] = l->items[j];
            j--;
        }
        l->items[j + 1] = tmp;
    }
}

static void beamlist_truncate(CBeamList *l, int keep)
{
    if (keep >= l->count) return;
    for (int i = keep; i < l->count; i++) beam_free(&l->items[i]);
    l->count = keep;
}

static int check_log_probs(const double *log_probs, int T, int K, int blank)
{
    if (!log_probs || T < 1 || K < 1) return -1;
    if (blank < 0 || blank >= K) return -1;
    for (int i = 0; i < T * K; i++) {
        const double v = log_probs[i];
        if (isnan(v) || v == INFINITY) return -1;
        /* -inf 与有限值均合法 */
    }
    return 0;
}

static int check_common(int T, int K, int beam_width,
                        double lm_weight, double length_penalty,
                        CtcBeamResult *out)
{
    if (T < 1 || K < 1) return -1;
    if (beam_width < 1) return -1;
    if (isnan(lm_weight) || !isfinite(lm_weight)) return -1;
    if (isnan(length_penalty) || !isfinite(length_penalty)) return -1;
    if (!out) return -1;
    return 0;
}

int ctc_prefix_beam_search(const double *log_probs, int T, int K, int blank,
                           int beam_width, CtcLmFn lm, void *lm_ctx,
                           double lm_weight, double length_penalty,
                           CtcBeamResult *out)
{
    CBeamList beams = {0, 0, 0, 0};
    int *scratch = NULL;
    int rc = -1;

    if (check_common(T, K, beam_width, lm_weight, length_penalty, out) != 0)
        return -1;
    memset(out, 0, sizeof(*out));
    if (check_log_probs(log_probs, T, K, blank) != 0) return -1;

    scratch = (int *)malloc((size_t)(T + 1) * sizeof(int));
    if (!scratch) return -1;
    if (beamlist_init(&beams, 1, T) != 0) {
        free(scratch);
        return -1;
    }
    if (beamlist_merge(&beams, NULL, 0, 0.0, CTC_NEG) != 0) goto done;

    for (int t = 0; t < T; t++) {
        CBeamList next = {0, 0, 0, 0};
        const int cap = (beams.count > 0 ? beams.count : 1) * K + 1;
        if (beamlist_init(&next, cap, T) != 0) goto done;

        for (int i = 0; i < beams.count; i++) {
            const CBeam *b = &beams.items[i];
            const double p_total = ctc_total(b);
            for (int k = 0; k < K; k++) {
                const double lp = log_probs[(size_t)t * (size_t)K + (size_t)k];
                if (lp == CTC_NEG) continue;
                if (k == blank) {
                    if (beamlist_merge(&next, b->tokens, b->n,
                                       p_total + lp, CTC_NEG) != 0) {
                        beamlist_free(&next);
                        goto done;
                    }
                } else if (b->n > 0 && k == b->tokens[b->n - 1]) {
                    /* 同字符重复：保持原前缀（非 blank 端）+ 扩展新前缀（blank 端） */
                    if (beamlist_merge(&next, b->tokens, b->n,
                                       CTC_NEG, b->p_nb + lp) != 0) {
                        beamlist_free(&next);
                        goto done;
                    }
                    if (b->n + 1 > T) {
                        beamlist_free(&next);
                        goto done;
                    }
                    if (b->n > 0)
                        memcpy(scratch, b->tokens, (size_t)b->n * sizeof(int));
                    scratch[b->n] = k;
                    if (beamlist_merge(&next, scratch, b->n + 1,
                                       CTC_NEG, b->p_b + lp) != 0) {
                        beamlist_free(&next);
                        goto done;
                    }
                } else {
                    if (b->n + 1 > T) {
                        beamlist_free(&next);
                        goto done;
                    }
                    if (b->n > 0)
                        memcpy(scratch, b->tokens, (size_t)b->n * sizeof(int));
                    scratch[b->n] = k;
                    if (beamlist_merge(&next, scratch, b->n + 1,
                                       CTC_NEG, p_total + lp) != 0) {
                        beamlist_free(&next);
                        goto done;
                    }
                }
            }
        }

        if (next.count == 0) {
            beamlist_free(&next);
            break;
        }
        beamlist_sort_desc(&next);
        beamlist_truncate(&next, beam_width);
        beamlist_free(&beams);
        beams = next;
    }

    if (beams.count == 0) {
        out->tokens = NULL;
        out->n_tokens = 0;
        out->score = 0.0;
        out->ctc_score = 0.0;
        out->lm_score = 0.0;
        rc = 0;
        goto done;
    }

    {
        /* 逐个前缀算 LM 与总分，再用稳定排序取最优 */
        double *totals = (double *)malloc((size_t)beams.count * sizeof(double));
        if (!totals) goto done;
        for (int i = 0; i < beams.count; i++) {
            CBeam *b = &beams.items[i];
            double ctc = ctc_total(b);
            double lm_score = 0.0;
            if (lm) {
                lm_score = lm(b->tokens, b->n, lm_ctx);
                if (isnan(lm_score) || !isfinite(lm_score)) {
                    free(totals);
                    goto done;
                }
            }
            totals[i] = ctc + lm_weight * lm_score
                        + length_penalty * (double)b->n;
        }
        for (int i = 1; i < beams.count; i++) {
            const double tmp_total = totals[i];
            CBeam tmp_beam = beams.items[i];
            int j = i - 1;
            while (j >= 0 && totals[j] < tmp_total) {
                totals[j + 1] = totals[j];
                beams.items[j + 1] = beams.items[j];
                j--;
            }
            totals[j + 1] = tmp_total;
            beams.items[j + 1] = tmp_beam;
        }
        {
            const CBeam *best = &beams.items[0];
            const double best_ctc = ctc_total(best);
            double best_lm = 0.0;
            if (lm) best_lm = lm(best->tokens, best->n, lm_ctx);
            out->tokens = (int *)malloc((size_t)(best->n > 0 ? best->n : 1)
                                        * sizeof(int));
            if (!out->tokens) {
                free(totals);
                goto done;
            }
            if (best->n > 0)
                memcpy(out->tokens, best->tokens,
                       (size_t)best->n * sizeof(int));
            out->n_tokens = best->n;
            out->ctc_score = best_ctc;
            out->lm_score = best_lm;
            out->score = totals[0];
        }
        free(totals);
        rc = 0;
    }

done:
    beamlist_free(&beams);
    free(scratch);
    return rc;
}

int ctc_beam_decode(const double *logits, int T, int K, int blank,
                    int beam_width, CtcLmFn lm, void *lm_ctx,
                    double lm_weight, double length_penalty,
                    CtcBeamResult *out)
{
    double *log_probs = NULL;
    int rc = -1;

    if (check_common(T, K, beam_width, lm_weight, length_penalty, out) != 0)
        return -1;
    if (!logits || blank < 0 || blank >= K) return -1;
    for (int i = 0; i < T * K; i++) {
        if (!isfinite(logits[i])) return -1;
    }
    log_probs = (double *)malloc((size_t)T * (size_t)K * sizeof(double));
    if (!log_probs) return -1;
    for (int t = 0; t < T; t++) {
        const double *row = logits + (size_t)t * (size_t)K;
        double m = row[0];
        for (int k = 1; k < K; k++) if (row[k] > m) m = row[k];
        double sum = 0.0;
        for (int k = 0; k < K; k++) sum += exp(row[k] - m);
        const double lse = m + log(sum);
        for (int k = 0; k < K; k++)
            log_probs[(size_t)t * (size_t)K + (size_t)k] = row[k] - lse;
    }
    rc = ctc_prefix_beam_search(log_probs, T, K, blank, beam_width,
                                lm, lm_ctx, lm_weight, length_penalty, out);
    free(log_probs);
    return rc;
}

void ctc_beam_result_free(CtcBeamResult *r)
{
    if (!r) return;
    free(r->tokens);
    memset(r, 0, sizeof(*r));
}
