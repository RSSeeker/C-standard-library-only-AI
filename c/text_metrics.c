/*
 * text_metrics.c —— 文本质量指标（METEOR / chrF / WER / CER）
 *
 * 与 AI.py 的同名实现逐位一致（双精度、同累加顺序），对拍见
 * c/tools/text_metrics_reference.py 与 c/tests/compare_text_metrics.c。
 *
 * 设计取舍：
 *   - 不依赖动态容器，全部用 malloc/realloc 一次算清；规模都在句子级别。
 *   - 字符串切分只认 ASCII 空白（Python 的 str.split() 还认 Unicode 空白，
 *     教学实现不覆盖；注释里已声明边界）。
 *   - 词干 / lower 只处理 ASCII 字母，非 ASCII 原样保留（中文用例照常可跑）。
 */
#include "text_metrics.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "text.h"

/* ---------- 小工具 ---------- */

static int tm_isspace_ascii(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static char *tm_strdup_lower(const char *s)
{
    if (!s) s = "";
    const size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        out[i] = (c < 0x80) ? (char)tolower(c) : (char)c;
    }
    out[n] = '\0';
    return out;
}

/* 按 ASCII 空白切分成 token 数组（对应 Python 的 str.split()，只覆盖 ASCII 空白） */
static char **tm_split_ascii(const char *s, int *out_n)
{
    *out_n = 0;
    if (!s) return NULL;
    int cap = 8;
    char **tokens = (char **)malloc(sizeof(char *) * (size_t)cap);
    if (!tokens) return NULL;
    const char *p = s;
    while (*p) {
        while (*p && tm_isspace_ascii((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && !tm_isspace_ascii((unsigned char)*p)) p++;
        const size_t len = (size_t)(p - start);
        char *tok = (char *)malloc(len + 1);
        if (!tok) {
            for (int i = 0; i < *out_n; i++) free(tokens[i]);
            free(tokens);
            return NULL;
        }
        memcpy(tok, start, len);
        tok[len] = '\0';
        if (*out_n == cap) {
            cap *= 2;
            char **grown = (char **)realloc(tokens, sizeof(char *) * (size_t)cap);
            if (!grown) {
                free(tok);
                for (int i = 0; i < *out_n; i++) free(tokens[i]);
                free(tokens);
                return NULL;
            }
            tokens = grown;
        }
        tokens[(*out_n)++] = tok;
    }
    return tokens;
}

static void tm_free_tokens(char **tokens, int n)
{
    if (!tokens) return;
    for (int i = 0; i < n; i++) free(tokens[i]);
    free(tokens);
}

/* UTF-8 解码成码点数组；返回码点个数，失败返回 -1 */
static int tm_to_codepoints(const char *s, int **out)
{
    *out = NULL;
    if (!s) s = "";
    const size_t bytes = strlen(s);
    int *cps = (int *)malloc(sizeof(int) * (bytes + 1));
    if (!cps) return -1;
    int n = 0;
    const char *p = s;
    while (*p) {
        int cp = 0;
        const int used = utf8_decode(p, &cp);
        if (used <= 0) {
            free(cps);
            return -1;
        }
        cps[n++] = cp;
        p += used;
    }
    *out = cps;
    return n;
}

/* Python 的 " ".join(s.lower().split())：lower(ASCII) -> 空白折叠为单个空格 -> 去首尾 */
static int tm_normalize_join(const char *s, int **out)
{
    *out = NULL;
    int n_cp = 0;
    int *cps = NULL;
    int tmp_n = tm_to_codepoints(s, &cps);
    if (tmp_n < 0) return -1;
    int *buf = (int *)malloc(sizeof(int) * (size_t)(tmp_n + 1));
    if (!buf) {
        free(cps);
        return -1;
    }
    int n = 0;
    int pending_space = 0;
    for (int i = 0; i < tmp_n; i++) {
        int cp = cps[i];
        if (cp < 0x80) cp = tolower((unsigned char)cp);
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f') {
            if (n > 0) pending_space = 1;
            continue;
        }
        if (pending_space) {
            buf[n++] = ' ';
            pending_space = 0;
        }
        buf[n++] = cp;
    }
    free(cps);
    n_cp = n;
    *out = buf;
    return n_cp;
}

/* 把 token 数组小写后以单空格 join 成码点序列（对应 Python 的 list 输入） */
static int tm_join_tokens(const char *const *tokens, int n_tokens, int **out)
{
    *out = NULL;
    int total = 0;
    int **parts = (int **)malloc(sizeof(int *) * (size_t)(n_tokens > 0 ? n_tokens : 1));
    int *lens = (int *)malloc(sizeof(int) * (size_t)(n_tokens > 0 ? n_tokens : 1));
    if (!parts || !lens) {
        free(parts);
        free(lens);
        return -1;
    }
    for (int i = 0; i < n_tokens; i++) {
        parts[i] = NULL;
        lens[i] = tm_to_codepoints(tokens[i], &parts[i]);
        if (lens[i] < 0) {
            for (int k = 0; k < i; k++) free(parts[k]);
            free(parts);
            free(lens);
            return -1;
        }
        for (int k = 0; k < lens[i]; k++) {
            if (parts[i][k] < 0x80) parts[i][k] = tolower((unsigned char)parts[i][k]);
        }
        total += lens[i] + (i > 0 ? 1 : 0);
    }
    int *buf = (int *)malloc(sizeof(int) * (size_t)(total + 1));
    if (!buf) {
        for (int i = 0; i < n_tokens; i++) free(parts[i]);
        free(parts);
        free(lens);
        return -1;
    }
    int pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) buf[pos++] = ' ';
        memcpy(buf + pos, parts[i], sizeof(int) * (size_t)lens[i]);
        pos += lens[i];
        free(parts[i]);
    }
    free(parts);
    free(lens);
    *out = buf;
    return total;
}

/* ---------- METEOR ---------- */

void tm_stem(const char *token, char *out, int out_size)
{
    if (!out || out_size <= 0) return;
    char *low = tm_strdup_lower(token);
    if (!low) {
        out[0] = '\0';
        return;
    }
    static const char *const suffixes[] = {"ing", "edly", "ed", "es", "s", "ly", "er", "est"};
    const int n_suf = (int)(sizeof(suffixes) / sizeof(suffixes[0]));
    const int len = (int)strlen(low);
    for (int k = 0; k < n_suf; k++) {
        const int sl = (int)strlen(suffixes[k]);
        if (len > sl + 2 && strcmp(low + len - sl, suffixes[k]) == 0) {
            low[len - sl] = '\0';
            break;
        }
    }
    snprintf(out, (size_t)out_size, "%s", low);
    free(low);
}

double tm_meteor_score(const char *const *pred, int n_pred,
                       const char *const *ref, int n_ref,
                       double alpha, double beta, double gamma)
{
    if (!(alpha >= 0.0 && alpha <= 1.0) || !(beta >= 0.0) || !(gamma >= 0.0)) return NAN;
    if (n_pred <= 0 || n_ref <= 0) return 0.0;
    unsigned char *used = (unsigned char *)calloc((size_t)n_ref, 1);
    if (!used) return NAN;
    int matched = 0;
    int chunks = 0;
    int prev = 0;
    for (int i = 0; i < n_pred; i++) {
        int hit = -1;
        for (int j = 0; j < n_ref; j++) {
            if (used[j]) continue;
            if (strcmp(pred[i], ref[j]) == 0) {
                hit = j;
                break;
            }
        }
        if (hit >= 0) {
            used[hit] = 1;
            matched++;
            if (!prev) chunks++;
            prev = 1;
        } else {
            prev = 0;
        }
    }
    free(used);
    if (matched == 0) return 0.0;
    const double precision = (double)matched / (double)n_pred;
    const double recall = (double)matched / (double)n_ref;
    const double denom = alpha * precision + (1.0 - alpha) * recall;
    if (denom <= 0.0) return 0.0;
    const double f_mean = (precision * recall) / denom;
    const double penalty = gamma * pow((double)chunks / (double)matched, beta);
    return f_mean * (1.0 - penalty);
}

/* 同义词组里是否包含某个 token（两边都按 ASCII 小写比较） */
static int tm_group_has(const TmSynonymGroup *g, const char *norm_token)
{
    for (int i = 0; i < g->n_tokens; i++) {
        char *low = tm_strdup_lower(g->tokens[i]);
        if (!low) return 0;
        const int eq = strcmp(low, norm_token) == 0;
        free(low);
        if (eq) return 1;
    }
    return 0;
}

static int tm_find_group(const TmSynonyms *syn, const char *norm_token)
{
    if (!syn) return -1;
    for (int g = 0; g < syn->n_groups; g++) {
        if (tm_group_has(&syn->groups[g], norm_token)) return g;
    }
    return -1;
}

static int tm_jaccard_cmp(const void *a, const void *b)
{
    const int *x = (const int *)a;
    const int *y = (const int *)b;
    if (x[0] != y[0]) return (x[0] < y[0]) ? -1 : 1;
    if (x[1] != y[1]) return (x[1] < y[1]) ? -1 : 1;
    return 0;
}

double tm_meteor_score_full(const char *const *pred, int n_pred,
                            const char *const *ref, int n_ref,
                            double alpha, double beta, double gamma,
                            const TmSynonyms *syn, int use_stem)
{
    if (!(alpha >= 0.0 && alpha <= 1.0) || !(beta >= 0.0) || !(gamma >= 0.0)) return NAN;
    if (n_pred <= 0 || n_ref <= 0) return 0.0;

    char **np = (char **)calloc((size_t)n_pred, sizeof(char *));
    char **nr = (char **)calloc((size_t)n_ref, sizeof(char *));
    unsigned char *used = (unsigned char *)calloc((size_t)n_ref, 1);
    int cap = n_pred < n_ref ? n_pred : n_ref;
    int(*pairs)[2] = (int(*)[2])malloc(sizeof(int[2]) * (size_t)(cap > 0 ? cap : 1));
    if (!np || !nr || !used || !pairs) {
        if (np) { for (int i = 0; i < n_pred; i++) free(np[i]); free(np); }
        if (nr) { for (int i = 0; i < n_ref; i++) free(nr[i]); free(nr); }
        free(used);
        free(pairs);
        return NAN;
    }
    for (int i = 0; i < n_pred; i++) np[i] = tm_strdup_lower(pred[i]);
    for (int j = 0; j < n_ref; j++) nr[j] = tm_strdup_lower(ref[j]);
    int n_pairs = 0;

    /* 第一遍：精确匹配（小写后比较） */
    for (int i = 0; i < n_pred; i++) {
        int hit = -1;
        for (int j = 0; j < n_ref; j++) {
            if (used[j]) continue;
            if (strcmp(np[i], nr[j]) == 0) { hit = j; break; }
        }
        if (hit >= 0) {
            used[hit] = 1;
            pairs[n_pairs][0] = i;
            pairs[n_pairs][1] = hit;
            n_pairs++;
        }
    }
    /* 第二遍：词干匹配 */
    if (use_stem) {
        for (int i = 0; i < n_pred; i++) {
            int already = 0;
            for (int k = 0; k < n_pairs; k++) {
                if (pairs[k][0] == i) { already = 1; break; }
            }
            if (already) continue;
            char sp[64] = {0};
            tm_stem(np[i], sp, (int)sizeof sp);
            int hit = -1;
            for (int j = 0; j < n_ref; j++) {
                if (used[j]) continue;
                char sr[64] = {0};
                tm_stem(nr[j], sr, (int)sizeof sr);
                if (strcmp(sp, sr) == 0) { hit = j; break; }
            }
            if (hit >= 0) {
                used[hit] = 1;
                pairs[n_pairs][0] = i;
                pairs[n_pairs][1] = hit;
                n_pairs++;
            }
        }
    }
    /* 第三遍：同义词匹配 */
    if (syn && syn->n_groups > 0) {
        for (int i = 0; i < n_pred; i++) {
            int already = 0;
            for (int k = 0; k < n_pairs; k++) {
                if (pairs[k][0] == i) { already = 1; break; }
            }
            if (already) continue;
            const int g = tm_find_group(syn, np[i]);
            if (g < 0) continue;
            int hit = -1;
            for (int j = 0; j < n_ref; j++) {
                if (used[j]) continue;
                if (tm_group_has(&syn->groups[g], nr[j])) { hit = j; break; }
            }
            if (hit >= 0) {
                used[hit] = 1;
                pairs[n_pairs][0] = i;
                pairs[n_pairs][1] = hit;
                n_pairs++;
            }
        }
    }
    /* 恢复 (候选下标, 参考下标) 的字典序（Python 里 pairs.sort()） */
    qsort(pairs, (size_t)n_pairs, sizeof pairs[0], tm_jaccard_cmp);

    int chunks = 0;
    int prev_i = -1, prev_j = -1;
    for (int k = 0; k < n_pairs; k++) {
        if (prev_i < 0 || pairs[k][0] != prev_i + 1 || pairs[k][1] != prev_j + 1) chunks++;
        prev_i = pairs[k][0];
        prev_j = pairs[k][1];
    }
    const int matched = n_pairs;

    for (int i = 0; i < n_pred; i++) free(np[i]);
    for (int j = 0; j < n_ref; j++) free(nr[j]);
    free(np);
    free(nr);
    free(used);
    free(pairs);

    if (matched == 0) return 0.0;
    const double precision = (double)matched / (double)n_pred;
    const double recall = (double)matched / (double)n_ref;
    const double denom = alpha * precision + (1.0 - alpha) * recall;
    if (denom <= 0.0) return 0.0;
    const double f_mean = (precision * recall) / denom;
    const double penalty = gamma * pow((double)chunks / (double)matched, beta);
    const double score = f_mean * (1.0 - penalty);
    return score > 0.0 ? score : 0.0;
}

/* ---------- chrF ---------- */

/* 码点序列上的字符 n-gram F-beta 平均（1..max_n） */
static double tm_chrf_cps(const int *p, int n_p, const int *r, int n_r,
                          int max_n, double beta)
{
    if (max_n < 1 || !(beta > 0.0)) return NAN;
    if (n_p <= 0 || n_r <= 0) return 0.0;
    const double b2 = beta * beta;
    double total = 0.0;
    int used_n = 0;
    for (int n = 1; n <= max_n; n++) {
        if (n > n_p || n > n_r) continue;
        const int pn = n_p - n + 1;
        const int rn = n_r - n + 1;
        if (pn <= 0 || rn <= 0) continue;
        unsigned char *used = (unsigned char *)calloc((size_t)rn, 1);
        if (!used) return NAN;
        int overlap = 0;
        for (int i = 0; i < pn; i++) {
            for (int j = 0; j < rn; j++) {
                if (used[j]) continue;
                if (memcmp(p + i, r + j, sizeof(int) * (size_t)n) == 0) {
                    used[j] = 1;
                    overlap++;
                    break;
                }
            }
        }
        free(used);
        const double precision = (double)overlap / (double)pn;
        const double recall = (double)overlap / (double)rn;
        if (precision + recall <= 0.0) continue;
        total += ((1.0 + b2) * precision * recall) / (b2 * precision + recall);
        used_n++;
    }
    if (used_n == 0) return 0.0;
    return total / (double)used_n;
}

double tm_chrf_score(const char *pred, const char *ref, int max_n, double beta)
{
    if (max_n < 1 || !(beta > 0.0)) return NAN;
    int *p = NULL;
    int *r = NULL;
    const int n_p = tm_normalize_join(pred, &p);
    const int n_r = tm_normalize_join(ref, &r);
    if (n_p < 0 || n_r < 0) {
        free(p);
        free(r);
        return NAN;
    }
    const double score = tm_chrf_cps(p, n_p, r, n_r, max_n, beta);
    free(p);
    free(r);
    return score;
}

double tm_chrf_score_tokens(const char *const *pred, int n_pred,
                            const char *const *ref, int n_ref,
                            int max_n, double beta)
{
    if (max_n < 1 || !(beta > 0.0)) return NAN;
    int *p = NULL;
    int *r = NULL;
    const int n_p = tm_join_tokens(pred, n_pred, &p);
    const int n_r = tm_join_tokens(ref, n_ref, &r);
    if (n_p < 0 || n_r < 0) {
        free(p);
        free(r);
        return NAN;
    }
    const double score = tm_chrf_cps(p, n_p, r, n_r, max_n, beta);
    free(p);
    free(r);
    return score;
}

/* ---------- 编辑距离 / WER / CER ---------- */

void tm_edit_ops_tokens(const char *const *reference, int n_ref,
                        const char *const *hypothesis, int n_hyp,
                        int *subs, int *dels, int *ins)
{
    if (subs) *subs = 0;
    if (dels) *dels = 0;
    if (ins) *ins = 0;
    if (n_ref < 0 || n_hyp < 0) return;
    const int cols = n_hyp + 1;
    int *dp = (int *)malloc(sizeof(int) * (size_t)((n_ref + 1) * cols));
    if (!dp) return;
#define TM_DP(i, j) dp[(size_t)(i) * (size_t)cols + (size_t)(j)]
    for (int i = 0; i <= n_ref; i++) TM_DP(i, 0) = i;
    for (int j = 0; j <= n_hyp; j++) TM_DP(0, j) = j;
    for (int i = 1; i <= n_ref; i++) {
        for (int j = 1; j <= n_hyp; j++) {
            if (strcmp(reference[i - 1], hypothesis[j - 1]) == 0) {
                TM_DP(i, j) = TM_DP(i - 1, j - 1);
            } else {
                int best = TM_DP(i - 1, j - 1);
                if (TM_DP(i - 1, j) < best) best = TM_DP(i - 1, j);
                if (TM_DP(i, j - 1) < best) best = TM_DP(i, j - 1);
                TM_DP(i, j) = 1 + best;
            }
        }
    }
    int i = n_ref, j = n_hyp;
    int s = 0, d = 0, in = 0;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && strcmp(reference[i - 1], hypothesis[j - 1]) == 0) {
            i--;
            j--;
        } else if (i > 0 && j > 0 && TM_DP(i, j) == TM_DP(i - 1, j - 1) + 1) {
            s++;
            i--;
            j--;
        } else if (i > 0 && TM_DP(i, j) == TM_DP(i - 1, j) + 1) {
            d++;
            i--;
        } else {
            in++;
            j--;
        }
    }
#undef TM_DP
    free(dp);
    if (subs) *subs = s;
    if (dels) *dels = d;
    if (ins) *ins = in;
}

void tm_edit_ops_int(const int *reference, int n_ref,
                     const int *hypothesis, int n_hyp,
                     int *subs, int *dels, int *ins)
{
    if (subs) *subs = 0;
    if (dels) *dels = 0;
    if (ins) *ins = 0;
    if (n_ref < 0 || n_hyp < 0) return;
    const int cols = n_hyp + 1;
    int *dp = (int *)malloc(sizeof(int) * (size_t)((n_ref + 1) * cols));
    if (!dp) return;
#define TM_DP(i, j) dp[(size_t)(i) * (size_t)cols + (size_t)(j)]
    for (int i = 0; i <= n_ref; i++) TM_DP(i, 0) = i;
    for (int j = 0; j <= n_hyp; j++) TM_DP(0, j) = j;
    for (int i = 1; i <= n_ref; i++) {
        for (int j = 1; j <= n_hyp; j++) {
            if (reference[i - 1] == hypothesis[j - 1]) {
                TM_DP(i, j) = TM_DP(i - 1, j - 1);
            } else {
                int best = TM_DP(i - 1, j - 1);
                if (TM_DP(i - 1, j) < best) best = TM_DP(i - 1, j);
                if (TM_DP(i, j - 1) < best) best = TM_DP(i, j - 1);
                TM_DP(i, j) = 1 + best;
            }
        }
    }
    int i = n_ref, j = n_hyp;
    int s = 0, d = 0, in = 0;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1]) {
            i--;
            j--;
        } else if (i > 0 && j > 0 && TM_DP(i, j) == TM_DP(i - 1, j - 1) + 1) {
            s++;
            i--;
            j--;
        } else if (i > 0 && TM_DP(i, j) == TM_DP(i - 1, j) + 1) {
            d++;
            i--;
        } else {
            in++;
            j--;
        }
    }
#undef TM_DP
    free(dp);
    if (subs) *subs = s;
    if (dels) *dels = d;
    if (ins) *ins = in;
}

static double tm_wer_from_ops(const char *const *ref, int n_ref,
                              const char *const *hyp, int n_hyp,
                              int is_token_input)
{
    (void)is_token_input;
    if (n_ref <= 0) return NAN;
    int s = 0, d = 0, in = 0;
    tm_edit_ops_tokens(ref, n_ref, hyp, n_hyp, &s, &d, &in);
    return (double)(s + d + in) / (double)n_ref;
}

double tm_word_error_rate_tokens(const char *const *reference, int n_ref,
                                 const char *const *hypothesis, int n_hyp)
{
    return tm_wer_from_ops(reference, n_ref, hypothesis, n_hyp, 1);
}

double tm_word_error_rate(const char *reference, const char *hypothesis)
{
    int n_ref = 0, n_hyp = 0;
    char **ref = tm_split_ascii(reference, &n_ref);
    char **hyp = tm_split_ascii(hypothesis, &n_hyp);
    if ((!ref && n_ref > 0) || (!hyp && n_hyp > 0)) {
        tm_free_tokens(ref, n_ref);
        tm_free_tokens(hyp, n_hyp);
        return NAN;
    }
    const double score = tm_wer_from_ops((const char *const *)ref, n_ref,
                                         (const char *const *)hyp, n_hyp, 0);
    tm_free_tokens(ref, n_ref);
    tm_free_tokens(hyp, n_hyp);
    return score;
}

static double tm_cer_from_cps(const int *ref, int n_ref, const int *hyp, int n_hyp)
{
    if (n_ref <= 0) return NAN;
    int s = 0, d = 0, in = 0;
    tm_edit_ops_int(ref, n_ref, hyp, n_hyp, &s, &d, &in);
    return (double)(s + d + in) / (double)n_ref;
}

double tm_character_error_rate(const char *reference, const char *hypothesis)
{
    int *ref = NULL;
    int *hyp = NULL;
    const int n_ref = tm_to_codepoints(reference, &ref);
    const int n_hyp = tm_to_codepoints(hypothesis, &hyp);
    if (n_ref < 0 || n_hyp < 0) {
        free(ref);
        free(hyp);
        return NAN;
    }
    const double score = tm_cer_from_cps(ref, n_ref, hyp, n_hyp);
    free(ref);
    free(hyp);
    return score;
}

double tm_character_error_rate_tokens(const char *const *reference, int n_ref,
                                      const char *const *hypothesis, int n_hyp)
{
    if (n_ref <= 0) return NAN;
    int s = 0, d = 0, in = 0;
    tm_edit_ops_tokens(reference, n_ref, hypothesis, n_hyp, &s, &d, &in);
    return (double)(s + d + in) / (double)n_ref;
}
