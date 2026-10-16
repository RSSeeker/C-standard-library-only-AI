/*
 * text.c —— 字符级文本处理器与文本生成
 */
#include "text.h"
#include "numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "text: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

/* ============================ UTF-8 ============================ */

int utf8_decode(const char *s, int *cp)
{
    const unsigned char *p = (const unsigned char *)s;
    unsigned char c = p[0];
    if (c == 0) return 0;
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) {
        if ((p[1] & 0xC0) != 0x80) return 0;
        *cp = ((c & 0x1F) << 6) | (p[1] & 0x3F);
        return (*cp >= 0x80) ? 2 : 0;             /* 拒绝过长编码 */
    }
    if ((c & 0xF0) == 0xE0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        *cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        return (*cp >= 0x800) ? 3 : 0;
    }
    if ((c & 0xF8) == 0xF0) {
        if ((p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return 0;
        *cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        return (*cp >= 0x10000 && *cp <= 0x10FFFF) ? 4 : 0;
    }
    return 0;
}

int utf8_encode(int cp, char *buf)
{
    if (cp < 0) return 0;
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

/* 取下一个码点：非法 UTF-8 字节按单字节码点处理，避免死循环 */
static int next_cp(const char *s, int *cp)
{
    int n = utf8_decode(s, cp);
    if (n > 0) return n;
    *cp = (unsigned char)s[0];
    return 1;
}

int utf8_length(const char *s)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        int cp;
        p += next_cp(p, &cp);
        n++;
    }
    return n;
}

int utf8_cp_at(const char *s, int idx)
{
    const char *p = s;
    int i = 0, cp = -1;
    while (*p) {
        int n = next_cp(p, &cp);
        if (i == idx) return cp;
        p += n;
        i++;
    }
    return -1;
}

/* ============================ TextProcessor ============================ */

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

TextProcessor *tp_create(const char *utf8_text)
{
    TextProcessor *tp = xmalloc(sizeof(TextProcessor));
    tp->vocab_size = 0;
    tp->chars = NULL;

    int cap = 16;
    int *all = xmalloc((size_t)cap * sizeof(int));
    int n = 0;
    const char *p = utf8_text;
    while (*p) {
        int cp;
        p += next_cp(p, &cp);
        if (n == cap) { cap *= 2; all = realloc(all, (size_t)cap * sizeof(int)); if (!all) exit(1); }
        all[n++] = cp;
    }
    if (n == 0) { tp->chars = NULL; tp->vocab_size = 0; free(all); return tp; }

    qsort(all, (size_t)n, sizeof(int), cmp_int);
    /* 去重 */
    int m = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || all[i] != all[i - 1]) all[m++] = all[i];
    tp->chars = all;
    tp->vocab_size = m;
    return tp;
}

void tp_free(TextProcessor *tp)
{
    if (!tp) return;
    free(tp->chars);
    free(tp);
}

/* 词表已排序：二分查找，等价于 Python 的 dict 查表 */
int tp_idx_of(const TextProcessor *tp, int cp)
{
    int lo = 0, hi = tp->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (tp->chars[mid] == cp) return mid;
        if (tp->chars[mid] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

int tp_cp_at(const TextProcessor *tp, int idx)
{
    if (idx < 0 || idx >= tp->vocab_size) return -1;
    return tp->chars[idx];
}

void tp_encode_one_hot_cp(const TextProcessor *tp, int cp, double *vec)
{
    for (int i = 0; i < tp->vocab_size; i++) vec[i] = 0.0;
    int idx = tp_idx_of(tp, cp);
    if (idx >= 0) vec[idx] = 1.0;
}

int tp_encode_indices(const TextProcessor *tp, const char *s, int *out)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        int cp;
        p += next_cp(p, &cp);
        int idx = tp_idx_of(tp, cp);
        out[n++] = (idx >= 0) ? idx : 0;          /* 未知字符回退到 0，同 Python */
    }
    return n;
}

int tp_decode_indices(const TextProcessor *tp, const int *idx, int n, char *buf, int bufsize)
{
    int len = 0;
    for (int i = 0; i < n; i++) {
        char tmp[4];
        int cp = tp_cp_at(tp, idx[i]);
        int nb;
        if (cp < 0) { tmp[0] = '?'; nb = 1; }
        else nb = utf8_encode(cp, tmp);
        if (len + nb + 1 > bufsize) break;
        memcpy(buf + len, tmp, (size_t)nb);
        len += nb;
    }
    buf[len] = '\0';
    return len;
}

/* ============================ 生成 ============================ */

void softmax_with_temp(const double *logits, int n, double temperature, double *probs)
{
    double mx;
    if (temperature != 1.0) {
        double first = logits[0] / temperature;
        mx = first;
        for (int i = 1; i < n; i++) {
            double v = logits[i] / temperature;
            if (v > mx) mx = v;
        }
        for (int i = 0; i < n; i++) probs[i] = exp(logits[i] / temperature - mx);
    } else {
        mx = logits[0];
        for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
        for (int i = 0; i < n; i++) probs[i] = exp(logits[i] - mx);
    }
    /* Python 用 sum(exps) -> Neumaier */
    double s = sum_neumaier(probs, n);
    for (int i = 0; i < n; i++) probs[i] /= s;
}

double rng_uniform(unsigned long long *state)
{
    unsigned long long x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (double)((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

int sample_index(const double *probs, int n, unsigned long long *rng_state)
{
    double r = rng_uniform(rng_state);
    double cumulative = 0.0;
    for (int i = 0; i < n; i++) {
        cumulative += probs[i];
        if (r < cumulative) return i;
    }
    return n - 1;
}

int generate_text(ForwardFn fwd, void *ctx, const TextProcessor *tp,
                  const char *start_str, int length, int window_size, double temp,
                  char *out_buf, int out_buf_size, unsigned long long *rng_state)
{
    const int V = tp->vocab_size;
    const int in_dim = V * window_size;

    int start_len = utf8_length(start_str);
    size_t start_bytes = strlen(start_str);
    if (start_len + length + 1 > out_buf_size) return -1;
    memcpy(out_buf, start_str, start_bytes + 1);

    int *ctx_idx = xmalloc((size_t)window_size * sizeof(int));
    double *x = xmalloc((size_t)in_dim * sizeof(double));
    double *logits = xmalloc((size_t)V * sizeof(double));
    double *probs = xmalloc((size_t)V * sizeof(double));
    char *cur = out_buf + start_bytes;
    int cur_len = start_len;

    for (int step = 0; step < length; step++) {
        /* 取最后 window_size 个字符；不足则用词表第一个字符左填充（同 Python 的 rjust） */
        int pad = window_size - cur_len;
        if (pad < 0) pad = 0;
        for (int i = 0; i < pad; i++) ctx_idx[i] = 0;
        int have = cur_len < window_size ? cur_len : window_size;
        for (int i = 0; i < have; i++) {
            int cp = utf8_cp_at(out_buf, cur_len - have + i);
            int idx = tp_idx_of(tp, cp);
            ctx_idx[pad + i] = (idx >= 0) ? idx : 0;
        }

        /* 拼接 one-hot */
        for (int w = 0; w < window_size; w++) {
            double *slot = x + (size_t)w * V;
            for (int v = 0; v < V; v++) slot[v] = 0.0;
            int idx = ctx_idx[w];
            if (idx >= 0 && idx < V) slot[idx] = 1.0;
        }

        fwd(ctx, x, in_dim, logits, V);
        softmax_with_temp(logits, V, temp, probs);
        int next_idx = sample_index(probs, V, rng_state);
        int cp = tp_cp_at(tp, next_idx);
        char tmp[4];
        int nb = (cp < 0) ? 0 : utf8_encode(cp, tmp);
        if (nb <= 0) break;
        if (strlen(out_buf) + (size_t)nb + 1 > (size_t)out_buf_size) break;
        memcpy(cur, tmp, (size_t)nb);
        cur += nb;
        *cur = '\0';
        cur_len++;
    }

    free(ctx_idx);
    free(x);
    free(logits);
    free(probs);
    return cur_len - start_len;
}
