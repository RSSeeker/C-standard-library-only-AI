/*
 * fim.c —— FIM（Fill-in-the-Middle）与 T5 span corruption 的 C 实现
 *
 * 见 fim.h 的模块说明。随机 span 抽样使用与 Python `random.Random(seed)`
 * 逐位一致的 MT19937（init_by_array 单/双 32 位 key）+ random() +
 * expovariate + getrandbits/_randbelow，从而同一 seed 下 span 与 Python 相同。
 */
#include "fim.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================= Python random.Random 兼容 RNG =========================
 *
 * Python 对整数 seed 的处理：取绝对值后按小端 32 位字拆成 key，
 * key 为 1 个字时等价于 init_by_array([seed])；本实现最多支持 2 个字
 * （seed 为 unsigned long long）。
 */
typedef struct {
    unsigned int mt[624];
    int index;
} FimRng;

static void fim_mt_init_genrand(unsigned int *mt, unsigned int seed)
{
    int i;
    mt[0] = seed;
    for (i = 1; i < 624; ++i)
        mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (unsigned int)i;
}

static void fim_mt_init_by_array(unsigned int *mt, const unsigned int *key,
                                 int key_length)
{
    int i = 1;
    int j = 0;
    int k;
    fim_mt_init_genrand(mt, 19650218U);
    k = (624 > key_length) ? 624 : key_length;
    for (; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U))
                + key[j] + (unsigned int)j;
        ++i;
        ++j;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = 623; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U))
                - (unsigned int)i;
        ++i;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000U;
}

static void fim_rng_seed(FimRng *rng, unsigned long long seed)
{
    unsigned int key[2];
    int keyused;
    int nbits = 0;
    unsigned long long t = seed;
    int i;
    while (t) {
        nbits++;
        t >>= 1;
    }
    keyused = nbits > 0 ? ((nbits - 1) / 32 + 1) : 1;
    for (i = 0; i < keyused; ++i)
        key[i] = (unsigned int)(seed >> (32 * i));
    fim_mt_init_by_array(rng->mt, key, keyused);
    rng->index = 624;
}

static unsigned int fim_mt_next(FimRng *rng)
{
    unsigned int y;
    static const unsigned int mag01[2] = {0x0U, 0x9908b0dfU};
    if (rng->index >= 624) {
        int i;
        for (i = 0; i < 624; ++i) {
            unsigned int y2 = (rng->mt[i] & 0x80000000U)
                            | (rng->mt[(i + 1) % 624] & 0x7fffffffU);
            rng->mt[i] = rng->mt[(i + 397) % 624] ^ (y2 >> 1) ^ mag01[y2 & 1U];
        }
        rng->index = 0;
    }
    y = rng->mt[rng->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

/* random()：两个 32 位输出拼 53 位 */
static double fim_rng_random(FimRng *rng)
{
    unsigned int a = fim_mt_next(rng) >> 5;
    unsigned int b = fim_mt_next(rng) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

/* getrandbits(k)，k <= 32（本模块的 n 远小于 2^32） */
static unsigned int fim_rng_getrandbits(FimRng *rng, int k)
{
    if (k <= 0) return 0;
    if (k > 32) k = 32;
    return fim_mt_next(rng) >> (32 - k);
}

/* Python random.Random._randbelow_with_getrandbits */
static int fim_rng_randbelow(FimRng *rng, int n)
{
    int k = 0;
    int r;
    int t;
    if (n <= 0) return 0;
    t = n;
    while (t) {
        k++;
        t >>= 1;
    }
    r = (int)fim_rng_getrandbits(rng, k);
    while (r >= n) r = (int)fim_rng_getrandbits(rng, k);
    return r;
}

/* int(round(x))：Python 3 用「四舍六入五取偶」，rint 在默认舍入模式下一致 */
static long long fim_round(double x)
{
    return (long long)rint(x);
}

/* ============================== 参数校验助手 ============================== */

/* 对齐 Python _fim_check_ids：非空、vocab_size > 1、每个 id 在 [0, vocab_size) */
static int fim_check_ids(const int *ids, int n_ids, int vocab_size)
{
    int i;
    if (!ids || n_ids < 1) return -1;
    if (vocab_size <= 1) return -1;
    for (i = 0; i < n_ids; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab_size) return -1;
    }
    return 0;
}

/* ================================ FIM 改写 ================================ */

int ai_fim_transform(const int *ids, int n_ids, int vocab_size, int mode,
                     double prefix_ratio, double middle_ratio,
                     AiFimExample *out)
{
    long long p, rest, m;
    int n;
    int i;

    if (!out) return -1;
    if (fim_check_ids(ids, n_ids, vocab_size) != 0) return -1;
    if (mode != FIM_MODE_PSM && mode != FIM_MODE_SPM) return -1;
    /* 注意 NaN 会让两个比较都为假，因此下面写法同时拒绝 NaN */
    if (!(prefix_ratio > 0.0 && prefix_ratio < 1.0)) return -1;
    if (!(middle_ratio > 0.0 && middle_ratio < 1.0)) return -1;
    n = n_ids;
    if (n < 3) return -1;
    if (n > INT_MAX - 3) return -1;

    p = fim_round((double)n * prefix_ratio);
    if (p < 1) p = 1;
    if (p > n - 2) p = n - 2;
    rest = n - p;
    m = fim_round((double)rest * middle_ratio);
    if (m < 1) m = 1;
    if (m > rest - 1) m = rest - 1;

    memset(out, 0, sizeof(*out));
    out->n_prefix = (int)p;
    out->n_middle = (int)m;
    out->n_suffix = (int)(rest - m);
    out->n_input = n + 3;
    out->middle_start = out->n_input - out->n_middle;
    out->mode = mode;
    out->prefix = (int *)malloc(sizeof(int) * (size_t)out->n_prefix);
    out->middle = (int *)malloc(sizeof(int) * (size_t)out->n_middle);
    out->suffix = (int *)malloc(sizeof(int) * (size_t)out->n_suffix);
    out->input_ids = (int *)malloc(sizeof(int) * (size_t)out->n_input);
    if (!out->prefix || !out->middle || !out->suffix || !out->input_ids) {
        ai_fim_example_free(out);
        return -1;
    }
    for (i = 0; i < out->n_prefix; ++i) out->prefix[i] = ids[i];
    for (i = 0; i < out->n_middle; ++i) out->middle[i] = ids[out->n_prefix + i];
    for (i = 0; i < out->n_suffix; ++i)
        out->suffix[i] = ids[out->n_prefix + out->n_middle + i];

    {
        int k = 0;
        const int sent_prefix = vocab_size + FIM_SENTINEL_PREFIX;
        const int sent_suffix = vocab_size + FIM_SENTINEL_SUFFIX;
        const int sent_middle = vocab_size + FIM_SENTINEL_MIDDLE;
        if (mode == FIM_MODE_PSM) {
            out->input_ids[k++] = sent_prefix;
            for (i = 0; i < out->n_prefix; ++i) out->input_ids[k++] = out->prefix[i];
            out->input_ids[k++] = sent_suffix;
            for (i = 0; i < out->n_suffix; ++i) out->input_ids[k++] = out->suffix[i];
            out->input_ids[k++] = sent_middle;
            for (i = 0; i < out->n_middle; ++i) out->input_ids[k++] = out->middle[i];
        } else {
            out->input_ids[k++] = sent_suffix;
            for (i = 0; i < out->n_suffix; ++i) out->input_ids[k++] = out->suffix[i];
            out->input_ids[k++] = sent_prefix;
            for (i = 0; i < out->n_prefix; ++i) out->input_ids[k++] = out->prefix[i];
            out->input_ids[k++] = sent_middle;
            for (i = 0; i < out->n_middle; ++i) out->input_ids[k++] = out->middle[i];
        }
        if (k != out->n_input) { /* 不应发生，防御性检查 */
            ai_fim_example_free(out);
            return -1;
        }
    }
    return 0;
}

void ai_fim_example_free(AiFimExample *ex)
{
    if (!ex) return;
    free(ex->input_ids);
    free(ex->prefix);
    free(ex->middle);
    free(ex->suffix);
    memset(ex, 0, sizeof(*ex));
}

int ai_fim_target_mask(const AiFimExample *ex, int *mask)
{
    int t;
    if (!ex || !ex->input_ids || ex->n_input <= 0 || !mask) return -1;
    if (ex->middle_start < 0 || ex->middle_start >= ex->n_input) return -1;
    for (t = 0; t < ex->n_input - 1; ++t)
        mask[t] = (t + 1 >= ex->middle_start) ? 1 : 0;
    return 0;
}

int ai_fim_batch(const int *flat_ids, const int *seq_lens, int n_seqs,
                 int vocab_size, int mode, double prefix_ratio,
                 double middle_ratio, AiFimExample **out, int *out_n)
{
    AiFimExample *examples;
    int s;
    int off = 0;
    if (!flat_ids || !seq_lens || n_seqs < 1 || !out || !out_n) return -1;
    for (s = 0; s < n_seqs; ++s) {
        if (seq_lens[s] < 1) return -1;
        if (off > INT_MAX - seq_lens[s]) return -1;
        off += seq_lens[s];
    }
    examples = (AiFimExample *)calloc((size_t)n_seqs, sizeof(AiFimExample));
    if (!examples) return -1;
    off = 0;
    for (s = 0; s < n_seqs; ++s) {
        if (ai_fim_transform(flat_ids + off, seq_lens[s], vocab_size, mode,
                             prefix_ratio, middle_ratio, &examples[s]) != 0) {
            ai_fim_batch_free(examples, n_seqs);
            return -1;
        }
        off += seq_lens[s];
    }
    *out = examples;
    *out_n = n_seqs;
    return 0;
}

void ai_fim_batch_free(AiFimExample *examples, int n_examples)
{
    int i;
    if (!examples) return;
    for (i = 0; i < n_examples; ++i) ai_fim_example_free(&examples[i]);
    free(examples);
}

/* ============================ T5 span corruption ============================ */

int ai_span_corruption_mask(int seq_len, double mask_ratio, double mean_span,
                            unsigned long long seed, AiFimSpan **out,
                            int *out_n)
{
    FimRng rng;
    char *chosen;
    AiFimSpan *spans;
    int cap = 8;
    int n_spans = 0;
    long long target;
    long long total = 0;
    int attempts = 0;
    int max_attempts;
    int i;

    if (!out || !out_n) return -1;
    if (seq_len <= 0) return -1;
    if (!(mask_ratio > 0.0 && mask_ratio <= 1.0)) return -1;
    if (!(mean_span >= 1.0)) return -1;

    fim_rng_seed(&rng, seed);
    target = fim_round((double)seq_len * mask_ratio);
    if (target < 1) target = 1;
    chosen = (char *)calloc((size_t)seq_len, 1);
    spans = (AiFimSpan *)malloc(sizeof(AiFimSpan) * (size_t)cap);
    if (!chosen || !spans) {
        free(chosen);
        free(spans);
        return -1;
    }
    max_attempts = 20 > seq_len * 4 ? 20 : seq_len * 4;

    while (total < target && attempts < max_attempts) {
        double extra;
        double lambd;
        int length;
        int start;
        int k;
        int overlap = 0;

        attempts++;
        /* Python: -math.log(1.0 - random()) / lambd，lambd = 1.0 / mean_span */
        lambd = 1.0 / mean_span;
        extra = -log(1.0 - fim_rng_random(&rng)) / lambd;
        length = (int)fim_round(1.0 + extra);
        {
            int cap_len = (int)fim_round(3.0 * mean_span);
            if (length < 1) length = 1;
            if (length > cap_len) length = cap_len;
        }
        if (length > seq_len) length = seq_len;
        start = fim_rng_randbelow(&rng, seq_len - length + 1);
        for (k = start; k < start + length; ++k) {
            if (chosen[k]) {
                overlap = 1;
                break;
            }
        }
        if (overlap) continue;
        for (k = start; k < start + length; ++k) chosen[k] = 1;
        if (n_spans == cap) {
            AiFimSpan *grown;
            cap *= 2;
            grown = (AiFimSpan *)realloc(spans, sizeof(AiFimSpan) * (size_t)cap);
            if (!grown) {
                free(chosen);
                free(spans);
                return -1;
            }
            spans = grown;
        }
        spans[n_spans].start = start;
        spans[n_spans].length = length;
        n_spans++;
        total += length;
    }
    free(chosen);

    /* Python spans.sort()：按 (start, length) 升序；start 唯一，插排足够 */
    for (i = 1; i < n_spans; ++i) {
        AiFimSpan tmp = spans[i];
        int j = i - 1;
        while (j >= 0 && (spans[j].start > tmp.start
                          || (spans[j].start == tmp.start
                              && spans[j].length > tmp.length))) {
            spans[j + 1] = spans[j];
            j--;
        }
        spans[j + 1] = tmp;
    }
    *out = spans;
    *out_n = n_spans;
    return 0;
}

void ai_fim_spans_free(AiFimSpan *spans)
{
    free(spans);
}

int ai_span_corruption_ids(const int *ids, int n_ids, int vocab_size,
                           double mask_ratio, double mean_span,
                           int sentinel_id, unsigned long long seed,
                           AiSpanCorruption *out)
{
    AiFimSpan *spans = NULL;
    int n_spans = 0;
    int explicit_sentinel;
    int sentinel;
    int masked = 0;
    int n_input;
    int n_target;
    int i;
    int si = 0;
    int ki = 0;
    int ti = 0;

    if (!out) return -1;
    if (fim_check_ids(ids, n_ids, vocab_size) != 0) return -1;
    explicit_sentinel = (sentinel_id != -1);
    if (sentinel_id < -1) return -1;
    sentinel = explicit_sentinel ? sentinel_id : vocab_size;
    if (sentinel < 0) return -1;
    if (explicit_sentinel && sentinel >= vocab_size) return -1;
    if (ai_span_corruption_mask(n_ids, mask_ratio, mean_span, seed, &spans,
                                &n_spans) != 0)
        return -1;

    for (i = 0; i < n_spans; ++i) masked += spans[i].length;
    if (n_ids > INT_MAX - n_spans) {
        ai_fim_spans_free(spans);
        return -1;
    }
    n_input = n_ids - masked + n_spans;
    n_target = masked + n_spans;

    memset(out, 0, sizeof(*out));
    out->input_ids = (int *)malloc(sizeof(int) * (size_t)(n_input ? n_input : 1));
    out->target_ids = (int *)malloc(sizeof(int) * (size_t)(n_target ? n_target : 1));
    if (!out->input_ids || !out->target_ids) {
        ai_span_corruption_free(out);
        ai_fim_spans_free(spans);
        return -1;
    }
    for (i = 0; i < n_ids; ++i) {
        if (si < n_spans && i == spans[si].start) {
            int k;
            out->input_ids[ki++] = sentinel;
            out->target_ids[ti++] = sentinel;
            for (k = 0; k < spans[si].length; ++k)
                out->target_ids[ti++] = ids[i + k];
            i += spans[si].length - 1;
            si++;
        } else {
            out->input_ids[ki++] = ids[i];
        }
    }
    if (ki != n_input || ti != n_target) {
        ai_span_corruption_free(out);
        ai_fim_spans_free(spans);
        return -1;
    }
    out->n_input = n_input;
    out->n_target = n_target;
    out->spans = spans;
    out->n_spans = n_spans;
    out->sentinel_id = sentinel;
    return 0;
}

void ai_span_corruption_free(AiSpanCorruption *sc)
{
    if (!sc) return;
    free(sc->input_ids);
    free(sc->target_ids);
    free(sc->spans);
    memset(sc, 0, sizeof(*sc));
}
