/*
 * model_merge.c —— 模型合并实现，见 model_merge.h。
 *
 * 运算顺序严格照搬 AI.py：
 *   - model_soup：对每个元素按张量下标顺序求 Neumaier 和再除以 n；
 *   - slerp_merge：na/nb 为 L2 范数，dot 为点积，cos 夹到 [-1, 1]；
 *     na <= eps / nb <= eps / cos 接近 ±1 时退化为线性插值；
 *   - ties_merge：阈值取全部 |task vector| 升序排序后的第 k-1 个（k =
 *     int(trim_fraction *总元素数)），只保留严格大于阈值的元素；符号和为正
 *     取正号组均值；density < 1 时按 (-|v|, 下标) 排序保留前 keep 个；
 *   - dare_merge：逐张量、逐元素按 rng.random() < p 丢弃，否则（可选）
 *     除以 1-p，再按张量下标累加、最后除以 n。
 */
#include "model_merge.h"

#include "numeric.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ===================== Python random.Random(seed) 兼容 MT19937 =====================
 * 与 fim.c / tts.c 相同的实现：整数 seed 取绝对值后按小端 32 位字拆 key。
 */
typedef struct {
    unsigned int mt[624];
    int index;
} MergeRng;

static void mm_mt_init_genrand(unsigned int *mt, unsigned int seed)
{
    int i;
    mt[0] = seed;
    for (i = 1; i < 624; ++i)
        mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (unsigned int)i;
}

static void mm_mt_init_by_array(unsigned int *mt, const unsigned int *key,
                                int key_length)
{
    int i = 1;
    int j = 0;
    int k;
    mm_mt_init_genrand(mt, 19650218U);
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

static void mm_rng_seed(MergeRng *rng, unsigned long long seed)
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
    mm_mt_init_by_array(rng->mt, key, keyused);
    rng->index = 624;
}

static unsigned int mm_mt_next(MergeRng *rng)
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
static double mm_rng_random(MergeRng *rng)
{
    unsigned int a = mm_mt_next(rng) >> 5;
    unsigned int b = mm_mt_next(rng) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

/* ================================ 通用校验 ================================ */

static int mm_finite(const double *v, int n)
{
    if (v == NULL || n < 1) return 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

static int mm_check_common(const double *weights, int n_tensors, int n_elem,
                           const double *out)
{
    if (out == NULL || n_tensors < 1 || n_elem < 1) return 0;
    if (n_tensors > INT_MAX / n_elem) return 0;
    return mm_finite(weights, n_tensors * n_elem);
}

/* ================================ model_soup ================================ */

int ai_model_soup(const double *weights, int n_tensors, int n_elem,
                  double *out)
{
    double *col;
    if (!mm_check_common(weights, n_tensors, n_elem, out)) return -1;

    col = (double *)malloc((size_t)n_tensors * sizeof(double));
    if (col == NULL) return -1;
    for (int i = 0; i < n_elem; i++) {
        for (int j = 0; j < n_tensors; j++) col[j] = weights[(size_t)j * n_elem + i];
        out[i] = sum_neumaier(col, n_tensors) / (double)n_tensors;
    }
    free(col);
    return 0;
}

/* ================================ slerp_merge ================================ */

static void mm_lerp(const double *a, const double *b, int n_elem, double t,
                    double *out)
{
    for (int i = 0; i < n_elem; i++) {
        out[i] = (1.0 - t) * a[i] + t * b[i];
    }
}

int ai_slerp_merge(const double *a, const double *b, int n_elem,
                   double t, double eps, double *out)
{
    double na, nb, dot, cos, theta, sin_theta, wa, wb;
    if (out == NULL || n_elem < 1) return -1;
    if (!isfinite(t) || t < 0.0 || t > 1.0) return -1;
    if (!isfinite(eps) || eps < 0.0) return -1;
    if (!mm_finite(a, n_elem) || !mm_finite(b, n_elem)) return -1;

    na = sqrt(dot_neumaier(a, a, n_elem));
    nb = sqrt(dot_neumaier(b, b, n_elem));
    if (na <= eps || nb <= eps) {
        mm_lerp(a, b, n_elem, t, out);
        return 0;
    }
    dot = dot_neumaier(a, b, n_elem);
    cos = dot / (na * nb);
    if (cos > 1.0) cos = 1.0;
    if (cos < -1.0) cos = -1.0;
    if (cos > 1.0 - 1e-7 || cos < -1.0 + 1e-7) {
        mm_lerp(a, b, n_elem, t, out);
        return 0;
    }
    theta = acos(cos);
    sin_theta = sin(theta);
    wa = sin((1.0 - t) * theta) / sin_theta;
    wb = sin(t * theta) / sin_theta;
    for (int i = 0; i < n_elem; i++) out[i] = wa * a[i] + wb * b[i];
    return 0;
}

/* ================================ ties_merge ================================ */

static int mm_abs_cmp(const void *pa, const void *pb)
{
    double a = *(const double *)pa;
    double b = *(const double *)pb;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

typedef struct {
    double key; /* -|merged|，升序即按绝对值降序 */
    int idx;
} MmOrder;

static int mm_order_cmp(const void *pa, const void *pb)
{
    const MmOrder *a = (const MmOrder *)pa;
    const MmOrder *b = (const MmOrder *)pb;
    if (a->key < b->key) return -1;
    if (a->key > b->key) return 1;
    if (a->idx < b->idx) return -1;
    if (a->idx > b->idx) return 1;
    return 0;
}

int ai_ties_merge(const double *weights, int n_tensors, int n_elem,
                  const double *base, double trim_fraction, double density,
                  double *out)
{
    double *tv;
    double *abs_vals;
    double *merged;
    double *same;
    int total;
    int k;
    double threshold;
    if (!mm_check_common(weights, n_tensors, n_elem, out)) return -1;
    if (base != NULL && !mm_finite(base, n_elem)) return -1;
    if (!isfinite(trim_fraction) || trim_fraction < 0.0 || trim_fraction >= 1.0)
        return -1;
    if (!isfinite(density) || density <= 0.0 || density > 1.0) return -1;

    tv = (double *)malloc((size_t)n_tensors * n_elem * sizeof(double));
    abs_vals = (double *)malloc((size_t)n_tensors * n_elem * sizeof(double));
    merged = (double *)malloc((size_t)n_elem * sizeof(double));
    same = (double *)malloc((size_t)n_tensors * sizeof(double));
    if (tv == NULL || abs_vals == NULL || merged == NULL || same == NULL) {
        free(tv);
        free(abs_vals);
        free(merged);
        free(same);
        return -1;
    }

    total = n_tensors * n_elem;
    for (int j = 0; j < n_tensors; j++) {
        for (int i = 0; i < n_elem; i++) {
            double b = base != NULL ? base[i] : 0.0;
            double d = weights[(size_t)j * n_elem + i] - b;
            tv[(size_t)j * n_elem + i] = d;
            abs_vals[(size_t)j * n_elem + i] = fabs(d);
        }
    }
    qsort(abs_vals, (size_t)total, sizeof(double), mm_abs_cmp);
    k = (int)(trim_fraction * (double)total);
    threshold = k > 0 ? abs_vals[k - 1] : 0.0;

    for (int i = 0; i < n_elem; i++) {
        long sign_sum = 0;
        int kept = 0;
        int same_n = 0;
        double sign;
        for (int j = 0; j < n_tensors; j++) {
            double d = tv[(size_t)j * n_elem + i];
            if (fabs(d) > threshold) kept++;
        }
        if (kept == 0) {
            merged[i] = 0.0;
            continue;
        }
        for (int j = 0; j < n_tensors; j++) {
            double d = tv[(size_t)j * n_elem + i];
            if (fabs(d) > threshold) sign_sum += (d > 0.0) ? 1 : -1;
        }
        if (sign_sum == 0) {
            merged[i] = 0.0;
            continue;
        }
        sign = sign_sum > 0 ? 1.0 : -1.0;
        for (int j = 0; j < n_tensors; j++) {
            double d = tv[(size_t)j * n_elem + i];
            if (fabs(d) > threshold && (d > 0.0) == (sign > 0.0)) {
                same[same_n++] = d;
            }
        }
        merged[i] = sum_neumaier(same, same_n) / (double)same_n;
    }

    if (density < 1.0) {
        int keep = (int)rint((double)n_elem * density);
        MmOrder *order;
        char *mask;
        if (keep < 1) keep = 1;
        order = (MmOrder *)malloc((size_t)n_elem * sizeof(MmOrder));
        mask = (char *)calloc((size_t)n_elem, sizeof(char));
        if (order == NULL || mask == NULL) {
            free(order);
            free(mask);
            free(tv);
            free(abs_vals);
            free(merged);
            free(same);
            return -1;
        }
        for (int i = 0; i < n_elem; i++) {
            order[i].key = -fabs(merged[i]);
            order[i].idx = i;
        }
        qsort(order, (size_t)n_elem, sizeof(MmOrder), mm_order_cmp);
        for (int i = 0; i < keep; i++) mask[order[i].idx] = 1;
        for (int i = 0; i < n_elem; i++) {
            if (!mask[i]) merged[i] = 0.0;
        }
        free(order);
        free(mask);
    }

    for (int i = 0; i < n_elem; i++) {
        out[i] = (base != NULL ? base[i] : 0.0) + merged[i];
    }
    free(tv);
    free(abs_vals);
    free(merged);
    free(same);
    return 0;
}

/* ================================ dare_merge ================================ */

int ai_dare_merge(const double *weights, int n_tensors, int n_elem,
                  const double *base, double drop_rate, int rescale,
                  unsigned long long seed, double *out)
{
    MergeRng rng;
    double *total;
    if (!mm_check_common(weights, n_tensors, n_elem, out)) return -1;
    if (base != NULL && !mm_finite(base, n_elem)) return -1;
    if (!isfinite(drop_rate) || drop_rate < 0.0 || drop_rate >= 1.0) return -1;

    total = (double *)calloc((size_t)n_elem, sizeof(double));
    if (total == NULL) return -1;
    mm_rng_seed(&rng, seed);
    for (int j = 0; j < n_tensors; j++) {
        for (int i = 0; i < n_elem; i++) {
            double b = base != NULL ? base[i] : 0.0;
            double delta = weights[(size_t)j * n_elem + i] - b;
            if (mm_rng_random(&rng) < drop_rate) {
                delta = 0.0;
            } else if (rescale) {
                delta = delta / (1.0 - drop_rate);
            }
            total[i] += delta;
        }
    }
    for (int i = 0; i < n_elem; i++) {
        out[i] = (base != NULL ? base[i] : 0.0) + total[i] / (double)n_tensors;
    }
    free(total);
    return 0;
}
