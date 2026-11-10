/* RetNet 风格语言模型前向实现，见 retnet.h。 */
#include "retnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rt_die(const char *msg) {
    fprintf(stderr, "RetNetModel: %s\n", msg);
    abort();
}

static double *rt_alloc(size_t n) {
    double *p = (double *)calloc(n ? n : 1, sizeof(double));
    if (!p) rt_die("内存不足");
    return p;
}

static void rt_dense(const double *W, const double *b, int out_n, int in_n,
                     const double *x, double *out) {
    for (int i = 0; i < out_n; i++) {
        const double *row = W + (size_t)i * in_n;
        double sum = 0.0;
        for (int j = 0; j < in_n; j++) sum += row[j] * x[j];
        out[i] = sum + b[i];
    }
}

static double rt_gauss(unsigned long long *st) {
    unsigned long long z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    double u1 = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    double u2 = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

static void rt_block_init(RetNetBlock *b, int d, int heads, int ff,
                          unsigned long long *st) {
    b->d_model = d;
    b->num_heads = heads;
    b->head_dim = d / heads;
    b->d_ff = ff;
    b->ln1 = layernorm_create(d, 1e-5);
    b->ln2 = layernorm_create(d, 1e-5);
    b->Wq = rt_alloc((size_t)d * d); b->bq = rt_alloc((size_t)d);
    b->Wk = rt_alloc((size_t)d * d); b->bk = rt_alloc((size_t)d);
    b->Wv = rt_alloc((size_t)d * d); b->bv = rt_alloc((size_t)d);
    b->Wo = rt_alloc((size_t)d * d); b->bo = rt_alloc((size_t)d);
    b->W1 = rt_alloc((size_t)ff * d); b->b1 = rt_alloc((size_t)ff);
    b->W2 = rt_alloc((size_t)d * ff); b->b2 = rt_alloc((size_t)d);
    b->gamma = rt_alloc((size_t)heads);
    for (int h = 0; h < heads; h++) b->gamma[h] = 1.0 - pow(2.0, -5.0 - h);
    b->x = b->h1 = b->q = b->k = b->v = b->concat = b->x1 = NULL;
    b->h2 = b->a2 = b->out2 = NULL;
    b->ret = b->scores = b->denom = NULL;
    double std = sqrt(2.0 / (double)d);
    for (int i = 0; i < d * d; i++) {
        b->Wq[i] = rt_gauss(st) * std;
        b->Wk[i] = rt_gauss(st) * std;
        b->Wv[i] = rt_gauss(st) * std;
        b->Wo[i] = rt_gauss(st) * std;
    }
    for (int i = 0; i < ff * d; i++) b->W1[i] = rt_gauss(st) * std;
    for (int i = 0; i < d * ff; i++) b->W2[i] = rt_gauss(st) * std;
}

static void rt_block_alloc(RetNetBlock *b, int max_len) {
    const int d = b->d_model;
    const int ff = b->d_ff;
    const int hd = b->head_dim;
    size_t T = (size_t)max_len;
    b->x = rt_alloc(T * d);
    b->h1 = rt_alloc(T * d);
    b->q = rt_alloc(T * d);
    b->k = rt_alloc(T * d);
    b->v = rt_alloc(T * d);
    b->concat = rt_alloc(T * d);
    b->x1 = rt_alloc(T * d);
    b->h2 = rt_alloc(T * d);
    b->a2 = rt_alloc(T * ff);
    b->out2 = rt_alloc(T * d);
    b->ret = rt_alloc(T * hd);
    b->scores = rt_alloc(T * T);
    b->denom = rt_alloc(T);
}

static void rt_block_free(RetNetBlock *b) {
    layernorm_free(b->ln1);
    layernorm_free(b->ln2);
    free(b->Wq); free(b->bq); free(b->Wk); free(b->bk);
    free(b->Wv); free(b->bv); free(b->Wo); free(b->bo);
    free(b->W1); free(b->b1); free(b->W2); free(b->b2);
    free(b->gamma);
    free(b->x); free(b->h1); free(b->q); free(b->k); free(b->v);
    free(b->concat); free(b->x1); free(b->h2); free(b->a2); free(b->out2);
    free(b->ret); free(b->scores); free(b->denom);
}

/* 单头保留：q/k/v 为 (T, hd)，返回 out (T, hd)。 */
static void rt_retention_head(const double *q, const double *k, const double *v,
                              int T, int hd, double gamma, double scale,
                              double *out, double *scores, double *denom,
                              double *ret_out) {
    double acc = 0.0;
    for (int t = 0; t < T; t++) {
        acc = (t == 0) ? 1.0 : acc * gamma + 1.0;
        denom[t] = acc;
    }
    for (int t = 0; t < T; t++) {
        const double *qt = q + (size_t)t * hd;
        double *row = scores + (size_t)t * T;
        for (int j = 0; j <= t; j++) {
            const double *kj = k + (size_t)j * hd;
            double s = 0.0;
            for (int i = 0; i < hd; i++) s += qt[i] * kj[i];
            row[j] = (s * scale) * pow(gamma, (double)(t - j));
        }
        double d = denom[t];
        for (int i = 0; i < hd; i++) {
            double total = 0.0;
            for (int j = 0; j <= t; j++) total += row[j] * v[(size_t)j * hd + i];
            ret_out[(size_t)t * hd + i] = total / d;
        }
    }
    (void)out;
}

static void rt_norm_rows(const RetNetModel *m, LayerNorm *ln, const double *x,
                         int T, double *out) {
    const int d = m->d_model;
    for (int t = 0; t < T; t++) {
        memcpy(m->t_row->data, x + (size_t)t * d, (size_t)d * sizeof(double));
        layernorm_forward(ln, m->t_row, m->t_ln);
        memcpy(out + (size_t)t * d, m->t_ln->data, (size_t)d * sizeof(double));
    }
}

RetNetModel *retnet_create(int vocab_size, int d_model, int num_heads, int d_ff,
                           int num_layers, int max_len, unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || num_heads < 1 || d_ff < 1 ||
        num_layers < 1 || max_len < 1) {
        rt_die("create: 参数必须是正整数");
    }
    if (d_model % num_heads != 0) rt_die("create: d_model 必须能被 num_heads 整除");
    RetNetModel *m = (RetNetModel *)calloc(1, sizeof(RetNetModel));
    if (!m) rt_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    unsigned long long st = seed ? seed : 0x12345ULL;
    double std = sqrt(2.0 / (double)d_model);
    m->emb = rt_alloc((size_t)vocab_size * d_model);
    for (int i = 0; i < vocab_size * d_model; i++) m->emb[i] = rt_gauss(&st) * std;

    m->blocks = (RetNetBlock *)calloc((size_t)num_layers, sizeof(RetNetBlock));
    if (!m->blocks) rt_die("create: 内存不足");
    for (int i = 0; i < num_layers; i++) {
        rt_block_init(&m->blocks[i], d_model, num_heads, d_ff, &st);
        rt_block_alloc(&m->blocks[i], max_len);
    }

    m->ln_out = layernorm_create(d_model, 1e-5);
    m->W_out = rt_alloc((size_t)vocab_size * d_model);
    m->b_out = rt_alloc((size_t)vocab_size);
    for (int i = 0; i < vocab_size * d_model; i++) m->W_out[i] = rt_gauss(&st) * std;

    int shape_d[1] = { d_model };
    m->t_row = tensor_new(1, shape_d);
    m->t_ln = tensor_new(1, shape_d);
    return m;
}

void retnet_free(RetNetModel *m) {
    if (!m) return;
    free(m->emb);
    for (int i = 0; i < m->num_layers; i++) rt_block_free(&m->blocks[i]);
    free(m->blocks);
    layernorm_free(m->ln_out);
    free(m->W_out); free(m->b_out);
    tensor_free(m->t_row);
    tensor_free(m->t_ln);
    free(m);
}

int retnet_param_count(const RetNetModel *m) {
    int d = m->d_model, ff = m->d_ff;
    int n = 2 * m->vocab_size * d + m->vocab_size + 2 * d;
    for (int i = 0; i < m->num_layers; i++) {
        n += 4 * d;                 /* ln1/ln2 gamma+beta */
        n += 4 * (d * d + d);       /* Wq/Wk/Wv/Wo + bias */
        n += ff * d + ff;           /* W1 + bias */
        n += d * ff + d;            /* W2 + bias */
    }
    return n;
}

void retnet_forward(RetNetModel *m, const int *ids, int T, double *logits) {
    if (!ids || T <= 0) rt_die("forward: ids 不能为空");
    if (T > m->max_len) rt_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) rt_die("forward: token id 越界");
    }
    const int d = m->d_model;
    const int ff = m->d_ff;
    const int heads = m->num_heads;
    const int hd = d / heads;

    for (int li = 0; li < m->num_layers; li++) {
        RetNetBlock *b = &m->blocks[li];
        /* 当前输入 x：第 0 层从 embedding 取；之后从上一层的输出 */
        if (li == 0) {
            for (int t = 0; t < T; t++) {
                memcpy(b->x + (size_t)t * d, m->emb + (size_t)ids[t] * d,
                       (size_t)d * sizeof(double));
            }
        } else {
            RetNetBlock *prev = &m->blocks[li - 1];
            memcpy(b->x, prev->x, (size_t)T * d * sizeof(double));
        }

        rt_norm_rows(m, b->ln1, b->x, T, b->h1);
        for (int t = 0; t < T; t++) {
            const double *h = b->h1 + (size_t)t * d;
            rt_dense(b->Wq, b->bq, d, d, h, b->q + (size_t)t * d);
            rt_dense(b->Wk, b->bk, d, d, h, b->k + (size_t)t * d);
            rt_dense(b->Wv, b->bv, d, d, h, b->v + (size_t)t * d);
        }

        /* 逐头保留，concat 到 (T, d) */
        double scale = 1.0 / sqrt((double)hd);
        for (int h = 0; h < heads; h++) {
            const int off = h * hd;
            /* 收集该头的 q/k/v 到连续缓冲（复用 ret/scores/denom） */
            /* 直接在 (T,d) 上按跨步计算：先拷贝到临时行缓冲 */
            double *qh = (double *)rt_alloc((size_t)T * hd);
            double *kh = (double *)rt_alloc((size_t)T * hd);
            double *vh = (double *)rt_alloc((size_t)T * hd);
            for (int t = 0; t < T; t++) {
                for (int i = 0; i < hd; i++) {
                    qh[(size_t)t * hd + i] = b->q[(size_t)t * d + off + i];
                    kh[(size_t)t * hd + i] = b->k[(size_t)t * d + off + i];
                    vh[(size_t)t * hd + i] = b->v[(size_t)t * d + off + i];
                }
            }
            rt_retention_head(qh, kh, vh, T, hd, b->gamma[h], scale,
                              NULL, b->scores, b->denom, b->ret);
            for (int t = 0; t < T; t++) {
                for (int i = 0; i < hd; i++) {
                    b->concat[(size_t)t * d + off + i] = b->ret[(size_t)t * hd + i];
                }
            }
            free(qh); free(kh); free(vh);
        }

        for (int t = 0; t < T; t++) {
            rt_dense(b->Wo, b->bo, d, d, b->concat + (size_t)t * d,
                     b->x1 + (size_t)t * d);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                b->x1[(size_t)t * d + j] =
                    b->x[(size_t)t * d + j] + b->x1[(size_t)t * d + j];
            }
        }

        rt_norm_rows(m, b->ln2, b->x1, T, b->h2);
        for (int t = 0; t < T; t++) {
            const double *h = b->h2 + (size_t)t * d;
            double *a2 = b->a2 + (size_t)t * ff;
            rt_dense(b->W1, b->b1, ff, d, h, a2);
            for (int j = 0; j < ff; j++) {
                if (a2[j] < 0.0) a2[j] = 0.0;
            }
        }
        for (int t = 0; t < T; t++) {
            rt_dense(b->W2, b->b2, d, ff, b->a2 + (size_t)t * ff,
                     b->out2 + (size_t)t * d);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                b->x[(size_t)t * d + j] =
                    b->x1[(size_t)t * d + j] + b->out2[(size_t)t * d + j];
            }
        }
    }

    double *h = rt_alloc((size_t)T * d);
    rt_norm_rows(m, m->ln_out, m->blocks[m->num_layers - 1].x, T, h);
    for (int t = 0; t < T; t++) {
        rt_dense(m->W_out, m->b_out, m->vocab_size, d, h + (size_t)t * d,
                 logits + (size_t)t * m->vocab_size);
    }
    free(h);
}
