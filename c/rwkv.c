/* RWKV 风格语言模型前向实现，见 rwkv.h。 */
#include "rwkv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rk_die(const char *msg) {
    fprintf(stderr, "RWKVModel: %s\n", msg);
    abort();
}

static double *rk_alloc(size_t n) {
    double *p = (double *)calloc(n ? n : 1, sizeof(double));
    if (!p) rk_die("内存不足");
    return p;
}

/* out[i] = b[i] + Σ_j W[i*in+j]*x[j] */
static void rk_dense(const double *W, const double *b, int out_n, int in_n,
                     const double *x, double *out) {
    for (int i = 0; i < out_n; i++) {
        const double *row = W + (size_t)i * in_n;
        double sum = 0.0;
        for (int j = 0; j < in_n; j++) sum += row[j] * x[j];
        out[i] = sum + b[i];
    }
}

static double rk_gauss(unsigned long long *st) {
    /* splitmix64 + Box-Muller */
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

static void rk_block_init(RwkvBlock *b, int d, int ff, unsigned long long *st) {
    b->d_model = d;
    b->d_ff = ff;
    b->ln1 = layernorm_create(d, 1e-5);
    b->ln2 = layernorm_create(d, 1e-5);
    b->Wr = rk_alloc((size_t)d * d); b->br = rk_alloc((size_t)d);
    b->Wk = rk_alloc((size_t)d * d); b->bk = rk_alloc((size_t)d);
    b->Wv = rk_alloc((size_t)d * d); b->bv = rk_alloc((size_t)d);
    b->Wo = rk_alloc((size_t)d * d); b->bo = rk_alloc((size_t)d);
    b->w_raw = rk_alloc((size_t)d);
    b->u = rk_alloc((size_t)d);
    b->Wr2 = rk_alloc((size_t)ff * d); b->br2 = rk_alloc((size_t)ff);
    b->Wk2 = rk_alloc((size_t)ff * d); b->bk2 = rk_alloc((size_t)ff);
    b->Wv2 = rk_alloc((size_t)ff * d); b->bv2 = rk_alloc((size_t)ff);
    b->Wo2 = rk_alloc((size_t)d * ff); b->bo2 = rk_alloc((size_t)d);

    /* 初值：decay≈0.9 对应 w_raw≈log(-log(0.9))；u=0.1；其余 gauss(std=sqrt(2/d)) */
    double std = sqrt(2.0 / (double)d);
    double w0 = log(-log(0.9));
    for (int i = 0; i < d; i++) b->w_raw[i] = w0 + rk_gauss(st) * 0.02;
    for (int i = 0; i < d; i++) b->u[i] = 0.1;
    for (int i = 0; i < d * d; i++) {
        b->Wr[i] = rk_gauss(st) * std;
        b->Wk[i] = rk_gauss(st) * std;
        b->Wv[i] = rk_gauss(st) * std;
        b->Wo[i] = rk_gauss(st) * std;
    }
    for (int i = 0; i < ff * d; i++) {
        b->Wr2[i] = rk_gauss(st) * std;
        b->Wk2[i] = rk_gauss(st) * std;
        b->Wv2[i] = rk_gauss(st) * std;
    }
    for (int i = 0; i < d * ff; i++) b->Wo2[i] = rk_gauss(st) * std;
}

static void rk_block_free(RwkvBlock *b) {
    layernorm_free(b->ln1);
    layernorm_free(b->ln2);
    free(b->Wr); free(b->br); free(b->Wk); free(b->bk);
    free(b->Wv); free(b->bv); free(b->Wo); free(b->bo);
    free(b->w_raw); free(b->u);
    free(b->Wr2); free(b->br2); free(b->Wk2); free(b->bk2);
    free(b->Wv2); free(b->bv2); free(b->Wo2); free(b->bo2);
}

static void rk_norm_rows(const RwkvModel *m, LayerNorm *ln, const double *x,
                         int T, double *out) {
    const int d = m->d_model;
    for (int t = 0; t < T; t++) {
        memcpy(m->t_row->data, x + (size_t)t * d, (size_t)d * sizeof(double));
        layernorm_forward(ln, m->t_row, m->t_ln);
        memcpy(out + (size_t)t * d, m->t_ln->data, (size_t)d * sizeof(double));
    }
}

RwkvModel *rwkv_create(int vocab_size, int d_model, int d_ff, int num_layers,
                       int max_len, unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || d_ff < 1 || num_layers < 1 || max_len < 1) {
        rk_die("create: 参数必须是正整数");
    }
    RwkvModel *m = (RwkvModel *)calloc(1, sizeof(RwkvModel));
    if (!m) rk_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    unsigned long long st = seed ? seed : 0x12345ULL;
    double std = sqrt(2.0 / (double)d_model);
    m->emb = rk_alloc((size_t)vocab_size * d_model);
    for (int i = 0; i < vocab_size * d_model; i++) m->emb[i] = rk_gauss(&st) * std;

    m->blocks = (RwkvBlock *)calloc((size_t)num_layers, sizeof(RwkvBlock));
    if (!m->blocks) rk_die("create: 内存不足");
    for (int i = 0; i < num_layers; i++) rk_block_init(&m->blocks[i], d_model, d_ff, &st);

    m->ln_out = layernorm_create(d_model, 1e-5);
    m->W_out = rk_alloc((size_t)vocab_size * d_model);
    m->b_out = rk_alloc((size_t)vocab_size);
    for (int i = 0; i < vocab_size * d_model; i++) m->W_out[i] = rk_gauss(&st) * std;

    size_t Tmax = (size_t)max_len;
    m->buf_x = rk_alloc(Tmax * d_model);
    m->buf_h1 = rk_alloc(Tmax * d_model);
    m->buf_r = rk_alloc(Tmax * d_model);
    m->buf_k = rk_alloc(Tmax * d_model);
    m->buf_v = rk_alloc(Tmax * d_model);
    m->buf_x1 = rk_alloc(Tmax * d_model);
    m->buf_h2 = rk_alloc(Tmax * d_model);
    m->buf_r2 = rk_alloc(Tmax * d_ff);
    m->buf_zk = rk_alloc(Tmax * d_ff);
    m->buf_v2 = rk_alloc(Tmax * d_ff);
    m->buf_y2 = rk_alloc(Tmax * d_ff);
    m->buf_out2 = rk_alloc(Tmax * d_model);

    int shape_d[1] = { d_model };
    m->t_row = tensor_new(1, shape_d);
    m->t_ln = tensor_new(1, shape_d);
    return m;
}

void rwkv_free(RwkvModel *m) {
    if (!m) return;
    free(m->emb);
    for (int i = 0; i < m->num_layers; i++) rk_block_free(&m->blocks[i]);
    free(m->blocks);
    layernorm_free(m->ln_out);
    free(m->W_out); free(m->b_out);
    free(m->buf_x); free(m->buf_h1); free(m->buf_r); free(m->buf_k); free(m->buf_v);
    free(m->buf_x1); free(m->buf_h2); free(m->buf_r2); free(m->buf_zk);
    free(m->buf_v2); free(m->buf_y2); free(m->buf_out2);
    tensor_free(m->t_row);
    tensor_free(m->t_ln);
    free(m);
}

int rwkv_param_count(const RwkvModel *m) {
    int d = m->d_model, ff = m->d_ff;
    int n = m->vocab_size * d + m->vocab_size;   /* emb + 词表头 bias 之外 */
    n += 2 * d;                                  /* ln_out */
    n += m->vocab_size * d;                      /* 词表头权重 */
    for (int i = 0; i < m->num_layers; i++) {
        n += 4 * d;                              /* ln1/ln2 各 gamma+beta */
        n += 4 * (d * d + d);                    /* Wr/Wk/Wv/Wo + bias */
        n += 2 * d;                              /* w_raw + u */
        n += 3 * (ff * d + ff);                  /* Wr2/Wk2/Wv2 + bias */
        n += d * ff + d;                         /* Wo2 + bias */
    }
    return n;
}

void rwkv_forward(RwkvModel *m, const int *ids, int T, double *logits) {
    if (!ids || T <= 0) rk_die("forward: ids 不能为空");
    if (T > m->max_len) rk_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) rk_die("forward: token id 越界");
    }
    const int d = m->d_model, ff = m->d_ff;

    for (int t = 0; t < T; t++) {
        memcpy(m->buf_x + (size_t)t * d, m->emb + (size_t)ids[t] * d,
               (size_t)d * sizeof(double));
    }

    for (int li = 0; li < m->num_layers; li++) {
        RwkvBlock *b = &m->blocks[li];
        rk_norm_rows(m, b->ln1, m->buf_x, T, m->buf_h1);
        for (int t = 0; t < T; t++) {
            const double *h = m->buf_h1 + (size_t)t * d;
            double *r = m->buf_r + (size_t)t * d;
            double *k = m->buf_k + (size_t)t * d;
            double *v = m->buf_v + (size_t)t * d;
            rk_dense(b->Wr, b->br, d, d, h, r);
            rk_dense(b->Wk, b->bk, d, d, h, k);
            rk_dense(b->Wv, b->bv, d, d, h, v);
        }
        /* time-mixing 状态递推 → y → Wo → 残差 */
        double *state = (double *)rk_alloc((size_t)d);
        for (int t = 0; t < T; t++) {
            const double *r = m->buf_r + (size_t)t * d;
            const double *k = m->buf_k + (size_t)t * d;
            const double *v = m->buf_v + (size_t)t * d;
            double *y = m->buf_x1 + (size_t)t * d;   /* 先暂存 y，之后再算 x1 */
            for (int j = 0; j < d; j++) {
                double decay = exp(-exp(b->w_raw[j]));
                state[j] = decay * state[j] + k[j] * v[j];
                y[j] = r[j] * (state[j] + b->u[j] * v[j]);
            }
        }
        free(state);
        for (int t = 0; t < T; t++) {
            double *out1 = m->buf_h1 + (size_t)t * d;   /* 复用 h1 存 out1 */
            rk_dense(b->Wo, b->bo, d, d, m->buf_x1 + (size_t)t * d, out1);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                m->buf_x1[(size_t)t * d + j] =
                    m->buf_x[(size_t)t * d + j] + m->buf_h1[(size_t)t * d + j];
            }
        }
        /* channel-mixing */
        rk_norm_rows(m, b->ln2, m->buf_x1, T, m->buf_h2);
        for (int t = 0; t < T; t++) {
            const double *h = m->buf_h2 + (size_t)t * d;
            double *r2 = m->buf_r2 + (size_t)t * ff;
            double *zk = m->buf_zk + (size_t)t * ff;
            double *v2 = m->buf_v2 + (size_t)t * ff;
            double *y2 = m->buf_y2 + (size_t)t * ff;
            rk_dense(b->Wr2, b->br2, ff, d, h, r2);
            rk_dense(b->Wk2, b->bk2, ff, d, h, zk);
            rk_dense(b->Wv2, b->bv2, ff, d, h, v2);
            for (int j = 0; j < ff; j++) {
                double kact = zk[j] > 0.0 ? zk[j] : 0.0;
                y2[j] = r2[j] * (kact * kact) * v2[j];
            }
        }
        for (int t = 0; t < T; t++) {
            rk_dense(b->Wo2, b->bo2, d, ff, m->buf_y2 + (size_t)t * ff,
                     m->buf_out2 + (size_t)t * d);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                m->buf_x[(size_t)t * d + j] =
                    m->buf_x1[(size_t)t * d + j] + m->buf_out2[(size_t)t * d + j];
            }
        }
    }

    /* 输出 LayerNorm + 词表头（逐行，prompt 式前向） */
    double *h = (double *)rk_alloc((size_t)T * d);
    rk_norm_rows(m, m->ln_out, m->buf_x, T, h);
    for (int t = 0; t < T; t++) {
        rk_dense(m->W_out, m->b_out, m->vocab_size, d, h + (size_t)t * d,
                 logits + (size_t)t * m->vocab_size);
    }
    free(h);
}
