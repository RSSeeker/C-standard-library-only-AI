/* Mamba 风格语言模型前向实现，见 mamba.h。 */
#include "mamba.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void mb_die(const char *msg) {
    fprintf(stderr, "MambaModel: %s\n", msg);
    abort();
}

static double *mb_alloc(size_t n) {
    double *p = (double *)calloc(n ? n : 1, sizeof(double));
    if (!p) mb_die("内存不足");
    return p;
}

static void mb_dense(const double *W, const double *b, int out_n, int in_n,
                     const double *x, double *out) {
    for (int i = 0; i < out_n; i++) {
        const double *row = W + (size_t)i * in_n;
        double sum = 0.0;
        for (int j = 0; j < in_n; j++) sum += row[j] * x[j];
        out[i] = sum + b[i];
    }
}

static double mb_gauss(unsigned long long *st) {
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

static double mb_silu(double v) { return v / (1.0 + exp(-v)); }

static double mb_softplus(double v) {
    if (v > 30.0) return v;
    return log1p(exp(v));
}

static void mb_block_init(MambaBlock *b, int d, int d_state, int d_conv, int expand,
                          unsigned long long *st) {
    b->d_model = d;
    b->d_state = d_state;
    b->d_conv = d_conv;
    b->expand = expand;
    b->d_inner = expand * d;
    const int di = b->d_inner;
    double std = sqrt(2.0 / (double)d);
    b->W_in = mb_alloc((size_t)2 * di * d); b->b_in = mb_alloc((size_t)2 * di);
    b->conv_w = mb_alloc((size_t)di * di * d_conv); b->conv_b = mb_alloc((size_t)di);
    b->W_delta = mb_alloc((size_t)di * di); b->b_delta = mb_alloc((size_t)di);
    b->W_b = mb_alloc((size_t)d_state * di); b->b_b = mb_alloc((size_t)d_state);
    b->W_c = mb_alloc((size_t)d_state * di); b->b_c = mb_alloc((size_t)d_state);
    b->A = mb_alloc((size_t)di * d_state); b->D = mb_alloc((size_t)di);
    b->W_out = mb_alloc((size_t)d * di); b->b_out = mb_alloc((size_t)d);
    for (int i = 0; i < 2 * di * d; i++) b->W_in[i] = mb_gauss(st) * std;
    for (int i = 0; i < di * di * d_conv; i++) b->conv_w[i] = mb_gauss(st) / sqrt((double)di * d_conv);
    for (int i = 0; i < di * di; i++) b->W_delta[i] = mb_gauss(st) * std;
    for (int i = 0; i < d_state * di; i++) {
        b->W_b[i] = mb_gauss(st) * std;
        b->W_c[i] = mb_gauss(st) * std;
    }
    for (int i = 0; i < di; i++) {
        for (int s = 0; s < d_state; s++) b->A[(size_t)i * d_state + s] = -(1.0 + s);
        b->D[i] = 1.0;
    }
    for (int i = 0; i < d * di; i++) b->W_out[i] = mb_gauss(st) * std;
    b->x = b->proj = b->conv_out = b->u = b->delta = b->Bij = b->Cij = NULL;
    b->y = b->gated = NULL;
    b->h = NULL;
}

static void mb_block_alloc(MambaBlock *b, int max_len) {
    const int di = b->d_inner;
    size_t T = (size_t)max_len;
    b->x = mb_alloc(T * b->d_model);
    b->proj = mb_alloc(T * 2 * di);
    b->conv_out = mb_alloc(T * di);
    b->u = mb_alloc(T * di);
    b->delta = mb_alloc(T * di);
    b->Bij = mb_alloc(T * b->d_state);
    b->Cij = mb_alloc(T * b->d_state);
    b->y = mb_alloc(T * di);
    b->gated = mb_alloc(T * di);
    b->h = mb_alloc((size_t)di * b->d_state);
}

static void mb_block_free(MambaBlock *b) {
    free(b->W_in); free(b->b_in); free(b->conv_w); free(b->conv_b);
    free(b->W_delta); free(b->b_delta); free(b->W_b); free(b->b_b);
    free(b->W_c); free(b->b_c); free(b->A); free(b->D);
    free(b->W_out); free(b->b_out);
    free(b->x); free(b->proj); free(b->conv_out); free(b->u); free(b->delta);
    free(b->Bij); free(b->Cij); free(b->y); free(b->gated); free(b->h);
}

MambaModel *mamba_create(int vocab_size, int d_model, int d_state, int d_conv,
                         int expand, int num_layers, int max_len,
                         unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || d_state < 1 || d_conv < 1 ||
        expand < 1 || num_layers < 1 || max_len < 1) {
        mb_die("create: 参数必须是正整数");
    }
    MambaModel *m = (MambaModel *)calloc(1, sizeof(MambaModel));
    if (!m) mb_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_layers = num_layers;
    m->max_len = max_len;

    unsigned long long st = seed ? seed : 0x12345ULL;
    double std = sqrt(2.0 / (double)d_model);
    m->emb = mb_alloc((size_t)vocab_size * d_model);
    for (int i = 0; i < vocab_size * d_model; i++) m->emb[i] = mb_gauss(&st) * std;

    m->blocks = (MambaBlock *)calloc((size_t)num_layers, sizeof(MambaBlock));
    if (!m->blocks) mb_die("create: 内存不足");
    for (int i = 0; i < num_layers; i++) {
        mb_block_init(&m->blocks[i], d_model, d_state, d_conv, expand, &st);
        mb_block_alloc(&m->blocks[i], max_len);
    }

    m->W_head = mb_alloc((size_t)vocab_size * d_model);
    m->b_head = mb_alloc((size_t)vocab_size);
    for (int i = 0; i < vocab_size * d_model; i++) m->W_head[i] = mb_gauss(&st) * std;
    return m;
}

void mamba_free(MambaModel *m) {
    if (!m) return;
    free(m->emb);
    for (int i = 0; i < m->num_layers; i++) mb_block_free(&m->blocks[i]);
    free(m->blocks);
    free(m->W_head); free(m->b_head);
    free(m);
}

int mamba_param_count(const MambaModel *m) {
    int d = m->d_model;
    int n = m->vocab_size * d;                 /* embedding（无 bias） */
    for (int li = 0; li < m->num_layers; li++) {
        const MambaBlock *b = &m->blocks[li];
        int di = b->d_inner;
        n += 2 * di * d + 2 * di;          /* in_proj */
        n += di * di * b->d_conv + di;     /* causal conv */
        n += di * di + di;                 /* delta_proj */
        n += b->d_state * di + b->d_state; /* b_proj */
        n += b->d_state * di + b->d_state; /* c_proj */
        n += di * b->d_state + di;         /* A + D */
        n += d * di + d;                   /* out_proj */
    }
    n += m->vocab_size * d + m->vocab_size; /* head */
    return n;
}

static void mb_causal_conv(const MambaBlock *b, int T) {
    const int di = b->d_inner;
    const int k = b->d_conv;
    for (int t = 0; t < T; t++) {
        for (int o = 0; o < di; o++) {
            double sum = b->conv_b[o];
            for (int kk = 0; kk < k; kk++) {
                int src = t - (k - 1) + kk;
                if (src < 0) continue;
                const double *xrow = b->proj + (size_t)src * 2 * di; /* x 部分 */
                const double *wrow = b->conv_w + ((size_t)o * di + 0) * k;
                for (int ic = 0; ic < di; ic++) {
                    const double *w = b->conv_w + (((size_t)o * di + ic) * k);
                    sum += xrow[ic] * w[kk];
                }
                (void)wrow;
            }
            b->conv_out[(size_t)t * di + o] = sum;
        }
    }
}

void mamba_forward(MambaModel *m, const int *ids, int T, double *logits) {
    if (!ids || T <= 0) mb_die("forward: ids 不能为空");
    if (T > m->max_len) mb_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) mb_die("forward: token id 越界");
    }
    const int d = m->d_model;

    for (int li = 0; li < m->num_layers; li++) {
        MambaBlock *b = &m->blocks[li];
        const int di = b->d_inner;
        if (li == 0) {
            for (int t = 0; t < T; t++) {
                memcpy(b->x + (size_t)t * d, m->emb + (size_t)ids[t] * d,
                       (size_t)d * sizeof(double));
            }
        } else {
            memcpy(b->x, m->blocks[li - 1].x, (size_t)T * d * sizeof(double));
        }

        for (int t = 0; t < T; t++) {
            mb_dense(b->W_in, b->b_in, 2 * di, d, b->x + (size_t)t * d,
                     b->proj + (size_t)t * 2 * di);
        }
        mb_causal_conv(b, T);
        for (int t = 0; t < T; t++) {
            const double *xs = b->proj + (size_t)t * 2 * di;
            const double *zs = xs + di;
            for (int i = 0; i < di; i++) {
                double u = mb_silu(b->conv_out[(size_t)t * di + i]);
                b->u[(size_t)t * di + i] = u;
                b->gated[(size_t)t * di + i] = mb_silu(zs[i]); /* 暂存 silu(z) */
            }
            (void)xs;
        }
        for (int t = 0; t < T; t++) {
            mb_dense(b->W_delta, b->b_delta, di, di, b->u + (size_t)t * di,
                     b->delta + (size_t)t * di);
            for (int i = 0; i < di; i++) {
                b->delta[(size_t)t * di + i] = mb_softplus(b->delta[(size_t)t * di + i]);
            }
            mb_dense(b->W_b, b->b_b, b->d_state, di, b->u + (size_t)t * di,
                     b->Bij + (size_t)t * b->d_state);
            mb_dense(b->W_c, b->b_c, b->d_state, di, b->u + (size_t)t * di,
                     b->Cij + (size_t)t * b->d_state);
        }

        memset(b->h, 0, (size_t)di * b->d_state * sizeof(double));
        for (int t = 0; t < T; t++) {
            const double *ut = b->u + (size_t)t * di;
            const double *dt = b->delta + (size_t)t * di;
            const double *Bt = b->Bij + (size_t)t * b->d_state;
            const double *Ct = b->Cij + (size_t)t * b->d_state;
            double *yt = b->y + (size_t)t * di;
            for (int i = 0; i < di; i++) {
                double *hi = b->h + (size_t)i * b->d_state;
                for (int s = 0; s < b->d_state; s++) {
                    double decay = exp(dt[i] * b->A[(size_t)i * b->d_state + s]);
                    hi[s] = decay * hi[s] + dt[i] * Bt[s] * ut[i];
                }
                double acc = 0.0;
                for (int s = 0; s < b->d_state; s++) acc += Ct[s] * hi[s];
                yt[i] = acc + b->D[i] * ut[i];
            }
        }
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < di; i++) {
                b->y[(size_t)t * di + i] *= b->gated[(size_t)t * di + i];
            }
        }
        for (int t = 0; t < T; t++) {
            mb_dense(b->W_out, b->b_out, d, di, b->y + (size_t)t * di,
                     b->proj + (size_t)t * 2 * di); /* 暂存 projected */
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                b->x[(size_t)t * d + j] =
                    b->x[(size_t)t * d + j] + b->proj[(size_t)t * 2 * di + j];
            }
        }
    }

    double *last = m->blocks[m->num_layers - 1].x;
    for (int t = 0; t < T; t++) {
        mb_dense(m->W_head, m->b_head, m->vocab_size, d, last + (size_t)t * d,
                 logits + (size_t)t * m->vocab_size);
    }
}
