/* xLSTM 风格语言模型前向实现，见 xlstm.h。 */
#include "xlstm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void xs_die(const char *msg) {
    fprintf(stderr, "XLSTMModel: %s\n", msg);
    abort();
}

static double *xs_alloc(size_t n) {
    double *p = (double *)calloc(n ? n : 1, sizeof(double));
    if (!p) xs_die("内存不足");
    return p;
}

static void xs_dense(const double *W, const double *b, int out_n, int in_n,
                     const double *x, double *out) {
    for (int i = 0; i < out_n; i++) {
        const double *row = W + (size_t)i * in_n;
        double sum = 0.0;
        for (int j = 0; j < in_n; j++) sum += row[j] * x[j];
        out[i] = sum + b[i];
    }
}

static double xs_gauss(unsigned long long *st) {
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

static void xs_block_init(XLSTMBlock *b, int d, int ff, unsigned long long *st) {
    b->d_model = d;
    b->d_ff = ff;
    b->eps = 1e-5;
    b->ln1 = layernorm_create(d, 1e-5);
    b->ln2 = layernorm_create(d, 1e-5);
    b->Wg = xs_alloc((size_t)4 * d * d); b->bg = xs_alloc((size_t)4 * d);
    b->Wo = xs_alloc((size_t)d * d); b->bo = xs_alloc((size_t)d);
    b->W1 = xs_alloc((size_t)ff * d); b->b1 = xs_alloc((size_t)ff);
    b->W2 = xs_alloc((size_t)d * ff); b->b2 = xs_alloc((size_t)d);
    b->x = b->h1 = b->gates = b->s = NULL;
    b->x1 = b->h2 = b->a2 = b->out2 = NULL;
    double std = sqrt(2.0 / (double)d);
    for (int i = 0; i < 4 * d * d; i++) b->Wg[i] = xs_gauss(st) * std;
    for (int i = 0; i < d * d; i++) b->Wo[i] = xs_gauss(st) * std;
    for (int i = 0; i < ff * d; i++) b->W1[i] = xs_gauss(st) * std;
    for (int i = 0; i < d * ff; i++) b->W2[i] = xs_gauss(st) * std;
}

static void xs_block_alloc(XLSTMBlock *b, int max_len) {
    const int d = b->d_model;
    const int ff = b->d_ff;
    size_t T = (size_t)max_len;
    b->x = xs_alloc(T * d);
    b->h1 = xs_alloc(T * d);
    b->gates = xs_alloc(T * 4 * d);
    b->s = xs_alloc(T * d);
    b->x1 = xs_alloc(T * d);
    b->h2 = xs_alloc(T * d);
    b->a2 = xs_alloc(T * ff);
    b->out2 = xs_alloc(T * d);
}

static void xs_block_free(XLSTMBlock *b) {
    layernorm_free(b->ln1);
    layernorm_free(b->ln2);
    free(b->Wg); free(b->bg); free(b->Wo); free(b->bo);
    free(b->W1); free(b->b1); free(b->W2); free(b->b2);
    free(b->x); free(b->h1); free(b->gates); free(b->s);
    free(b->x1); free(b->h2); free(b->a2); free(b->out2);
}

static void xs_norm_rows(const XLSTMModel *m, LayerNorm *ln, const double *x,
                         int T, double *out) {
    const int d = m->d_model;
    for (int t = 0; t < T; t++) {
        memcpy(m->t_row->data, x + (size_t)t * d, (size_t)d * sizeof(double));
        layernorm_forward(ln, m->t_row, m->t_ln);
        memcpy(out + (size_t)t * d, m->t_ln->data, (size_t)d * sizeof(double));
    }
}

XLSTMModel *xlstm_create(int vocab_size, int d_model, int d_ff, int num_layers,
                         int max_len, unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || d_ff < 1 || num_layers < 1 || max_len < 1) {
        xs_die("create: 参数必须是正整数");
    }
    XLSTMModel *m = (XLSTMModel *)calloc(1, sizeof(XLSTMModel));
    if (!m) xs_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    unsigned long long st = seed ? seed : 0x12345ULL;
    double std = sqrt(2.0 / (double)d_model);
    m->emb = xs_alloc((size_t)vocab_size * d_model);
    for (int i = 0; i < vocab_size * d_model; i++) m->emb[i] = xs_gauss(&st) * std;

    m->blocks = (XLSTMBlock *)calloc((size_t)num_layers, sizeof(XLSTMBlock));
    if (!m->blocks) xs_die("create: 内存不足");
    for (int i = 0; i < num_layers; i++) {
        xs_block_init(&m->blocks[i], d_model, d_ff, &st);
        xs_block_alloc(&m->blocks[i], max_len);
    }

    m->ln_out = layernorm_create(d_model, 1e-5);
    m->W_out = xs_alloc((size_t)vocab_size * d_model);
    m->b_out = xs_alloc((size_t)vocab_size);
    for (int i = 0; i < vocab_size * d_model; i++) m->W_out[i] = xs_gauss(&st) * std;

    int shape_d[1] = { d_model };
    m->t_row = tensor_new(1, shape_d);
    m->t_ln = tensor_new(1, shape_d);
    return m;
}

void xlstm_free(XLSTMModel *m) {
    if (!m) return;
    free(m->emb);
    for (int i = 0; i < m->num_layers; i++) xs_block_free(&m->blocks[i]);
    free(m->blocks);
    layernorm_free(m->ln_out);
    free(m->W_out); free(m->b_out);
    tensor_free(m->t_row);
    tensor_free(m->t_ln);
    free(m);
}

int xlstm_param_count(const XLSTMModel *m) {
    int d = m->d_model, ff = m->d_ff;
    int n = 2 * m->vocab_size * d + m->vocab_size + 2 * d;
    for (int i = 0; i < m->num_layers; i++) {
        n += 4 * d;                 /* ln1/ln2 gamma+beta */
        n += 4 * d * d + 4 * d;     /* Wg + bias */
        n += d * d + d;             /* Wo + bias */
        n += ff * d + ff;           /* W1 + bias */
        n += d * ff + d;            /* W2 + bias */
    }
    return n;
}

static void xs_cell_forward(const XLSTMBlock *b, int T, double *h) {
    const int d = b->d_model;
    double *c_prev = (double *)xs_alloc((size_t)d);
    double *n_prev = (double *)xs_alloc((size_t)d);
    double *c_t = (double *)xs_alloc((size_t)d);
    double *n_t = (double *)xs_alloc((size_t)d);
    for (int t = 0; t < T; t++) {
        const double *row = b->gates + (size_t)t * 4 * d;
        double *ht = h + (size_t)t * d;
        for (int j = 0; j < d; j++) {
            double i_pre = row[j];
            double f_pre = row[d + j];
            double z_pre = row[2 * d + j];
            double o_pre = row[3 * d + j];
            double ii = exp(fmax(-10.0, fmin(5.0, i_pre)));
            double ff = 1.0 / (1.0 + exp(-f_pre));
            double zz = tanh(z_pre);
            double oo = 1.0 / (1.0 + exp(-o_pre));
            c_t[j] = ff * c_prev[j] + ii * zz;
            n_t[j] = ff * n_prev[j] + ii;
            double denom = fabs(n_t[j]);
            if (denom < b->eps) denom = b->eps;
            ht[j] = oo * c_t[j] / denom;
        }
        memcpy(c_prev, c_t, (size_t)d * sizeof(double));
        memcpy(n_prev, n_t, (size_t)d * sizeof(double));
    }
    free(c_prev); free(n_prev); free(c_t); free(n_t);
}

void xlstm_forward(XLSTMModel *m, const int *ids, int T, double *logits) {
    if (!ids || T <= 0) xs_die("forward: ids 不能为空");
    if (T > m->max_len) xs_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) xs_die("forward: token id 越界");
    }
    const int d = m->d_model;
    const int ff = m->d_ff;

    for (int li = 0; li < m->num_layers; li++) {
        XLSTMBlock *b = &m->blocks[li];
        if (li == 0) {
            for (int t = 0; t < T; t++) {
                memcpy(b->x + (size_t)t * d, m->emb + (size_t)ids[t] * d,
                       (size_t)d * sizeof(double));
            }
        } else {
            memcpy(b->x, m->blocks[li - 1].x, (size_t)T * d * sizeof(double));
        }

        xs_norm_rows(m, b->ln1, b->x, T, b->h1);
        for (int t = 0; t < T; t++) {
            xs_dense(b->Wg, b->bg, 4 * d, d, b->h1 + (size_t)t * d,
                     b->gates + (size_t)t * 4 * d);
        }
        xs_cell_forward(b, T, b->s);
        for (int t = 0; t < T; t++) {
            xs_dense(b->Wo, b->bo, d, d, b->s + (size_t)t * d,
                     b->x1 + (size_t)t * d);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                b->x1[(size_t)t * d + j] =
                    b->x[(size_t)t * d + j] + b->x1[(size_t)t * d + j];
            }
        }

        xs_norm_rows(m, b->ln2, b->x1, T, b->h2);
        for (int t = 0; t < T; t++) {
            double *a2 = b->a2 + (size_t)t * ff;
            xs_dense(b->W1, b->b1, ff, d, b->h2 + (size_t)t * d, a2);
            for (int j = 0; j < ff; j++) {
                if (a2[j] < 0.0) a2[j] = 0.0;
            }
        }
        for (int t = 0; t < T; t++) {
            xs_dense(b->W2, b->b2, d, ff, b->a2 + (size_t)t * ff,
                     b->out2 + (size_t)t * d);
        }
        for (int t = 0; t < T; t++) {
            for (int j = 0; j < d; j++) {
                b->x[(size_t)t * d + j] =
                    b->x1[(size_t)t * d + j] + b->out2[(size_t)t * d + j];
            }
        }
    }

    double *h = xs_alloc((size_t)T * d);
    xs_norm_rows(m, m->ln_out, m->blocks[m->num_layers - 1].x, T, h);
    for (int t = 0; t < T; t++) {
        xs_dense(m->W_out, m->b_out, m->vocab_size, d, h + (size_t)t * d,
                 logits + (size_t)t * m->vocab_size);
    }
    free(h);
}
