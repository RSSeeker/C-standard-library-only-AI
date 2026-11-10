/* DeBERTa 风格语言模型前向实现（与 AI.py 的 DeBERTaBlock/DeBERTaModel 对齐）。 */
#include "deberta.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DB_DIE(msg) do { fprintf(stderr, "%s\n", (msg)); exit(1); } while (0)

static void *db_xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) DB_DIE("deberta: 内存不足");
    return p;
}

static void db_dense(int out_dim, int in_dim, const double *W, const double *b,
                     const double *x, double *y)
{
    for (int o = 0; o < out_dim; o++) {
        double s = b ? b[o] : 0.0;
        const double *row = W + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) s += row[i] * x[i];
        y[o] = s;
    }
}

static void db_softmax_row(double *row, int n)
{
    double mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        row[i] = exp(row[i] - mx);
        sum += row[i];
    }
    for (int i = 0; i < n; i++) row[i] /= sum;
}

static void db_norm_rows(LayerNorm *ln, Tensor *tin, Tensor *tout,
                         const double *x, int T, int d, double *out)
{
    for (int t = 0; t < T; t++) {
        memcpy(tin->data, x + (size_t)t * d, (size_t)d * sizeof(double));
        layernorm_forward(ln, tin, tout);
        memcpy(out + (size_t)t * d, tout->data, (size_t)d * sizeof(double));
    }
}

static void db_block_init(DebertaBlock *b, int d_model, int d_ff, int max_rel, int max_len)
{
    memset(b, 0, sizeof(*b));
    b->d_model = d_model;
    b->d_ff = d_ff;
    b->max_rel = max_rel;
    b->n_rel = 2 * max_rel + 1;
    int d = d_model, ff = d_ff;
    b->ln1 = layernorm_create(d, 1e-5);
    b->ln2 = layernorm_create(d, 1e-5);
    b->Wq = (double *)db_xcalloc((size_t)d * d, sizeof(double));
    b->Wk = (double *)db_xcalloc((size_t)d * d, sizeof(double));
    b->Wv = (double *)db_xcalloc((size_t)d * d, sizeof(double));
    b->Wo = (double *)db_xcalloc((size_t)d * d, sizeof(double));
    b->bq = (double *)db_xcalloc((size_t)d, sizeof(double));
    b->bk = (double *)db_xcalloc((size_t)d, sizeof(double));
    b->bv = (double *)db_xcalloc((size_t)d, sizeof(double));
    b->bo = (double *)db_xcalloc((size_t)d, sizeof(double));
    b->Rk = (double *)db_xcalloc((size_t)b->n_rel * d, sizeof(double));
    b->Rq = (double *)db_xcalloc((size_t)b->n_rel * d, sizeof(double));
    b->W1 = (double *)db_xcalloc((size_t)ff * d, sizeof(double));
    b->W2 = (double *)db_xcalloc((size_t)d * ff, sizeof(double));
    b->b1 = (double *)db_xcalloc((size_t)ff, sizeof(double));
    b->b2 = (double *)db_xcalloc((size_t)d, sizeof(double));
    b->h1 = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->q = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->k = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->v = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->scores = (double *)db_xcalloc((size_t)max_len * max_len, sizeof(double));
    b->attn = (double *)db_xcalloc((size_t)max_len * max_len, sizeof(double));
    b->vmix = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->orows = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->x1 = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->h2 = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->a2 = (double *)db_xcalloc((size_t)max_len * ff, sizeof(double));
    b->ffn = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    b->out = (double *)db_xcalloc((size_t)max_len * d, sizeof(double));
    int shape[1] = { d };
    b->tin = tensor_new(1, shape);
    b->tout = tensor_new(1, shape);
}

static void db_block_free(DebertaBlock *b)
{
    if (!b) return;
    free(b->Wq); free(b->Wk); free(b->Wv); free(b->Wo);
    free(b->bq); free(b->bk); free(b->bv); free(b->bo);
    free(b->Rk); free(b->Rq);
    free(b->W1); free(b->W2); free(b->b1); free(b->b2);
    free(b->h1); free(b->q); free(b->k); free(b->v);
    free(b->scores); free(b->attn); free(b->vmix); free(b->orows);
    free(b->x1); free(b->h2); free(b->a2); free(b->ffn); free(b->out);
    tensor_free(b->tin); tensor_free(b->tout);
}

static void db_block_forward(DebertaBlock *b, const double *x, int T, double *out)
{
    int d = b->d_model, ff = b->d_ff;
    double scale = 1.0 / sqrt((double)d);

    db_norm_rows(b->ln1, b->tin, b->tout, x, T, d, b->h1);
    for (int t = 0; t < T; t++) {
        const double *row = b->h1 + (size_t)t * d;
        db_dense(d, d, b->Wq, b->bq, row, b->q + (size_t)t * d);
        db_dense(d, d, b->Wk, b->bk, row, b->k + (size_t)t * d);
        db_dense(d, d, b->Wv, b->bv, row, b->v + (size_t)t * d);
    }

    for (int t = 0; t < T; t++) {
        double *sc = b->scores + (size_t)t * T;
        const double *q_t = b->q + (size_t)t * d;
        for (int j = 0; j <= t; j++) {
            int delta = t - j;
            if (delta > b->max_rel) delta = b->max_rel;
            const double *rk = b->Rk + (size_t)delta * d;
            const double *rq = b->Rq + (size_t)delta * d;
            const double *k_j = b->k + (size_t)j * d;
            double cc = 0.0, cp = 0.0, pc = 0.0;
            for (int i = 0; i < d; i++) {
                cc += q_t[i] * k_j[i];
                cp += q_t[i] * rk[i];
                pc += k_j[i] * rq[i];
            }
            sc[j] = scale * (cc + cp + pc);
        }
        for (int j = t + 1; j < T; j++) sc[j] = -1e30;
        db_softmax_row(sc, T);
        double *arow = b->attn + (size_t)t * T;
        memcpy(arow, sc, (size_t)T * sizeof(double));
    }

    for (int t = 0; t < T; t++) {
        double *mix = b->vmix + (size_t)t * d;
        memset(mix, 0, (size_t)d * sizeof(double));
        const double *arow = b->attn + (size_t)t * T;
        for (int j = 0; j <= t; j++) {
            double a = arow[j];
            const double *vj = b->v + (size_t)j * d;
            for (int i = 0; i < d; i++) mix[i] += a * vj[i];
        }
    }
    for (int t = 0; t < T; t++) {
        db_dense(d, d, b->Wo, b->bo, b->vmix + (size_t)t * d, b->orows + (size_t)t * d);
        for (int i = 0; i < d; i++)
            b->x1[(size_t)t * d + i] = x[(size_t)t * d + i] + b->orows[(size_t)t * d + i];
    }
    db_norm_rows(b->ln2, b->tin, b->tout, b->x1, T, d, b->h2);
    for (int t = 0; t < T; t++) {
        db_dense(ff, d, b->W1, b->b1, b->h2 + (size_t)t * d, b->a2 + (size_t)t * ff);
        for (int i = 0; i < ff; i++) {
            double v = b->a2[(size_t)t * ff + i];
            b->a2[(size_t)t * ff + i] = v > 0.0 ? v : 0.0;
        }
        db_dense(d, ff, b->W2, b->b2, b->a2 + (size_t)t * ff, b->ffn + (size_t)t * d);
        for (int i = 0; i < d; i++)
            out[(size_t)t * d + i] = b->x1[(size_t)t * d + i] + b->ffn[(size_t)t * d + i];
    }
}

DebertaModel *deberta_create(int vocab_size, int d_model, int d_ff, int num_layers,
                             int max_len, int max_rel, unsigned long long seed)
{
    (void)seed;
    if (vocab_size < 1 || d_model < 1 || num_layers < 1 || max_len < 1 || max_rel < 1)
        DB_DIE("deberta_create: 参数非法");
    if (d_ff <= 0) d_ff = 4 * d_model;
    DebertaModel *m = (DebertaModel *)db_xcalloc(1, sizeof(DebertaModel));
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_layers = num_layers;
    m->max_len = max_len;
    m->max_rel = max_rel;
    m->emb = (double *)db_xcalloc((size_t)vocab_size * d_model, sizeof(double));
    m->blocks = (DebertaBlock *)db_xcalloc((size_t)num_layers, sizeof(DebertaBlock));
    for (int i = 0; i < num_layers; i++)
        db_block_init(&m->blocks[i], d_model, d_ff, max_rel, max_len);
    m->ln_out = layernorm_create(d_model, 1e-5);
    m->W_out = (double *)db_xcalloc((size_t)vocab_size * d_model, sizeof(double));
    m->b_out = (double *)db_xcalloc((size_t)vocab_size, sizeof(double));
    m->seq_buf = (double *)db_xcalloc((size_t)max_len * d_model, sizeof(double));
    return m;
}

void deberta_free(DebertaModel *m)
{
    if (!m) return;
    free(m->emb);
    for (int i = 0; i < m->num_layers; i++)
        db_block_free(&m->blocks[i]);
    free(m->blocks);
    layernorm_free(m->ln_out);
    free(m->W_out);
    free(m->b_out);
    free(m->seq_buf);
    free(m);
}

int deberta_param_count(const DebertaModel *m)
{
    int d = m->d_model;
    int ff = m->blocks ? m->blocks->d_ff : 0;
    int n_rel = 2 * m->max_rel + 1;
    int n = m->vocab_size * d;
    for (int li = 0; li < m->num_layers; li++) {
        n += 4 * (d * d + d);            /* Wq/Wk/Wv/Wo + biases */
        n += ff * d + ff;                /* W1/b1 */
        n += d * ff + d;                 /* W2/b2 */
        n += 4 * d;                      /* ln1 + ln2 gamma/beta */
        n += 2 * n_rel * d;              /* Rk + Rq */
    }
    n += 2 * d;                          /* ln_out */
    n += m->vocab_size * d + m->vocab_size; /* head */
    return n;
}

void deberta_forward(DebertaModel *m, const int *ids, int T, double *logits)
{
    int d = m->d_model;
    for (int t = 0; t < T; t++) {
        memcpy(m->seq_buf + (size_t)t * d, m->emb + (size_t)ids[t] * d,
               (size_t)d * sizeof(double));
    }
    for (int li = 0; li < m->num_layers; li++) {
        db_block_forward(&m->blocks[li], m->seq_buf, T, m->seq_buf);
    }
    Tensor *tin = m->blocks[0].tin;
    Tensor *tout = m->blocks[0].tout;
    db_norm_rows(m->ln_out, tin, tout, m->seq_buf, T, d, m->seq_buf);
    for (int t = 0; t < T; t++) {
        db_dense(m->vocab_size, d, m->W_out, m->b_out,
                 m->seq_buf + (size_t)t * d, logits + (size_t)t * m->vocab_size);
    }
}
