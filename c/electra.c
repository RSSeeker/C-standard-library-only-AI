/* ELECTRA 判别器前向实现，见 electra.h。 */
#include "electra.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void ed_die(const char *msg) {
    fprintf(stderr, "ElectraDiscriminator: %s\n", msg);
    abort();
}

ElectraDiscriminator *electra_disc_create(int vocab_size, int d_model, int num_heads,
                                          int d_ff, int num_layers, int max_len,
                                          unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || num_heads < 1 || d_ff < 1 ||
        num_layers < 1 || max_len < 1) {
        ed_die("create: 参数必须是正整数");
    }
    if (d_model % num_heads != 0) ed_die("create: d_model 必须能被 num_heads 整除");

    ElectraDiscriminator *m = (ElectraDiscriminator *)calloc(1, sizeof(ElectraDiscriminator));
    if (!m) ed_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    m->emb = emb_create(vocab_size + 1, d_model, max_len);
    m->pe = pe_create(d_model, max_len);
    m->encoder = te_create(d_model, num_heads, d_ff, num_layers, max_len, 0.0, seed);
    te_set_train(m->encoder, 0);

    m->W_d = (double *)calloc((size_t)2 * (size_t)d_model, sizeof(double));
    m->b_d = (double *)calloc(2, sizeof(double));
    if (!m->W_d || !m->b_d) ed_die("create: 内存不足");

    int shape[2] = { max_len, d_model };
    m->t_emb = tensor_new(2, shape);
    m->t_x = tensor_new(2, shape);
    m->t_out = tensor_new(2, shape);
    return m;
}

void electra_disc_free(ElectraDiscriminator *m) {
    if (!m) return;
    emb_free(m->emb);
    pe_free(m->pe);
    te_free(m->encoder);
    free(m->W_d);
    free(m->b_d);
    tensor_free(m->t_emb);
    tensor_free(m->t_x);
    tensor_free(m->t_out);
    free(m);
}

int electra_disc_param_count(const ElectraDiscriminator *m) {
    int n = emb_param_count(m->emb);
    n += te_param_count(m->encoder);
    n += 2 * m->d_model + 2;
    return n;
}

Embedding *electra_disc_embedding(ElectraDiscriminator *m) { return m->emb; }
PositionalEncoding *electra_disc_positional(ElectraDiscriminator *m) { return m->pe; }
TransformerEncoder *electra_disc_encoder(ElectraDiscriminator *m) { return m->encoder; }

void electra_disc_set_head(ElectraDiscriminator *m, const double *W_d, const double *b_d) {
    memcpy(m->W_d, W_d, (size_t)2 * (size_t)m->d_model * sizeof(double));
    memcpy(m->b_d, b_d, (size_t)2 * sizeof(double));
}

void electra_disc_forward(ElectraDiscriminator *m, const int *ids, int T,
                          double *score, double *prob) {
    if (!ids || T <= 0) ed_die("forward: ids 不能为空");
    if (T > m->max_len) ed_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] > m->vocab_size) {
            ed_die("forward: token id 越界（允许 vocab_size，即 mask token）");
        }
    }
    if (!score || !prob) ed_die("forward: 输出缓冲不能为空");

    m->t_emb->shape[0] = T; m->t_emb->size = T * m->d_model;
    m->t_x->shape[0] = T; m->t_x->size = T * m->d_model;
    m->t_out->shape[0] = T; m->t_out->size = T * m->d_model;

    emb_forward(m->emb, ids, T, m->t_emb);
    pe_forward(m->pe, m->t_emb, m->t_x);
    te_forward(m->encoder, m->t_x, m->t_out);

    const int d = m->d_model;
    const double base = m->b_d[1] - m->b_d[0];
    for (int t = 0; t < T; t++) {
        const double *h = m->t_out->data + (size_t)t * d;
        double s = base;
        for (int j = 0; j < d; j++) {
            s += (m->W_d[d + j] - m->W_d[j]) * h[j];
        }
        score[t] = s;
        if (s >= 0.0) {
            prob[t] = 1.0 / (1.0 + exp(-s));
        } else {
            double e = exp(s);
            prob[t] = e / (1.0 + e);
        }
    }
}
