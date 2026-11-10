/* BERT 式模型前向实现：见 bert.h 的设计约定。 */
#include "bert.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double vb_uniform01(unsigned long long *state) {
    /* splitmix64 → [0,1) */
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

static double vb_gauss(unsigned long long *state) {
    /* Box-Muller（与 llama.c 同思路，但独立实现，避免跨模块耦合） */
    double u1 = vb_uniform01(state);
    double u2 = vb_uniform01(state);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

static void bert_die(const char *msg) {
    fprintf(stderr, "BertModel: %s\n", msg);
    abort();
}

static void bert_check_ids(const BertModel *m, const int *ids, const int *seg, int T) {
    if (!ids || T <= 0) bert_die("forward: ids 不能为空");
    if (T > m->max_len) bert_die("forward: 序列长度超过 max_len");
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) bert_die("forward: token id 越界");
        if (seg && seg[t] != 0 && seg[t] != 1) bert_die("forward: segment id 只能是 0/1");
    }
}

BertModel *bert_create(int vocab_size, int d_model, int num_heads, int d_ff,
                       int num_layers, int max_len, unsigned long long seed) {
    if (vocab_size < 1 || d_model < 1 || num_heads < 1 || d_ff < 1 ||
        num_layers < 1 || max_len < 1) {
        bert_die("create: 参数必须是正整数");
    }
    if (d_model % num_heads != 0) bert_die("create: d_model 必须能被 num_heads 整除");

    BertModel *m = (BertModel *)calloc(1, sizeof(BertModel));
    if (!m) bert_die("create: 内存不足");
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    m->tok_emb = emb_create(vocab_size, d_model, max_len);
    m->seg_emb = emb_create(2, d_model, max_len);
    m->pe = pe_create(d_model, max_len);
    m->encoder = te_create(d_model, num_heads, d_ff, num_layers, max_len, 0.0, seed);
    te_set_train(m->encoder, 0);   /* 前向推理语义（dropout 恒 0，仅为一致性） */

    size_t w_mlm = (size_t)vocab_size * (size_t)d_model;
    size_t w_nsp = (size_t)2 * (size_t)d_model;
    m->W_mlm = (double *)calloc(w_mlm, sizeof(double));
    m->b_mlm = (double *)calloc((size_t)vocab_size, sizeof(double));
    m->W_nsp = (double *)calloc(w_nsp, sizeof(double));
    m->b_nsp = (double *)calloc(2, sizeof(double));
    if (!m->W_mlm || !m->b_mlm || !m->W_nsp || !m->b_nsp) bert_die("create: 内存不足");

    unsigned long long st = seed ? seed : 0x123456789ABCDEFULL;
    double std = sqrt(2.0 / (double)d_model);
    for (size_t i = 0; i < w_mlm; i++) m->W_mlm[i] = vb_gauss(&st) * std;
    for (size_t i = 0; i < w_nsp; i++) m->W_nsp[i] = vb_gauss(&st) * std;

    int shape[2] = { max_len, d_model };
    m->t_x = tensor_new(2, shape);
    m->t_sum = tensor_new(2, shape);
    m->t_pe = tensor_new(2, shape);
    return m;
}

void bert_free(BertModel *m) {
    if (!m) return;
    emb_free(m->tok_emb);
    emb_free(m->seg_emb);
    pe_free(m->pe);
    te_free(m->encoder);
    free(m->W_mlm);
    free(m->b_mlm);
    free(m->W_nsp);
    free(m->b_nsp);
    tensor_free(m->t_x);
    tensor_free(m->t_sum);
    tensor_free(m->t_pe);
    free(m);
}

int bert_param_count(const BertModel *m) {
    int n = 0;
    n += emb_param_count(m->tok_emb);
    n += emb_param_count(m->seg_emb);
    n += te_param_count(m->encoder);
    n += m->vocab_size * m->d_model + m->vocab_size;
    n += 2 * m->d_model + 2;
    return n;
}

Embedding *bert_token_embedding(BertModel *m) { return m->tok_emb; }
Embedding *bert_segment_embedding(BertModel *m) { return m->seg_emb; }
PositionalEncoding *bert_positional_encoding(BertModel *m) { return m->pe; }
TransformerEncoder *bert_encoder(BertModel *m) { return m->encoder; }

void bert_set_mlm(BertModel *m, const double *W, const double *b) {
    memcpy(m->W_mlm, W, (size_t)m->vocab_size * (size_t)m->d_model * sizeof(double));
    memcpy(m->b_mlm, b, (size_t)m->vocab_size * sizeof(double));
}

void bert_set_nsp(BertModel *m, const double *W, const double *b) {
    memcpy(m->W_nsp, W, (size_t)2 * (size_t)m->d_model * sizeof(double));
    memcpy(m->b_nsp, b, (size_t)2 * sizeof(double));
}

void bert_forward(BertModel *m, const int *ids, const int *seg, int T, Tensor *hidden) {
    bert_check_ids(m, ids, seg, T);
    if (!hidden || hidden->ndim != 2 || hidden->shape[0] != T || hidden->shape[1] != m->d_model) {
        bert_die("forward: hidden 必须是 (T, d_model) 张量");
    }

    Tensor *tx = m->t_x;
    Tensor *ts = m->t_sum;
    Tensor *tp = m->t_pe;
    tx->shape[0] = T; tx->shape[1] = m->d_model; tx->size = T * m->d_model;
    ts->shape[0] = T; ts->shape[1] = m->d_model; ts->size = T * m->d_model;
    tp->shape[0] = T; tp->shape[1] = m->d_model; tp->size = T * m->d_model;

    emb_forward(m->tok_emb, ids, T, tx);
    int *seg_ids = (int *)malloc((size_t)T * sizeof(int));
    if (!seg_ids) bert_die("forward: 内存不足");
    for (int t = 0; t < T; t++) seg_ids[t] = seg ? seg[t] : 0;
    emb_forward(m->seg_emb, seg_ids, T, ts);
    free(seg_ids);

    /* token + segment embedding */
    for (int i = 0; i < T * m->d_model; i++) tp->data[i] = tx->data[i] + ts->data[i];
    /* 正弦位置编码 */
    pe_forward(m->pe, tp, tp);
    /* TransformerEncoder */
    te_forward(m->encoder, tp, hidden);
}

void bert_mlm_logits(const BertModel *m, const Tensor *hidden, double *out) {
    if (!hidden || hidden->ndim != 2 || hidden->shape[1] != m->d_model) {
        bert_die("mlm_logits: hidden 必须是 (T, d_model) 张量");
    }
    int T = hidden->shape[0];
    for (int t = 0; t < T; t++) {
        const double *h = hidden->data + (size_t)t * (size_t)m->d_model;
        double *o = out + (size_t)t * (size_t)m->vocab_size;
        for (int v = 0; v < m->vocab_size; v++) {
            const double *row = m->W_mlm + (size_t)v * (size_t)m->d_model;
            double sum = 0.0;
            for (int j = 0; j < m->d_model; j++) sum += row[j] * h[j];
            o[v] = sum + m->b_mlm[v];
        }
    }
}

void bert_nsp_logits(const BertModel *m, const Tensor *hidden, double *out) {
    if (!hidden || hidden->ndim != 2 || hidden->shape[1] != m->d_model || hidden->shape[0] < 1) {
        bert_die("nsp_logits: hidden 必须是 (T, d_model) 张量");
    }
    const double *h0 = hidden->data;
    for (int v = 0; v < 2; v++) {
        const double *row = m->W_nsp + (size_t)v * (size_t)m->d_model;
        double sum = 0.0;
        for (int j = 0; j < m->d_model; j++) sum += row[j] * h0[j];
        out[v] = sum + m->b_nsp[v];
    }
}
