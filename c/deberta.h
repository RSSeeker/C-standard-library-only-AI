/* DeBERTa 风格语言模型前向（单头解耦相对注意力，教学简化版）。
 * 对应 Python 的 AI.DeBERTaModel / AI.DeBERTaBlock（只提供前向）。
 *
 * 注意力分数（因果，j <= t）：
 *   s(t, j) = scale · ( q_t·k_j + q_t·Rk[d] + k_j·Rq[d] ),  d = min(t-j, max_rel)
 * Rk 为 content-to-position 相对位置表，Rq 为 position-to-content 表。
 */
#ifndef AI_DEBERTA_H
#define AI_DEBERTA_H

#include "tensor.h"
#include "layers_norm.h"

typedef struct {
    int d_model;
    int d_ff;
    int max_rel;
    int n_rel;

    LayerNorm *ln1;
    LayerNorm *ln2;

    double *Wq, *bq;
    double *Wk, *bk;
    double *Wv, *bv;
    double *Wo, *bo;
    double *Rk, *Rq;        /* (n_rel, d_model) */
    double *W1, *b1;        /* (d_ff, d_model), (d_ff) */
    double *W2, *b2;        /* (d_model, d_ff), (d_model) */

    /* 工作缓冲（按 max_len 预分配） */
    double *h1, *q, *k, *v;
    double *scores;         /* (max_len, max_len) */
    double *attn;           /* (max_len, max_len) */
    double *vmix, *orows, *x1, *h2, *a2, *ffn, *out;
    Tensor *tin, *tout;
} DebertaBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int num_layers;
    int max_len;
    int max_rel;

    double *emb;            /* (vocab, d_model) */
    DebertaBlock *blocks;
    LayerNorm *ln_out;
    double *W_out, *b_out;  /* (vocab, d_model), (vocab) */
    double *seq_buf;        /* (max_len, d_model) */
} DebertaModel;

DebertaModel *deberta_create(int vocab_size, int d_model, int d_ff, int num_layers,
                             int max_len, int max_rel, unsigned long long seed);
void deberta_free(DebertaModel *m);
int deberta_param_count(const DebertaModel *m);
void deberta_forward(DebertaModel *m, const int *ids, int T, double *logits);

#endif /* AI_DEBERTA_H */
