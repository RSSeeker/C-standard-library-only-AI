/* ELECTRA 判别器前向：双向 Transformer 编码 + 逐位置二分类打分。
 * 对应 Python 的 AI.ElectraDiscriminator（只提供前向）。
 *
 * 打分定义（与 Python 一致）：score[t] = (b_d[1]-b_d[0]) +
 *   Σ_j (W_d[1][j]-W_d[0][j]) * h[t][j]；probabilities = sigmoid(score)。
 * Embedding 词表是 vocab_size+1（多出的行是 mask token）。
 */
#ifndef AI_ELECTRA_H
#define AI_ELECTRA_H

#include "attention.h"
#include "tensor.h"

typedef struct {
    int vocab_size;          /* 原始词表大小；embedding 行数为 vocab_size+1 */
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    int max_len;

    Embedding *emb;          /* (vocab+1, d_model) */
    PositionalEncoding *pe;
    TransformerEncoder *encoder;

    double *W_d;             /* (2, d_model) */
    double *b_d;             /* (2) */

    Tensor *t_emb;           /* (T, d_model) */
    Tensor *t_x;             /* (T, d_model) */
    Tensor *t_out;           /* (T, d_model) */
} ElectraDiscriminator;

ElectraDiscriminator *electra_disc_create(int vocab_size, int d_model, int num_heads,
                                          int d_ff, int num_layers, int max_len,
                                          unsigned long long seed);
void electra_disc_free(ElectraDiscriminator *m);

int electra_disc_param_count(const ElectraDiscriminator *m);

/* forward：ids 长度 T（允许 id == vocab_size，即 mask token），score/prob 长度 T。 */
void electra_disc_forward(ElectraDiscriminator *m, const int *ids, int T,
                          double *score, double *prob);

/* 权重访问器。 */
Embedding *electra_disc_embedding(ElectraDiscriminator *m);
PositionalEncoding *electra_disc_positional(ElectraDiscriminator *m);
TransformerEncoder *electra_disc_encoder(ElectraDiscriminator *m);
void electra_disc_set_head(ElectraDiscriminator *m, const double *W_d, const double *b_d);

#endif /* AI_ELECTRA_H */
