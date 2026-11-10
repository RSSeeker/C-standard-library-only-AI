/* RetNet 风格语言模型前向：Embedding → N×RetNetBlock → LayerNorm → 词表头。
 * 对应 Python 的 AI.RetNetModel（只移植前向）。
 *
 * 每个 block：pre-norm 两残差
 *   retention：ln1 → Wq/Wk/Wv ∈ R^{d×d} → 逐头（gamma_h = 1 - 2^(-5-h)）
 *              out[t] = Σ_{j<=t} gamma^(t-j)·(q_t·k_j)·v_j / denom_t
 *              denom_t = Σ_{j<=t} gamma^(t-j)（递推 denom = denom*gamma + 1）
 *              再 W_o → 残差
 *   FFN：ln2 → W1(relu) → W2 → 残差
 */
#ifndef AI_RETNET_H
#define AI_RETNET_H

#include "layers_norm.h"
#include "tensor.h"

typedef struct {
    int d_model;
    int num_heads;
    int head_dim;
    int d_ff;

    LayerNorm *ln1;
    LayerNorm *ln2;

    double *Wq, *bq;
    double *Wk, *bk;
    double *Wv, *bv;
    double *Wo, *bo;
    double *W1, *b1;
    double *W2, *b2;

    double *gamma;   /* (num_heads) */

    /* 工作缓冲（按 max_len 预分配） */
    double *x, *h1, *q, *k, *v, *concat, *x1, *h2, *a2, *out2;
    double *ret;     /* (T, head_dim) 每头复用 */
    double *scores;  /* (T, T) 每头复用 */
    double *denom;   /* (T) */
} RetNetBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    int max_len;

    double *emb;              /* (vocab, d) */
    RetNetBlock *blocks;
    LayerNorm *ln_out;
    double *W_out, *b_out;    /* (vocab, d), (vocab) */

    Tensor *t_row, *t_ln;
} RetNetModel;

RetNetModel *retnet_create(int vocab_size, int d_model, int num_heads, int d_ff,
                           int num_layers, int max_len, unsigned long long seed);
void retnet_free(RetNetModel *m);

int retnet_param_count(const RetNetModel *m);

/* forward: ids 长度 T，logits 为 (T, vocab) 行主序。 */
void retnet_forward(RetNetModel *m, const int *ids, int T, double *logits);

#endif /* AI_RETNET_H */
