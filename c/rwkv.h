/* RWKV 风格语言模型前向：Embedding → N×RWKVBlock → LayerNorm → 词表头。
 * 对应 Python 的 AI.RWKVModel（只提供前向，无反向/训练/生成）。
 *
 * 每个 block（与 Python 一致的简化 RWKV-4）：
 *   time-mixing：ln1 → r/k/v 线性 → 逐通道状态 state=decay*state+k*v →
 *                y=r*(state+u*v) → Wo 线性 → 残差
 *   channel-mixing：ln2 → r2/k2/v2 线性（到 d_ff）→ y2=r2*relu(k2)^2*v2 →
 *                Wo2 线性 → 残差
 * decay = exp(-exp(w_raw))；w_raw 与 u 都是长度 d_model 的可学习向量。
 */
#ifndef AI_RWKV_H
#define AI_RWKV_H

#include "layers_norm.h"
#include "tensor.h"

typedef struct {
    int d_model;
    int d_ff;

    LayerNorm *ln1;
    LayerNorm *ln2;

    double *Wr, *br;     /* (d, d), (d) */
    double *Wk, *bk;
    double *Wv, *bv;
    double *Wo, *bo;
    double *w_raw;       /* (d) */
    double *u;           /* (d) */

    double *Wr2, *br2;   /* (d_ff, d), (d_ff) */
    double *Wk2, *bk2;
    double *Wv2, *bv2;
    double *Wo2, *bo2;   /* (d, d_ff), (d) */
} RwkvBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int d_ff;
    int num_layers;
    int max_len;

    double *emb;                 /* (vocab, d) */
    RwkvBlock *blocks;           /* num_layers */
    LayerNorm *ln_out;
    double *W_out, *b_out;       /* (vocab, d), (vocab) */

    double *buf_x, *buf_h1, *buf_r, *buf_k, *buf_v;
    double *buf_x1, *buf_h2, *buf_r2, *buf_zk, *buf_v2, *buf_y2, *buf_out2;
    Tensor *t_row, *t_ln;        /* 逐行 LayerNorm 用的 1D 张量 */
} RwkvModel;

RwkvModel *rwkv_create(int vocab_size, int d_model, int d_ff, int num_layers,
                       int max_len, unsigned long long seed);
void rwkv_free(RwkvModel *m);

int rwkv_param_count(const RwkvModel *m);

/* forward：ids 长度 T（0 <= id < vocab），logits 为 (T, vocab) 行主序。 */
void rwkv_forward(RwkvModel *m, const int *ids, int T, double *logits);

#endif /* AI_RWKV_H */
