/* Mamba 风格语言模型前向：Embedding → N×MambaBlock → 词表头。
 * 对应 Python 的 AI.MambaModel（只提供前向）。
 *
 * MambaBlock（简化版，与 Python 一致）：
 *   in_proj → 拆成 x/z → 因果一维卷积（左补 k-1）→ SiLU 得 u；
 *   delta = softplus(u @ delta_proj)；B = u @ b_proj；C = u @ c_proj；
 *   选择性扫描（对角 A）：
 *     h_t = exp(delta_t ⊙ A) ⊙ h_{t-1} + (delta_t ⊙ B_t) ⊙ u_t
 *     y_t = C_t · h_t + D ⊙ u_t
 *   out = x + out_proj(y ⊙ SiLU(z))
 * A 直接作为参数（不是 -exp(A_log) 参数化）；时间维顺序递推。
 */
#ifndef AI_MAMBA_H
#define AI_MAMBA_H

#include "tensor.h"

typedef struct {
    int d_model;
    int d_state;
    int d_conv;
    int expand;
    int d_inner;

    double *W_in, *b_in;        /* (2*d_inner, d_model), (2*d_inner) */
    double *conv_w, *conv_b;    /* (d_inner, d_inner, d_conv), (d_inner) */
    double *W_delta, *b_delta;  /* (d_inner, d_inner), (d_inner) */
    double *W_b, *b_b;          /* (d_state, d_inner), (d_state) */
    double *W_c, *b_c;          /* (d_state, d_inner), (d_state) */
    double *A;                  /* (d_inner, d_state) */
    double *D;                  /* (d_inner) */
    double *W_out, *b_out;      /* (d_model, d_inner), (d_model) */

    /* 工作缓冲（按 max_len 预分配） */
    double *x, *proj, *conv_out, *u, *delta, *Bij, *Cij, *y, *gated;
    double *h;                  /* (d_inner, d_state) 扫描状态 */
} MambaBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int num_layers;
    int max_len;

    double *emb;                /* (vocab, d_model) */
    MambaBlock *blocks;
    double *W_head, *b_head;    /* (vocab, d_model), (vocab) */
} MambaModel;

MambaModel *mamba_create(int vocab_size, int d_model, int d_state, int d_conv,
                         int expand, int num_layers, int max_len,
                         unsigned long long seed);
void mamba_free(MambaModel *m);
int mamba_param_count(const MambaModel *m);
void mamba_forward(MambaModel *m, const int *ids, int T, double *logits);

#endif /* AI_MAMBA_H */
