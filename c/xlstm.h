/* xLSTM 风格语言模型前向：Embedding → N×XLSTMBlock → LayerNorm → 词表头。
 * 对应 Python 的 AI.XLSTMModel（只移植前向）。
 *
 * XLSTMBlock：pre-norm sLSTM 残差 + pre-norm FFN 残差
 *   sLSTM cell：Wg(d→4d) 得到 i/f/z/o 四组 pre-activation：
 *     i = exp(clamp(i_pre,-10,5))，f = sigmoid(f_pre)，z = tanh(z_pre)，o = sigmoid(o_pre)
 *     c_t = f*c_prev + i*z；n_t = f*n_prev + i；h_t = o*c_t / max(|n_t|, eps)
 *   W_o 输出后加残差；FFN 为 ln2 → W1(relu) → W2 → 残差。
 */
#ifndef AI_XLSTM_H
#define AI_XLSTM_H

#include "layers_norm.h"
#include "tensor.h"

typedef struct {
    int d_model;
    int d_ff;
    double eps;

    LayerNorm *ln1;
    LayerNorm *ln2;

    double *Wg, *bg;  /* (4d, d), (4d) */
    double *Wo, *bo;  /* (d, d), (d) */
    double *W1, *b1;  /* (d_ff, d), (d_ff) */
    double *W2, *b2;  /* (d, d_ff), (d) */

    double *x, *h1;   /* (T, d) */
    double *gates;    /* (T, 4d) */
    double *s;        /* (T, d) cell 输出 */
    double *x1, *h2, *a2, *out2;
} XLSTMBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int d_ff;
    int num_layers;
    int max_len;

    double *emb;
    XLSTMBlock *blocks;
    LayerNorm *ln_out;
    double *W_out, *b_out;

    Tensor *t_row, *t_ln;
} XLSTMModel;

XLSTMModel *xlstm_create(int vocab_size, int d_model, int d_ff, int num_layers,
                         int max_len, unsigned long long seed);
void xlstm_free(XLSTMModel *m);
int xlstm_param_count(const XLSTMModel *m);
void xlstm_forward(XLSTMModel *m, const int *ids, int T, double *logits);

#endif /* AI_XLSTM_H */
