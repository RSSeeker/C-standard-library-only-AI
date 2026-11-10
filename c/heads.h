/* ==========================================================================
 *  heads.h：序列标注头 / 多标签分类头 / 稀疏专家层（MoE）的 C17 实现
 *
 *  对应 Python（AI.py）：
 *    - TokenClassificationHead（45407）：(T, d) -> (T, L) logits，
 *      softmax + 交叉熵，mask=False 的 token 不进 loss、梯度恰为 0。
 *    - MultiLabelHead（45577）：(fan_in,) -> (num_labels,) sigmoid 概率，
 *      BCE-with-logits（数值稳定），多个标签可以同时为正。
 *    - MoE（15740）：out_t = Σ_{e ∈ top-k} q_e · FFN_e(x_t)，q 只在**被选中**
 *      的专家上做 softmax（Switch/GShard 口径）；未选中专家梯度为 0。
 *
 *  数值约定（与 Python 逐位一致的前提）：
 *    - 矩阵-向量内积用 dot_neumaier（Python 的 sum(map(_mul, ...))）；
 *    - softmax 的分母用 sum_neumaier；
 *    - 梯度累积、loss 累加、专家输出加权求和一律朴素 +=；
 *    - sigmoid 用 z >= 0 分支的稳定写法。
 * ========================================================================== */
#ifndef AI_HEADS_H
#define AI_HEADS_H

#include "tensor.h"
#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * TokenClassificationHead
 * -------------------------------------------------------------------------- */
typedef struct TokenHead {
    int d_model;
    int num_labels;
    double *W;      /* (num_labels, d_model) */
    double *b;      /* (num_labels) */
    double *gW;
    double *gb;
    double *x;      /* (T, d_model) 前向缓存 */
    int T;
    int cap;        /* x 的容量（按 T 增长） */
} TokenHead;

TokenHead *token_head_create(int d_model, int num_labels);
void token_head_free(TokenHead *h);
void token_head_zero_grad(TokenHead *h);
void token_head_set_params(TokenHead *h, const double *W, const double *b);
int  token_head_param_count(const TokenHead *h);
/* (T, d) -> (T, L)；x 行主序 */
void token_head_forward(TokenHead *h, const double *x, int T, double *logits);
/* softmax + 交叉熵，mask 为 NULL 表示全部有效；返回平均 loss，dlogits 可为 NULL */
double token_head_loss_grad(TokenHead *h, const double *logits, const int *targets,
                            const int *mask, int T, double *dlogits);
/* 累积 gW/gb，返回 dx（(T, d)，可为 NULL） */
void token_head_backward(TokenHead *h, const double *dlogits, int T, double *dx);
/* argmax；mask 为假的位置写 -1 */
void token_head_predict(const TokenHead *h, const double *logits, int T,
                        const int *mask, int *out);

/* --------------------------------------------------------------------------
 * MultiLabelHead
 * -------------------------------------------------------------------------- */
typedef struct MultiLabelHead {
    int fan_in;
    int num_labels;
    double *W;      /* (num_labels, fan_in) */
    double *b;      /* (num_labels) */
    double *gW;
    double *gb;
    double *x;      /* (fan_in) 前向缓存 */
    double *logits; /* (num_labels) */
    double *probs;  /* (num_labels) */
    int ready;
} MultiLabelHead;

MultiLabelHead *multilabel_head_create(int fan_in, int num_labels);
void multilabel_head_free(MultiLabelHead *h);
void multilabel_head_zero_grad(MultiLabelHead *h);
void multilabel_head_set_params(MultiLabelHead *h, const double *W, const double *b);
int  multilabel_head_param_count(const MultiLabelHead *h);
/* 返回 sigmoid 概率（长度 num_labels） */
void multilabel_head_forward(MultiLabelHead *h, const double *x, double *probs);
/* BCE-with-logits，返回平均 loss；dlogits 为 (num_labels) 或 NULL */
double multilabel_head_loss_grad(MultiLabelHead *h, const double *probs,
                                 const double *targets, double *dlogits);
void multilabel_head_backward(MultiLabelHead *h, const double *dlogits, double *dx);
void multilabel_head_predict(const MultiLabelHead *h, const double *probs,
                             double threshold, int *out);

/* --------------------------------------------------------------------------
 * MoE（稀疏专家层，top-k 路由）
 * -------------------------------------------------------------------------- */
typedef struct MoE {
    int d_model;
    int d_ff;
    int n_experts;
    int top_k;
    int act;
    int max_seq;
    FeedForward **experts;

    double *W_gate;  /* (n_experts, d_model) */
    double *b_gate;  /* (n_experts) */
    double *gW_gate;
    double *gb_gate;

    double *x;       /* (T, d_model) 缓存 */
    int T;

    int *route_e;    /* (T, top_k)：被选中的专家下标，按 logit 降序 */
    double *route_w; /* (T, top_k)：对应权重（在选中集合上 softmax） */
    int *group_idx;  /* (n_experts, T)：每个专家负责的 token 下标 */
    double *group_w; /* (n_experts, T)：对应路由权重 */
    int *group_len;  /* (n_experts) */
    int *group_pos;  /* (n_experts, T)：token 在该专家组内的位置，-1 表示不在组内 */
    double *group_out; /* (n_experts, T, d_model)：专家输出，供门控反向使用 */
    int ready;
} MoE;

MoE *moe_create(int d_model, int d_ff, int n_experts, int top_k, int act, int max_seq);
void moe_free(MoE *m);
void moe_zero_grad(MoE *m);
int  moe_param_count(const MoE *m);
/* 取第 e 个专家（用于 set_params / 检查梯度） */
FeedForward *moe_expert(const MoE *m, int e);
void moe_set_gate(MoE *m, const double *W_gate, const double *b_gate);
/* (T, d) -> (T, d) */
void moe_forward(MoE *m, const double *x, int T, double *out);
/* 累积所有专家与门控的梯度，返回 dx；dout 为 (T, d) */
void moe_backward(MoE *m, const double *dout, int T, double *dx);

#ifdef __cplusplus
}
#endif

#endif /* AI_HEADS_H */
