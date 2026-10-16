/*
 * attention.h —— Transformer 组件（移植自 AI.py）
 *
 * 与 Python 版的对应关系：
 *   MultiHeadAttention     -> MultiHeadAttention
 *   FeedForward            -> FeedForward
 *   TransformerEncoderLayer-> TransformerEncoderLayer
 *   TransformerEncoder     -> TransformerEncoder
 *   PositionalEncoding     -> PositionalEncoding
 *   Embedding              -> Embedding
 *   Dropout / LayerNorm    -> TfDropout / TfLayerNorm（本文件自带的私有实现，见下方说明）
 *
 * 设计约定（与 ai.h / tensor.h 一致，且刻意对齐 Python 的运算顺序）：
 *   1. 一律 (seq_len, d_model) 的二维张量（Tensor 的 ndim=2 形状 [seq_len, d_model]），
 *      行主序、double；权重矩阵按 Python 的 self.W[j][i] 顺序展平（W 的第 j 行 = 第 j 个输出单元）。
 *   2. 所有点积 / 矩阵乘都走 Neumaier 补偿求和（numeric.h），并且 **累加下标升序**，
 *      与 Python 的 _matmul（内部 sum(map(mul, row, col))）逐位一致。
 *   3. forward / backward 热路径零 malloc：所有中间缓冲（Q/K/V 各头切片、attn、拼接、
 *      反向缓冲……）都在 create 时按 max_seq_len 一次性分配好挂在结构体上。
 *   4. 形状不匹配一律 tensor_check_shape 报错退出，绝不静默截断
 *      （Python 版 PositionalEncoding 用 min(len(x), len(pe)) 静默截断，见 pe_forward 注释）。
 *
 * 关于 LayerNorm / Dropout：这两个组件按分工属于 layers_norm.h（另一位同事负责）。
 * 该文件目前尚不存在，为了让 TransformerEncoderLayer 能独立编译与自测，本文件自带一份
 * 私有实现（TfLayerNorm / TfDropout 与 tf_* / tfd_* 函数），数学与 AI.py 的
 * LayerNorm / Dropout 完全一致，且刻意保持接口可替换：待 layers_norm.h 落地后，
 * 只需把这两个结构体与函数替换成那边的同名实现即可（依赖见答复说明）。
 */
#ifndef AI_ATTENTION_H
#define AI_ATTENTION_H

#include <stddef.h>

#include "numeric.h"
#include "tensor.h"

/* ==========================================================================
 *  私有 LayerNorm（与 layers_norm.h 的 LayerNorm 重复，后续统一）
 *
 *  Python 版 LayerNorm 作用在「单个 token 的 d 维向量」上，TransformerEncoderLayer
 *  逐 token 调用它。这里保留同样的逐行语义：行号由调用方显式给出，缓存按行号索引。
 * ========================================================================== */
typedef struct {
    int d;              /* 特征维度 num_features */
    int max_seq;        /* 预分配的最大 token 数 */
    double eps;         /* 与 Python 一致，默认 1e-5 */
    double *gamma;      /* d */
    double *beta;       /* d */
    double *ggamma;     /* 累积梯度 d（对应 Python 的 grads["ln?_gamma"]） */
    double *gbeta;      /* 累积梯度 d */
    double *mean;       /* 前向缓存 (max_seq) */
    double *inv_std;    /* 前向缓存 (max_seq) */
    double *x_hat;      /* 前向缓存 (max_seq, d) */
    double *tmp;        /* (d) 反向临时 */
    double *tmp2;       /* (d) 反向临时 */
} TfLayerNorm;

void tfln_init(TfLayerNorm *ln, int d, int max_seq, double eps);
void tfln_free(TfLayerNorm *ln);
void tfln_zero_grad(TfLayerNorm *ln);
/* 单 token 前向：x 长度 d → out 长度 d（与 LayerNorm.forward 同序） */
void tfln_forward_row(TfLayerNorm *ln, int row, const double *x, double *out);
/* 单 token 反向：dout 长度 d → dx 长度 d；同时把 grad_gamma/grad_beta 返回给调用方
 * （与 Python 一样是「返回」而不是「累加」，累积由 TransformerEncoderLayer 负责） */
void tfln_backward_row(TfLayerNorm *ln, int row, const double *dout,
                       double *grad_gamma, double *grad_beta, double *dx);

/* ==========================================================================
 *  私有 Dropout（与 layers_norm.h 的 Dropout 重复，后续统一）
 *
 *  mask 按 token 存储（max_seq, d），与 Python 的 dropout?_masks[t][j] 一一对应。
 *  注意：C 侧无法复现 Python 的 Mersenne Twister 随机序列，因此 p>0 时数值上不可能
 *  与 Python 逐位对齐；p=0 时 mask 恒为 1.0，可与 Python 完全对齐。
 * ========================================================================== */
typedef struct {
    double p;
    int train;              /* 1 = 训练（随机 mask）；0 = 推理（mask 全 1、输出原样） */
    int max_seq;
    int d;
    unsigned long long rng; /* 自带的 splitmix64 状态（与 Python 的 random 不同源） */
    double *mask;           /* (max_seq, d) */
} TfDropout;

void tfd_init(TfDropout *dp, double p, int d, int max_seq, unsigned long long seed);
void tfd_free(TfDropout *dp);
/* 逐 token 前向：x 长度 d → out 长度 d（可原地：out == x），mask 存进 dp->mask[row] */
void tfd_forward_row(TfDropout *dp, int row, const double *x, double *out);
void tfd_backward_row(const TfDropout *dp, int row, const double *dout, double *dx);

/* ==========================================================================
 *  MultiHeadAttention
 * ========================================================================== */
typedef struct {
    int d_model;
    int num_heads;
    int d_k;                /* d_model / num_heads */
    int max_seq_len;        /* 预分配的最大序列长度 */
    int seq_len;            /* 最近一次 forward 的序列长度 */
    double dropout_p;       /* 与 Python 一致地保存，但 forward 里未使用（Python 同样未使用） */

    /* 参数（行主序）：W_? 为 (d_model, d_model)，按 Python 的 self.W_?[j][i] 展平 */
    double *W_q, *W_k, *W_v, *W_o;
    double *b_q, *b_k, *b_v, *b_o;
    /* 参数梯度（累积） */
    double *grad_W_q, *grad_W_k, *grad_W_v, *grad_W_o;
    double *grad_b_q, *grad_b_k, *grad_b_v, *grad_b_o;

    /* ---------- 前向缓存 ---------- */
    double *x;          /* (T, d_model) 输入副本 */
    double *q_full;     /* (T, d_model) */
    double *k_full;
    double *v_full;
    double *Q;          /* (h, T, d_k) —— 与 Python 的 self.Q[head] 一一对应 */
    double *K;
    double *V;
    double *attn;       /* (h, T, T) —— 每头的注意力权重矩阵 */
    double *head_out;   /* (h, T, d_k) */
    double *concat;     /* (T, d_model) */

    /* ---------- 反向缓冲（全部预分配，热路径零 malloc） ---------- */
    double *d_concat;   /* (T, d_model) */
    double *d_head;     /* (h, T, d_k) */
    double *d_scores;   /* (h, T, T)：先装 d_attn，随后原地覆盖为 d_scores */
    double *dQ;         /* (h, T, d_k) */
    double *dK;
    double *dV;
    double *dQ_full;    /* (T, d_model) */
    double *dK_full;
    double *dV_full;
    double *tmp;        /* (T, d_model) 通用临时 */
} MultiHeadAttention;

/* dropout_p 仅保存不使用，保持与 Python 构造签名一致；max_seq_len 为预分配上限 */
MultiHeadAttention *mha_create(int d_model, int num_heads, int max_seq_len, double dropout_p);
void mha_free(MultiHeadAttention *m);
void mha_zero_grad(MultiHeadAttention *m);
/* x: (seq_len, d_model) → out: (seq_len, d_model) */
void mha_forward(MultiHeadAttention *m, const Tensor *x, Tensor *out);
/* dout: (seq_len, d_model) → dx: (seq_len, d_model)，参数梯度累积到 grad_* */
void mha_backward(MultiHeadAttention *m, const Tensor *dout, Tensor *dx);
/* 灌入权重（跨语言验证用）：8 个指针都必须非 NULL，
 * W_* 为 (d_model, d_model) 行主序（按 Python 的 self.W_*[j][i] 展平），b_* 长度 d_model */
void mha_set_params(MultiHeadAttention *m,
                    const double *W_q, const double *b_q,
                    const double *W_k, const double *b_k,
                    const double *W_v, const double *b_v,
                    const double *W_o, const double *b_o);
int  mha_param_count(const MultiHeadAttention *m);
/* 包装成 tensor.h 的 Module（供 Sequential / ResidualBlock 使用）。
 * 所有权：包装器接管 m，module_free() 会连同底层一起释放；不要重复 free。 */
Module *mha_module(MultiHeadAttention *m);

/* ==========================================================================
 *  FeedForward：Linear → ReLU → Linear
 * ========================================================================== */
typedef struct {
    int d_model;
    int d_ff;
    int max_seq_len;
    int seq_len;

    double *W1;     /* (d_ff, d_model) */
    double *b1;     /* (d_ff) */
    double *W2;     /* (d_model, d_ff) */
    double *b2;     /* (d_model) */
    double *grad_W1, *grad_b1, *grad_W2, *grad_b2;

    double *x;      /* (T, d_model) 输入副本 */
    double *h;      /* (T, d_ff) 缓存（ReLU 之后的激活，Python 的 self.h） */
    double *dh;     /* (T, d_ff) 反向缓冲 */
} FeedForward;

FeedForward *ff_create(int d_model, int d_ff, int max_seq_len);
void ff_free(FeedForward *f);
void ff_zero_grad(FeedForward *f);
void ff_forward(FeedForward *f, const Tensor *x, Tensor *out);
void ff_backward(FeedForward *f, const Tensor *dout, Tensor *dx);
void ff_set_params(FeedForward *f, const double *W1, const double *b1,
                   const double *W2, const double *b2);
int  ff_param_count(const FeedForward *f);
Module *ff_module(FeedForward *f);

/* ==========================================================================
 *  TransformerEncoderLayer：Self-Attn + Dropout + Residual + LN1 + FFN + Dropout + Residual + LN2
 * ========================================================================== */
typedef struct {
    MultiHeadAttention *self_attn;
    FeedForward *ffn;
    TfLayerNorm ln1, ln2;
    TfDropout dropout1, dropout2;

    int d_model;
    int max_seq_len;
    int seq_len;

    /* 三块 (T, d_model) 缓冲，前向/反向复用（前向数据在反向开始时已全部无用：
     * LN / FFN / Attention 各自把需要的量缓存在自己内部）。反向的角色分配：
     *   buf3 = d_res2（未加 mask）→ 之后复用为 d_res1（未加 mask）
     *   buf1 = d_ffn（masked）   → 之后复用为 d_attn（masked）
     *   buf2 = d_ln1（FFN 反传结果，再加直通分支 d_res2）
     * 为什么 dropout 不能原地做：Python 里残差直通分支加的是「未加 mask」的 d_res2 / d_res1，
     * 原地覆盖会把直通分支也一起 mask 掉（数值梯度会立刻查出来）。 */
    double *buf1;
    double *buf2;
    double *buf3;
    double *ln_dgamma;  /* (d_model) LN 逐 token 反传的临时 */
    double *ln_dbeta;   /* (d_model) */
} TransformerEncoderLayer;

TransformerEncoderLayer *tel_create(int d_model, int num_heads, int d_ff,
                                    int max_seq_len, double dropout_p,
                                    unsigned long long seed);
void tel_free(TransformerEncoderLayer *l);
void tel_zero_grad(TransformerEncoderLayer *l);
void tel_forward(TransformerEncoderLayer *l, const Tensor *x, Tensor *out);
void tel_backward(TransformerEncoderLayer *l, const Tensor *dout, Tensor *dx);
/* 用累积梯度做一次 SGD：param -= lr * grad（对应 Python 的 TransformerEncoderLayer.sgd_step） */
void tel_sgd_step(TransformerEncoderLayer *l, double lr);
void tel_set_train(TransformerEncoderLayer *l, int train);
int  tel_param_count(const TransformerEncoderLayer *l);
Module *tel_module(TransformerEncoderLayer *l);

/* ==========================================================================
 *  TransformerEncoder：堆叠 num_layers 层
 * ========================================================================== */
typedef struct {
    int num_layers;
    int d_model;
    int max_seq_len;
    int seq_len;
    TransformerEncoderLayer **layers;
    Tensor *buf_a;  /* (T, d_model) 乒乓缓冲 */
    Tensor *buf_b;
} TransformerEncoder;

TransformerEncoder *te_create(int d_model, int num_heads, int d_ff, int num_layers,
                              int max_seq_len, double dropout_p, unsigned long long seed);
void te_free(TransformerEncoder *e);
void te_zero_grad(TransformerEncoder *e);
void te_forward(TransformerEncoder *e, const Tensor *x, Tensor *out);
void te_backward(TransformerEncoder *e, const Tensor *dout, Tensor *dx);
void te_sgd_step(TransformerEncoder *e, double lr);
void te_set_train(TransformerEncoder *e, int train);
int  te_param_count(const TransformerEncoder *e);
Module *te_module(TransformerEncoder *e);

/* ==========================================================================
 *  PositionalEncoding：正弦/余弦位置编码（无参数，加到输入上）
 * ========================================================================== */
typedef struct {
    int d_model;
    int max_len;
    double *pe;     /* (max_len, d_model) */
} PositionalEncoding;

PositionalEncoding *pe_create(int d_model, int max_len);
void pe_free(PositionalEncoding *p);
/* out = x + pe[0..T)；T > max_len 时报错（Python 版是 min() 静默截断，这里按项目约定报错） */
void pe_forward(PositionalEncoding *p, const Tensor *x, Tensor *out);
/* 无参数，直通：dx = dout */
void pe_backward(const Tensor *dout, Tensor *dx);
int  pe_param_count(const PositionalEncoding *p);
Module *pe_module(PositionalEncoding *p);

/* ==========================================================================
 *  Embedding：查表
 * ========================================================================== */
typedef struct {
    int vocab_size;
    int embed_dim;
    int max_len;
    int seq_len;
    double *weight;         /* (vocab_size, embed_dim) */
    double *grad_weight;    /* (vocab_size, embed_dim) 累积 */
    int *indices;           /* (max_len) 最近一次 forward 的索引缓存 */
} Embedding;

Embedding *emb_create(int vocab_size, int embed_dim, int max_len);
void emb_free(Embedding *e);
void emb_zero_grad(Embedding *e);
/* 原生接口：indices 长度 n（0 <= idx < vocab_size）→ out: (n, embed_dim) */
void emb_forward(Embedding *e, const int *indices, int n, Tensor *out);
/* dx 可为 NULL（Python 版返回 None，即没有对输入的梯度）；
 * 非 NULL 时必须是 (n) 的一维张量（= Module 包装器输入的形状），会被写成 0，
 * 这样 Sequential / ResidualBlock 就能用同一套「backward 把输入梯度写进 dx」的契约 */
void emb_backward(Embedding *e, const Tensor *dout, Tensor *dx);
void emb_set_weight(Embedding *e, const double *W);
int  emb_param_count(const Embedding *e);
/* Module 包装：输入是 ndim=1 的索引张量（double 里存整数索引），输出 (n, embed_dim) */
Module *emb_module(Embedding *e);

#endif /* AI_ATTENTION_H */
