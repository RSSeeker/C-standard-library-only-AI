/*
 * llama.h —— LLaMA 式因果语言模型（前向推理，移植自 AI.py 的 LlamaBlock / LlamaModel）
 *
 * 结构（与 Python 逐项对应）：
 *   Embedding → N × LlamaBlock → RMSNorm → 词表头（W_out / b_out）
 *
 * LlamaBlock（pre-norm，两个残差）：
 *   h1 = RMSNorm1(x)
 *   x  = x + GQA_Attn(h1)           # 因果 + RoPE（MHA 内置）
 *   h2 = RMSNorm2(x)
 *   out = x + SwiGLU(h2)
 *
 * 与 Python 的差异（本批只移植前向）：
 *   - 不实现 backward / 训练；反向与 Adam 留给后续批次。
 *   - dropout 固定为 0（Python 默认值也是 0）。
 *   - 权重初始化用 gauss(splitmix64) 只保证分布一致；跨语言对拍必须用
 *     llama_model_set_* 系列函数灌入 Python 导出的同一组权重。
 *
 * 所有权约定：所有 create / setters 内部复制调用方数据；调用方不需要保持缓冲存活。
 */
#ifndef AI_LLAMA_H
#define AI_LLAMA_H

#include <stddef.h>

#include "attention.h"
#include "layers_norm.h"
#include "tensor.h"

/* ============================== SwiGLU ============================== */
typedef struct SwiGLU {
    int d_model;
    int d_ff;
    double *W1; /* (d_ff, d_model) 行主序，按 Python self.W1[j][i] 展平 */
    double *b1; /* (d_ff) */
    double *Wg; /* (d_ff, d_model) */
    double *bg; /* (d_ff) */
    double *W2; /* (d_model, d_ff) */
    double *b2; /* (d_model) */
    double *h1; /* (T, d_ff) 前向缓存 */
    double *g;  /* (T, d_ff) 门控缓存 */
    double *h;  /* (T, d_ff) h1 ⊙ g */
    int cap_h1, cap_g, cap_h; /* 三个缓存各自按 token 的最大容量 */
} SwiGLU;

SwiGLU *swiglu_create(int d_model, int d_ff);
void swiglu_free(SwiGLU *s);
/* 灌入权重（跨语言验证用）：6 个指针都必须非 NULL */
void swiglu_set_params(SwiGLU *s, const double *W1, const double *b1,
                       const double *Wg, const double *bg,
                       const double *W2, const double *b2);
int swiglu_param_count(const SwiGLU *s);
/* x: (T, d_model) 行主序 → out: (T, d_model) 行主序 */
void swiglu_forward(SwiGLU *s, const double *x, int T, double *out);

/* ============================== LlamaBlock ============================== */
typedef struct LlamaBlock {
    int d_model;
    int num_heads;
    int num_kv_heads;
    int d_ff;
    int max_seq_len;

    MultiHeadAttention *attn;
    RMSNorm *ln1;
    RMSNorm *ln2;
    SwiGLU *ffn;

    /* 前向缓冲（T, d_model）：h1 / res1 / h2 */
    double *buf1;
    double *buf2;
    double *buf3;
} LlamaBlock;

/* num_kv_heads == 0 → 默认等于 num_heads（普通多头） */
LlamaBlock *llama_block_create(int d_model, int num_heads, int num_kv_heads,
                               int d_ff, int max_seq_len, double rope_base);
void llama_block_free(LlamaBlock *b);
int llama_block_param_count(const LlamaBlock *b);
/* x: (T, d_model) → out: (T, d_model) */
void llama_block_forward(LlamaBlock *b, const double *x, int T, double *out);

/* 访问器：加载权重 / 单测用 */
MultiHeadAttention *llama_block_attn(LlamaBlock *b);
RMSNorm *llama_block_ln1(LlamaBlock *b);
RMSNorm *llama_block_ln2(LlamaBlock *b);
SwiGLU *llama_block_ffn(LlamaBlock *b);

/* ============================== LlamaModel ============================== */
typedef struct LlamaModel {
    int vocab_size;
    int d_model;
    int num_heads;
    int num_kv_heads;
    int d_ff;
    int num_layers;
    int max_len;

    Embedding *emb;
    LlamaBlock **blocks;
    RMSNorm *norm;
    double *W_out; /* (vocab_size, d_model) */
    double *b_out; /* (vocab_size) */

    Tensor *t_x;   /* (T, d_model) 工作缓冲 */
    Tensor *t_y;
    double *buf_a;
    double *buf_b;
    int cap;
} LlamaModel;

LlamaModel *llama_create(int vocab_size, int d_model, int num_heads, int d_ff,
                         int num_layers, int num_kv_heads, int max_len,
                         double rope_base);
void llama_free(LlamaModel *m);
int llama_param_count(const LlamaModel *m);
/* ids: 长度 T（0 <= id < vocab_size）；logits: (T, vocab_size) 行主序（未 softmax） */
void llama_forward(LlamaModel *m, const int *ids, int T, double *logits);

Embedding *llama_emb(LlamaModel *m);
LlamaBlock *llama_block(LlamaModel *m, int index);
RMSNorm *llama_norm(LlamaModel *m);
void llama_set_head(LlamaModel *m, const double *W_out, const double *b_out);

#endif /* AI_LLAMA_H */
