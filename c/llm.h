/*
 * llm.h —— 因果语言模型 CausalLM（移植自 AI.py 的 CausalLM）
 *
 * 结构（与 Python 逐项对应）：
 *     Embedding(词表) → PositionalEncoding → TransformerEncoder(causal=True)
 *     → 词表头 logits = h @ W_out^T + b_out
 *
 * 训练目标：teacher forcing。给定 ids，位置 t 的 logits 与 ids[t+1] 做交叉熵，
 * 对预测位置取平均；反向一路传到词向量、每层参数和输出头。
 * 优化器：内部 Adam（0.9 / 0.999 / 1e-8），运算顺序与 Python 的 _adam_apply 一致。
 *
 * 关键：因果掩码（mha->causal）让位置 t 只能看到 0..t，因此一次前向就能并行
 * 算出所有位置的「下一个 token」预测 —— 这是 GPT 式预训练的做法，不需要串行递推。
 *
 * 与 Python 的差异（有意为之）：
 *   - 随机数源不同（C 用 splitmix64，Python 用 Mersenne Twister），所以 p>0 的
 *     dropout 无法逐位对齐；p=0 时可完全对齐。构造权重同理：C 自己的初始化只保证
 *     分布一致，要与 Python 比对必须用 llm_param_blocks 灌入同一组权重。
 *   - 注意力权重 dropout 尚未移植（Python 已实现），本文件只把 p 传给两层 Dropout。
 * 热路径零 malloc：forward / backward / adam_step 只用 create 时预分配的缓冲。
 */
#ifndef AI_LLM_H
#define AI_LLM_H

#include <stddef.h>

#include "attention.h"
#include "tensor.h"

/* 一个「参数块」= 一段连续参数 + 同形状梯度 + Adam 的一阶/二阶矩。
 * 名字与 Python CausalLM._named_params() 完全一致，跨语言比对时按名字对齐。 */
typedef struct {
    const char *name;
    double *param;
    double *grad;
    int n;
} LLMParamBlock;

typedef struct {
    int vocab_size;
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    int max_len;
    double dropout;
    int use_abs_pos;    /* 1 = embed 之后加绝对位置编码（默认）；0 = 跳过（RoPE 打开时） */

    Embedding *emb;
    PositionalEncoding *pe;
    TransformerEncoder *enc;

    /* 词表头：W_out 为 (vocab_size, d_model)，按 Python 的 self.W_out[v][j] 展平 */
    double *W_out;
    double *b_out;
    double *grad_W_out;
    double *grad_b_out;

    /* 参数块（顺序与 Python 的 _named_params 一致），Adam 状态按块延迟分配 */
    LLMParamBlock *blocks;
    int n_blocks;
    double **adam_m;
    double **adam_v;
    int step_count;

    /* ---------- 前向/反向缓存（create 时按 max_len 预分配） ---------- */
    int seq_len;
    Tensor *emb_buf;    /* (T, d_model) 词向量输出 */
    Tensor *x_buf;      /* (T, d_model) 位置编码之后 */
    Tensor *h_buf;      /* (T, d_model) 编码器输出（反向要用） */
    Tensor *logits_buf; /* (T, vocab_size) */
    Tensor *dh_buf;     /* (T, d_model) 词表头反向的输出梯度 */
    Tensor *dx_buf;     /* (T, d_model) 编码器反向的输入梯度 */
    double *dlogits;    /* (T, vocab_size) */
    double *probs;      /* (vocab_size) 临时 */
    double *tmp_row;    /* (max(vocab_size, d_model)) 临时 */
    unsigned char *keep; /* (vocab_size) nucleus 采样的保留标记，仅 top_p < 1 时使用 */
} CausalLM;

/* 全部参数非正时报错退出；dropout 须在 [0,1) 内 */
CausalLM *llm_create(int vocab_size, int d_model, int num_heads, int d_ff,
                     int num_layers, int max_len, double dropout,
                     unsigned long long seed);
void llm_free(CausalLM *m);

/* 训练/推理模式（只影响两层 Dropout） */
void llm_set_train(CausalLM *m, int train);

/* RoPE（旋转位置编码）总开关，对应 Python 的 CausalLM(rope=True, rope_base=...)：
 *   enable != 0：每层的 MultiHeadAttention 打开 RoPE，同时把 use_abs_pos 置 0
 *                 —— 前向跳过 PositionalEncoding（位置信息已由 q/k 的旋转携带）；
 *   enable == 0：恢复默认路径（RoPE 关闭 + 绝对位置编码），与 llm_create 后一致。
 * base 必须 > 0（常用 10000.0）。必须在 forward/训练之前调用。
 * 默认路径不受影响：llm_create 之后 use_abs_pos == 1、各层 rope == 0，代码路径与旧版逐位相同。 */
void llm_set_rope(CausalLM *m, int enable, double base);

/* 清零全部梯度（词向量、各层、输出头） */
void llm_zero_grad(CausalLM *m);

/* ids 长度 T（1 <= T <= max_len，且每个索引在 [0, vocab_size)）→ logits (T, vocab) */
void llm_forward(CausalLM *m, const int *ids, int T, Tensor *logits);

/* teacher forcing：返回平均交叉熵，并把梯度累积到各参数块；T >= 2 */
double llm_loss_and_backward(CausalLM *m, const int *ids, int T);

/* 用累积梯度对全部参数块做一次 Adam 更新 */
void llm_adam_step(CausalLM *m, double lr);

/* 不更新参数的平均交叉熵（eval 模式；按预测位置总数平均，不是按序列平均） */
double llm_evaluate(CausalLM *m, const int *const *seqs, const int *lens, int nseq);

/* 每个序列一步：zero_grad → loss_and_backward → adam_step；
 * history 可为 NULL，否则写入 epochs 个 epoch 的平均 loss。返回最后一个 epoch 的 loss。 */
double llm_fit(CausalLM *m, const int *const *seqs, const int *lens, int nseq,
               int epochs, double lr, int shuffle, unsigned long long seed,
               double *history);

/* 自回归生成：prompt 之后生成 max_new 个 token，写进 out（容量 >= prompt_len + max_new），
 * 返回总长度。temperature > 0；top_k <= 0 或 >= vocab_size 表示不过滤。 */
int llm_generate(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                 double temperature, int top_k, int *out, unsigned long long seed);

/* 带 nucleus（top-p）采样的生成，对应 Python 的 CausalLM._sample(top_p=...)：
 *   top_p >= 1.0：与 llm_generate 完全等价 —— 走同一条路径、抽同样多的随机数、
 *                 运算顺序不变，因此输出逐位相同；
 *   0 < top_p < 1：在 top_k 过滤与温度缩放之后，按概率降序遍历，取出累积概率刚达到
 *                 top_p 的最小 token 集合（并列时小下标在前），其余置 -INFINITY，
 *                 再重新 softmax 采样。
 * top_p 必须落在 (0, 1] 内，否则报错退出。 */
int llm_generate_p(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                   double temperature, int top_k, double top_p, int *out,
                   unsigned long long seed);

/* 困惑度 = exp(平均交叉熵)，对应 Python 的 perplexity()；
 * 平均 loss > 700 时返回 INFINITY（避免 exp 溢出）。 */
double llm_perplexity(CausalLM *m, const int *const *seqs, const int *lens, int nseq);

int llm_param_count(const CausalLM *m);
/* 取出参数块表（顺序与 Python 一致），返回块数；cap 不够时报错退出 */
int llm_param_blocks(CausalLM *m, LLMParamBlock *out, int cap);

#endif /* AI_LLM_H */
