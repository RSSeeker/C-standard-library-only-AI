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
    int *penalty_counts; /* (vocab_size) 生成罚分用的出现次数统计，仅启用罚分时使用 */
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

/* 完整版采样入口：在 llm_generate_p 的 top_k / top_p 之外，再加 Python
 * CausalLM._sample / _apply_generation_penalties 的五项能力。运算顺序与 Python 一致：
 *   罚分（用**完整历史** ids）→ top_k → 温度 → top_p → min_p → softmax → 采样。
 *   min_p：温度缩放后 softmax 概率 < min_p * max(prob) 的 token 置 -INFINITY；
 *          取值 [0,1)，0 表示不启用（与旧路径逐位相同）。
 *   repetition_penalty：出现过的 token，logit > 0 时除以该值，否则乘以该值；必须 > 0，
 *          1.0 表示不惩罚。
 *   presence_penalty：出现过的 token 的 logit 减去一次该值。
 *   frequency_penalty：出现过的 token 的 logit 减去 count * frequency_penalty。
 *   no_repeat_ngram_size：若最近 n-1 个 token 的片段在历史中出现过，把后继 token 的
 *          logit 置 -INFINITY；0 关闭，1 非法（会禁掉所有已出现 token）。
 * 全部取默认值（min_p=0, repetition=1, presence=0, frequency=0, ngram=0）时，
 * 本函数与 llm_generate_p 逐位相同：额外代码只做分支判断，不改任何数值。 */
int llm_generate_ex(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                    double temperature, int top_k, double top_p, double min_p,
                    double repetition_penalty, double presence_penalty,
                    double frequency_penalty, int no_repeat_ngram_size,
                    int *out, unsigned long long seed);

/* ---------------- 增量 KV cache 生成（推理加速） ----------------
 * 与 llm_generate_p 语义完全相同的另一条实现：预填充一次后用单 token 增量前向，
 * 不再对整段窗口重算。逐位一致的保证：
 *   - 采样逻辑与 llm_generate_p 共用同一个 sample_row（同样的 top_k / top_p /
 *     温度 / RNG 消耗顺序）；
 *   - 单 token 前向与整段前向的最后一行逐位相同（Neumaier 点积、同一 softmax）；
 *   - 窗口滑动时按新窗口重建 cache（位置重基为 0..max_len-1），与重算路径一致。
 * 代价：窗口滑动那一步要重建 max_len 个 token（与 Python _generate_cached 相同），
 * 所以 max_len 远大于生成长度时收益最大。
 * 默认行为不变：llm_generate / llm_generate_p 原样保留，本组函数是新入口。 */

/* 清空所有层的 KV cache（生成开始 / 窗口重建前调用） */
void llm_cache_reset(CausalLM *m);

/* 单 token 前向：token 在窗口内的位置 pos，logits 写 V 个数。
 * 调用约定：先 llm_cache_reset，再按 pos = 0,1,2,... 顺序调用；
 * pos 必须等于内部 cache_pos（不匹配会报错退出，防止窗口滑动后忘记重建）。 */
void llm_forward_step(CausalLM *m, int token, int pos, double *logits);

/* 增量生成：参数与返回值同 llm_generate_p；out 容量 >= prompt_len + max_new。
 * 与 llm_generate_p(prompt, ..., top_p, ...) 在 eval 模式下逐位相同。 */
int llm_generate_cached(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                        double temperature, int top_k, double top_p, int *out,
                        unsigned long long seed);

/* 增量版完整采样：参数、校验与语义同 llm_generate_ex，只是走 KV cache 前向。
 * 因采样顺序、罚分用的完整历史、RNG 消耗次数都一致，结果与 llm_generate_ex 逐位相同
 * （窗口滑动时按新窗口重建 cache，与整段重算的窗口重基一致）。 */
int llm_generate_cached_ex(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                           double temperature, int top_k, double top_p, double min_p,
                           double repetition_penalty, double presence_penalty,
                           double frequency_penalty, int no_repeat_ngram_size,
                           int *out, unsigned long long seed);

/* 诊断 / 投机解码用：最近一次 llm_forward / kv 前向的隐藏状态 h（T 行 × d_model，行主序）。
 * 返回内部缓冲，指针在下一次前向之前有效；未前向时返回 NULL。 */
const double *llm_hidden(const CausalLM *m, int *T_out);

/* 诊断 / 跨语言对拍用：把 min_p 截尾作用在一行 logits（长度 n，不修改原数组），
 * 返回保留的候选数。语义与 Python 的 _mask_min_p 相同（min_p <= 0 时返回 n）。 */
int llm_min_p_candidates(const double *logits, int n, double min_p);

/* 诊断 / 自定义采样用：把 Python _apply_generation_penalties 的五项罚分就地作用到
 * 一行 logits（长度 vocab_size）。ids 是**完整历史**（不是窗口）。默认参数全关时
 * 直接返回、不改 row。生成路径内部用的就是这个函数。 */
void llm_apply_generation_penalties(CausalLM *m, double *logits, const int *ids, int len,
                                    double repetition_penalty, double presence_penalty,
                                    double frequency_penalty, int no_repeat_ngram_size);

/* 困惑度 = exp(平均交叉熵)，对应 Python 的 perplexity()；
 * 平均 loss > 700 时返回 INFINITY（避免 exp 溢出）。 */
double llm_perplexity(CausalLM *m, const int *const *seqs, const int *lens, int nseq);

int llm_param_count(const CausalLM *m);
/* 取出参数块表（顺序与 Python 一致），返回块数；cap 不够时报错退出 */
int llm_param_blocks(CausalLM *m, LLMParamBlock *out, int cap);

#endif /* AI_LLM_H */
