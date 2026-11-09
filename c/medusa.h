/*
 * medusa.h —— 投机解码：Medusa 多头草稿 / EAGLE 特征级草稿
 *
 * 与 Python 版 AI.py 的对应关系：
 *   MedusaHeads  -> MedusaHeads：K 个独立 linear 头，同一隐藏状态同时预测未来 K 个 token
 *   EagleDrafter -> EagleDrafter：由目标模型隐藏状态 + 上一 token 词嵌入预测下一 token
 *   medusa_decode / eagle_decode -> 草稿 + 目标模型贪心验证的完整解码循环
 *
 * 数据约定：隐藏状态按行主序（n_rows × d_model）。
 */
#ifndef MEDUSA_H
#define MEDUSA_H

#include "ai.h"
#include "llm.h"

typedef struct {
    int d_model;
    int vocab_size;
    int num_heads;
    MLP **heads;        /* 每头一个单层 linear MLP（d_model → vocab_size） */
} MedusaHeads;

typedef struct {
    int d_model;
    int vocab_size;
    int hidden;
    double *Wf;         /* hidden × d_model，行主序 */
    double *We;         /* hidden × d_model，行主序 */
    double *bh;         /* hidden */
    double *Wh;         /* vocab_size × hidden，行主序 */
    double *bo;         /* vocab_size */
} EagleDrafter;

typedef struct {
    int drafted;
    int accepted;
    int target_calls;
    double accept_rate;
} SpecStats;

MedusaHeads *medusa_heads_create(int d_model, int vocab_size, int num_heads,
                                 unsigned long long seed);
void medusa_heads_free(MedusaHeads *h);
long medusa_heads_param_count(const MedusaHeads *h);
void medusa_heads_zero_grad(MedusaHeads *h);
void medusa_heads_sgd_step(MedusaHeads *h, double lr);

/* hidden_rows: n_rows × d_model；out: num_heads × n_rows × vocab_size。 */
void medusa_heads_forward(MedusaHeads *h, const double *hidden_rows, int n_rows,
                          double *out);

/* targets: num_heads × n_rows 的 token 下标；返回平均交叉熵并累积梯度。 */
double medusa_heads_loss_and_backward(MedusaHeads *h, const double *hidden_rows,
                                      int n_rows, const int *targets);

/* 单行预测：probs 为 num_heads × vocab_size。 */
void medusa_heads_predict(MedusaHeads *h, const double *hidden_row, double *probs);

EagleDrafter *eagle_drafter_create(int d_model, int vocab_size, int hidden,
                                   unsigned long long seed);
void eagle_drafter_free(EagleDrafter *d);
long eagle_drafter_param_count(const EagleDrafter *d);

/* 特征 + 上一 token 嵌入 → 下一 token logits（长度 vocab_size）。 */
void eagle_drafter_forward(const EagleDrafter *d, const double *feature,
                           const double *prev_emb, double *logits);

/* 解码：返回 out 中的总长度（prompt_len + 新 token 数），失败返回 -1。
 * out 容量至少 prompt_len + max_new；out 内容为 prompt + 新 token。
 * eos_id < 0 表示不提前停止。 */
int medusa_decode(CausalLM *target, const int *prompt, int prompt_len, int max_new,
                  int eos_id, MedusaHeads *heads, int *out, int out_cap,
                  SpecStats *stats);
int eagle_decode(CausalLM *target, const int *prompt, int prompt_len, int max_new,
                 int gamma, int eos_id, EagleDrafter *drafter, int *out, int out_cap,
                 SpecStats *stats);

#endif /* MEDUSA_H */
