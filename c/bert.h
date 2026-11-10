/* BERT 式模型前向（token + segment embedding → 正弦位置编码 → TransformerEncoder
 * → MLM / NSP 两个线性头）。对应 Python 的 AI.BertModel（只提供前向，无反向/训练）。
 *
 * 设计约定：
 *   - 输入是 (T) 的 int token id 与 0/1 segment id；输出 hidden 是 (T, d_model) 张量。
 *   - 权重通过访问器灌入（跨语言逐位对拍必须如此）：Embedding 用 emb_set_weight，
 *     TransformerEncoder 内部各层用 mha_set_params / ff_set_params / 直接写
 *     TransformerEncoderLayer.ln1/ln2 的 gamma/beta（TfLayerNorm 结构公开）。
 *   - 位置编码按 Python PositionalEncoding 的 sin/cos 方案，T > max_len 报错。
 */
#ifndef AI_BERT_H
#define AI_BERT_H

#include "attention.h"
#include "tensor.h"

typedef struct {
    int vocab_size;
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    int max_len;

    Embedding *tok_emb;            /* (vocab, d) */
    Embedding *seg_emb;            /* (2, d) */
    PositionalEncoding *pe;        /* (max_len, d) */
    TransformerEncoder *encoder;

    double *W_mlm;                 /* (vocab, d) */
    double *b_mlm;                 /* (vocab) */
    double *W_nsp;                 /* (2, d) */
    double *b_nsp;                 /* (2) */

    Tensor *t_x;                   /* (T, d) 工作缓冲（max_len） */
    Tensor *t_sum;                 /* (T, d) */
    Tensor *t_pe;                  /* (T, d) */
} BertModel;

/* seed 只用于 MLM/NSP 头的初始权重（对拍时会被灌入值覆盖）。 */
BertModel *bert_create(int vocab_size, int d_model, int num_heads, int d_ff,
                       int num_layers, int max_len, unsigned long long seed);
void bert_free(BertModel *m);

int bert_param_count(const BertModel *m);

/* forward：ids/seg 长度 T，seg 可为 NULL（视为全 0）。hidden 必须是 (T, d_model) 张量。 */
void bert_forward(BertModel *m, const int *ids, const int *seg, int T,
                  Tensor *hidden);

/* MLM 词表 logits：out 长度 T*vocab_size，行主序 (token, vocab)。 */
void bert_mlm_logits(const BertModel *m, const Tensor *hidden, double *out);
/* NSP 二分类 logits（取第 0 个位置）：out 长度 2。 */
void bert_nsp_logits(const BertModel *m, const Tensor *hidden, double *out);

/* 权重访问器（对拍时逐块灌入）。 */
Embedding *bert_token_embedding(BertModel *m);
Embedding *bert_segment_embedding(BertModel *m);
PositionalEncoding *bert_positional_encoding(BertModel *m);
TransformerEncoder *bert_encoder(BertModel *m);
void bert_set_mlm(BertModel *m, const double *W, const double *b);
void bert_set_nsp(BertModel *m, const double *W, const double *b);

#endif /* AI_BERT_H */
