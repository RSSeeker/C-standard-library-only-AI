/*
 * text.h —— 字符级文本处理器与文本生成（对应 Python 的 TextProcessor / generate_text）
 *
 * 与 Python 版的一致性要点：
 *   - Python 的 `sorted(list(set(text)))` 是按 **Unicode 码点**排序去重，
 *     所以 C 版也按码点处理（UTF-8 解码 -> 排序 -> 二分查找），而不是按字节。
 *   - `encode_indices` 对未知字符回退到 0；`decode_indices` 对未知索引用 '?'
 *   - `_softmax_with_temp` 先除温度、再减最大值、再 exp，最后用 **sum()**（Neumaier）
 *   - `sample_index` 用累积分布采样；C 版用自己的 xorshift，不追求与 MT 逐位一致
 */
#ifndef AI_TEXT_H
#define AI_TEXT_H

#include <stddef.h>

typedef struct {
    int *chars;        /* 升序排列的 Unicode 码点 */
    int vocab_size;
} TextProcessor;

/* ---------- UTF-8 工具 ---------- */
/* 解码一个码点，返回消耗的字节数；失败返回 0 */
int utf8_decode(const char *s, int *cp);
/* 编码一个码点到 buf，返回字节数（buf 至少 4 字节）；非法码点返回 0 */
int utf8_encode(int cp, char *buf);
/* 字符串里的码点个数（按 Python 的 len() 语义，不是字节数） */
int utf8_length(const char *s);
/* 取第 idx 个码点（0 基）；越界返回 -1 */
int utf8_cp_at(const char *s, int idx);

/* ---------- TextProcessor ---------- */
TextProcessor *tp_create(const char *utf8_text);
void tp_free(TextProcessor *tp);
int  tp_idx_of(const TextProcessor *tp, int cp);        /* 未找到返回 -1 */
int  tp_cp_at(const TextProcessor *tp, int idx);        /* 越界返回 -1 */
/* 把一个码点写成 one-hot（vec 长度必须为 vocab_size）；未知字符全 0 */
void tp_encode_one_hot_cp(const TextProcessor *tp, int cp, double *vec);
/* 把字符串写成索引序列，返回个数（写入 out，容量至少 utf8_length(s)） */
int  tp_encode_indices(const TextProcessor *tp, const char *s, int *out);
/* 索引 -> UTF-8 字符串（未找到的索引用 '?'），返回写入的字节数 */
int  tp_decode_indices(const TextProcessor *tp, const int *idx, int n, char *buf, int bufsize);

/* ---------- 生成 ---------- */
/* 带温度的 softmax（输出写入 probs，长度 n） */
void softmax_with_temp(const double *logits, int n, double temperature, double *probs);
/* 按概率分布采样一个索引；rng_state 为 xorshift64 状态 */
int sample_index(const double *probs, int n, unsigned long long *rng_state);
/* [0,1) 均匀随机（xorshift64*） */
double rng_uniform(unsigned long long *state);

/* 任意"可前向"模型的回调：把 n_in 维输入写进 out（n_out 维） */
typedef void (*ForwardFn)(void *ctx, const double *x, int n_in, double *out, int n_out);

/*
 * 自回归文本生成（对应 generate_text_with_mlp / generate_text）。
 *   start_str  : 起始字符串（UTF-8）
 *   length     : 生成多少个字符（按码点计）
 *   window_size: 上下文窗口大小
 *   temp       : 采样温度
 *   out_buf    : 输出缓冲，生成结果 = start_str + length 个字符
 * 返回生成出来的字符个数（不含 start_str），失败返回 -1。
 */
int generate_text(ForwardFn fwd, void *ctx, const TextProcessor *tp,
                  const char *start_str, int length, int window_size, double temp,
                  char *out_buf, int out_buf_size, unsigned long long *rng_state);

#endif /* AI_TEXT_H */
