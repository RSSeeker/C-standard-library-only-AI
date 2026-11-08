/*
 * text_tok.h —— 字节级 BPE 分词器的 C17 移植（Python `BytePairTokenizer`）
 *
 * 与 Python 版一致的三条语义：
 *   1. 词表从 256 个字节 id 起长，每条合并规则 (a, b) -> 256 + k，只能引用
 *      已定义的 id（因此规则表天然是「按定义的拓扑序」）；
 *   2. 训练/编码前先做 pretokenize：字母数字连写 / 空白连写 / 其它符号连写
 *      分成词块，只块内学习与合并 —— 避免跨词边界生成 `the  cat` 这种 token；
 *   3. 训练挑下一个合并对的次序是「出现次数降序，再按 (a, b) 升序」
 *      （Python 的 `key=lambda kv: (kv[1], -kv[0][0], -kv[0][1])`）。
 *
 * 与 Python 版的唯一差异（已写进 README）：`_bpe_pretokenize` 在 Python 里用
 * `str.isalnum()` / `str.isspace()`，是完整的 Unicode 判定；C 侧只用一张
 * 覆盖拉丁/希腊/西里尔/CJK/假名/谚文的子表，表外码点（含 emoji）按「其它符号」
 * 处理。对表内字符两边结果逐位相同，表外脚本的切块边界可能不同。
 */
#ifndef AI_TEXT_TOK_H
#define AI_TEXT_TOK_H

#include <stddef.h>

typedef struct {
    int a;
    int b;
} BpePair;

typedef struct {
    int n_merges;
    int cap_merges;
    BpePair *merges;
    int vocab_size;                 /* 256 + n_merges */
    unsigned char **bytes_of;       /* [vocab_size]，decode 时直接拼接 */
    int *len_of;
    /* (a, b) -> rank 的开放寻址表，编码时 O(1) 查优先级 */
    long long *keys;
    int *ranks;
    int hash_cap;
} BpeTokenizer;

/* 空分词器：只有 256 个字节 id。 */
BpeTokenizer *bpe_create(void);
/* 用既有的合并规则构造（规则非法 —— 引用未定义 id / 自我循环 —— 返回 NULL）。 */
BpeTokenizer *bpe_create_from_merges(const int *pairs, int n_merges);
void bpe_free(BpeTokenizer *t);

int  bpe_vocab_size(const BpeTokenizer *t);
int  bpe_merge_count(const BpeTokenizer *t);
/* 取第 k 条规则；越界返回 0，否则 1。 */
int  bpe_merge_at(const BpeTokenizer *t, int k, int *a, int *b);

/* 在 text 上学习合并规则，返回学到的条数（失败返回 -1）。
 * vocab_size 是目标词表大小（>= 256），min_frequency 是停止合并的频次门槛（>= 1）。 */
int bpe_train(BpeTokenizer *t, const char *text, int vocab_size, int min_frequency);

/* pretokenize：把文本切词块，返回块数；*out 由调用方用 bpe_pretokenize_free 释放。
 * 失败（非法 UTF-8）返回 -1。 */
int  bpe_pretokenize(const char *text, char ***out);
void bpe_pretokenize_free(char **pieces, int n);

/* 编码：写入至多 cap 个 id，返回写入数；cap 不够返回 -1，非法 UTF-8 返回 -2。 */
int bpe_encode(const BpeTokenizer *t, const char *text, int *ids, int cap);

/* 解码：返回写入的字节数（不含结尾 NUL）；非法 id 返回 -1，
 * 拼出来不是合法 UTF-8 返回 -2，cap 不够返回 -3。总会写结尾 NUL（cap > 0 时）。 */
int bpe_decode(const BpeTokenizer *t, const int *ids, int n, char *out, int cap);

/* 存档："format": "ai_bpe_v1", "merges": [[a, b], ...]。成功返回 0。 */
int bpe_save(const BpeTokenizer *t, const char *path);
/* 读取；失败返回 NULL 并把原因写进 err（可为 NULL）。 */
BpeTokenizer *bpe_load(const char *path, char *err, size_t errlen);

#endif /* AI_TEXT_TOK_H */
