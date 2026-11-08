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

/* ======================================================================
 *  WordPiece（Python `WordPieceTokenizer`）
 *
 *  词表是**字符串**列表：前五个固定为 [PAD] [UNK] [CLS] [SEP] [MASK]，
 *  其余是字符 / 子词；词内非首字符带 `##` 前缀。训练按
 *  `freq / (left_freq * right_freq)` 打分，编码用最长前缀匹配。
 * ====================================================================== */

typedef struct {
    char **tokens;              /* 拥有所有字符串 */
    int n;
    int cap;
    int unk_id, pad_id, cls_id, sep_id, mask_id;
} WpVocab;

WpVocab *wp_create_default(void);
/* 用给定词表构造（缺特殊标记 / 有重复 / 空 token 时返回 NULL 并写 err）。 */
WpVocab *wp_create(const char *const *tokens, int n, char *err, size_t errlen);
void wp_free(WpVocab *v);

int  wp_vocab_size(const WpVocab *v);
const char *wp_token(const WpVocab *v, int id);
/* 查 token 的 id；不存在返回 -1。 */
int  wp_token_id(const WpVocab *v, const char *token);

/* 切词块（字母数字 / 空白 / 其它），返回块数；失败 -1。 */
int  wp_split_words(const char *text, char ***out);
/* 词 -> WordPiece 初始切片（首字符原样，其余 `##` 前缀），返回片数；失败 -1。 */
int  wp_to_pieces(const char *word, char ***out);

/* 编码：最大词长按**码点**计（与 Python 的 len(str) 一致）。返回 id 数，
 * 缓冲不足 -1，非法 UTF-8 -2。 */
int wp_encode(const WpVocab *v, const char *text, int *ids, int cap,
              int max_chars_per_word);
/* [CLS] a [SEP] b [SEP]。 */
int wp_encode_pair(const WpVocab *v, const char *a, const char *b, int *ids, int cap,
                   int max_chars_per_word);
/* 解码：返回字节数；id 越界 -1，缓冲不足 -3。 */
int wp_decode(const WpVocab *v, const int *ids, int n, char *out, int cap);

/* 训练并返回新词表；失败返回 NULL 并写 err。 */
WpVocab *wp_train(const char *text, int vocab_size, int min_frequency,
                  char *err, size_t errlen);

int  wp_save(const WpVocab *v, const char *path);
WpVocab *wp_load(const char *path, char *err, size_t errlen);

#endif /* AI_TEXT_TOK_H */
