/* ============================================================================
 *  fim.h —— FIM（Fill-in-the-Middle）与 T5 span corruption 的 C17 实现
 *
 *  对齐 Python 版 AI.py「训练目标：FIM（Fill-in-the-Middle）与 T5 span
 *  corruption」一节：
 *    fim_transform / fim_target_mask / fim_batch
 *    span_corruption_mask / span_corruption_ids
 *    常量 FIM_SENTINEL_IDS = {prefix: 0, suffix: 1, middle: 2}
 *
 *  FIM 的三个哨兵 id 固定为 vocab_size / vocab_size+1 / vocab_size+2
 *  （依次 prefix / suffix / middle），所以词表实际需要 vocab_size+3 行。
 *
 *  随机性：Python 的 span_corruption_mask 用 `random.Random(seed)` 抽 span，
 *  本模块内置同款 MT19937（init_by_array + random() + expovariate +
 *  getrandbits/_randbelow），只要 seed 相同，抽出的 span 与 Python 逐位一致。
 *  与 augment.h「随机性外部注入」的约定不同：这里的 seed 是函数语义的一部分，
 *  因此直接实现 Python 兼容 RNG。Python 的 seed=None 走 urandom、不可复现，
 *  C 侧要求显式 seed（0 是合法值），参考脚本只用非负 seed。
 *  另外 Python 对负 seed 取绝对值，C 侧 seed 为无符号数，负值由调用方自行取绝对值。
 *
 *  所有函数返回 0 成功、-1 参数非法；输出数组由 malloc 分配，调用方用对应的
 *  *_free 释放。int 参数模拟 Python 的关键字参数：mode 取 FIM_MODE_PSM/SPM，
 *  sentinel_id == -1 表示 Python 的 None（缺省为 vocab_size）。
 * ==========================================================================*/
#ifndef AI_C_FIM_H
#define AI_C_FIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* FIM 模式（对齐 Python mode="psm"/"spm"） */
#define FIM_MODE_PSM 0
#define FIM_MODE_SPM 1

/* FIM_SENTINEL_IDS（哨兵在 vocab_size 之上的偏移） */
#define FIM_SENTINEL_PREFIX 0
#define FIM_SENTINEL_SUFFIX 1
#define FIM_SENTINEL_MIDDLE 2

/* fim_transform 的返回样本：PSM 为 <prefix>P<suffix>S<middle>M，
 * SPM 为 <suffix>S<prefix>P<middle>M；middle 是「要预测」的那段。 */
typedef struct {
    int *input_ids; /* 长度 n_input（= 原序列长度 + 3） */
    int n_input;
    int *prefix;
    int n_prefix;
    int *middle;
    int n_middle;
    int *suffix;
    int n_suffix;
    int middle_start; /* middle 在 input_ids 中的起点，= n_input - n_middle */
    int mode;         /* FIM_MODE_PSM / FIM_MODE_SPM */
} AiFimExample;

/* span corruption 的一段掩码（按起点升序） */
typedef struct {
    int start;
    int length;
} AiFimSpan;

/* span_corruption_ids 的返回：input_ids 长度 = n - 被掩码 token 数 + span 数，
 * target_ids 长度 = 被掩码 token 数 + span 数。 */
typedef struct {
    int *input_ids;
    int n_input;
    int *target_ids;
    int n_target;
    AiFimSpan *spans; /* malloc；调用方 free（或整体用 ai_span_corruption_free） */
    int n_spans;
    int sentinel_id;
} AiSpanCorruption;

/* ---- 三种 FIM 改写 ---- */

/* 把 ids[0..n_ids) 改写成 FIM 训练样本（确定性按比例切分；Python 的 seed 参数
 * 在当前实现里不影响结果，因此 C 侧未保留）。成功返回 0，参数非法返回 -1。 */
int ai_fim_transform(const int *ids, int n_ids, int vocab_size, int mode,
                     double prefix_ratio, double middle_ratio,
                     AiFimExample *out);
void ai_fim_example_free(AiFimExample *ex);

/* 逐位置损失掩码：mask 长度必须为 ex->n_input - 1（causal shift），
 * 只有预测 middle token 的位置为 1。成功返回 0，参数非法返回 -1。 */
int ai_fim_target_mask(const AiFimExample *ex, int *mask);

/* 批量改写：flat_ids 按 seq_lens[0..n_seqs) 切分（与 build_draft_tree 的
 * 扁平入参约定一致）。成功返回 0 并写 out 与 out_n（malloc），
 * 某一条非法则整体返回 -1；调用方用 ai_fim_batch_free 释放。 */
int ai_fim_batch(const int *flat_ids, const int *seq_lens, int n_seqs,
                 int vocab_size, int mode, double prefix_ratio,
                 double middle_ratio, AiFimExample **out, int *out_n);
void ai_fim_batch_free(AiFimExample *examples, int n_examples);

/* ---- T5 span corruption ---- */

/* 生成 T5 风格随机 span 掩码（按起点升序）。seed 与 Python
 * random.Random(seed) 的整数 seed 语义一致（非负；负 seed 请传绝对值）。
 * 成功返回 0 并写 out 与 out_n（malloc），参数非法返回 -1。 */
int ai_span_corruption_mask(int seq_len, double mask_ratio, double mean_span,
                            unsigned long long seed, AiFimSpan **out,
                            int *out_n);
void ai_fim_spans_free(AiFimSpan *spans);

/* 把 ids 改写成 span corruption 的 input_ids/target_ids：
 * 每个 span 在 input_ids 中用同一个哨兵替换（span 内 token 丢弃），
 * target_ids 为「哨兵 + 该 span 原始 token」拼接。
 * sentinel_id == -1 表示 Python 的 None（缺省 vocab_size）；
 * 显式哨兵必须在 [0, vocab_size) 内。成功返回 0，参数非法返回 -1。 */
int ai_span_corruption_ids(const int *ids, int n_ids, int vocab_size,
                           double mask_ratio, double mean_span,
                           int sentinel_id, unsigned long long seed,
                           AiSpanCorruption *out);
void ai_span_corruption_free(AiSpanCorruption *sc);

#ifdef __cplusplus
}
#endif

#endif /* AI_C_FIM_H */
