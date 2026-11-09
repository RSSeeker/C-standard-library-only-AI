/*
 * text_metrics.h —— 文本质量指标（与 AI.py 的同名函数逐位一致）
 *
 * 对应 Python 侧 `meteor_score` / `meteor_score_full` / `chrf_score` /
 * `word_error_rate` / `character_error_rate`（AI.py 52440-53039）。
 *
 * 一致性由 c/tools/text_metrics_reference.py + c/tests/compare_text_metrics.c 保证：
 * 参考值从 AI.py 现场导出，C 侧逐位比对（容差 0，双精度）。
 *
 * 约定：
 *   - token 数组是 `const char *` 数组，长度由调用方给出（对应 Python 的 token 列表）。
 *   - 字符串 API 按 Python 的语义切分：WER 按空白切词、CER/chrF 按 Unicode 码点切字符。
 *     切分只认 ASCII 空白（Python 的 str.split() 还认 Unicode 空白，这里不覆盖）。
 *   - `_mt_stem` 是教学版英文词干；`_mt_norm` 的 lower() 这里只处理 ASCII 字母，
 *     非 ASCII 字符原样保留（测试用例覆盖中文）。
 *   - 参数非法时返回 NAN（C 侧没有异常；compare 脚本只喂合法输入）。
 */
#ifndef AI_TEXT_METRICS_H
#define AI_TEXT_METRICS_H

/* 同义词组：一组互相等价的 token（对应 Python 的 synonyms 组列表） */
typedef struct {
    const char *const *tokens;
    int n_tokens;
} TmSynonymGroup;

typedef struct {
    const TmSynonymGroup *groups;
    int n_groups;
} TmSynonyms;

/* ---------- 词干 ---------- */
/* 把 token 转小写并剥离常见英文后缀，结果写入 out（含结尾 0）。
 * 对应 Python 的 _mt_stem；out_size 不足时按 0 截断。 */
void tm_stem(const char *token, char *out, int out_size);

/* ---------- METEOR ---------- */
/* 简化版 METEOR：精确匹配 + 片段惩罚（对应 Python 的 meteor_score）。
 * pred/ref 为 token 数组。非法 alpha（>1 或 <0）或 beta/gamma<0 时返回 NAN。 */
double tm_meteor_score(const char *const *pred, int n_pred,
                       const char *const *ref, int n_ref,
                       double alpha, double beta, double gamma);

/* 完整版 METEOR：精确 -> 词干 -> 同义词贪心对齐，chunk 需在两端同时连续
 * （对应 Python 的 meteor_score_full）；syn 可为 NULL。 */
double tm_meteor_score_full(const char *const *pred, int n_pred,
                            const char *const *ref, int n_ref,
                            double alpha, double beta, double gamma,
                            const TmSynonyms *syn, int use_stem);

/* ---------- chrF ---------- */
/* 字符串版：小写化 -> 按空白切分再以单空格 join -> 字符 n-gram（1..max_n）F-beta 平均。
 * 对应 Python chrf_score(str, str, max_n, beta)。 */
double tm_chrf_score(const char *pred, const char *ref, int max_n, double beta);
/* token 数组版：每个 token 小写后以单空格 join（对应 Python list 输入）。 */
double tm_chrf_score_tokens(const char *const *pred, int n_pred,
                            const char *const *ref, int n_ref,
                            int max_n, double beta);

/* ---------- 编辑距离 / WER / CER ---------- */
/* 整数序列的最小编辑路径计数：返回 (替换, 删除, 插入) 到 subs/dels/ins。 */
void tm_edit_ops_int(const int *reference, int n_ref,
                     const int *hypothesis, int n_hyp,
                     int *subs, int *dels, int *ins);

/* 字符串 token 序列的编辑路径计数（按 strcmp 比较）。 */
void tm_edit_ops_tokens(const char *const *reference, int n_ref,
                        const char *const *hypothesis, int n_hyp,
                        int *subs, int *dels, int *ins);

/* WER = (替换 + 删除 + 插入) / 参考词数；字符串按空白切词。
 * 参考为空或参数非法时返回 NAN。 */
double tm_word_error_rate(const char *reference, const char *hypothesis);
double tm_word_error_rate_tokens(const char *const *reference, int n_ref,
                                 const char *const *hypothesis, int n_hyp);

/* CER = 字符级编辑距离 / 参考字符数；字符串按 Unicode 码点拆（中文友好）。 */
double tm_character_error_rate(const char *reference, const char *hypothesis);
double tm_character_error_rate_tokens(const char *const *reference, int n_ref,
                                      const char *const *hypothesis, int n_hyp);

#endif /* AI_TEXT_METRICS_H */
