/*
 * ctc_beam.h —— CTC 前缀 beam search（可选字符级 LM 浅融合）的 C 实现
 *
 * 与 Python 版 AI.py 的 ctc_prefix_beam_search / ctc_beam_decode 对齐：
 * Graves 前缀 beam 的 p_blank / p_nonblank 合并算法，保留 beam_width 个前缀，
 * 最终按 `ctc_score + lm_weight*lm_score + length_penalty*len(tokens)` 取最优。
 *
 * 边界：
 *   - 只返回最优前缀及其分数（Python 返回的完整 beams 列表未暴露）；
 *   - LM 通过函数指针注入（C 没有闭包），返回 log 概率，越大越好；
 *   - log_probs 允许 -INFINITY；其他非法参数返回 -1。
 */
#ifndef CTC_BEAM_H
#define CTC_BEAM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*CtcLmFn)(const int *tokens, int n_tokens, void *ctx);

typedef struct {
    int *tokens;   /* malloc 分配；用 ctc_beam_result_free 释放，可为 NULL */
    int n_tokens;
    double score;
    double ctc_score;
    double lm_score;
} CtcBeamResult;

/* log_probs: T×K 行主序（已取 log）；blank ∈ [0,K)；
 * beam_width >= 1；lm 可为 NULL；lm_weight/length_penalty 必须有限。
 * 返回 0 成功，-1 失败。 */
int ctc_prefix_beam_search(const double *log_probs, int T, int K, int blank,
                           int beam_width, CtcLmFn lm, void *lm_ctx,
                           double lm_weight, double length_penalty,
                           CtcBeamResult *out);

/* 对未归一化 logits 逐帧做 log_softmax 后调用上面的函数。 */
int ctc_beam_decode(const double *logits, int T, int K, int blank,
                    int beam_width, CtcLmFn lm, void *lm_ctx,
                    double lm_weight, double length_penalty,
                    CtcBeamResult *out);

void ctc_beam_result_free(CtcBeamResult *r);

#ifdef __cplusplus
}
#endif

#endif /* CTC_BEAM_H */
