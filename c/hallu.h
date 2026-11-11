/*
 * hallu.h —— 基于向量一致性的幻觉检测 / NLI 三分类（C 实现）
 *
 * 纯数值函数：余弦相似度、句向量相似度矩阵、NLI 三分类比例、
 * token 级幻觉检测与报告。不依赖模型调用。
 */
#ifndef AI_HALLU_H
#define AI_HALLU_H

#ifdef __cplusplus
extern "C" {
#endif

/* 两个 n 维向量的余弦相似度；任一为零向量时返回 0。 */
double ai_cosine_similarity(const double *a, const double *b, int n);

/* 句向量相似度矩阵：out 为 n_hyp × n_pre 行主序。返回 0 成功。 */
int ai_sentence_similarity_matrix(const double *premise, int n_premise,
                                  const double *hypothesis, int n_hypothesis,
                                  int dim, double *out);

/*
 * NLI 三分类：每个假设句取与前提句的最大余弦。
 * labels[i]：0=entail，1=contradict，2=neutral。
 * 比例写入 entail_ratio/contradiction_ratio/neutral_ratio，score=entail-contradiction。
 */
int ai_nli_consistency(const double *premise, int n_premise,
                       const double *hypothesis, int n_hypothesis, int dim,
                       double entail_threshold, double contradiction_threshold,
                       double *entail_ratio, double *contradiction_ratio,
                       double *neutral_ratio, double *score, int *labels);

/*
 * token 级幻觉检测：每个答案 token 取与源 token 的最大余弦。
 * flagged/sims 长度至少 n_answer；spans 长度至少 n_answer（成对 [start,end] 展平）。
 * 返回 0 成功；失败返回 -1。
 */
int ai_hallucination_detect(const double *answer, int n_answer,
                            const double *source, int n_source, int dim,
                            double threshold, int *flagged, int *n_flagged,
                            int *spans, int *n_spans, double *sims);

/*
 * 幻觉报告：返回 verdict（0=supported，1=partially_supported，2=unsupported），
 * unsupported 下标写入 out_unsupported（长度至少 n_answer）。
 */
int ai_hallucination_report(const double *answer, int n_answer,
                            const double *source, int n_source, int dim,
                            double threshold, int *out_unsupported,
                            int *n_unsupported);

#ifdef __cplusplus
}
#endif

#endif /* AI_HALLU_H */
