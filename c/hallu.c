/*
 * hallu.c —— 基于向量一致性的幻觉检测 / NLI 三分类（C 实现）
 */
#include "hallu.h"

#include <math.h>
#include <stdlib.h>

double ai_cosine_similarity(const double *a, const double *b, int n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    if (!a || !b || n < 1) return 0.0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int ai_sentence_similarity_matrix(const double *premise, int n_premise,
                                  const double *hypothesis, int n_hypothesis,
                                  int dim, double *out)
{
    if (!premise || !hypothesis || !out || n_premise < 1 || n_hypothesis < 1 ||
        dim < 1)
        return -1;
    for (int h = 0; h < n_hypothesis; h++) {
        for (int p = 0; p < n_premise; p++) {
            out[(size_t)h * (size_t)n_premise + (size_t)p] =
                ai_cosine_similarity(&hypothesis[(size_t)h * (size_t)dim],
                                     &premise[(size_t)p * (size_t)dim], dim);
        }
    }
    return 0;
}

int ai_nli_consistency(const double *premise, int n_premise,
                       const double *hypothesis, int n_hypothesis, int dim,
                       double entail_threshold, double contradiction_threshold,
                       double *entail_ratio, double *contradiction_ratio,
                       double *neutral_ratio, double *score, int *labels)
{
    int n_entail = 0, n_contradict = 0;
    if (!premise || !hypothesis || !labels || !entail_ratio ||
        !contradiction_ratio || !neutral_ratio || !score)
        return -1;
    if (n_premise < 1 || n_hypothesis < 1 || dim < 1) return -1;
    if (!isfinite(entail_threshold) || !(0.0 <= entail_threshold &&
                                         entail_threshold <= 1.0))
        return -1;
    if (!isfinite(contradiction_threshold) ||
        !(0.0 <= contradiction_threshold && contradiction_threshold <= 1.0))
        return -1;
    if (contradiction_threshold > entail_threshold) return -1;
    for (int h = 0; h < n_hypothesis; h++) {
        double best = -2.0;
        for (int p = 0; p < n_premise; p++) {
            double s = ai_cosine_similarity(
                &hypothesis[(size_t)h * (size_t)dim],
                &premise[(size_t)p * (size_t)dim], dim);
            if (s > best) best = s;
        }
        if (best >= entail_threshold) {
            labels[h] = 0;
            n_entail++;
        } else if (best <= contradiction_threshold) {
            labels[h] = 1;
            n_contradict++;
        } else {
            labels[h] = 2;
        }
    }
    *entail_ratio = (double)n_entail / (double)n_hypothesis;
    *contradiction_ratio = (double)n_contradict / (double)n_hypothesis;
    *neutral_ratio = 1.0 - *entail_ratio - *contradiction_ratio;
    *score = *entail_ratio - *contradiction_ratio;
    return 0;
}

int ai_hallucination_detect(const double *answer, int n_answer,
                            const double *source, int n_source, int dim,
                            double threshold, int *flagged, int *n_flagged,
                            int *spans, int *n_spans, double *sims)
{
    int nf = 0, ns = 0;
    if (!answer || !source || !flagged || !n_flagged || !spans || !n_spans ||
        !sims)
        return -1;
    if (n_answer < 1 || n_source < 1 || dim < 1) return -1;
    if (!isfinite(threshold) || !(0.0 <= threshold && threshold <= 1.0))
        return -1;
    for (int i = 0; i < n_answer; i++) {
        double best = -2.0;
        for (int j = 0; j < n_source; j++) {
            double s = ai_cosine_similarity(
                &answer[(size_t)i * (size_t)dim],
                &source[(size_t)j * (size_t)dim], dim);
            if (s > best) best = s;
        }
        sims[i] = best;
        if (best < threshold) {
            flagged[nf++] = i;
        }
    }
    for (int k = 0; k < nf; k++) {
        const int i = flagged[k];
        if (ns > 0 && i == spans[2 * (ns - 1) + 1] + 1) {
            spans[2 * (ns - 1) + 1] = i;
        } else {
            spans[2 * ns] = i;
            spans[2 * ns + 1] = i;
            ns++;
        }
    }
    *n_flagged = nf;
    *n_spans = ns;
    return 0;
}

int ai_hallucination_report(const double *answer, int n_answer,
                            const double *source, int n_source, int dim,
                            double threshold, int *out_unsupported,
                            int *n_unsupported)
{
    int *flagged = NULL, *spans = NULL;
    int n_flagged = 0, n_spans = 0;
    int i, verdict;
    double *sims = NULL;
    if (!out_unsupported || !n_unsupported) return -1;
    flagged = (int *)malloc(sizeof(int) * (size_t)(n_answer > 0 ? n_answer : 1));
    spans = (int *)malloc(sizeof(int) * 2 * (size_t)(n_answer > 0 ? n_answer : 1));
    sims = (double *)malloc(sizeof(double) * (size_t)(n_answer > 0 ? n_answer : 1));
    if (!flagged || !spans || !sims) {
        free(flagged);
        free(spans);
        free(sims);
        return -1;
    }
    if (ai_hallucination_detect(answer, n_answer, source, n_source, dim,
                                threshold, flagged, &n_flagged, spans,
                                &n_spans, sims) != 0) {
        free(flagged);
        free(spans);
        free(sims);
        return -1;
    }
    for (i = 0; i < n_flagged; i++) out_unsupported[i] = flagged[i];
    *n_unsupported = n_flagged;
    if (n_flagged == 0) verdict = 0;
    else if ((double)n_flagged / (double)n_answer <= 0.5) verdict = 1;
    else verdict = 2;
    free(flagged);
    free(spans);
    free(sims);
    return verdict;
}
