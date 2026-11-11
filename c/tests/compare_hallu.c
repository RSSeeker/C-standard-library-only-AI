/*
 * compare_hallu.c —— 幻觉检测 / NLI 三分类与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_hallu [hallu_reference.json]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hallu.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_hallu] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_hallu] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/hallu_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static int arr_len(const JsonValue *v) { return json_array_len(v); }

static double *darr(const JsonValue *v, int n)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    return out;
}

static int *iarr(const JsonValue *v, int n)
{
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
    return out;
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double d = fabs(got - want);
        if (d > g_worst) g_worst = d;
        if (d <= tol) return;
        printf("  %-28s C=%.17g py=%.17g [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-28s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

static void case_cosine(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *av = need(cs, "a", name);
    const JsonValue *bv = need(cs, "b", name);
    const int n = arr_len(av);
    double *a = darr(av, n);
    double *b = darr(bv, n);
    check_num("cosine", ai_cosine_similarity(a, b, n),
              jnum(need(cs, "out", name)), tol);
    free(a);
    free(b);
}

static void case_matrix(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *pv = need(cs, "premise", name);
    const JsonValue *hv = need(cs, "hypothesis", name);
    const int n_pre = jint(need(cs, "n_premise", name));
    const int n_hyp = jint(need(cs, "n_hypothesis", name));
    const int dim = jint(need(cs, "dim", name));
    double *pre = darr(pv, n_pre * dim);
    double *hyp = darr(hv, n_hyp * dim);
    double *got = (double *)malloc(sizeof(double) * (size_t)(n_hyp * n_pre));
    double *want = darr(need(cs, "out", name), n_hyp * n_pre);
    if (!got) die("out of memory");
    if (ai_sentence_similarity_matrix(pre, n_pre, hyp, n_hyp, dim, got) != 0)
        die("ai_sentence_similarity_matrix 失败");
    for (int i = 0; i < n_hyp * n_pre; i++) check_num("matrix", got[i], want[i], tol);
    free(pre);
    free(hyp);
    free(got);
    free(want);
}

static void case_nli(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *pv = need(cs, "premise", name);
    const JsonValue *hv = need(cs, "hypothesis", name);
    const int n_pre = jint(need(cs, "n_premise", name));
    const int n_hyp = jint(need(cs, "n_hypothesis", name));
    const int dim = jint(need(cs, "dim", name));
    const double ent = jnum(need(cs, "entail_threshold", name));
    const double con = jnum(need(cs, "contradiction_threshold", name));
    double *pre = darr(pv, n_pre * dim);
    double *hyp = darr(hv, n_hyp * dim);
    int *labels = (int *)malloc(sizeof(int) * (size_t)n_hyp);
    int *want_labels = iarr(need(cs, "labels", name), n_hyp);
    double er, cr, nr, sc;
    if (!labels) die("out of memory");
    if (ai_nli_consistency(pre, n_pre, hyp, n_hyp, dim, ent, con, &er, &cr, &nr,
                           &sc, labels) != 0)
        die("ai_nli_consistency 失败");
    check_num("entail_ratio", er, jnum(need(cs, "entail_ratio", name)), tol);
    check_num("contradiction_ratio", cr,
              jnum(need(cs, "contradiction_ratio", name)), tol);
    check_num("neutral_ratio", nr, jnum(need(cs, "neutral_ratio", name)), tol);
    check_num("score", sc, jnum(need(cs, "score", name)), tol);
    for (int i = 0; i < n_hyp; i++) {
        char label[64];
        snprintf(label, sizeof label, "nli_label[%d]", i);
        check_int(label, labels[i], want_labels[i]);
    }
    free(pre);
    free(hyp);
    free(labels);
    free(want_labels);
}

static void case_detect(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *av = need(cs, "answer", name);
    const JsonValue *sv = need(cs, "source", name);
    const int n_ans = jint(need(cs, "n_answer", name));
    const int n_src = jint(need(cs, "n_source", name));
    const int dim = jint(need(cs, "dim", name));
    const double thr = jnum(need(cs, "threshold", name));
    double *ans = darr(av, n_ans * dim);
    double *src = darr(sv, n_src * dim);
    int *got_flagged = (int *)malloc(sizeof(int) * (size_t)n_ans);
    int *got_spans = (int *)malloc(sizeof(int) * 2 * (size_t)n_ans);
    double *got_sims = (double *)malloc(sizeof(double) * (size_t)n_ans);
    int nf = 0, ns = 0;
    const JsonValue *want_flagged = need(cs, "flagged", name);
    const int want_nf = arr_len(want_flagged);
    const JsonValue *want_spans = need(cs, "spans", name);
    const int want_ns = jint(need(cs, "n_spans", name));
    const JsonValue *want_sims = need(cs, "similarities", name);
    if (!got_flagged || !got_spans || !got_sims) die("out of memory");
    if (ai_hallucination_detect(ans, n_ans, src, n_src, dim, thr, got_flagged,
                                &nf, got_spans, &ns, got_sims) != 0)
        die("ai_hallucination_detect 失败");
    check_int("n_flagged", nf, want_nf);
    for (int i = 0; i < want_nf; i++) {
        char label[64];
        snprintf(label, sizeof label, "flagged[%d]", i);
        check_int(label, got_flagged[i], jint(json_at(want_flagged, i)));
    }
    check_int("n_spans", ns, want_ns);
    for (int i = 0; i < want_ns * 2; i++) {
        char label[64];
        snprintf(label, sizeof label, "span[%d]", i);
        check_int(label, got_spans[i], jint(json_at(want_spans, i)));
    }
    for (int i = 0; i < n_ans; i++) {
        char label[64];
        snprintf(label, sizeof label, "sim[%d]", i);
        check_num(label, got_sims[i], jnum(json_at(want_sims, i)), tol);
    }
    check_num("hallucination_ratio",
              (double)want_nf / (double)n_ans,
              jnum(need(cs, "hallucination_ratio", name)), tol);
    free(ans);
    free(src);
    free(got_flagged);
    free(got_spans);
    free(got_sims);
}

static void case_report(const JsonValue *cs, const char *name, double tol)
{
    (void)tol;
    const JsonValue *av = need(cs, "answer", name);
    const JsonValue *sv = need(cs, "source", name);
    const int n_ans = jint(need(cs, "n_answer", name));
    const int n_src = jint(need(cs, "n_source", name));
    const int dim = jint(need(cs, "dim", name));
    const double thr = jnum(need(cs, "threshold", name));
    double *ans = darr(av, n_ans * dim);
    double *src = darr(sv, n_src * dim);
    int *unsupported = (int *)malloc(sizeof(int) * (size_t)n_ans);
    int n_unsup = 0;
    int verdict;
    if (!unsupported) die("out of memory");
    verdict = ai_hallucination_report(ans, n_ans, src, n_src, dim, thr,
                                      unsupported, &n_unsup);
    if (verdict < 0) die("ai_hallucination_report 失败");
    check_int("verdict", verdict, jint(need(cs, "verdict", name)));
    check_int("n_unsupported", n_unsup,
              jint(need(cs, "n_unsupported", name)));
    free(ans);
    free(src);
    free(unsupported);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "hallu_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_hallu] 读不了 %s：%s\n"
                        "请先运行 python tools/hallu_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = arr_len(cases);
    printf("[compare_hallu] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        printf("\n用例 %-22s kind=%s\n", name, kind);
        if (strcmp(kind, "cosine") == 0) case_cosine(cs, name, tol);
        else if (strcmp(kind, "matrix") == 0) case_matrix(cs, name, tol);
        else if (strcmp(kind, "nli") == 0) case_nli(cs, name, tol);
        else if (strcmp(kind, "detect") == 0) case_detect(cs, name, tol);
        else if (strcmp(kind, "report") == 0) case_report(cs, name, tol);
        else die("未知 kind");
    }
    printf("\n=== compare_hallu：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
