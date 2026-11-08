/*
 * compare_metrics.c —— 把 C 的指标函数与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_metrics [metrics_reference.json]
 *
 * 分类指标是整数/精确值，回归指标与 AI.py 同累加顺序，因此全部要求容差 0。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../metrics.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_metrics] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/metrics_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static void ints_from(const JsonValue *v, int *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_metrics] %s 长度不符：期望 %d，JSON 里是 %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = (int)json_num(json_at(v, i), 0.0);
}

static void nums_from(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_metrics] %s 长度不符：期望 %d，JSON 里是 %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    const double d = fabs(got - want);
    if (d > g_worst) g_worst = d;
    if (d <= tol) {
        printf("  %-16s C=%.17g py=%.17g 差=%.3e [容差 %.0e 内]\n",
               what, got, want, d, tol);
        return;
    }
    printf("  %-16s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
    g_fail++;
}

static MetricAverage parse_average(const JsonValue *cs, const char *who)
{
    const char *avg = json_str(need(cs, "average", who), "macro");
    if (strcmp(avg, "macro") == 0) return METRIC_AVG_MACRO;
    if (strcmp(avg, "micro") == 0) return METRIC_AVG_MICRO;
    if (strcmp(avg, "weighted") == 0) return METRIC_AVG_WEIGHTED;
    if (strcmp(avg, "none") == 0) return METRIC_AVG_NONE;
    fprintf(stderr, "[compare_metrics] 未知 average \"%s\"（用例 %s）\n", avg, who);
    exit(2);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "metrics_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_metrics] 读不了 %s：%s\n"
                        "请先运行 python tools/metrics_reference.py\n", path, err);
        return 2;
    }

    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_metrics] %s：%d 个用例\n", path, n_cases);

    int *yt = NULL, *yp = NULL;
    double *dt = NULL, *dp = NULL, *vals = NULL;
    int cap = 0;

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *func = json_str(json_get(cs, "func"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        const int has_true = json_get(cs, "y_true") != NULL;
        const int n = has_true ? json_array_len(json_get(cs, "y_true")) : 0;
        const int need_n = (n > 0) ? n : 1;

        if (need_n > cap) {
            free(yt); free(yp); free(dt); free(dp); free(vals);
            cap = need_n;
            yt = (int *)malloc(sizeof(int) * (size_t)cap);
            yp = (int *)malloc(sizeof(int) * (size_t)cap);
            dt = (double *)malloc(sizeof(double) * (size_t)cap);
            dp = (double *)malloc(sizeof(double) * (size_t)cap);
            vals = (double *)malloc(sizeof(double) * (size_t)cap);
            if (!yt || !yp || !dt || !dp || !vals) { fprintf(stderr, "out of memory\n"); return 2; }
        }

        printf("\n用例 %-26s func=%s\n", name, func);

        if (strcmp(func, "confusion_matrix") == 0) {
            ints_from(need(cs, "y_true", name), yt, n, name);
            ints_from(need(cs, "y_pred", name), yp, n, name);
            const int nc = (int)json_num(json_get(cs, "num_classes"), 0.0);
            int got_n = 0;
            int *mat = confusion_matrix(yt, yp, n, nc, &got_n);
            const int want_n = (int)json_num(need(cs, "n", name), 0.0);
            report("n", (double)got_n, (double)want_n, 0.0);
            const JsonValue *jm = need(cs, "matrix", name);
            if (json_array_len(jm) != want_n) {
                fprintf(stderr, "[compare_metrics] %s 的 matrix 行数不符\n", name);
                return 2;
            }
            for (int r = 0; r < want_n; r++) {
                const JsonValue *row = json_at(jm, r);
                if (json_array_len(row) != want_n) {
                    fprintf(stderr, "[compare_metrics] %s 的 matrix 第 %d 行列数不符\n", name, r);
                    return 2;
                }
                for (int col = 0; col < want_n; col++) {
                    const double got = (double)mat[(size_t)r * (size_t)want_n + (size_t)col];
                    const double want = json_num(json_at(row, col), 0.0);
                    if (got != want) {
                        printf("  [%d][%d] C=%.0f py=%.0f [不等]\n", r, col, got, want);
                        g_checks++;
                        g_fail++;
                    } else {
                        g_checks++;
                        g_exact++;
                    }
                }
            }
            free(mat);
        } else if (strcmp(func, "accuracy_score") == 0) {
            ints_from(need(cs, "y_true", name), yt, n, name);
            ints_from(need(cs, "y_pred", name), yp, n, name);
            report("accuracy", accuracy_score(yt, yp, n),
                   json_num(need(cs, "value", name), 0.0), tol);
        } else if (strcmp(func, "precision_recall_f1") == 0) {
            ints_from(need(cs, "y_true", name), yt, n, name);
            ints_from(need(cs, "y_pred", name), yp, n, name);
            const int nc = (int)json_num(json_get(cs, "num_classes"), 0.0);
            const MetricAverage avg = parse_average(cs, name);
            double out3[3] = {0.0, 0.0, 0.0};
            precision_recall_f1(yt, yp, n, nc, avg, out3);
            report("precision", out3[0], json_num(need(cs, "precision", name), 0.0), tol);
            report("recall", out3[1], json_num(need(cs, "recall", name), 0.0), tol);
            report("f1", out3[2], json_num(need(cs, "f1", name), 0.0), tol);

            const int k = (int)json_num(need(cs, "n", name), 0.0);
            double *pp = (double *)calloc((size_t)k, sizeof(double));
            double *rr = (double *)calloc((size_t)k, sizeof(double));
            double *ff = (double *)calloc((size_t)k, sizeof(double));
            double *want_pp = (double *)calloc((size_t)k, sizeof(double));
            double *want_rr = (double *)calloc((size_t)k, sizeof(double));
            double *want_ff = (double *)calloc((size_t)k, sizeof(double));
            if (!pp || !rr || !ff || !want_pp || !want_rr || !want_ff) {
                fprintf(stderr, "out of memory\n");
                return 2;
            }
            precision_recall_f1_per_class(yt, yp, n, nc, pp, rr, ff, NULL);
            nums_from(need(cs, "per_precision", name), want_pp, k, name);
            nums_from(need(cs, "per_recall", name), want_rr, k, name);
            nums_from(need(cs, "per_f1", name), want_ff, k, name);
            for (int i = 0; i < k; i++) {
                char label[64];
                snprintf(label, sizeof label, "per_precision[%d]", i);
                report(label, pp[i], want_pp[i], tol);
                snprintf(label, sizeof label, "per_recall[%d]", i);
                report(label, rr[i], want_rr[i], tol);
                snprintf(label, sizeof label, "per_f1[%d]", i);
                report(label, ff[i], want_ff[i], tol);
            }
            free(pp); free(rr); free(ff);
            free(want_pp); free(want_rr); free(want_ff);
        } else if (strcmp(func, "matthews_corrcoef") == 0) {
            ints_from(need(cs, "y_true", name), yt, n, name);
            ints_from(need(cs, "y_pred", name), yp, n, name);
            const int nc = (int)json_num(json_get(cs, "num_classes"), 0.0);
            report("mcc", matthews_corrcoef(yt, yp, n, nc),
                   json_num(need(cs, "value", name), 0.0), tol);
        } else if (strcmp(func, "top_k_accuracy") == 0) {
            ints_from(need(cs, "y_true", name), yt, n, name);
            const int nc = (int)json_num(need(cs, "num_classes", name), 0.0);
            const int k = (int)json_num(need(cs, "k", name), 1.0);
            const JsonValue *js = need(cs, "scores", name);
            const int rows = json_array_len(js);
            if (rows != n) {
                fprintf(stderr, "[compare_metrics] %s 的 scores 行数不符\n", name);
                return 2;
            }
            double *sc = (double *)calloc((size_t)n * (size_t)nc, sizeof(double));
            if (!sc) { fprintf(stderr, "out of memory\n"); return 2; }
            for (int r = 0; r < n; r++) {
                nums_from(json_at(js, r), sc + (size_t)r * (size_t)nc, nc, name);
            }
            report("top_k", top_k_accuracy(yt, sc, n, nc, k),
                   json_num(need(cs, "value", name), 0.0), tol);
            free(sc);
        } else if (strcmp(func, "roc_auc_score") == 0) {
            const int m = json_array_len(need(cs, "labels", name));
            ints_from(need(cs, "labels", name), yt, m, name);
            nums_from(need(cs, "scores", name), vals, m, name);
            report("roc_auc", roc_auc_score(yt, vals, m),
                   json_num(need(cs, "value", name), 0.0), tol);
        } else if (strcmp(func, "mean_absolute_error") == 0 ||
                   strcmp(func, "root_mean_squared_error") == 0 ||
                   strcmp(func, "mean_absolute_percentage_error") == 0 ||
                   strcmp(func, "r2_score") == 0) {
            nums_from(need(cs, "y_true", name), dt, n, name);
            nums_from(need(cs, "y_pred", name), dp, n, name);
            double got = 0.0;
            if (strcmp(func, "mean_absolute_error") == 0) {
                got = mean_absolute_error(dt, dp, n);
            } else if (strcmp(func, "root_mean_squared_error") == 0) {
                got = root_mean_squared_error(dt, dp, n);
            } else if (strcmp(func, "mean_absolute_percentage_error") == 0) {
                const double eps = json_num(json_get(cs, "eps"), 1e-12);
                got = mean_absolute_percentage_error(dt, dp, n, eps);
            } else {
                got = r2_score(dt, dp, n);
            }
            report(func, got, json_num(need(cs, "value", name), 0.0), tol);
        } else {
            fprintf(stderr, "[compare_metrics] 未知 func \"%s\"（用例 %s）\n", func, name);
            return 2;
        }
    }

    printf("\n[汇总] %d 项，其中 %d 项逐位相同；全局最大差异 %.3e\n",
           g_checks, g_exact, g_worst);
    if (g_fail == 0) {
        printf("[OK] 分类/回归指标与 AI.py 逐位一致\n");
    } else {
        printf("[FAIL] 有 %d 处不逐位相同\n", g_fail);
    }

    free(yt); free(yp); free(dt); free(dp); free(vals);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
