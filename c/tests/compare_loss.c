/*
 * compare_loss.c —— 把 C 的 focal / bce / bce_with_logits / huber 与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_loss [loss_reference.json]
 *       默认读当前目录的 loss_reference.json
 *
 * 规则：所有用例都要求逐位相同（tol 固定 0），任何一处 memcmp 不等就打印
 * 「下标 / C 值 / Python 值 / 差值」并返回 1。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../losses.h"

static int g_arrays;
static int g_exact;
static int g_fail;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_loss] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/loss_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_loss] %s 长度不符：期望 %d，JSON 里是 %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, double got, double want, double tol)
{
    g_arrays++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    double d = fabs(got - want);
    if (d > g_worst) g_worst = d;
    if (d <= tol) {
        printf("  %-12s C=%.17g py=%.17g 差=%.3e [容差 %.0e 内]\n",
               what, got, want, d, tol);
        return;
    }
    printf("  %-12s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
    g_fail++;
}

static void report_vec(const char *what, const double *got, const double *want, int n,
                       double tol)
{
    int neq = 0;
    int first = -1;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            neq++;
            if (first < 0) first = i;
            double d = fabs(got[i] - want[i]);
            if (d > worst) worst = d;
        }
    }
    g_arrays++;
    if (neq == 0) {
        g_exact++;
        printf("  %-12s 逐位 %d/%d\n", what, n, n);
        return;
    }
    if (worst > g_worst) g_worst = worst;
    if (worst <= tol) {
        printf("  %-12s 逐位 %d/%d 最大差异 %.3e [容差 %.0e 内]\n",
               what, n - neq, n, worst, tol);
        return;
    }
    printf("  %-12s 逐位 %d/%d 最大差异 %.3e [不等]\n", what, n - neq, n, worst);
    for (int i = 0, shown = 0; i < n && shown < 5; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            printf("      下标 %d: C=%.17g py=%.17g 差=%.3e\n",
                   i, got[i], want[i], fabs(got[i] - want[i]));
            shown++;
        }
    }
    g_fail++;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "loss_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_loss] 读不了 %s：%s\n"
                        "请先运行 python tools/loss_reference.py\n", path, err);
        return 2;
    }

    const JsonValue *cases = need(root, "cases", "root");
    int n_cases = json_array_len(cases);
    printf("[compare_loss] %s：%d 个用例\n", path, n_cases);

    double *a = NULL, *b = NULL, *got = NULL, *want = NULL;
    int cap = 0;

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *func = json_str(json_get(cs, "func"), "?");
        const double loss_want = json_num(need(cs, "loss", name), 0.0);
        const JsonValue *jgrad = need(cs, "grad", name);
        int ng = json_array_len(jgrad);

        const int is_logits = (strcmp(func, "bce_logits") == 0);
        const JsonValue *ja = need(cs, is_logits ? "logits" : "pred", name);
        int n = json_array_len(ja);
        int need_n = (n > ng) ? n : ng;
        if (need_n > cap) {
            free(a); free(b); free(got); free(want);
            cap = need_n;
            a = (double *)malloc(sizeof(double) * (size_t)cap);
            b = (double *)malloc(sizeof(double) * (size_t)cap);
            got = (double *)malloc(sizeof(double) * (size_t)cap);
            want = (double *)malloc(sizeof(double) * (size_t)cap);
            if (!a || !b || !got || !want) { fprintf(stderr, "out of memory\n"); return 2; }
        }
        arr_vals(ja, a, n, name);
        arr_vals(jgrad, want, ng, name);
        for (int i = 0; i < ng; i++) got[i] = 0.0;

        printf("\n用例 %-24s func=%-10s n=%d\n", name, func, n);

        double loss = 0.0;
        if (strcmp(func, "focal") == 0) {
            const int idx = (int)json_num(need(cs, "target_idx", name), 0.0);
            const double gamma = json_num(need(cs, "gamma", name), 0.0);
            const double alpha = json_num(need(cs, "alpha", name), -1.0);
            loss = focal_loss_grad(a, n, idx, gamma, alpha, got);
        } else if (strcmp(func, "bce") == 0) {
            arr_vals(need(cs, "target", name), b, ng, name);
            const double eps = json_num(need(cs, "eps", name), 1e-12);
            loss = bce_loss_grad(a, b, ng, eps, got);
        } else if (strcmp(func, "bce_logits") == 0) {
            arr_vals(need(cs, "logits", name), a, n, name);
            arr_vals(need(cs, "target", name), b, ng, name);
            loss = bce_with_logits_loss_grad(a, b, ng, got);
        } else if (strcmp(func, "huber") == 0) {
            arr_vals(need(cs, "target", name), b, ng, name);
            const double delta = json_num(need(cs, "delta", name), 1.0);
            loss = huber_loss_grad(a, b, ng, delta, got);
        } else {
            fprintf(stderr, "[compare_loss] 未知 func \"%s\"（用例 %s）\n", func, name);
            return 2;
        }

        const double tol = json_num(json_get(cs, "tol"), 0.0);
        report("loss", loss, loss_want, tol);
        report_vec("grad", got, want, ng, tol);
    }

    printf("\n[汇总] %d 个标量/数组，其中 %d 个逐位相同；全局最大差异 %.3e\n",
           g_arrays, g_exact, g_worst);
    if (g_fail == 0) {
        printf("[OK] focal/bce/bce_with_logits/huber 与 AI.py 逐位一致\n");
    } else {
        printf("[FAIL] 有 %d 处不逐位相同\n", g_fail);
    }

    free(a); free(b); free(got); free(want);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
