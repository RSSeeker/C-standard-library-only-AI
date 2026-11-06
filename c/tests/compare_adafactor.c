/*
 * compare_adafactor.c —— Adafactor 的跨语言对拍（C vs AI.py）
 *
 * 用法（在 c/ 目录执行，先跑 python tools/adafactor_reference.py）：
 *   ./tests/compare_adafactor.exe adafactor_reference.json
 *
 * 比较项：最终 w/b + 行二阶矩 R/列二阶矩 C/偏置二阶矩 V_b + step_count。
 * 所有用例 tol=0.0（唯一 libm 调用是 pow(t,-0.8) 与 sqrt；用例步数都 <= 12，
 * 实测该区间 msvcrt 与 UCRT 逐位一致）。任一处超差打印下标/期望/实际/差值，
 * 全部通过才打印 [OK]。
 */
#include "../json.h"
#include "../optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static int g_arrays;
static int g_exact_arrays;
static double g_worst;
static double g_tol_override = -1.0;

static void die(const char *msg)
{
    printf("[错误] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        printf("[错误] %s 缺少字段 '%s'（请重跑 tools/adafactor_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *a, double *out, int n, const char *who)
{
    const int len = json_array_len(a);
    if (len != n) {
        printf("[错误] %s 长度 %d != 期望 %d（JSON 与 C 侧形状不一致）\n", who, len, n);
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(a, i), 0.0);
}

static int report(const char *what, const double *got, const double *want, int n, double tol)
{
    int nident = 0, first_bad = -1, nbad = 0;
    int at_worst = 0;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) == 0) nident++;
        const double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at_worst = i; }
        if (!(d <= tol)) { if (first_bad < 0) first_bad = i; nbad++; }
    }
    const int exact = (nident == n);
    const int ok = exact || (tol > 0.0 && worst <= tol);

    printf("    %-8s 逐位 %d/%d  最大差异 %.3e  容差 %.0e  %s%s\n",
           what, nident, n, worst, tol, ok ? "[OK]" : "[超差]",
           exact ? "  (逐位相同)" : "");
    if (!ok) {
        printf("      下标 %d: 期望 %.17g  实际 %.17g  差值 %.3e（共 %d 处超差，最大在 %d）\n",
               first_bad, want[first_bad], got[first_bad],
               fabs(got[first_bad] - want[first_bad]), nbad, at_worst);
        g_fail++;
    }
    g_arrays++;
    if (exact) g_exact_arrays++;
    if (worst > g_worst) g_worst = worst;
    return ok;
}

static void free_case(double **w, double **b, double **gw, double **gb, OptLayer *ls, int nl)
{
    for (int k = 0; k < nl; k++) {
        free(w[k]); free(b[k]); free(gw[k]); free(gb[k]);
    }
    free(w); free(b); free(gw); free(gb); free(ls);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "adafactor_reference.json";
    if (argc > 2) g_tol_override = atof(argv[2]);

    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        printf("[错误] 读 %s 失败: %s\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "顶层");
    const int nc = json_array_len(cases);
    printf("Adafactor 对拍：%s（%d 个用例）\n", path, nc);

    int n_ok_cases = 0;
    for (int ci = 0; ci < nc; ci++) {
        const JsonValue *c = json_at(cases, ci);
        char who[96];
        snprintf(who, sizeof who, "case[%d]", ci);

        const char *name = json_str(need(c, "name", who), "?");
        const char *note = json_str(json_get(c, "note"), NULL);
        double tol = json_num(json_get(c, "tol"), 0.0);
        if (g_tol_override >= 0.0) tol = g_tol_override;
        const int steps = json_int(need(c, "steps", who), -1);
        const int want_step_count = json_int(need(c, "step_count", who), -1);
        const JsonValue *hyper = need(c, "hyper", who);
        const double lr = json_num(need(hyper, "lr", who), 0.0);
        const double eps = json_num(need(hyper, "eps", who), 0.0);
        const double eps2 = json_num(need(hyper, "eps2", who), 0.0);
        const double b2 = json_num(need(hyper, "beta2", who), 0.0);
        const double wd = json_num(need(hyper, "weight_decay", who), 0.0);

        printf("\n[用例 %d] %s  steps=%d  lr=%g  eps=%g  eps2=%g  b2=%g  wd=%g  tol=%.0e\n",
               ci, name, steps, lr, eps, eps2, b2, wd, tol);
        if (note) printf("  备注: %s\n", note);

        const JsonValue *jlayers = need(c, "layers", who);
        const int nl = json_array_len(jlayers);
        if (nl < 1) die("用例里没有层");

        OptLayer *ls = calloc((size_t)nl, sizeof(OptLayer));
        double **w = calloc((size_t)nl, sizeof(double *));
        double **b = calloc((size_t)nl, sizeof(double *));
        double **gw = calloc((size_t)nl, sizeof(double *));
        double **gb = calloc((size_t)nl, sizeof(double *));
        if (!ls || !w || !b || !gw || !gb) die("out of memory");

        int before = g_fail;

        for (int k = 0; k < nl; k++) {
            const JsonValue *jl = json_at(jlayers, k);
            char lw[96];
            snprintf(lw, sizeof lw, "case[%d].layers[%d]", ci, k);
            const int fo = json_int(need(jl, "fan_out", lw), 0);
            const int fi = json_int(need(jl, "fan_in", lw), 0);
            if (fo < 1 || fi < 1) die("fan_out/fan_in 必须 >= 1");
            const size_t nw = (size_t)fo * (size_t)fi, nb = (size_t)fo;
            w[k] = malloc(nw * sizeof(double));
            b[k] = malloc(nb * sizeof(double));
            gw[k] = malloc(nw * sizeof(double));
            gb[k] = malloc(nb * sizeof(double));
            if (!w[k] || !b[k] || !gw[k] || !gb[k]) die("out of memory");
            arr_vals(need(jl, "w", lw), w[k], (int)nw, lw);
            arr_vals(need(jl, "b", lw), b[k], (int)nb, lw);
            memset(gw[k], 0, nw * sizeof(double));
            memset(gb[k], 0, nb * sizeof(double));
            ls[k] = (OptLayer){w[k], b[k], gw[k], gb[k], fo, fi};
        }

        const char *cerr = NULL;
        Adafactor *opt = adafactor_create(ls, nl, lr, eps, eps2, b2, wd, &cerr);
        if (!opt) {
            printf("  [错误] adafactor_create 失败: %s\n", cerr ? cerr : "?");
            free_case(w, b, gw, gb, ls, nl);
            g_fail++;
            continue;
        }

        for (int s = 0; s < steps; s++) {
            for (int k = 0; k < nl; k++) {
                const JsonValue *jl = json_at(jlayers, k);
                char lw[120];
                snprintf(lw, sizeof lw, "case[%d].layers[%d].gw_steps[%d]", ci, k, s);
                const int nw = ls[k].fan_out * ls[k].fan_in;
                arr_vals(json_at(need(jl, "gw_steps", lw), s), gw[k], nw, lw);
                snprintf(lw, sizeof lw, "case[%d].layers[%d].gb_steps[%d]", ci, k, s);
                arr_vals(json_at(need(jl, "gb_steps", lw), s), gb[k], ls[k].fan_out, lw);
            }
            if (adafactor_step(opt, &cerr) != 0) {
                printf("  [错误] adafactor_step 失败: %s\n", cerr ? cerr : "?");
                g_fail++;
                break;
            }
        }

        if (adafactor_step_count(opt) != want_step_count) {
            printf("  [错误] step_count = %d，期望 %d\n",
                   adafactor_step_count(opt), want_step_count);
            g_fail++;
        }
        if (adafactor_num_layers(opt) != nl) {
            printf("  [错误] num_layers = %d，期望 %d\n", adafactor_num_layers(opt), nl);
            g_fail++;
        }

        for (int k = 0; k < nl; k++) {
            const JsonValue *ex = need(json_at(jlayers, k), "expect", "层");
            const int nw = ls[k].fan_out * ls[k].fan_in;
            const int nb = ls[k].fan_out;
            const int nc_ = ls[k].fan_in;
            double *tmp = malloc((size_t)(nw > nb ? nw : nb) * sizeof(double));
            if (!tmp) die("out of memory");

            printf("  层 %d（%d x %d）\n", k, ls[k].fan_out, ls[k].fan_in);
            arr_vals(need(ex, "w", "expect"), tmp, nw, "expect.w");
            report("w", w[k], tmp, nw, tol);
            arr_vals(need(ex, "b", "expect"), tmp, nb, "expect.b");
            report("b", b[k], tmp, nb, tol);
            arr_vals(need(ex, "R", "expect"), tmp, nb, "expect.R");
            report("R", adafactor_row(opt, k), tmp, nb, tol);
            arr_vals(need(ex, "C", "expect"), tmp, nc_, "expect.C");
            report("C", adafactor_col(opt, k), tmp, nc_, tol);
            arr_vals(need(ex, "V_b", "expect"), tmp, nb, "expect.V_b");
            report("V_b", adafactor_v_b(opt, k), tmp, nb, tol);
            free(tmp);
        }

        if (g_fail == before) n_ok_cases++;
        adafactor_free(opt);
        free_case(w, b, gw, gb, ls, nl);
    }

    printf("\n[汇总] %d 个用例中 %d 个通过；%d 个数组中 %d 个逐位相同；全局最大差异 %.3e\n",
           nc, n_ok_cases, g_arrays, g_exact_arrays, g_worst);
    if (g_fail == 0) printf("[OK] Adafactor 与 AI.py 完全一致\n");
    else printf("[失败] 有 %d 个数组超出容差\n", g_fail);

    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
