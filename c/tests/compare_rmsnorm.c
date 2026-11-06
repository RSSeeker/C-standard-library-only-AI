/*
 * compare_rmsnorm.c —— RMSNorm 的跨语言逐位对拍
 *
 * 用法（在 c/ 目录执行，先跑 python tools/rmsnorm_reference.py）：
 *   ./tests/compare_rmsnorm.exe rmsnorm_reference.json
 *
 * 每个用例：固定 gamma，若干组固定 (x, dout)。逐位比较
 *   forward 的 y / inv_rms，backward 的 grad_gamma / dx。
 * RMSNorm 只用到 dot 求和、乘除与 sqrt（硬件指令），理论上必须逐位相同，
 * 所以 JSON 里所有用例 tol=0.0。
 */
#include "../json.h"
#include "../layers_norm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static int g_arrays;
static int g_exact_arrays;
static int g_values;
static int g_exact_values;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        printf("[错误] %s 缺少字段 '%s'（请重跑 tools/rmsnorm_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static double arr_at(const JsonValue *a, int i)
{
    return json_num(json_at(a, i), 0.0);
}

static void report_array(const char *what, const double *got, const JsonValue *want,
                         int n, double tol)
{
    int exact = 0;
    double worst = 0.0;
    int worst_at = -1;
    for (int i = 0; i < n; i++) {
        const double w = arr_at(want, i);
        const double g = got[i];
        const double d = fabs(g - w);
        if (memcmp(&g, &w, sizeof(double)) == 0) exact++;
        if (d > worst) { worst = d; worst_at = i; }
    }
    const int ok = (exact == n) || (worst <= tol);
    printf("    %-40s 逐位 %d/%d  最大差异 %.3e  %s%s\n",
           what, exact, n, worst, ok ? "[OK]" : "[超差]",
           exact == n ? "  (逐位相同)" : "");
    if (!ok) {
        printf("      首个越界点 index %d: C=%.17g py=%.17g 差 %.3e\n",
               worst_at, got[worst_at], arr_at(want, worst_at), worst);
        g_fail++;
    }
    g_arrays++;
    if (exact == n) g_exact_arrays++;
    for (int i = 0; i < n; i++) {
        g_values++;
        const double w = arr_at(want, i);
        if (memcmp(&got[i], &w, sizeof(double)) == 0) g_exact_values++;
    }
    if (worst > g_worst) g_worst = worst;
}

static void report_scalar(const char *what, double got, double want, double tol)
{
    const int exact = (memcmp(&got, &want, sizeof(double)) == 0);
    const double d = fabs(got - want);
    const int ok = exact || d <= tol;
    printf("    %-40s 期望 %-22.17g 实际 %-22.17g %s%s\n",
           what, want, got, ok ? "[OK]" : "[超差]", exact ? "  (逐位相同)" : "");
    if (!ok) g_fail++;
    g_values++;
    if (exact) g_exact_values++;
    if (d > g_worst) g_worst = d;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "rmsnorm_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        printf("[错误] 读 %s 失败: %s\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "顶层");
    const int nc = json_array_len(cases);
    printf("RMSNorm 对拍：%s（%d 个用例）\n", path, nc);

    for (int ci = 0; ci < nc; ci++) {
        const JsonValue *c = json_at(cases, ci);
        char who[96];
        snprintf(who, sizeof who, "cases[%d]", ci);
        const char *name = json_str(need(c, "name", who), "?");
        const char *note = json_str(json_get(c, "note"), NULL);
        const double tol = json_num(json_get(c, "tol"), 0.0);
        const int d = json_int(need(c, "d", who), 0);
        const double eps = json_num(need(c, "eps", who), 0.0);
        const JsonValue *jgamma = need(c, "gamma", who);
        const JsonValue *samples = need(c, "samples", who);
        const int ns = json_array_len(samples);

        printf("\n[%d] %s  d=%d eps=%g samples=%d\n", ci, name, d, eps, ns);
        if (note) printf("  备注: %s\n", note);
        if (json_array_len(jgamma) != d) {
            printf("  [错误] gamma 长度 %d != d=%d\n", json_array_len(jgamma), d);
            g_fail++;
            continue;
        }

        RMSNorm *rn = rmsnorm_create(d, eps);
        double *gamma = (double *)malloc((size_t)d * sizeof(double));
        double *xbuf = (double *)malloc((size_t)d * sizeof(double));
        double *dbuf = (double *)malloc((size_t)d * sizeof(double));
        double *ggot = (double *)malloc((size_t)d * sizeof(double));
        double *dxgot = (double *)malloc((size_t)d * sizeof(double));
        for (int j = 0; j < d; j++) gamma[j] = arr_at(jgamma, j);
        rmsnorm_set_params(rn, gamma);

        const int shape[1] = { d };
        Tensor *x = tensor_new(1, shape);
        Tensor *y = tensor_new(1, shape);
        Tensor *dout = tensor_new(1, shape);
        Tensor *dx = tensor_new(1, shape);

        for (int si = 0; si < ns; si++) {
            const JsonValue *s = json_at(samples, si);
            char swho[128];
            snprintf(swho, sizeof swho, "%s.samples[%d]", who, si);
            const JsonValue *jx = need(s, "x", swho);
            const JsonValue *jd = need(s, "dout", swho);
            const JsonValue *jy = need(s, "expect_y", swho);
            const JsonValue *jgg = need(s, "expect_grad_gamma", swho);
            const JsonValue *jdx = need(s, "expect_dx", swho);
            const double jr = json_num(need(s, "expect_inv_rms", swho), 0.0);
            if (json_array_len(jx) != d || json_array_len(jd) != d) {
                printf("    [错误] 样本 %d 的 x/dout 长度与 d 不符\n", si);
                g_fail++;
                continue;
            }
            for (int j = 0; j < d; j++) {
                xbuf[j] = arr_at(jx, j);
                dbuf[j] = arr_at(jd, j);
            }
            memcpy(x->data, xbuf, (size_t)d * sizeof(double));
            memcpy(dout->data, dbuf, (size_t)d * sizeof(double));

            printf("  样本 %d:\n", si);
            rmsnorm_forward(rn, x, y);
            report_array("y", y->data, jy, d, tol);
            report_scalar("inv_rms", rmsnorm_inv_rms(rn), jr, tol);

            rmsnorm_backward(rn, dout, dx);
            memcpy(ggot, rmsnorm_grad_gamma(rn), (size_t)d * sizeof(double));
            report_array("grad_gamma", ggot, jgg, d, tol);
            report_array("dx", dx->data, jdx, d, tol);
        }

        /* 参数个数与 Module 包装的 to_string */
        if (rmsnorm_param_count(rn) != d) {
            printf("  [错误] rmsnorm_param_count=%d != d=%d\n", rmsnorm_param_count(rn), d);
            g_fail++;
        }
        Module *m = rmsnorm_as_module(rn);
        char namebuf[64];
        m->vt->to_string(m, namebuf, sizeof namebuf);
        printf("    Module: %s  参数 %d\n", namebuf, module_param_count(m));
        module_free(m);   /* 同时释放 rn */

        tensor_free(x);
        tensor_free(y);
        tensor_free(dout);
        tensor_free(dx);
        free(gamma);
        free(xbuf);
        free(dbuf);
        free(ggot);
        free(dxgot);
    }

    printf("\n[汇总] %d 个数组（%d 个整体逐位相同）+ %d/%d 个值逐位相同；全局最大差异 %.3e\n",
           g_arrays, g_exact_arrays, g_exact_values, g_values, g_worst);
    if (g_fail == 0) printf("[OK] RMSNorm 与 AI.py 完全一致\n");
    else printf("[失败] 有 %d 处不一致\n", g_fail);

    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
