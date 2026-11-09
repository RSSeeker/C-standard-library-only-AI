/*
 * compare_diffusion.c —— C 侧扩散采样器与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_diffusion [diffusion_reference.json]
 *
 * 参考用例（tools/diffusion_reference.py 生成）提供 sigma 调度、Euler/Heun/
 * DPM-Solver/Consistency 采样、一致性蒸馏损失，以及参数非法的错误用例。
 * 去噪回调同构为 D(x, sigma) = scale/(1+sigma) * x + bias。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../diffusion.h"
#include "../json.h"

static int g_checks;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_diffusion] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_diffusion] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/diffusion_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static double jnum(const JsonValue *v) { return json_num(v, 0.0); }
static int jint(const JsonValue *v) { return json_int(v, 0); }

static double *darr(const JsonValue *v, int *n_out)
{
    int n = v ? v->n_items : 0;
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    if (n_out) *n_out = n;
    return out;
}

typedef struct {
    double scale;
    const double *bias;
    int n;
} LinCtx;

static void lin_denoise(const double *x, int n, double sigma, double *out, void *ctx)
{
    const LinCtx *c = (const LinCtx *)ctx;
    double k = c->scale / (1.0 + sigma);
    for (int i = 0; i < n; i++) out[i] = k * x[i] + c->bias[i % c->n];
}

static void check(const char *name, int idx, double got, double want, double tol)
{
    g_checks++;
    double diff = fabs(got - want);
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    double rel = diff / scale;
    if (diff > g_worst) g_worst = diff;
    if (rel > tol) {
        g_fail++;
        printf("  %-28s [%d] C=%.17g py=%.17g rel=%.3e [不等]\n",
               name, idx, got, want, rel);
    }
}

static void check_rc(const char *name, int rc)
{
    g_checks++;
    if (rc >= 0) {
        g_fail++;
        printf("  %-28s rc=%d 期望 <0 [不等]\n", name, rc);
    }
}

static void run_case(const JsonValue *c)
{
    const char *name = json_str(need(c, "name", "case"), "");
    const char *func = json_str(need(c, "func", name), "");

    if (strcmp(func, "error") == 0) {
        const char *target = json_str(need(c, "target", name), "");
        int rc = -2;
        if (strcmp(target, "sigmas") == 0) {
            int steps = json_get(c, "steps") ? jint(json_get(c, "steps")) : 5;
            double smin = json_get(c, "sigma_min") ? jnum(json_get(c, "sigma_min")) : 0.002;
            double smax = json_get(c, "sigma_max") ? jnum(json_get(c, "sigma_max")) : 80.0;
            double rho = json_get(c, "rho") ? jnum(json_get(c, "rho")) : 7.0;
            double buf[64];
            rc = diffusion_sigmas(steps, smin, smax, rho, buf, 64);
        } else if (strcmp(target, "edm_euler_sampler") == 0) {
            double x[2] = {0.0, 0.0};
            double out[2];
            LinCtx ctx = {0.8, x, 2};
            int steps = json_get(c, "steps") ? jint(json_get(c, "steps")) : 4;
            double smin = json_get(c, "sigma_min") ? jnum(json_get(c, "sigma_min")) : 0.01;
            double smax = json_get(c, "sigma_max") ? jnum(json_get(c, "sigma_max")) : 5.0;
            double rho = json_get(c, "rho") ? jnum(json_get(c, "rho")) : 2.0;
            rc = edm_euler_sampler(lin_denoise, &ctx, x, 2, steps, smin, smax, rho, out);
        } else if (strcmp(target, "consistency_training_loss") == 0) {
            double x[1] = {0.0};
            double out[1];
            LinCtx ctx = {0.8, x, 1};
            double sigma = json_get(c, "sigma") ? jnum(json_get(c, "sigma")) : 1.0;
            double sigma_next = json_get(c, "sigma_next") ? jnum(json_get(c, "sigma_next")) : 0.0;
            double loss = 0.0;
            rc = consistency_training_loss(lin_denoise, &ctx, lin_denoise, &ctx,
                                           x, 1, sigma, sigma_next, &loss);
            (void)out;
        }
        check_rc(name, rc);
        return;
    }

    if (strcmp(func, "sigmas") == 0) {
        int steps = jint(need(c, "steps", name));
        double smin = jnum(need(c, "sigma_min", name));
        double smax = jnum(need(c, "sigma_max", name));
        double rho = jnum(need(c, "rho", name));
        int n = 0;
        double *want = darr(need(c, "expected", name), &n);
        double *got = (double *)malloc(sizeof(double) * (size_t)n);
        if (!got) die("out of memory");
        int rc = diffusion_sigmas(steps, smin, smax, rho, got, n);
        if (rc != n) {
            g_fail++;
            printf("  %-28s 返回 %d 期望 %d [不等]\n", name, rc, n);
        }
        for (int i = 0; i < n; i++) check(name, i, got[i], want[i], 1e-12);
        free(got); free(want);
        return;
    }

    int xn = 0;
    double *x = darr(need(c, "x", name), &xn);

    if (strcmp(func, "consistency_training_loss") == 0) {
        int sn = 0, tn = 0;
        double *sb = darr(need(c, "student_bias", name), &sn);
        double *tb = darr(need(c, "teacher_bias", name), &tn);
        LinCtx sctx = {jnum(need(c, "student_scale", name)), sb, sn};
        LinCtx tctx = {jnum(need(c, "teacher_scale", name)), tb, tn};
        double sigma = jnum(need(c, "sigma", name));
        double sigma_next = jnum(need(c, "sigma_next", name));
        int want_n = 0;
        double *want = darr(need(c, "expected", name), &want_n);
        double loss = 0.0;
        int rc = consistency_training_loss(lin_denoise, &sctx, lin_denoise, &tctx,
                                          x, xn, sigma, sigma_next, &loss);
        if (rc != 0) {
            g_fail++;
            printf("  %-28s rc=%d [不等]\n", name, rc);
        }
        if (want_n > 0) check(name, 0, loss, want[0], 1e-12);
        free(sb); free(tb); free(want);
    } else {
        int bn = 0;
        double *bias = darr(need(c, "bias", name), &bn);
        LinCtx ctx = {jnum(need(c, "scale", name)), bias, bn};
        int steps = jint(need(c, "steps", name));
        double smin = jnum(need(c, "sigma_min", name));
        double smax = jnum(need(c, "sigma_max", name));
        double rho = jnum(need(c, "rho", name));
        int want_n = 0;
        double *want = darr(need(c, "expected", name), &want_n);
        double *got = (double *)malloc(sizeof(double) * (size_t)xn);
        if (!got) die("out of memory");
        int rc;
        if (strcmp(func, "edm_euler_sampler") == 0) {
            rc = edm_euler_sampler(lin_denoise, &ctx, x, xn, steps, smin, smax, rho, got);
        } else if (strcmp(func, "edm_heun_sampler") == 0) {
            rc = edm_heun_sampler(lin_denoise, &ctx, x, xn, steps, smin, smax, rho, got);
        } else if (strcmp(func, "dpm_solver_sampler") == 0) {
            rc = dpm_solver_sampler(lin_denoise, &ctx, x, xn, steps, smin, smax, rho, got);
        } else if (strcmp(func, "consistency_sampler") == 0) {
            rc = consistency_sampler(lin_denoise, &ctx, x, xn, steps, smin, smax, rho, got);
        } else {
            die("未知 func");
            return;
        }
        if (rc != 0 || want_n != xn) {
            g_fail++;
            printf("  %-28s rc=%d 或长度 %d/%d [不等]\n", name, rc, want_n, xn);
        }
        for (int i = 0; i < xn; i++) check(name, i, got[i], want[i], 1e-12);
        free(got); free(want); free(bias);
    }
    free(x);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "diffusion_reference.json";
    char err[256] = {0};
    JsonValue *doc = json_parse_file(path, err, sizeof err);
    if (!doc) {
        fprintf(stderr, "[compare_diffusion] JSON 解析失败：%s\n", err);
        return 2;
    }
    const JsonValue *cases = json_get(doc, "cases");
    if (!cases) die("缺少 cases");
    for (int i = 0; i < cases->n_items; i++) run_case(json_at(cases, i));

    printf("[compare_diffusion] %d 项检查：失败 %d，最大绝对差 %.3e\n",
           g_checks, g_fail, g_worst);
    json_free(doc);
    return g_fail == 0 ? 0 : 1;
}
