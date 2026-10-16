/*
 * compare_norm.c —— 读取 tools/norm_reference.py 导出的参考数据，
 * 用 C 实现重算 BatchNorm1d / LayerNorm / Dropout 并逐项比对。
 *
 * 用法: compare_norm [norm_reference.json]
 * 退出码: 0 = 全部在容差内（默认要求逐位相同），1 = 存在超差项
 */
#include "../json.h"
#include "../layers_norm.h"
#include "../tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static double g_tol = 0.0;          /* 默认要求逐位相同 */

/* 取对象里的必需字段；缺失就报错退出。
 * 注意：这里绝不能默默按 0 处理——曾因为导出的 JSON 漏了一个 "dout" 字段，
 * 导致 C 侧拿到全 0 输入，"反向梯度全是 0"看起来像实现的 bug。 */
static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/norm_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

/* ---------- 取数组的小工具 ---------- */
static void arr_vals(const JsonValue *v, double *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = json_num(json_at(v, i), 0.0);
}

/* 二维数组（行优先）展平 */
static void arr2_vals(const JsonValue *v, double *dst, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        const JsonValue *row = json_at(v, i);
        for (int j = 0; j < cols; j++)
            dst[i * cols + j] = json_num(json_at(row, j), 0.0);
    }
}

static void report(const char *what, const double *got, const double *want, int n)
{
    double worst = 0.0;
    int at = -1, exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    int ok = (worst <= g_tol);
    if (!ok) g_fail++;
    printf("  %-34s 最大差异 %.3e  %s%s\n", what, worst,
           ok ? "[OK]" : "[超出容差]", exact ? "  (逐位相同)" : "");
    if (!ok) printf("      index %d: C=%.17g py=%.17g\n", at, got[at], want[at]);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "norm_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/norm_reference.py %s\n", path);
        return 1;
    }
    printf("参考数据: %s（容差 %.1e）\n\n", path, g_tol);

    /* ===================== BatchNorm1d 训练路径 ===================== */
    {
        const JsonValue *bn = json_get(root, "bn");
        int d = json_int(need(bn, "d", "bn"), 0);
        int n = json_int(need(bn, "n", "bn"), 0);
        double eps = json_num(need(bn, "eps", "bn"), 1e-5);
        double mom = json_num(need(bn, "momentum", "bn"), 0.1);

        BatchNorm1d *b = bn1d_create(d, eps, mom);
        bn1d_reserve(b, n);
        double *gamma = malloc((size_t)d * sizeof(double));
        double *beta = malloc((size_t)d * sizeof(double));
        double *rm = malloc((size_t)d * sizeof(double));
        double *rv = malloc((size_t)d * sizeof(double));
        arr_vals(need(bn, "gamma", "bn"), gamma, d);
        arr_vals(need(bn, "beta", "bn"), beta, d);
        arr_vals(need(bn, "running_mean", "bn"), rm, d);
        arr_vals(need(bn, "running_var", "bn"), rv, d);
        bn1d_set_params(b, gamma, beta);
        bn1d_set_running(b, rm, rv);

        int xs[2] = { n, d };
        Tensor *x = tensor_new(2, xs), *out = tensor_new(2, xs);
        Tensor *dout = tensor_new(2, xs), *dx = tensor_new(2, xs);
        arr2_vals(need(bn, "x", "bn"), x->data, n, d);
        arr2_vals(need(bn, "dout", "bn"), dout->data, n, d);

        printf("BatchNorm1d 训练路径 (n=%d, d=%d, eps=%g, momentum=%g)\n", n, d, eps, mom);
        bn1d_forward_batch(b, x, out);
        arr2_vals(need(bn, "out", "bn"), dx->data, n, d);      /* 暂存期望值 */
        report("forward (N,D)", out->data, dx->data, n * d);

        bn1d_backward(b, dout, dx);
        double *want = malloc((size_t)n * d * sizeof(double));
        arr2_vals(need(bn, "dx", "bn"), want, n, d);
        report("backward dx", dx->data, want, n * d);

        double *w1 = malloc((size_t)d * sizeof(double));
        arr_vals(need(bn, "grad_gamma", "bn"), w1, d);
        report("grad_gamma", bn1d_grad_gamma(b), w1, d);
        arr_vals(need(bn, "grad_beta", "bn"), w1, d);
        report("grad_beta", bn1d_grad_beta(b), w1, d);

        /* running 统计量是否按 Python 的公式更新 */
        arr_vals(need(bn, "running_mean_after", "bn"), w1, d);
        report("running_mean 更新后", b->running_mean, w1, d);
        arr_vals(need(bn, "running_var_after", "bn"), w1, d);
        report("running_var 更新后", b->running_var, w1, d);

        /* 中间量：mean / var / inv_std / x_hat（x_hat 是列主序 d×n） */
        arr_vals(need(bn, "mean", "bn"), w1, d);
        report("batch mean", b->mean, w1, d);
        arr_vals(need(bn, "var", "bn"), w1, d);
        report("batch var（有偏）", b->var, w1, d);
        arr_vals(need(bn, "inv_std", "bn"), w1, d);
        report("inv_std", b->inv_std, w1, d);

        const JsonValue *xh = need(bn, "x_hat", "bn");        /* Python 是 n×d */
        double *want_xh = malloc((size_t)n * d * sizeof(double));
        arr2_vals(xh, want_xh, n, d);
        double *got_xh = malloc((size_t)n * d * sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                got_xh[i * d + j] = b->x_hat[j * n + i];     /* 列主序 -> 行主序 */
        report("x_hat", got_xh, want_xh, n * d);

        free(gamma); free(beta); free(rm); free(rv); free(want); free(w1);
        free(want_xh); free(got_xh);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        bn1d_free(b);
    }

    /* ===================== BatchNorm1d 推理路径 ===================== */
    {
        const JsonValue *bi = json_get(root, "bn_infer");
        int d = json_int(json_get(bi, "d"), 0);
        double eps = json_num(json_get(bi, "eps"), 1e-5);

        BatchNorm1d *b = bn1d_create(d, eps, 0.1);
        double *gamma = malloc((size_t)d * sizeof(double));
        double *beta = malloc((size_t)d * sizeof(double));
        double *rm = malloc((size_t)d * sizeof(double));
        double *rv = malloc((size_t)d * sizeof(double));
        arr_vals(json_get(bi, "gamma"), gamma, d);
        arr_vals(json_get(bi, "beta"), beta, d);
        bn1d_set_params(b, gamma, beta);
        arr_vals(json_get(bi, "running_mean"), rm, d);
        arr_vals(json_get(bi, "running_var"), rv, d);
        bn1d_set_running(b, rm, rv);

        int xs[1] = { d };
        Tensor *x = tensor_new(1, xs), *out = tensor_new(1, xs);
        arr_vals(json_get(bi, "x"), x->data, d);
        double *want = malloc((size_t)d * sizeof(double));
        arr_vals(json_get(bi, "out"), want, d);

        printf("\nBatchNorm1d 推理路径 (d=%d)\n", d);
        bn1d_forward(b, x, out);
        report("forward (D,) 用 running 统计量", out->data, want, d);

        free(gamma); free(beta); free(rm); free(rv); free(want);
        tensor_free(x); tensor_free(out);
        bn1d_free(b);
    }

    /* ===================== LayerNorm ===================== */
    {
        const JsonValue *lj = json_get(root, "ln");
        int d = json_int(need(lj, "d", "ln"), 0);
        double eps = json_num(need(lj, "eps", "ln"), 1e-5);

        LayerNorm *ln = layernorm_create(d, eps);
        double *gamma = malloc((size_t)d * sizeof(double));
        double *beta = malloc((size_t)d * sizeof(double));
        arr_vals(need(lj, "gamma", "ln"), gamma, d);
        arr_vals(need(lj, "beta", "ln"), beta, d);
        layernorm_set_params(ln, gamma, beta);

        int xs[1] = { d };
        Tensor *x = tensor_new(1, xs), *out = tensor_new(1, xs);
        Tensor *dout = tensor_new(1, xs), *dx = tensor_new(1, xs);
        arr_vals(need(lj, "x", "ln"), x->data, d);
        arr_vals(need(lj, "dout", "ln"), dout->data, d);

        printf("\nLayerNorm (d=%d, eps=%g)\n", d, eps);
        layernorm_forward(ln, x, out);
        double *want = malloc((size_t)d * sizeof(double));
        arr_vals(need(lj, "out", "ln"), want, d);
        report("forward", out->data, want, d);

        layernorm_backward(ln, dout, dx);
        arr_vals(need(lj, "dx", "ln"), want, d);
        report("backward dx", dx->data, want, d);
        arr_vals(need(lj, "grad_gamma", "ln"), want, d);
        report("grad_gamma", layernorm_grad_gamma(ln), want, d);
        arr_vals(need(lj, "grad_beta", "ln"), want, d);
        report("grad_beta", layernorm_grad_beta(ln), want, d);

        arr_vals(need(lj, "x_hat", "ln"), want, d);
        report("x_hat", ln->x_hat, want, d);
        double mv[1];
        mv[0] = json_num(need(lj, "mean", "ln"), 0.0);
        report("mean", &ln->mean, mv, 1);
        mv[0] = json_num(need(lj, "var", "ln"), 0.0);
        report("var（有偏）", &ln->var, mv, 1);
        mv[0] = json_num(need(lj, "inv_std", "ln"), 0.0);
        report("inv_std", &ln->inv_std, mv, 1);

        free(gamma); free(beta); free(want);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        layernorm_free(ln);
    }

    /* ===================== Dropout（灌入 mask） ===================== */
    {
        const JsonValue *dj = json_get(root, "dropout");
        int n = json_int(need(dj, "n", "dropout"), 0);
        double p = json_num(need(dj, "p", "dropout"), 0.5);

        Dropout *dp = dropout_create(p);
        dropout_reserve(dp, n);
        int xs[1] = { n };
        Tensor *x = tensor_new(1, xs), *out = tensor_new(1, xs);
        Tensor *dout = tensor_new(1, xs), *dx = tensor_new(1, xs);
        arr_vals(need(dj, "x", "dropout"), x->data, n);
        arr_vals(need(dj, "dout", "dropout"), dout->data, n);

        double *mask = malloc((size_t)n * sizeof(double));
        arr_vals(need(dj, "mask", "dropout"), mask, n);
        dropout_set_mask(dp, mask);

        printf("\nDropout (p=%g, n=%d，灌入 Python 生成的 mask)\n", p, n);
        dropout_forward(dp, x, out);
        double *want = malloc((size_t)n * sizeof(double));
        arr_vals(need(dj, "out", "dropout"), want, n);
        report("forward = x*mask", out->data, want, n);

        report("内部 mask", dropout_mask(dp), mask, n);

        dropout_backward(dp, dout, dx);
        arr_vals(need(dj, "dx", "dropout"), want, n);
        report("backward = dout*mask", dx->data, want, n);

        free(mask); free(want);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        dropout_free(dp);
    }

    json_free(root);
    printf("\n结论: %s（容差 %.1e）\n", g_fail ? "存在超差项" : "C 与 Python 数值一致", g_tol);
    return g_fail ? 1 : 0;
}
