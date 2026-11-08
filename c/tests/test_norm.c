/*
 * test_norm.c —— Dropout / BatchNorm1d / LayerNorm 的 C 移植自测
 *
 * 编译（在 D:\vscode\AI_C\c 下）：
 *   gcc -std=c17 -O2 -Wall -Wextra -o tests\test_norm.exe tests\test_norm.c layers_norm.c tensor.c -lm
 * 运行：tests\test_norm.exe   （全部通过返回 0，并打印每项最大误差）
 *
 * 校验手段：
 *   1. BatchNorm1d 训练路径：L = Σ_ij w_j * out_ij（于是 dL/dout_ij = w_j），
 *      对 dx / gamma / beta 做中心差分（eps=1e-6，阈值 1e-4）。扰动输入后 batch
 *      统计量会跟着变——这正是 forward_batch 应有的行为，必须一起被差分覆盖。
 *   2. LayerNorm：同样的中心差分（单样本，x 变化时 mean/var 随之变化）。
 *   3. Dropout：灌入确定 mask 验证 forward 缩放与 backward 只回传未置零位置，
 *      再验证 RNG 路径的取值只能是 {0, scale}，最后验证 training=0 透传。
 *   4. Module vtable：module_forward/backward/param_count/zero_grad/free 与直接调用逐位一致。
 */
#include "../layers_norm.h"

#include "../numeric.h"   /* 测试里的参考和也走 Neumaier，避免自己引入额外误差 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS_FD 1e-6
#define TOL    1e-4
#define TOL_EXACT 1e-15   /* "同表达式同顺序"应当逐位一致，留 1 ulp 余量 */

static int g_failures = 0;

static void check(const char *name, double err, double tol)
{
    const int ok = (err <= tol);
    printf("  %-46s max|err| = %.3e  %s\n", name, err, ok ? "OK" : "*** FAIL ***");
    if (!ok) g_failures++;
}

static void check_true(const char *name, int cond)
{
    printf("  %-46s %s\n", name, cond ? "OK" : "*** FAIL ***");
    if (!cond) g_failures++;
}

static double max_abs_diff(const double *a, const double *b, int n)
{
    double w = 0.0;
    for (int i = 0; i < n; i++) {
        const double e = fabs(a[i] - b[i]);
        if (e > w) w = e;
    }
    return w;
}

static int all_zero(const double *a, int n)
{
    for (int i = 0; i < n; i++) if (a[i] != 0.0) return 0;
    return 1;
}

/* 确定性伪随机数据（LCG），保证每次运行的输入完全一致 */
static unsigned long long g_rng = 0x123456789ABCDEFULL;

static double frand(void)
{
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_rng >> 11) * (1.0 / 4503599627370496.0) - 1.0; /* [-1, 1) */
}

/* 视图张量：借用栈上数组，owns_data = 0（tensor.h 里就是这么约定的） */
static Tensor view1(int n, double *data)
{
    Tensor t = { .ndim = 1, .shape = { n, 1, 1, 1 }, .size = n, .data = data, .owns_data = 0 };
    return t;
}

static Tensor view2(int r, int c, double *data)
{
    Tensor t = { .ndim = 2, .shape = { r, c, 1, 1 }, .size = r * c, .data = data, .owns_data = 0 };
    return t;
}

/* L = Σ_i Σ_j w_j * out_ij，与 Python 测试里 sum(wi*o for row in out for wi,o in zip(w,row))
 * 同序同算法（Neumaier），使差分本身不引入额外误差 */
/* 三维视图（C,H,W），用于 BatchNorm2d / GroupNorm / InstanceNorm */
static Tensor view3(int c, int h, int w, double *data)
{
    Tensor t = { .ndim = 3, .shape = { c, h, w, 1 }, .size = c * h * w,
                 .data = data, .owns_data = 0 };
    return t;
}

/* 扁平内积 sum(a[i]*b[i])，用 Neumaier 补偿求和（与 Python 的 sum() 同口径） */
static double dot_flat(const double *a, const double *b, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = a[i] * b[i];
        const double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

static double bn_loss(const Tensor *out, const double *w, int d)
{
    const int n = out->shape[0];
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        const double *row = out->data + (size_t)i * d;
        for (int j = 0; j < d; j++) {
            const double x = row[j] * w[j];
            const double t = s + x;
            if (fabs(s) >= fabs(x)) c += (s - t) + x;
            else                    c += (x - t) + s;
            s = t;
        }
    }
    return s + c;
}

/* ==================== 1. BatchNorm1d 训练路径：dx 数值梯度 ==================== */

static void test_bn1d_numeric_dx(void)
{
    enum { N = 5, D = 3 };
    double xv[N * D], outv[N * D], doutv[N * D], dxv[N * D];
    const double gamma[D] = { 0.7, -1.3, 0.4 };
    const double beta[D]  = { 0.1, -0.2, 0.35 };
    const double w[D]     = { 0.9, -0.6, 1.2 };

    printf("\n[1] BatchNorm1d 训练路径的 dx 数值梯度（n=%d, d=%d）\n", (int)N, (int)D);
    for (int i = 0; i < N * D; i++) xv[i] = frand();

    BatchNorm1d *bn = bn1d_create(D, 1e-5, 0.1);
    bn1d_set_params(bn, gamma, beta);

    Tensor X = view2(N, D, xv), OUT = view2(N, D, outv);
    Tensor DOUT = view2(N, D, doutv), DX = view2(N, D, dxv);

    bn1d_forward_batch(bn, &X, &OUT);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) doutv[i * D + j] = w[j];  /* dL/dout_ij = w_j */

    bn1d_zero_grad(bn);
    bn1d_backward(bn, &DOUT, &DX);

    double worst = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            const int k = i * D + j;
            const double old = xv[k];
            xv[k] = old + EPS_FD;
            bn1d_forward_batch(bn, &X, &OUT);   /* batch 统计量随扰动改变：正确行为 */
            const double lp = bn_loss(&OUT, w, D);
            xv[k] = old - EPS_FD;
            bn1d_forward_batch(bn, &X, &OUT);
            const double lm = bn_loss(&OUT, w, D);
            xv[k] = old;
            const double err = fabs((lp - lm) / (2 * EPS_FD) - dxv[k]);
            if (err > worst) worst = err;
        }
    }
    check("dx vs 中心差分（eps=1e-6, 阈值 1e-4）", worst, TOL);

    bn1d_free(bn);
}

/* ==================== 2. BatchNorm1d 训练路径：gamma / beta 数值梯度 + 累加语义 ==================== */

static void test_bn1d_numeric_gamma_beta(void)
{
    enum { N = 5, D = 3 };
    double xv[N * D], outv[N * D], doutv[N * D], dxv[N * D];
    const double gamma0[D] = { 0.7, -1.3, 0.4 };
    const double beta0[D]  = { 0.1, -0.2, 0.35 };
    const double w[D]      = { 0.9, -0.6, 1.2 };

    printf("\n[2] BatchNorm1d 训练路径的 gamma / beta 数值梯度\n");
    for (int i = 0; i < N * D; i++) xv[i] = frand();

    BatchNorm1d *bn = bn1d_create(D, 1e-5, 0.1);
    bn1d_set_params(bn, gamma0, beta0);

    Tensor X = view2(N, D, xv), OUT = view2(N, D, outv);
    Tensor DOUT = view2(N, D, doutv), DX = view2(N, D, dxv);

    bn1d_forward_batch(bn, &X, &OUT);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++) doutv[i * D + j] = w[j];

    bn1d_zero_grad(bn);
    bn1d_backward(bn, &DOUT, &DX);
    const double *gg = bn1d_grad_gamma(bn);
    const double *gb = bn1d_grad_beta(bn);

    double worst_g = 0.0, worst_b = 0.0;
    for (int j = 0; j < D; j++) {
        /* struct 是公开的：直接改字段，等价于 Python 里的 bn.gamma[j] = ... */
        const double old_g = bn->gamma[j];
        bn->gamma[j] = old_g + EPS_FD;
        bn1d_forward_batch(bn, &X, &OUT);
        const double lp = bn_loss(&OUT, w, D);
        bn->gamma[j] = old_g - EPS_FD;
        bn1d_forward_batch(bn, &X, &OUT);
        const double lm = bn_loss(&OUT, w, D);
        bn->gamma[j] = old_g;
        const double eg = fabs((lp - lm) / (2 * EPS_FD) - gg[j]);
        if (eg > worst_g) worst_g = eg;

        const double old_b = bn->beta[j];
        bn->beta[j] = old_b + EPS_FD;
        bn1d_forward_batch(bn, &X, &OUT);
        const double lp2 = bn_loss(&OUT, w, D);
        bn->beta[j] = old_b - EPS_FD;
        bn1d_forward_batch(bn, &X, &OUT);
        const double lm2 = bn_loss(&OUT, w, D);
        bn->beta[j] = old_b;
        const double eb = fabs((lp2 - lm2) / (2 * EPS_FD) - gb[j]);
        if (eb > worst_b) worst_b = eb;
    }
    check("d gamma vs 中心差分", worst_g, TOL);
    check("d beta  vs 中心差分", worst_b, TOL);

    /* 累加语义（tensor.h 的 Module 约定）：连续两次 backward，梯度翻倍 */
    bn1d_zero_grad(bn);
    bn1d_forward_batch(bn, &X, &OUT);
    bn1d_backward(bn, &DOUT, &DX);
    double first[D];
    memcpy(first, bn1d_grad_gamma(bn), sizeof first);
    bn1d_backward(bn, &DOUT, &DX);
    double twice[D];
    for (int j = 0; j < D; j++) twice[j] = 2.0 * first[j];
    check("grad_gamma 是累加而非覆盖", max_abs_diff(bn1d_grad_gamma(bn), twice, D), TOL_EXACT);

    bn1d_free(bn);
}

/* ==================== 3. BatchNorm1d：归一化统计量 + running 更新 ==================== */

static void test_bn1d_running_and_stats(void)
{
    enum { N = 64, D = 4 };
    double xv[N * D], outv[N * D];
    double col[N], dev[N];

    printf("\n[3] BatchNorm1d 归一化统计量与 running 更新（n=%d, d=%d）\n", (int)N, (int)D);
    for (int i = 0; i < N * D; i++) xv[i] = frand();

    BatchNorm1d *bn = bn1d_create(D, 1e-5, 0.1); /* gamma=1, beta=0, running=(0,1) */
    Tensor X = view2(N, D, xv), OUT = view2(N, D, outv);
    bn1d_forward_batch(bn, &X, &OUT);

    /* 输出每列均值≈0、方差≈1；输出列 - 输入列 = 归一化（gamma=1,beta=0） */
    double worst_m = 0.0, worst_v = 0.0;
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) col[i] = outv[i * D + j];
        const double m = sum_neumaier(col, N) / N;
        for (int i = 0; i < N; i++) dev[i] = col[i] - m;
        const double v = dot_neumaier(dev, dev, N) / N;
        if (fabs(m) > worst_m) worst_m = fabs(m);
        if (fabs(v - 1.0) > worst_v) worst_v = fabs(v - 1.0);
    }
    check("输出列均值 ~ 0", worst_m, 1e-9);
    check("输出列方差 ~ 1", worst_v, 1e-3);

    /* running 统计量：rm = 0.9*0 + 0.1*m；rv = 0.9*1 + 0.1*var*n/(n-1) */
    double worst_rm = 0.0, worst_rv = 0.0;
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) col[i] = xv[i * D + j];
        const double m = sum_neumaier(col, N) / N;
        for (int i = 0; i < N; i++) dev[i] = col[i] - m;
        const double v = dot_neumaier(dev, dev, N) / N;
        const double exp_rm = 0.1 * m;
        const double exp_rv = 0.9 * 1.0 + 0.1 * v * (double)N / (double)(N - 1);
        const double e_rm = fabs(bn->running_mean[j] - exp_rm);
        const double e_rv = fabs(bn->running_var[j] - exp_rv);
        if (e_rm > worst_rm) worst_rm = e_rm;
        if (e_rv > worst_rv) worst_rv = e_rv;
    }
    check("running_mean = 0.9*rm + 0.1*batch_mean", worst_rm, TOL_EXACT);
    check("running_var  = 0.9*rv + 0.1*var*n/(n-1)", worst_rv, TOL_EXACT);

    /* n=1：无偏估计无定义 → 走有偏 fallback；此时列方差恒为 0，x_hat 全 0、out=beta=0 */
    BatchNorm1d *bn1 = bn1d_create(D, 1e-5, 0.1);
    double x1[D] = { 0.4, -1.1, 2.0, 0.25 }, o1[D];
    Tensor X1 = view2(1, D, x1), O1 = view2(1, D, o1);
    bn1d_forward_batch(bn1, &X1, &O1);
    double worst_n1 = 0.0;
    for (int j = 0; j < D; j++) {
        const double e1 = fabs(o1[j] - 0.0);                                  /* out = beta = 0 */
        const double e2 = fabs(bn1->running_mean[j] - 0.1 * x1[j]);           /* 0.9*0 + 0.1*m */
        const double e3 = fabs(bn1->running_var[j] - 0.9 * 1.0);              /* 0.9*1 + 0.1*0 */
        if (e1 > worst_n1) worst_n1 = e1;
        if (e2 > worst_n1) worst_n1 = e2;
        if (e3 > worst_n1) worst_n1 = e3;
    }
    check("n=1 fallback（有偏 var + 输出=beta）", worst_n1, TOL_EXACT);

    bn1d_free(bn1);
    bn1d_free(bn);
}

/* ==================== 4. BatchNorm1d 推理路径 ==================== */

static void test_bn1d_eval_forward(void)
{
    enum { D = 4 };
    double xv[D] = { 0.5, -0.2, 0.8, 1.4 }, outv[D], ref[D];
    const double gamma[D] = { 0.8, 1.1, -0.5, 1.3 };
    const double beta[D]  = { -0.1, 0.2, 0.05, -0.3 };
    const double rm[D]    = { 0.3, -0.7, 0.15, 1.0 };
    const double rv[D]    = { 1.2, 0.6, 2.5, 0.9 };
    const double eps = 1e-5;
    char buf[64];

    printf("\n[4] BatchNorm1d 推理路径（用 running 统计量）\n");
    BatchNorm1d *bn = bn1d_create(D, eps, 0.1);
    bn1d_set_params(bn, gamma, beta);
    bn1d_set_running(bn, rm, rv);
    bn1d_set_training(bn, 0);

    Tensor X = view1(D, xv), OUT = view1(D, outv);
    bn1d_forward(bn, &X, &OUT);

    /* 与 Python 的 forward 完全同表达式：(x - rm) / sqrt(rv + eps) 再 gamma*xh + beta */
    for (int j = 0; j < D; j++) {
        const double xh = (xv[j] - rm[j]) / sqrt(rv[j] + eps);
        ref[j] = gamma[j] * xh + beta[j];
    }
    check("forward(x) == Python 表达式（逐位）", max_abs_diff(outv, ref, D), TOL_EXACT);

    /* 顺手验证 vtable 的 to_string */
    Module *m = bn1d_as_module(bn);
    m->vt->to_string(m, buf, sizeof buf);
    printf("  to_string: %s\n", buf);
    module_free(m); /* 同时释放 bn */

    LayerNorm *ln = layernorm_create(4, 1e-5);
    Module *mln = layernorm_as_module(ln);
    mln->vt->to_string(mln, buf, sizeof buf);
    printf("  to_string: %s\n", buf);
    module_free(mln);

    Dropout *dp = dropout_create(0.25);
    Module *mdp = dropout_as_module(dp);
    mdp->vt->to_string(mdp, buf, sizeof buf);
    printf("  to_string: %s\n", buf);
    module_free(mdp);
}

/* ==================== 5. LayerNorm：数值梯度 ==================== */

static void test_layernorm_numeric(void)
{
    enum { D = 5 };
    double xv[D], outv[D], doutv[D], dxv[D];
    double gamma[D], beta[D], w[D];
    const double eps = 1e-5;

    printf("\n[5] LayerNorm 数值梯度（d=%d）\n", (int)D);
    for (int j = 0; j < D; j++) {
        xv[j] = frand();
        gamma[j] = 0.5 + 0.4 * (double)j;       /* 非平凡的 gamma/beta，避免退化用例 */
        beta[j] = -0.3 + 0.15 * (double)j;
        w[j] = 1.0 - 0.25 * (double)j;
    }

    LayerNorm *ln = layernorm_create(D, eps);
    layernorm_set_params(ln, gamma, beta);

    Tensor X = view1(D, xv), OUT = view1(D, outv);
    Tensor DOUT = view1(D, doutv), DX = view1(D, dxv);

    layernorm_forward(ln, &X, &OUT);
    for (int j = 0; j < D; j++) doutv[j] = w[j];   /* L = Σ_j w_j*out_j => dL/dout_j = w_j */

    layernorm_zero_grad(ln);
    layernorm_backward(ln, &DOUT, &DX);
    const double *gg = layernorm_grad_gamma(ln);
    const double *gb = layernorm_grad_beta(ln);

    double worst_x = 0.0;
    for (int j = 0; j < D; j++) {
        const double old = xv[j];
        xv[j] = old + EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lp = dot_neumaier(w, outv, D);
        xv[j] = old - EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lm = dot_neumaier(w, outv, D);
        xv[j] = old;
        const double err = fabs((lp - lm) / (2 * EPS_FD) - dxv[j]);
        if (err > worst_x) worst_x = err;
    }
    check("dx     vs 中心差分（eps=1e-6, 阈值 1e-4）", worst_x, TOL);

    double worst_g = 0.0, worst_b = 0.0;
    for (int j = 0; j < D; j++) {
        const double old_g = ln->gamma[j];
        ln->gamma[j] = old_g + EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lp = dot_neumaier(w, outv, D);
        ln->gamma[j] = old_g - EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lm = dot_neumaier(w, outv, D);
        ln->gamma[j] = old_g;
        const double eg = fabs((lp - lm) / (2 * EPS_FD) - gg[j]);
        if (eg > worst_g) worst_g = eg;

        const double old_b = ln->beta[j];
        ln->beta[j] = old_b + EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lp2 = dot_neumaier(w, outv, D);
        ln->beta[j] = old_b - EPS_FD;
        layernorm_forward(ln, &X, &OUT);
        const double lm2 = dot_neumaier(w, outv, D);
        ln->beta[j] = old_b;
        const double eb = fabs((lp2 - lm2) / (2 * EPS_FD) - gb[j]);
        if (eb > worst_b) worst_b = eb;
    }
    check("d gamma vs 中心差分", worst_g, TOL);
    check("d beta  vs 中心差分", worst_b, TOL);

    /* LayerNorm 的 grad_gamma 是逐元素 dout[j]*x_hat[j]（不对 batch 求和）：
     * 这里 dout=w，所以 grad_gamma 应当等于 w_j * x_hat_j */
    double ref_g[D];
    for (int j = 0; j < D; j++) ref_g[j] = w[j] * ln->x_hat[j];
    check("grad_gamma == dout[j]*x_hat[j]（非 batch 求和）", max_abs_diff(gg, ref_g, D), TOL_EXACT);

    layernorm_free(ln);
}

/* ==================== 6. LayerNorm：输出统计量 ==================== */

static void test_layernorm_stats(void)
{
    enum { D = 5 };
    double xv[D] = { 1.0, 2.0, 3.0, 4.0, 5.0 }, outv[D];

    printf("\n[6] LayerNorm 输出统计量（gamma=1, beta=0）\n");
    LayerNorm *ln = layernorm_create(D, 1e-5);
    Tensor X = view1(D, xv), OUT = view1(D, outv);
    layernorm_forward(ln, &X, &OUT);

    const double m = sum_neumaier(outv, D) / D;
    const double v = dot_neumaier(outv, outv, D) / D - m * m; /* 描述性统计，够用 */
    check("输出均值 ~ 0", fabs(m), 1e-9);
    check("输出方差 ~ 1", fabs(v - 1.0), 1e-3);
    layernorm_free(ln);
}

/* ==================== 7. Dropout ==================== */

static void test_dropout(void)
{
    enum { N = 6, BIG = 1000 };
    double xv[N] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const double mask[N] = { 0.0, 2.0, 2.0, 0.0, 2.0, 0.0 }; /* p=0.5 -> scale=2.0 */
    double outv[N], doutv[N], dxv[N], ref[N];
    static double bx[BIG], bo[BIG];

    printf("\n[7] Dropout\n");
    Dropout *dp = dropout_create(0.5);
    Tensor X = view1(N, xv), OUT = view1(N, outv);
    Tensor DOUT = view1(N, doutv), DX = view1(N, dxv);

    /* 7.1 灌入确定 mask：forward 做 inverted scaling */
    dropout_set_mask(dp, mask);
    dropout_forward(dp, &X, &OUT);
    for (int i = 0; i < N; i++) ref[i] = xv[i] * mask[i];
    check("forward = x*mask（灌入 mask）", max_abs_diff(outv, ref, N), TOL_EXACT);
    check("内部 mask 与灌入值一致", max_abs_diff(dropout_mask(dp), mask, N), 0.0);
    check_true("mask 长度 = 输入 size", dropout_mask_len(dp) == N);

    /* 7.2 backward 只回传给未置零的位置 */
    for (int i = 0; i < N; i++) doutv[i] = 1.0 + (double)i;
    for (int i = 0; i < N; i++) ref[i] = doutv[i] * mask[i];
    dropout_backward(dp, &DOUT, &DX);
    check("backward = dout*mask", max_abs_diff(dxv, ref, N), TOL_EXACT);
    int dropped_ok = 1;
    for (int i = 0; i < N; i++) if (mask[i] == 0.0 && dxv[i] != 0.0) dropped_ok = 0;
    check_true("被丢弃位置的梯度严格为 0", dropped_ok);

    /* 7.3 RNG 路径：p=0.5 → mask 只能取 {0, 2.0}（inverted scaling） */
    for (int i = 0; i < BIG; i++) bx[i] = 1.0;
    dropout_set_seed(dp, 12345);
    Tensor BX = view1(BIG, bx), BO = view1(BIG, bo);
    dropout_forward(dp, &BX, &BO);
    int zeros = 0, bad = 0;
    for (int i = 0; i < BIG; i++) {
        if (bo[i] == 0.0) zeros++;
        else if (bo[i] != 2.0 * bx[i]) bad++;
    }
    printf("  Dropout(p=0.5) 在 %d 个元素上丢弃 %d 个（固定种子，结果确定）\n", (int)BIG, zeros);
    check_true("RNG mask 取值只能是 {0, 1/(1-p)}", bad == 0);
    check_true("丢弃比例在 [400,600]/1000 内", zeros >= 400 && zeros <= 600);

    /* 7.4 p=0.2 → scale=1.25，且 backward 与 forward 用同一张 mask */
    dropout_set_p(dp, 0.2);
    dropout_forward(dp, &BX, &BO);
    int bad2 = 0, zeros2 = 0;
    for (int i = 0; i < BIG; i++) {
        if (bo[i] == 0.0) zeros2++;
        else if (bo[i] != 1.25 * bx[i]) bad2++;
    }
    printf("  Dropout(p=0.2) 在 %d 个元素上丢弃 %d 个（期望约 200，scale=1.25）\n", (int)BIG, zeros2);
    check_true("scale = 1/(1-p) = 1.25", bad2 == 0);
    /* 丢弃比例 ≈ p：p=0.2 时应在 200/1000 附近（固定种子，结果是确定的） */
    check_true("丢弃比例在 [100,300]/1000 内", zeros2 >= 100 && zeros2 <= 300);

    /* 7.5 推理模式（training=0）透传，且 backward 直通 */
    dropout_set_training(dp, 0);
    dropout_forward(dp, &X, &OUT);
    check("training=0 时 forward 透传", max_abs_diff(outv, xv, N), 0.0);
    dropout_backward(dp, &DOUT, &DX);
    check("training=0 时 backward 直通", max_abs_diff(dxv, doutv, N), 0.0);
    check_true("training=0 时 mask 全 1", dropout_mask(dp) != NULL && dropout_mask(dp)[0] == 1.0);

    dropout_free(dp);
}

/* ==================== 8. Module vtable 适配层 ==================== */

static void test_module_vtables(void)
{
    enum { N = 4, D = 3 };
    double xv[N * D], oa[N * D], ob[N * D], doutv[N * D], dxa[N * D], dxb[N * D];
    double x1[D] = { 0.2, -0.9, 1.3 }, o1[D] = { 0 }, o2[D] = { 0 }, d1[D] = { 1.0, 2.0, 3.0 }, dx1[D];
    const double gamma[D] = { 0.6, -1.2, 0.9 };
    const double beta[D]  = { 0.05, 0.3, -0.15 };
    const double rm[D]    = { 0.1, -0.4, 0.7 };
    const double rv[D]    = { 0.8, 1.6, 1.1 };

    printf("\n[8] Module vtable 适配层\n");
    for (int i = 0; i < N * D; i++) { xv[i] = frand(); doutv[i] = frand(); }

    /* BatchNorm1d：training=1 时 vtable 走 (N,D) 的 forward_batch，结果与直接调用逐位一致 */
    BatchNorm1d *a = bn1d_create(D, 1e-5, 0.1);
    BatchNorm1d *b = bn1d_create(D, 1e-5, 0.1);
    bn1d_set_params(a, gamma, beta);
    bn1d_set_params(b, gamma, beta);
    bn1d_set_running(a, rm, rv);
    bn1d_set_running(b, rm, rv);
    bn1d_set_training(a, 1);
    bn1d_set_training(b, 1);

    Module *ma = bn1d_as_module(a);
    Tensor X = view2(N, D, xv), OA = view2(N, D, oa), OB = view2(N, D, ob);
    Tensor DOUT = view2(N, D, doutv), DXA = view2(N, D, dxa), DXB = view2(N, D, dxb);

    module_forward(ma, &X, &OA);
    bn1d_forward_batch(b, &X, &OB);
    check("module_forward(training=1) == forward_batch", max_abs_diff(oa, ob, N * D), 0.0);
    check_true("module_param_count(BatchNorm1d) == 2d", module_param_count(ma) == 2 * D);

    module_backward(ma, &DOUT, &DXA);
    bn1d_backward(b, &DOUT, &DXB);
    check("module_backward == bn1d_backward", max_abs_diff(dxa, dxb, N * D), 0.0);
    check("module_backward 参数梯度一致", max_abs_diff(bn1d_grad_gamma(a), bn1d_grad_gamma(b), D), 0.0);
    module_zero_grad(ma);
    check_true("module_zero_grad 清零 grad_gamma/beta",
               all_zero(bn1d_grad_gamma(a), D) && all_zero(bn1d_grad_beta(a), D));

    /* training=0 时 vtable 走 (D,) 的推理路径 */
    bn1d_set_training(a, 0);
    bn1d_set_training(b, 0);
    Tensor X1 = view1(D, x1), O1 = view1(D, o1), O2 = view1(D, o2);
    module_forward(ma, &X1, &O1);
    bn1d_forward(b, &X1, &O2);
    check("module_forward(training=0) == forward(D,)", max_abs_diff(o1, o2, D), 0.0);

    module_free(ma);   /* 连带释放 a */
    bn1d_free(b);

    /* LayerNorm / Dropout 的包装 */
    LayerNorm *ln = layernorm_create(D, 1e-5);
    Module *mln = layernorm_as_module(ln);
    check_true("module_param_count(LayerNorm) == 2d", module_param_count(mln) == 2 * D);
    module_forward(mln, &X1, &O1);
    layernorm_forward(ln, &X1, &O2);   /* 同一个对象再算一次，确定性检查 */
    check("LayerNorm module_forward 可重复", max_abs_diff(o1, o2, D), 0.0);
    Tensor D1 = view1(D, d1), DX1 = view1(D, dx1);
    module_forward(mln, &X1, &O1);
    module_backward(mln, &D1, &DX1);
    check_true("LayerNorm module_backward 产出有限值", isfinite(dx1[0]) && isfinite(dx1[2]));
    module_free(mln);

    Dropout *dp = dropout_create(0.5);
    Module *mdp = dropout_as_module(dp);
    check_true("module_param_count(Dropout) == 0", module_param_count(mdp) == 0);
    module_forward(mdp, &X1, &O1);     /* training=1，随机 mask */
    module_forward(mdp, &X1, &O1);     /* 再跑一次，确保不崩 */
    module_backward(mdp, &D1, &DX1);
    int mask_ok = 1;
    for (int j = 0; j < D; j++) {
        const double m = dropout_mask(dp)[j];
        if (!(m == 0.0 || m == 2.0)) mask_ok = 0;
        if (dx1[j] != d1[j] * m) mask_ok = 0;
    }
    check_true("Dropout module 路径的 mask/梯度自洽", mask_ok);
    module_free(mdp);
}

/* ==================== 8. BatchNorm2d：dx / gamma / beta 的数值梯度 ==================== */

static void test_bn2d_numeric(void)
{
    const int C = 2, H = 2, W = 3;
    const int m = H * W, n = C * m;
    double xd[12], dd[12], gamma[2], beta[2];
    for (int i = 0; i < n; i++) { xd[i] = frand(); dd[i] = frand(); }
    for (int c = 0; c < C; c++) { gamma[c] = 0.7 + 0.3 * c; beta[c] = -0.2 + 0.1 * c; }

    BatchNorm2d *bn = bn2d_create(C, 1e-5, 0.2);
    bn2d_set_params(bn, gamma, beta);

    double xs[12], ds[12];
    memcpy(xs, xd, sizeof(xs));
    memcpy(ds, dd, sizeof(ds));
    Tensor x = view3(C, H, W, xs), out = view3(C, H, W, xd);
    Tensor dout = view3(C, H, W, ds), dx = view3(C, H, W, xs);
    double *ox = malloc((size_t)n * sizeof(double));
    double *dxx = malloc((size_t)n * sizeof(double));
    Tensor o = view3(C, H, W, ox), dxt = view3(C, H, W, dxx);

    bn2d_forward(bn, &x, &o);
    bn2d_backward(bn, &dout, &dxt);

    /* dx 的中心差分 */
    const double eps = 1e-6;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double orig = xs[i];
        xs[i] = orig + eps;
        bn2d_forward(bn, &x, &o);
        const double lp = dot_flat(ds, ox, n);
        xs[i] = orig - eps;
        bn2d_forward(bn, &x, &o);
        const double lm = dot_flat(ds, ox, n);
        xs[i] = orig;
        const double num = (lp - lm) / (2.0 * eps);
        const double d = fabs(num - dxx[i]);
        if (d > worst) worst = d;
    }
    check("BatchNorm2d 训练 dx 数值梯度", worst, 1e-5);

    /* gamma / beta 的中心差分（重新 forward 一次让缓存回到原值） */
    bn2d_forward(bn, &x, &o);
    double worst_g = 0.0, worst_b = 0.0;
    for (int c = 0; c < C; c++) {
        const double og = gamma[c];
        gamma[c] = og + eps;
        bn2d_set_params(bn, gamma, beta);
        bn2d_forward(bn, &x, &o);
        const double lp = dot_flat(ds, ox, n);
        gamma[c] = og - eps;
        bn2d_set_params(bn, gamma, beta);
        bn2d_forward(bn, &x, &o);
        const double lm = dot_flat(ds, ox, n);
        gamma[c] = og;
        bn2d_set_params(bn, gamma, beta);
        const double num = (lp - lm) / (2.0 * eps);
        double d = fabs(num - bn2d_grad_gamma(bn)[c]);
        if (d > worst_g) worst_g = d;

        const double ob = beta[c];
        beta[c] = ob + eps;
        bn2d_set_params(bn, gamma, beta);
        bn2d_forward(bn, &x, &o);
        const double lp2 = dot_flat(ds, ox, n);
        beta[c] = ob - eps;
        bn2d_set_params(bn, gamma, beta);
        bn2d_forward(bn, &x, &o);
        const double lm2 = dot_flat(ds, ox, n);
        beta[c] = ob;
        bn2d_set_params(bn, gamma, beta);
        const double num2 = (lp2 - lm2) / (2.0 * eps);
        double d2 = fabs(num2 - bn2d_grad_beta(bn)[c]);
        if (d2 > worst_b) worst_b = d2;
    }
    check("BatchNorm2d 训练 grad_gamma 数值梯度", worst_g, 1e-5);
    check("BatchNorm2d 训练 grad_beta 数值梯度", worst_b, 1e-5);

    free(ox); free(dxx); (void)out; (void)dx; (void)m;
    bn2d_free(bn);
}

/* ==================== 9. BatchNorm2d：推理路径、running 统计量、单元素通道 ==================== */

static void test_bn2d_eval_and_running(void)
{
    const int C = 2, H = 3, W = 2;
    const int m = H * W, n = C * m;
    double xs[12], ds[12], gamma[2] = { 1.3, 0.6 }, beta[2] = { 0.1, -0.4 };
    double rm0[2] = { 0.5, -0.25 }, rv0[2] = { 2.0, 0.8 };
    for (int i = 0; i < n; i++) { xs[i] = frand(); ds[i] = frand(); }

    /* 推理路径：dx = inv * dout * gamma，且不动 running 统计量 */
    BatchNorm2d *bn = bn2d_create(C, 1e-5, 0.1);
    bn2d_set_params(bn, gamma, beta);
    bn2d_set_running(bn, rm0, rv0);
    bn2d_set_training(bn, 0);

    double ox[12], dxx[12];
    Tensor x = view3(C, H, W, xs), dout = view3(C, H, W, ds);
    Tensor o = view3(C, H, W, ox), dxt = view3(C, H, W, dxx);
    bn2d_forward(bn, &x, &o);
    bn2d_backward(bn, &dout, &dxt);

    double worst = 0.0;
    for (int c = 0; c < C; c++) {
        const double inv = 1.0 / sqrt(rv0[c] + 1e-5);
        for (int i = 0; i < m; i++) {
            const double want = inv * ds[c * m + i] * gamma[c];
            const double d = fabs(dxx[c * m + i] - want);
            if (d > worst) worst = d;
        }
    }
    check("BatchNorm2d 推理 dx = inv*d*gamma", worst, 1e-15);

    int untouched = 1;
    for (int c = 0; c < C; c++)
        if (bn2d_running_mean(bn)[c] != rm0[c] || bn2d_running_var(bn)[c] != rv0[c])
            untouched = 0;
    check_true("BatchNorm2d 推理不更新 running 统计量", untouched);
    bn2d_free(bn);

    /* momentum = 1：running 直接被写成当前统计量（mean、无偏 var） */
    BatchNorm2d *b1 = bn2d_create(C, 1e-5, 1.0);
    bn2d_set_params(b1, gamma, beta);
    bn2d_set_running(b1, rm0, rv0);
    bn2d_forward(b1, &x, &o);
    double worst_rm = 0.0, worst_rv = 0.0;
    for (int c = 0; c < C; c++) {
        double mean = 0.0;
        for (int i = 0; i < m; i++) mean += xs[c * m + i];
        mean /= (double)m;
        double var = 0.0;
        for (int i = 0; i < m; i++) { const double d = xs[c * m + i] - mean; var += d * d; }
        var /= (double)m;
        const double unb = var * (double)m / (double)(m - 1);
        double d1 = fabs(bn2d_running_mean(b1)[c] - mean);
        double d2 = fabs(bn2d_running_var(b1)[c] - unb);
        if (d1 > worst_rm) worst_rm = d1;
        if (d2 > worst_rv) worst_rv = d2;
    }
    check("momentum=1 running_mean = batch mean", worst_rm, 1e-15);
    check("momentum=1 running_var = 无偏 batch var", worst_rv, 1e-15);
    bn2d_free(b1);

    /* momentum = 0：running 完全不变 */
    BatchNorm2d *b0 = bn2d_create(C, 1e-5, 0.0);
    bn2d_set_params(b0, gamma, beta);
    bn2d_set_running(b0, rm0, rv0);
    bn2d_forward(b0, &x, &o);
    int unchanged = 1;
    for (int c = 0; c < C; c++)
        if (bn2d_running_mean(b0)[c] != rm0[c] || bn2d_running_var(b0)[c] != rv0[c])
            unchanged = 0;
    check_true("momentum=0 running 统计量不变", unchanged);
    bn2d_free(b0);

    /* H = W = 1：m_size = 1，无偏方差退化为有偏（不能除 0 出 inf/nan） */
    double x1[2] = { 0.75, -1.25 }, o1[2];
    Tensor xs1 = view3(C, 1, 1, x1), os1 = view3(C, 1, 1, o1);
    BatchNorm2d *bs = bn2d_create(C, 1e-5, 0.5);
    bn2d_set_params(bs, gamma, beta);
    bn2d_set_running(bs, rm0, rv0);
    bn2d_forward(bs, &xs1, &os1);
    int finite = 1;
    for (int c = 0; c < C; c++)
        if (!isfinite(bn2d_running_var(bs)[c])) finite = 0;
    check_true("m_size=1 时 running_var 仍是有限值", finite);
    bn2d_free(bs);
}

/* ==================== 10. GroupNorm / InstanceNorm：数值梯度 ==================== */

static void test_gn_numeric(void)
{
    const int G = 2, C = 4, H = 2, W = 2;
    const int n = C * H * W;
    double xs[16], ds[16], gamma[4], beta[4];
    for (int i = 0; i < n; i++) { xs[i] = frand(); ds[i] = frand(); }
    for (int c = 0; c < C; c++) { gamma[c] = 0.5 + 0.2 * c; beta[c] = 0.15 - 0.05 * c; }

    GroupNorm *gn = gn_create(G, C, 1e-5);
    gn_set_params(gn, gamma, beta);

    double ox[16], dxx[16];
    Tensor x = view3(C, H, W, xs), dout = view3(C, H, W, ds);
    Tensor o = view3(C, H, W, ox), dxt = view3(C, H, W, dxx);
    gn_forward(gn, &x, &o);
    gn_backward(gn, &dout, &dxt);

    const double eps = 1e-6;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double orig = xs[i];
        xs[i] = orig + eps;
        gn_forward(gn, &x, &o);
        const double lp = dot_flat(ds, ox, n);
        xs[i] = orig - eps;
        gn_forward(gn, &x, &o);
        const double lm = dot_flat(ds, ox, n);
        xs[i] = orig;
        const double num = (lp - lm) / (2.0 * eps);
        const double d = fabs(num - dxx[i]);
        if (d > worst) worst = d;
    }
    check("GroupNorm dx 数值梯度", worst, 1e-5);

    gn_forward(gn, &x, &o);
    double worst_g = 0.0, worst_b = 0.0;
    for (int c = 0; c < C; c++) {
        const double og = gamma[c];
        gamma[c] = og + eps;
        gn_set_params(gn, gamma, beta);
        gn_forward(gn, &x, &o);
        const double lp = dot_flat(ds, ox, n);
        gamma[c] = og - eps;
        gn_set_params(gn, gamma, beta);
        gn_forward(gn, &x, &o);
        const double lm = dot_flat(ds, ox, n);
        gamma[c] = og;
        gn_set_params(gn, gamma, beta);
        double d = fabs((lp - lm) / (2.0 * eps) - gn_grad_gamma(gn)[c]);
        if (d > worst_g) worst_g = d;

        const double ob = beta[c];
        beta[c] = ob + eps;
        gn_set_params(gn, gamma, beta);
        gn_forward(gn, &x, &o);
        const double lp2 = dot_flat(ds, ox, n);
        beta[c] = ob - eps;
        gn_set_params(gn, gamma, beta);
        gn_forward(gn, &x, &o);
        const double lm2 = dot_flat(ds, ox, n);
        beta[c] = ob;
        gn_set_params(gn, gamma, beta);
        double d2 = fabs((lp2 - lm2) / (2.0 * eps) - gn_grad_beta(gn)[c]);
        if (d2 > worst_b) worst_b = d2;
    }
    check("GroupNorm grad_gamma 数值梯度", worst_g, 1e-5);
    check("GroupNorm grad_beta 数值梯度", worst_b, 1e-5);
    gn_free(gn);

    /* InstanceNorm：每组一个通道，等价于逐通道标准化 */
    GroupNorm *in = inorm_create(C, 1e-5);
    gn_set_params(in, gamma, beta);
    gn_forward(in, &x, &o);
    double worst_i = 0.0;
    for (int c = 0; c < C; c++) {
        double mean = 0.0;
        for (int i = 0; i < H * W; i++) mean += xs[c * H * W + i];
        mean /= (double)(H * W);
        double var = 0.0;
        for (int i = 0; i < H * W; i++) {
            const double d = xs[c * H * W + i] - mean;
            var += d * d;
        }
        var /= (double)(H * W);
        const double inv = 1.0 / sqrt(var + 1e-5);
        for (int i = 0; i < H * W; i++) {
            const double want = gamma[c] * (xs[c * H * W + i] - mean) * inv + beta[c];
            const double d = fabs(ox[c * H * W + i] - want);
            if (d > worst_i) worst_i = d;
        }
    }
    check("InstanceNorm 逐通道标准化闭式", worst_i, 1e-15);
    gn_free(in);
}

/* ==================== 11. DropPath：factor 规则与反向 ==================== */

static void test_droppath(void)
{
    const int n = 5;
    double xd[5], dd[5], ox[5], dxx[5];
    for (int i = 0; i < n; i++) { xd[i] = frand(); dd[i] = frand(); }
    Tensor x = view1(n, xd), o = view1(n, ox), dout = view1(n, dd), dxt = view1(n, dxx);

    /* 丢弃：输出全 0（含 -0.0），反向全 0 */
    DropPath *dp = droppath_create(0.25, 1);
    droppath_set_keep(dp, 0);
    droppath_forward(dp, &x, &o);
    check("DropPath 丢弃 factor = 0", droppath_factor(dp), 1e-15);
    int allz = 1;
    for (int i = 0; i < n; i++) if (ox[i] != 0.0) allz = 0;
    check_true("DropPath 丢弃时输出全 0", allz);
    droppath_backward(dp, &dout, &dxt);
    check_true("DropPath 丢弃时反向全 0", all_zero(dxx, n));

    /* 保留：factor = 1/(1-p)，反向 = factor*dout */
    droppath_set_keep(dp, 1);
    droppath_forward(dp, &x, &o);
    check("DropPath 保留 factor = 1/(1-p)", droppath_factor(dp) - 1.0 / 0.75, 1e-15);
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = fabs(ox[i] - xd[i] / 0.75);
        if (d > worst) worst = d;
    }
    check("DropPath 保留 out = x/(1-p)", worst, 1e-15);
    droppath_backward(dp, &dout, &dxt);
    worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = fabs(dxx[i] - dd[i] / 0.75);
        if (d > worst) worst = d;
    }
    check("DropPath 保留 backward = dout/(1-p)", worst, 1e-15);

    /* scale_by_keep = 0：保留时不放大 */
    DropPath *ns = droppath_create(0.25, 0);
    droppath_set_keep(ns, 1);
    droppath_forward(ns, &x, &o);
    check("scale_by_keep=0 时 factor = 1", droppath_factor(ns) - 1.0, 1e-15);
    worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = fabs(ox[i] - xd[i]);
        if (d > worst) worst = d;
    }
    check("scale_by_keep=0 时 out = x", worst, 0.0);
    droppath_free(ns);

    /* eval 与 p = 0：恒等 */
    droppath_set_training(dp, 0);
    droppath_forward(dp, &x, &o);
    check("DropPath eval factor = 1", droppath_factor(dp) - 1.0, 1e-15);
    worst = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = fabs(ox[i] - xd[i]);
        if (d > worst) worst = d;
    }
    check("DropPath eval 时 out = x", worst, 0.0);
    droppath_free(dp);

    DropPath *p0 = droppath_create(0.0, 1);
    droppath_forward(p0, &x, &o);
    check("p=0 时 factor = 1", droppath_factor(p0) - 1.0, 1e-15);
    droppath_free(p0);
}

/* ==================== 12. 新模块的构造期校验（不崩即可） ==================== */

static void test_new_norm_vtables(void)
{
    BatchNorm2d *bn = bn2d_create(3, 1e-5, 0.1);
    Module *m = bn2d_as_module(bn);
    char buf[128];
    m->vt->to_string(m, buf, sizeof(buf));
    check_true("BatchNorm2d to_string 含模块名", strstr(buf, "BatchNorm2d") != NULL);
    check_true("BatchNorm2d param_count = 2C", module_param_count(m) == 6);
    module_free(m);

    GroupNorm *gn = gn_create(2, 4, 1e-5);
    Module *mg = gn_as_module(gn);
    mg->vt->to_string(mg, buf, sizeof(buf));
    check_true("GroupNorm to_string 含 groups", strstr(buf, "groups=2") != NULL ||
                                              strstr(buf, "GroupNorm(2, 4") != NULL);
    check_true("GroupNorm param_count = 2C", module_param_count(mg) == 8);
    module_free(mg);

    GroupNorm *in = inorm_create(3, 1e-5);
    Module *mi = inorm_as_module(in);
    mi->vt->to_string(mi, buf, sizeof(buf));
    check_true("InstanceNorm to_string 含模块名", strstr(buf, "InstanceNorm") != NULL);
    module_free(mi);

    DropPath *dp = droppath_create(0.1, 1);
    Module *md = droppath_as_module(dp);
    md->vt->to_string(md, buf, sizeof(buf));
    check_true("DropPath to_string 含 p", strstr(buf, "p=0.1") != NULL);
    check_true("DropPath param_count = 0", module_param_count(md) == 0);
    module_free(md);
}

int main(void)
{
    printf("== Dropout / BatchNorm1d / LayerNorm C17 自测 ==\n");

    test_bn1d_numeric_dx();
    test_bn1d_numeric_gamma_beta();
    test_bn1d_running_and_stats();
    test_bn1d_eval_forward();
    test_layernorm_numeric();
    test_layernorm_stats();
    test_dropout();
    test_module_vtables();
    test_bn2d_numeric();
    test_bn2d_eval_and_running();
    test_gn_numeric();
    test_droppath();
    test_new_norm_vtables();

    printf("\n%s（失败项 = %d）\n", g_failures == 0 ? "全部通过" : "*** 存在失败项 ***", g_failures);
    return g_failures == 0 ? 0 : 1;
}
