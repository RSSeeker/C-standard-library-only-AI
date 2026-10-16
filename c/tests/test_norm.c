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
    check("输出列均值 ≈ 0", worst_m, 1e-9);
    check("输出列方差 ≈ 1", worst_v, 1e-3);

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
    check("输出均值 ≈ 0", fabs(m), 1e-9);
    check("输出方差 ≈ 1", fabs(v - 1.0), 1e-3);
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

    printf("\n%s（失败项 = %d）\n", g_failures == 0 ? "全部通过" : "*** 存在失败项 ***", g_failures);
    return g_failures == 0 ? 0 : 1;
}
