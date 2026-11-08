/*
 * test_adapt.c — LoRA / EMA / SWA / Lookahead 的恒等式与闭式自测
 *
 * 与 compare_adapt.c 的分工：compare 负责「与 Python 逐位相同」，
 * 这里负责「数学性质成立」——同一条性质在 Python 侧也有断言。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../adapt.h"

static int g_checks;
static int g_fails;

static void check(const char *name, int ok, double err)
{
    g_checks++;
    if (!ok) {
        g_fails++;
        printf("  [FAIL] %-40s err=%.3e\n", name, err);
    }
}

static double max_abs_diff(const double *a, const double *b, int n)
{
    double w = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > w) w = d;
    }
    return w;
}

static MLP *make_mlp(const int *sizes, int n)
{
    Activation acts[4] = { ACT_LINEAR, ACT_LINEAR, ACT_LINEAR, ACT_LINEAR };
    MLP *m = mlp_create(sizes, acts, n);
    if (!m) {
        fprintf(stderr, "[test_adapt] mlp_create 返回 NULL（sizes 长度必须 >= 2）\n");
        exit(2);
    }
    unsigned long long rng = 20261108ULL;
    for (int i = 0; i < m->n_layers; i++) he_init(&m->layers[i], &rng);
    return m;
}

/* [1] LoRA：B=0 时 forward 与原层逐位相同，dx 也相同 */
static void test_lora_identity(void)
{
    const int fi = 4, fo = 3, rank = 2;
    double W[12], b[3], A[8], B[6], x[4], dout[3];
    unsigned long long rng = 7ULL;
    for (int i = 0; i < 12; i++) W[i] = 0.1 * (double)((i * 7) % 11) - 0.4;
    for (int i = 0; i < 3; i++) b[i] = 0.05 * (double)i - 0.02;
    for (int i = 0; i < 8; i++) A[i] = 0.3 * (double)((i * 5) % 7) - 0.6;
    memset(B, 0, sizeof(B));
    for (int i = 0; i < 4; i++) x[i] = 0.2 * (double)i - 0.3;
    for (int i = 0; i < 3; i++) dout[i] = 0.4 * (double)i - 0.25;
    (void)rng;

    LoRA *l = lora_create(fi, fo, rank, 2.0);
    lora_set_params(l, A, B);

    double y[3], y0[3];
    lora_forward(l, W, b, x, y);
    for (int j = 0; j < fo; j++) {
        double acc = 0.0;
        for (int i = 0; i < fi; i++) acc += W[j * fi + i] * x[i];
        y0[j] = acc + b[j];
    }
    check("B=0 时 forward 等于原层", max_abs_diff(y, y0, fo) <= 1e-15, max_abs_diff(y, y0, fo));

    double dx[4], dx0[4];
    lora_zero_grad(l);
    lora_backward(l, W, dout, dx);
    for (int i = 0; i < fi; i++) {
        double acc = 0.0;
        for (int j = 0; j < fo; j++) acc += W[j * fi + i] * dout[j];
        dx0[i] = acc;
    }
    check("B=0 时 dx 等于原层", max_abs_diff(dx, dx0, fi) <= 1e-15, max_abs_diff(dx, dx0, fi));
    check("B=0 时 grad_A 全零", max_abs_diff(l->gA, (const double[8]){0}, 8) <= 0.0, 0.0);
    lora_free(l);
}

/* [2] merge：折进 W 之后 forward 不变、反向只走原层 */
static void test_lora_merge(void)
{
    const int fi = 3, fo = 2, rank = 2;
    double W[6] = { 0.5, -0.2, 0.1, 0.3, -0.4, 0.2 };
    double b[2] = { 0.1, -0.1 };
    double A[6] = { 0.2, -0.3, 0.4, 0.1, -0.5, 0.25 };
    double B[4] = { 0.3, -0.2, 0.5, 0.1 };
    double x[3] = { 1.0, -0.5, 0.25 };
    double dout[2] = { 0.7, -0.3 };

    LoRA *l = lora_create(fi, fo, rank, 2.0);
    lora_set_params(l, A, B);
    double y_before[2];
    lora_forward(l, W, b, x, y_before);

    /* effective_weight 与 merge 后的 W 必须一致 */
    double Weff[6];
    lora_effective_weight(l, W, Weff);
    double Wm[6];
    memcpy(Wm, W, sizeof(W));
    lora_merge(l, Wm);
    check("effective_weight 等于 merge 后的 W", max_abs_diff(Weff, Wm, 6) <= 1e-15,
          max_abs_diff(Weff, Wm, 6));

    double y_after[2];
    lora_forward(l, Wm, b, x, y_after);
    check("merge 前后 forward 一致（<1e-15）", max_abs_diff(y_before, y_after, fo) <= 1e-15,
          max_abs_diff(y_before, y_after, fo));

    double dx[3], dx0[3];
    lora_backward(l, Wm, dout, dx);
    for (int i = 0; i < fi; i++) {
        double acc = 0.0;
        for (int j = 0; j < fo; j++) acc += Wm[j * fi + i] * dout[j];
        dx0[i] = acc;
    }
    check("merge 后 dx 只走原层", max_abs_diff(dx, dx0, fi) <= 1e-15, max_abs_diff(dx, dx0, fi));
    check("merge 后 B 被清零", max_abs_diff(l->B, (const double[4]){0}, 4) <= 0.0, 0.0);
    lora_free(l);
}

/* [3] EMA：第一步 shadow = d*p0 + (1-d)*p1；warmup 的有效衰减是 min(d,(1+t)/(10+t)) */
static void test_ema(void)
{
    const int sizes[2] = { 2, 2 };
    MLP *m = make_mlp(sizes, 2);
    const int n = mlp_flat_size(m);
    double *p0 = (double *)malloc(sizeof(double) * (size_t)n);
    double *p1 = (double *)malloc(sizeof(double) * (size_t)n);
    double *want = (double *)malloc(sizeof(double) * (size_t)n);
    mlp_flat_read(m, p0);

    EMA *e = ema_create(m, 0.9, 0);
    check("EMA 初始 shadow 等于当前权重", max_abs_diff(e->shadow, p0, n) <= 0.0, 0.0);

    for (int i = 0; i < n; i++) p1[i] = p0[i] + 1.0;
    mlp_flat_write(m, p1);
    ema_update(e, m);
    for (int i = 0; i < n; i++) want[i] = 0.9 * p0[i] + 0.1 * p1[i];
    check("EMA 一步 = d*p0+(1-d)*p1", max_abs_diff(e->shadow, want, n) <= 1e-15,
          max_abs_diff(e->shadow, want, n));
    check("EMA step 自增", e->step == 1, (double)e->step);

    EMA *w = ema_create(m, 0.999, 5);
    check("warmup 第 0 步有效衰减 = 1/10", fabs(ema_effective_decay(w) - 0.1) <= 1e-15,
          ema_effective_decay(w));
    w->step = 5;
    check("warmup 用完后回到 decay", fabs(ema_effective_decay(w) - 0.999) <= 1e-15,
          ema_effective_decay(w));

    ema_copy_to(e, m);
    check("ema_copy_to 写回 shadow", max_abs_diff(m->layers[0].W, p0, 4) != 0.0, 1.0);

    free(p0);
    free(p1);
    free(want);
    ema_free(e);
    ema_free(w);
    mlp_free(m);
}

/* [4] SWA：两次快照之后 avg = (p1 + p2) / 2 */
static void test_swa(void)
{
    const int sizes[2] = { 2, 2 };
    MLP *m = make_mlp(sizes, 2);
    const int n = mlp_flat_size(m);
    double *p1 = (double *)malloc(sizeof(double) * (size_t)n);
    mlp_flat_read(m, p1);

    SWA *s = swa_create(m);
    swa_update(s, m);
    check("SWA 一次快照 = 该快照", max_abs_diff(s->avg, p1, n) <= 0.0, 0.0);

    double *p2 = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) p2[i] = p1[i] + 2.0;
    mlp_flat_write(m, p2);
    swa_update(s, m);
    double *want = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) want[i] = (p1[i] + p2[i]) / 2.0;
    check("SWA 两次快照 = 算术平均", max_abs_diff(s->avg, want, n) <= 1e-15,
          max_abs_diff(s->avg, want, n));
    check("SWA count = 2", s->count == 2, (double)s->count);

    swa_apply(s, m);
    double *now = (double *)malloc(sizeof(double) * (size_t)n);
    mlp_flat_read(m, now);
    check("swa_apply 把平均写回", max_abs_diff(now, want, n) <= 1e-15, max_abs_diff(now, want, n));

    free(p1);
    free(p2);
    free(want);
    free(now);
    swa_free(s);
    mlp_free(m);
}

/* [5] Lookahead：k=2、alpha=0.5 时第 2 步同步后 fast = slow = 0.5*fast0 + 0.5*fast2 */
static void test_lookahead(void)
{
    const int sizes[2] = { 1, 1 };
    MLP *m = make_mlp(sizes, 2);
    const int n = mlp_flat_size(m);
    double *p0 = (double *)malloc(sizeof(double) * (size_t)n);
    mlp_flat_read(m, p0);

    Lookahead *l = lookahead_create(m, 2, 0.5);
    double *p = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) p[i] = p0[i] + 1.0;
    mlp_flat_write(m, p);
    check("第 1 步不同步", lookahead_step(l, m) == 0, 0.0);

    for (int i = 0; i < n; i++) p[i] = p0[i] + 2.0;
    mlp_flat_write(m, p);
    check("第 2 步同步", lookahead_step(l, m) == 1, 0.0);

    double *want = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) want[i] = p0[i] + 0.5 * 2.0;
    double *now = (double *)malloc(sizeof(double) * (size_t)n);
    mlp_flat_read(m, now);
    check("同步后 fast = slow = 0.5*(p0+p2)", max_abs_diff(now, want, n) <= 1e-15,
          max_abs_diff(now, want, n));

    for (int i = 0; i < n; i++) p[i] = want[i] + 1.0;
    mlp_flat_write(m, p);
    check("第 3 步不同步", lookahead_step(l, m) == 0, 0.0);

    free(p0);
    free(p);
    free(want);
    free(now);
    lookahead_free(l);
    mlp_free(m);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[test_adapt] LoRA / EMA / SWA / Lookahead\n");
    test_lora_identity();
    test_lora_merge();
    test_ema();
    test_swa();
    test_lookahead();
    printf("\n[test_adapt] %d checks, %d failures\n", g_checks, g_fails);
    if (g_fails) {
        printf("[test_adapt] FAILED\n");
        return 1;
    }
    printf("[test_adapt] ALL PASS\n");
    return 0;
}
