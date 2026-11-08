/*
 * test_augment.c —— 数据增广内核的自测（不依赖参考 JSON）
 *
 * 覆盖：亮度夹取、对比度恒等/单调、平移的零填充与可逆性、rotate90 的 4 次恒等
 * 与单次索引、cutout 的洞位置与边界裁剪、cutmix 的框复制与有效 lam 公式、
 * mixup 的两个端点与软标签、randaugment 的空序列/恒等序列/尺寸变化。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../augment.h"

static int g_fail;

static void check(const char *what, double err, double tol)
{
    if (err <= tol) {
        printf("  %-46s max|err| = %.3e  OK\n", what, err);
    } else {
        printf("  %-46s max|err| = %.3e  *** FAIL ***\n", what, err);
        g_fail++;
    }
}

static void check_true(const char *what, int cond)
{
    if (cond) {
        printf("  %-46s  OK\n", what);
    } else {
        printf("  %-46s  *** FAIL ***\n", what);
        g_fail++;
    }
}

static double *mk(int n, double v)
{
    double *p = (double *)malloc(sizeof(double) * (size_t)n);
    if (p == NULL) {
        fprintf(stderr, "test_augment: 内存不足\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) p[i] = v;
    return p;
}

static void fill_seq(double *p, int n)
{
    for (int i = 0; i < n; i++) p[i] = 0.01 * (double)(i % 97);
}

static void test_brightness(void)
{
    int n = 8;
    double *a = mk(n, 0.0);
    double *b = mk(n, 0.0);
    fill_seq(a, n);
    memcpy(b, a, sizeof(double) * (size_t)n);
    aug_brightness(b, n, 3.0);
    {
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            double v = 3.0 * a[i];
            if (v > 1.0) v = 1.0;
            if (v < 0.0) v = 0.0;
            worst = fmax(worst, fabs(b[i] - v));
        }
        check("brightness factor=3 与夹取公式一致", worst, 0.0);
    }
    aug_brightness(b, n, 0.0);
    {
        double worst = 0.0;
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(b[i]));
        check("brightness factor=0 全 0", worst, 0.0);
    }
    free(a);
    free(b);
}

static void test_contrast(void)
{
    int C = 2, H = 3, W = 4;
    int n = C * H * W;
    double *a = mk(n, 0.0);
    double *b = mk(n, 0.0);
    fill_seq(a, n);
    memcpy(b, a, sizeof(double) * (size_t)n);
    aug_contrast(b, C * H, W, 1.0);
    {
        double worst = 0.0;
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(b[i] - a[i]));
        check("contrast factor=1 近似恒等（仅舍入）", worst, 1e-15);
    }
    /* factor=0 把所有像素拉向均值：方差显著变小 */
    memcpy(b, a, sizeof(double) * (size_t)n);
    aug_contrast(b, C * H, W, 0.0);
    {
        double m1 = 0.0, m2 = 0.0, v1 = 0.0, v2 = 0.0;
        for (int i = 0; i < n; i++) {
            m1 += a[i] / n;
            m2 += b[i] / n;
        }
        for (int i = 0; i < n; i++) {
            v1 += (a[i] - m1) * (a[i] - m1) / n;
            v2 += (b[i] - m2) * (b[i] - m2) / n;
        }
        check("contrast factor=0 方差压到 0", fabs(v2), 1e-30);
        check("contrast 保持均值", fabs(m1 - m2), 1e-15);
        check_true("contrast 原方差非零", v1 > 1e-6);
    }
    free(a);
    free(b);
}

static void test_shift_rotate(void)
{
    int C = 1, H = 3, W = 4;
    int n = C * H * W;
    double *a = mk(n, 0.0);
    double *b = mk(n, 0.0);
    double *c = mk(n, 0.0);
    for (int i = 0; i < n; i++) a[i] = (double)(i + 1);
    aug_shift(a, C, H, W, 1, 0, b);
    check("shift(1,0) 首行补 0", fabs(b[0]), 0.0);
    check("shift(1,0) 第二行 = 原首行", fabs(b[W] - a[0]), 0.0);
    aug_shift(b, C, H, W, -1, 0, c);
    {
        double worst = 0.0;
        for (int y = 0; y < H - 1; y++)
            for (int x = 0; x < W; x++)
                worst = fmax(worst, fabs(c[y * W + x] - a[y * W + x]));
        check("shift 上下往返", worst, 0.0);
    }
    /* rotate90：new[y][x] = old[H-1-x][y]，尺寸变成 W×H */
    {
        int oh = 0, ow = 0;
        aug_rotate90(a, 1, H, W, 1, b, &oh, &ow);
        check_true("rotate90 尺寸互换", oh == W && ow == H);
        check("rotate90 索引 new[0][0] = old[2][0]", fabs(b[0] - a[2 * W + 0]), 0.0);
        check("rotate90 索引 new[0][1] = old[1][0]", fabs(b[1] - a[1 * W + 0]), 0.0);
    }
    /* 4 次恒等（中间两步尺寸是 W×H，必须把尺寸跟着传） */
    {
        double *p = mk(n, 0.0), *q = mk(n, 0.0);
        int ch = H, cw = W;
        memcpy(p, a, sizeof(double) * (size_t)n);
        for (int t = 0; t < 4; t++) {
            int oh = 0, ow = 0;
            aug_rotate90(p, 1, ch, cw, 1, q, &oh, &ow);
            memcpy(p, q, sizeof(double) * (size_t)n);
            ch = oh;
            cw = ow;
        }
        check_true("rotate90 转 4 次后尺寸回到原值", ch == H && cw == W);
        {
            double worst = 0.0;
            for (int i = 0; i < n; i++) worst = fmax(worst, fabs(p[i] - a[i]));
            check("rotate90 转 4 次恒等", worst, 0.0);
        }
        free(p);
        free(q);
    }
    free(a);
    free(b);
    free(c);
}

static void test_cutout(void)
{
    int C = 2, H = 5, W = 6;
    int n = C * H * W;
    double *a = mk(n, 0.7);
    int xs[2] = {1, 4};
    int ys[2] = {2, 0};
    aug_cutout(a, C, H, W, 2, 0.4, -1.5, xs, ys);
    /* side = round(min(5,6)*0.4) = round(2.0) = 2 */
    check("cutout 洞内被填成 fill", fabs(a[(0 * H + 2) * W + 1] + 1.5), 0.0);
    check("cutout 洞外不变", fabs(a[(0 * H + 0) * W + 0] - 0.7), 0.0);
    check("cutout 第二个通道同样生效", fabs(a[((size_t)1 * H + 3) * W + 2] + 1.5), 0.0);
    /* 贴边：x0=4, side=2 → 只填到 W=6，不越界 */
    check("cutout 贴右边界不越界", fabs(a[(0 * H + 0) * W + 5] + 1.5), 0.0);
    check("cutout 洞范围外未被写", fabs(a[(0 * H + 2) * W + 5] - 0.7), 0.0);
    free(a);
}

static void test_cutmix(void)
{
    int C = 1, H = 4, W = 4;
    int n = C * H * W;
    double *a = mk(n, 1.0);
    double *b = mk(n, 2.0);
    double *o = mk(n, 0.0);
    double lam;
    /* lam=0.75 → cut = rint(4*sqrt(0.25)) = 2，框以 (2,1) 为中心 → x∈[1,3), y∈[0,2) */
    lam = aug_cutmix(a, b, C, H, W, 0.75, 2, 1, o);
    check("cutmix 有效 lam = 1 - 面积/16", fabs(lam - (1.0 - 4.0 / 16.0)), 0.0);
    check("cutmix 框内取 b", fabs(o[0 * W + 1] - 2.0), 0.0);
    check("cutmix 框外保 a", fabs(o[3 * W + 3] - 1.0), 0.0);
    /* lam=1 → 空框，输出等于 a、有效 lam=1 */
    {
        double l2 = aug_cutmix(a, b, C, H, W, 1.0, 0, 0, o);
        double worst = 0.0;
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(o[i] - a[i]));
        check("cutmix lam=1 输出等于 xa", worst, 0.0);
        check("cutmix lam=1 有效 lam 仍为 1", fabs(l2 - 1.0), 0.0);
    }
    free(a);
    free(b);
    free(o);
}

static void test_mixup(void)
{
    double x1[4] = {0.1, 0.2, -0.3, 0.4};
    double x2[4] = {0.9, -0.5, 0.25, 0.0};
    double out[4];
    double soft[3];
    aug_mixup(x1, x2, 4, 1.0, out);
    {
        double worst = 0.0;
        for (int i = 0; i < 4; i++) worst = fmax(worst, fabs(out[i] - x1[i]));
        check("mixup lam=1 等于 x1", worst, 0.0);
    }
    aug_mixup(x1, x2, 4, 0.0, out);
    {
        double worst = 0.0;
        for (int i = 0; i < 4; i++) worst = fmax(worst, fabs(out[i] - x2[i]));
        check("mixup lam=0 等于 x2", worst, 0.0);
    }
    aug_mixup(x1, x2, 4, 0.25, out);
    check("mixup lam=0.25 逐元素线性", fabs(out[2] - (0.25 * x1[2] + 0.75 * x2[2])), 0.0);
    aug_mixup_soft(1, 2, 3, 0.3, soft);
    check("mixup 软标签和为 1", fabs(soft[0] + soft[1] + soft[2] - 1.0), 0.0);
    check("mixup 软标签 y1 分到 lam", fabs(soft[1] - 0.3), 0.0);
    aug_mixup_soft(1, 1, 3, 0.3, soft);
    check("mixup y1==y2 时该位为 1", fabs(soft[1] - 1.0), 0.0);
    check_true("mixup 其余位为 0", soft[0] == 0.0 && soft[2] == 0.0);
}

static void test_randaugment(void)
{
    int C = 1, H = 3, W = 4;
    int n = C * H * W;
    double *a = mk(n, 0.0);
    double *o = mk(n, 0.0);
    int oh = 0, ow = 0;
    for (int i = 0; i < n; i++) a[i] = 0.05 * (double)i;

    /* n=0：原样复制 */
    aug_randaugment(a, C, H, W, 0, NULL, NULL, o, &oh, &ow);
    {
        double worst = 0.0;
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(o[i] - a[i]));
        check("randaugment n=0 输出 = 输入", worst, 0.0);
        check_true("randaugment n=0 尺寸不变", oh == H && ow == W);
    }
    /* 单步 identity */
    {
        int ops[1] = {AUG_OP_IDENTITY};
        double ps[2] = {0.0, 0.0};
        double worst = 0.0;
        aug_randaugment(a, C, H, W, 1, ops, ps, o, &oh, &ow);
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(o[i] - a[i]));
        check("randaugment 单步 identity 不变", worst, 0.0);
    }
    /* 单步 brightness 与直接调用一致 */
    {
        int ops[1] = {AUG_OP_BRIGHTNESS};
        double ps[2] = {0.5, 0.0};
        double *ref = mk(n, 0.0);
        double worst = 0.0;
        memcpy(ref, a, sizeof(double) * (size_t)n);
        aug_brightness(ref, n, 0.5);
        aug_randaugment(a, C, H, W, 1, ops, ps, o, &oh, &ow);
        for (int i = 0; i < n; i++) worst = fmax(worst, fabs(o[i] - ref[i]));
        check("randaugment brightness 与内核一致", worst, 0.0);
        free(ref);
    }
    /* translate + rotate90：尺寸先不变、再互换 */
    {
        int ops[2] = {AUG_OP_TRANSLATE, AUG_OP_ROTATE90};
        double ps[4] = {1.0, 0.0, 1.0, 0.0};
        aug_randaugment(a, C, H, W, 2, ops, ps, o, &oh, &ow);
        check_true("randaugment translate+rotate90 后尺寸互换", oh == W && ow == H);
    }
    check_true("aug_op_name 覆盖四个操作与兜底",
               strcmp(aug_op_name(AUG_OP_BRIGHTNESS), "brightness") == 0 &&
               strcmp(aug_op_name(AUG_OP_CONTRAST), "contrast") == 0 &&
               strcmp(aug_op_name(AUG_OP_TRANSLATE), "translate") == 0 &&
               strcmp(aug_op_name(AUG_OP_ROTATE90), "rotate90") == 0 &&
               strcmp(aug_op_name(99), "identity") == 0);
    free(a);
    free(o);
}

int main(void)
{
    printf("== 数据增广内核自测 ==\n");
    test_brightness();
    test_contrast();
    test_shift_rotate();
    test_cutout();
    test_cutmix();
    test_mixup();
    test_randaugment();
    if (g_fail == 0) {
        printf("全部通过（失败项 = 0）\n");
        return 0;
    }
    printf("存在失败项（失败项 = %d）\n", g_fail);
    return 1;
}
