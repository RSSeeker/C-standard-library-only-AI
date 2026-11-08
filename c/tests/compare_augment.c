/*
 * compare_augment.c —— 把 C 的增广内核与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_augment [augment_reference.json]
 *       默认读当前目录的 augment_reference.json
 *
 * 随机量（lam / 裁剪框 / 洞位置 / 操作序列与强度）由参考脚本注入，
 * 因此这里所有数值都要求**逐位相同**（容差 0）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../augment.h"
#include "../json.h"

static int g_items;
static int g_exact;
static int g_fail;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_augment] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/augment_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int num_int(const JsonValue *o, const char *key, const char *who)
{
    return (int)json_num(need(o, key, who), 0.0);
}

static double num_dbl(const JsonValue *o, const char *key, const char *who)
{
    return json_num(need(o, key, who), 0.0);
}

static void arr_vals(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_augment] %s 长度不符：期望 %d，JSON 里是 %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static double *alloc_vals(const JsonValue *v, int n)
{
    double *p = (double *)malloc(sizeof(double) * (size_t)n);
    if (p == NULL) {
        fprintf(stderr, "[compare_augment] 内存不足\n");
        exit(2);
    }
    arr_vals(v, p, n, "数组");
    return p;
}

static void report_vec(const char *what, const double *got, const double *want,
                       int n, int *bad)
{
    int diff = 0, first = -1;
    for (int i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            diff++;
            if (first < 0) first = i;
        }
    }
    g_items++;
    if (diff == 0) {
        g_exact++;
        return;
    }
    {
        double d = fabs(got[first] - want[first]);
        if (d > g_worst) g_worst = d;
        printf("  %-22s %d / %d 个元素不等，首个下标 %d：C=%.17g py=%.17g 差=%.3e [不等]\n",
               what, diff, n, first, got[first], want[first], d);
    }
    g_fail++;
    *bad = 1;
}

static void run_case(const JsonValue *c, int idx)
{
    const char *name = json_str(need(c, "name", "case"), "?");
    const char *func = json_str(need(c, "func", "case"), "?");
    int bad = 0;

    if (strcmp(func, "brightness") == 0 || strcmp(func, "contrast") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        double factor = num_dbl(c, "factor", name);
        double *x = alloc_vals(need(c, "x", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        if (func[0] == 'b') aug_brightness(x, n, factor);
        else aug_contrast(x, C * H, W, factor);
        report_vec(name, x, want, n, &bad);
        free(x);
        free(want);
    } else if (strcmp(func, "shift") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        double *x = alloc_vals(need(c, "x", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        double *got = (double *)calloc((size_t)n, sizeof(double));
        if (got == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        aug_shift(x, C, H, W, num_int(c, "dy", name), num_int(c, "dx", name), got);
        report_vec(name, got, want, n, &bad);
        free(x); free(want); free(got);
    } else if (strcmp(func, "rotate90") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        int oh = 0, ow = 0;
        double *x = alloc_vals(need(c, "x", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        double *got = (double *)calloc((size_t)n, sizeof(double));
        if (got == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        aug_rotate90(x, C, H, W, num_int(c, "turns", name), got, &oh, &ow);
        if (oh != num_int(c, "oh", name) || ow != num_int(c, "ow", name)) {
            printf("  %-22s 输出尺寸不符：C=%dx%d py=%dx%d [不等]\n", name, oh, ow,
                   num_int(c, "oh", name), num_int(c, "ow", name));
            g_fail++;
        } else {
            report_vec(name, got, want, n, &bad);
        }
        free(x); free(want); free(got);
    } else if (strcmp(func, "cutout") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        int holes = num_int(c, "num_holes", name);
        double *x = alloc_vals(need(c, "x", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        int *xs = (int *)calloc((size_t)(holes > 0 ? holes : 1), sizeof(int));
        int *ys = (int *)calloc((size_t)(holes > 0 ? holes : 1), sizeof(int));
        if (xs == NULL || ys == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        for (int i = 0; i < holes; i++) {
            xs[i] = (int)json_num(json_at(need(c, "xs", name), i), 0.0);
            ys[i] = (int)json_num(json_at(need(c, "ys", name), i), 0.0);
        }
        aug_cutout(x, C, H, W, holes, num_dbl(c, "hole_size", name),
                   num_dbl(c, "fill", name), xs, ys);
        report_vec(name, x, want, n, &bad);
        free(x); free(want); free(xs); free(ys);
    } else if (strcmp(func, "cutmix") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        int ny = json_array_len(need(c, "y1", name));
        double *xa = alloc_vals(need(c, "xa", name), n);
        double *xb = alloc_vals(need(c, "xb", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        double *got = (double *)calloc((size_t)n, sizeof(double));
        double *want_lab = alloc_vals(need(c, "label", name), ny);
        double *got_lab = (double *)calloc((size_t)ny, sizeof(double));
        double *y1 = alloc_vals(need(c, "y1", name), ny);
        double *y2 = alloc_vals(need(c, "y2", name), ny);
        double lam;
        if (got == NULL || got_lab == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        lam = aug_cutmix(xa, xb, C, H, W, num_dbl(c, "lam0", name),
                         num_int(c, "cx", name), num_int(c, "cy", name), got);
        report_vec(name, got, want, n, &bad);
        /* 标签：lam*float(u) + (1-lam)*float(v)，用 C 返回的有效 lam */
        for (int i = 0; i < ny; i++) got_lab[i] = lam * y1[i] + (1.0 - lam) * y2[i];
        {
            char buf[96];
            snprintf(buf, sizeof buf, "%s.label", name);
            report_vec(buf, got_lab, want_lab, ny, &bad);
        }
        free(xa); free(xb); free(want); free(got);
        free(y1); free(y2); free(want_lab); free(got_lab);
    } else if (strcmp(func, "mixup") == 0) {
        int n = num_int(c, "n", name);
        int n_cls = num_int(c, "n_cls", name);
        double lam = num_dbl(c, "lam", name);
        double *x1 = alloc_vals(need(c, "x1", name), n);
        double *x2 = alloc_vals(need(c, "x2", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        double *want_soft = alloc_vals(need(c, "soft", name), n_cls);
        double *got = (double *)calloc((size_t)n, sizeof(double));
        double *got_soft = (double *)calloc((size_t)n_cls, sizeof(double));
        if (got == NULL || got_soft == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        aug_mixup(x1, x2, n, lam, got);
        report_vec(name, got, want, n, &bad);
        aug_mixup_soft(num_int(c, "y1", name), num_int(c, "y2", name), n_cls, lam, got_soft);
        {
            char buf[96];
            snprintf(buf, sizeof buf, "%s.soft", name);
            report_vec(buf, got_soft, want_soft, n_cls, &bad);
        }
        free(x1); free(x2); free(want); free(got); free(want_soft); free(got_soft);
    } else if (strcmp(func, "randaugment") == 0) {
        int C = num_int(c, "C", name), H = num_int(c, "H", name), W = num_int(c, "W", name);
        int n = C * H * W;
        int steps = num_int(c, "n", name);
        int oh = 0, ow = 0;
        double *x = alloc_vals(need(c, "x", name), n);
        double *want = alloc_vals(need(c, "out", name), n);
        double *got = (double *)calloc((size_t)n, sizeof(double));
        int *ops = (int *)calloc((size_t)(steps > 0 ? steps : 1), sizeof(int));
        double *params = (double *)calloc((size_t)(steps > 0 ? steps : 1) * 2, sizeof(double));
        if (got == NULL || ops == NULL || params == NULL) {
            fprintf(stderr, "[compare_augment] 内存不足\n");
            exit(2);
        }
        for (int i = 0; i < steps; i++)
            ops[i] = (int)json_num(json_at(need(c, "ops", name), i), 0.0);
        for (int i = 0; i < steps * 2; i++)
            params[i] = json_num(json_at(need(c, "params", name), i), 0.0);
        aug_randaugment(x, C, H, W, steps, ops, params, got, &oh, &ow);
        if (oh != num_int(c, "oh", name) || ow != num_int(c, "ow", name)) {
            printf("  %-22s 输出尺寸不符：C=%dx%d py=%dx%d [不等]\n", name, oh, ow,
                   num_int(c, "oh", name), num_int(c, "ow", name));
            g_fail++;
        } else {
            report_vec(name, got, want, n, &bad);
        }
        free(x); free(want); free(got); free(ops); free(params);
    } else {
        fprintf(stderr, "[compare_augment] 第 %d 个用例 %s 的 func=%s 未知\n", idx, name, func);
        exit(2);
    }
    (void)bad;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "augment_reference.json";
    char err[512];
    JsonValue *root = json_parse_file(path, err, sizeof err);
    const JsonValue *cases;
    int n;

    if (root == NULL) {
        fprintf(stderr, "[compare_augment] 解析 %s 失败：%s\n", path, err);
        return 2;
    }
    cases = need(root, "cases", "root");
    n = json_array_len(cases);
    printf("数据增广内核：%s（%d 个用例）\n", path, n);
    for (int i = 0; i < n; i++) run_case(json_at(cases, i), i);

    printf("\n%d 项检查，%d 项逐位相同", g_items, g_exact);
    if (g_fail == 0) {
        printf("，全部 [OK]（最大差异 %.3e）\n", g_worst);
    } else {
        printf("，%d 项 [不等]（最大差异 %.3e）\n", g_fail, g_worst);
    }
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
