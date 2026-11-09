/*
 * compare_gan_variants.c —— 读取 tools/gan_variants_reference.py 生成的参考数据，
 * 用 C 实现重算 GAN 变体辅助函数并逐项比对。
 *
 * 用法: compare_gan_variants [gan_variants_reference.json] [容差]
 */
#include "../gan_variants.h"
#include "../json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_fail = 0;
static int g_exact = 0;
static double g_tol = 1e-12;

static const JsonValue *need(const JsonValue *obj, const char *key)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/gan_variants_reference.py）\n", key);
        exit(2);
    }
    return v;
}

static double *alloc_doubles(const JsonValue *arr, int n)
{
    if (!arr || json_array_len(arr) != n) {
        fprintf(stderr, "compare_gan_variants: 数组长度不符合预期\n");
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_gan_variants: 内存不足\n");
        exit(2);
    }
    for (int i = 0; i < n; i++) {
        p[i] = json_num(json_at(arr, i), 0.0);
    }
    return p;
}

static void report(const char *what, const double *got, const double *want, int n)
{
    g_checks++;
    double worst = 0.0;
    int at = -1;
    int exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            exact = 0;
        }
        double d = fabs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    if (exact) {
        g_exact++;
    }
    if (worst > g_tol) {
        g_fail++;
        printf("    %-30s 最大差异 %.3e [超出容差] index %d C=%.17g py=%.17g\n",
               what, worst, at, got[at], want[at]);
    } else {
        printf("    %-30s 最大差异 %.3e [OK]%s\n", what, worst, exact ? " (逐位相同)" : "");
    }
}

static void report_scalar(const char *what, double got, double want)
{
    report(what, &got, &want, 1);
}

/* 把嵌套数组（数字）递归展平，返回展平长度；buf 为 NULL 时只计数。 */
static int flatten_num(const JsonValue *v, double *buf, int off)
{
    if (!v) {
        return off;
    }
    if (v->kind == JSON_ARRAY) {
        for (int i = 0; i < v->n_items; i++) {
            off = flatten_num(v->items[i], buf, off);
        }
        return off;
    }
    if (buf) {
        buf[off] = json_num(v, 0.0);
    }
    return off + 1;
}

static double *flatten_alloc(const JsonValue *v, int *out_n)
{
    int n = flatten_num(v, NULL, 0);
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_gan_variants: 内存不足\n");
        exit(2);
    }
    flatten_num(v, p, 0);
    *out_n = n;
    return p;
}

static const char *case_func(const JsonValue *c)
{
    return json_str(need(c, "func"), "");
}

static double *case_expected(const JsonValue *c, int *n)
{
    return flatten_alloc(need(c, "expected"), n);
}

static int case_i(const JsonValue *c, const char *k)
{
    return json_int(need(c, k), 0);
}

static double case_d(const JsonValue *c, const char *k)
{
    return json_num(need(c, k), 0.0);
}

static void run_case(const JsonValue *c)
{
    const char *func = case_func(c);
    if (strcmp(func, "fade_in") == 0) {
        int ch = case_i(c, "c");
        int lh = case_i(c, "lh");
        int lw = case_i(c, "lw");
        double alpha = case_d(c, "alpha");
        int nlow = ch * lh * lw;
        int nout = ch * lh * 2 * lw * 2;
        double *low = alloc_doubles(need(c, "low"), nlow);
        double *high = alloc_doubles(need(c, "high"), nout);
        int en = 0;
        double *want = case_expected(c, &en);
        double *out = (double *)calloc((size_t)nout, sizeof(double));
        int rc = gv_fade_in(low, high, ch, lh, lw, alpha, out);
        report("fade_in 返回码", &(double){rc == 0 ? 0.0 : 1.0}, &(double){0.0}, 1);
        report("fade_in", out, want, nout);
        free(low);
        free(high);
        free(want);
        free(out);
    } else if (strcmp(func, "progressive_alpha") == 0) {
        double alpha = -1.0;
        int rc = gv_progressive_alpha(case_i(c, "step"), case_i(c, "total"), &alpha);
        report_scalar("progressive_alpha 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        report_scalar("progressive_alpha", alpha, case_d(c, "expected"));
    } else if (strcmp(func, "progressive_stages") == 0) {
        const JsonValue *res = need(c, "resolutions");
        int n = json_array_len(res);
        int *r = (int *)malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; i++) {
            r[i] = json_int(json_at(res, i), 0);
        }
        int fade_steps = case_i(c, "fade_steps");
        int cap = (n - 1) * (fade_steps + 1);
        int *low = (int *)malloc((size_t)cap * sizeof(int));
        int *high = (int *)malloc((size_t)cap * sizeof(int));
        double *alpha = (double *)malloc((size_t)cap * sizeof(double));
        int rc = gv_progressive_stages(r, n, fade_steps, low, high, alpha, cap);
        report_scalar("progressive_stages 返回阶段数", rc, (double)cap);
        double *got = (double *)malloc((size_t)(cap * 3) * sizeof(double));
        for (int i = 0; i < cap; i++) {
            got[3 * i] = low[i];
            got[3 * i + 1] = high[i];
            got[3 * i + 2] = alpha[i];
        }
        int en = 0;
        double *want = case_expected(c, &en);
        report("progressive_stages", got, want, cap * 3);
        free(r);
        free(low);
        free(high);
        free(alpha);
        free(got);
        free(want);
    } else if (strcmp(func, "adain_style") == 0) {
        int ch = case_i(c, "c");
        int h = case_i(c, "h");
        int w = case_i(c, "w");
        int n = ch * h * w;
        double *feat = alloc_doubles(need(c, "features"), n);
        double *style = alloc_doubles(need(c, "style"), 2 * ch);
        double eps = case_d(c, "eps");
        double *out = (double *)calloc((size_t)n, sizeof(double));
        double *means = (double *)calloc((size_t)ch, sizeof(double));
        double *stds = (double *)calloc((size_t)ch, sizeof(double));
        int rc = gv_adain_style(feat, ch, h, w, style, eps, out, means, stds);
        report_scalar("adain_style 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        int en = 0;
        double *want = case_expected(c, &en);
        report("adain_style", out, want, n);
        double *wm = alloc_doubles(need(c, "stats_mean"), ch);
        double *ws = alloc_doubles(need(c, "stats_std"), ch);
        report("adain_style stats_mean", means, wm, ch);
        report("adain_style stats_std", stds, ws, ch);
        free(feat);
        free(style);
        free(out);
        free(means);
        free(stds);
        free(want);
        free(wm);
        free(ws);
    } else if (strcmp(func, "truncation_trick") == 0) {
        const JsonValue *wv = need(c, "w");
        int n = json_array_len(wv);
        double *w = alloc_doubles(wv, n);
        double *w_avg = alloc_doubles(need(c, "w_avg"), n);
        double *out = (double *)calloc((size_t)n, sizeof(double));
        int rc = gv_truncation_trick(w, w_avg, n, case_d(c, "psi"), out);
        report_scalar("truncation_trick 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        int en = 0;
        double *want = case_expected(c, &en);
        report("truncation_trick", out, want, n);
        free(w);
        free(w_avg);
        free(out);
        free(want);
    } else if (strcmp(func, "cycle_consistency_loss") == 0) {
        int ch = case_i(c, "c");
        int h = case_i(c, "h");
        int w = case_i(c, "w");
        int n = ch * h * w;
        double *a = alloc_doubles(need(c, "a"), n);
        double *a_rec = alloc_doubles(need(c, "a_rec"), n);
        double *b = alloc_doubles(need(c, "b"), n);
        double *b_rec = alloc_doubles(need(c, "b_rec"), n);
        double loss = 0.0;
        double *ga = (double *)calloc((size_t)n, sizeof(double));
        double *gb = (double *)calloc((size_t)n, sizeof(double));
        int rc = gv_cycle_consistency_loss(a, a_rec, b, b_rec, ch, h, w,
                                          case_d(c, "lambda_cycle"), &loss, ga, gb);
        report_scalar("cycle_consistency_loss 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        report_scalar("cycle_consistency_loss", loss, case_d(c, "expected"));
        int en = 0;
        double *wa = alloc_doubles(need(c, "grad_a"), n);
        double *wb = alloc_doubles(need(c, "grad_b"), n);
        report("cycle_consistency_loss grad_a", ga, wa, n);
        report("cycle_consistency_loss grad_b", gb, wb, n);
        (void)en;
        free(a);
        free(a_rec);
        free(b);
        free(b_rec);
        free(ga);
        free(gb);
        free(wa);
        free(wb);
    } else if (strcmp(func, "identity_mapping_loss") == 0) {
        int ch = case_i(c, "c");
        int h = case_i(c, "h");
        int w = case_i(c, "w");
        int n = ch * h * w;
        double *a = alloc_doubles(need(c, "a"), n);
        double *g_ba = alloc_doubles(need(c, "g_ba"), n);
        double *b = alloc_doubles(need(c, "b"), n);
        double *g_ab = alloc_doubles(need(c, "g_ab"), n);
        double loss = 0.0;
        int rc = gv_identity_mapping_loss(a, g_ba, b, g_ab, ch, h, w,
                                          case_d(c, "lambda_identity"), &loss);
        report_scalar("identity_mapping_loss 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        report_scalar("identity_mapping_loss", loss, case_d(c, "expected"));
        free(a);
        free(g_ba);
        free(b);
        free(g_ab);
    } else if (strcmp(func, "patchgan_targets") == 0) {
        int h = case_i(c, "h");
        int w = case_i(c, "w");
        const JsonValue *rv = json_get(c, "real");
        int real = (rv && rv->kind == JSON_BOOL) ? rv->boolean : json_int(rv, 1);
        int n = h * w;
        double *out = (double *)calloc((size_t)n, sizeof(double));
        int rc = gv_patchgan_targets(h, w, real, out);
        report_scalar("patchgan_targets 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        int en = 0;
        double *want = case_expected(c, &en);
        report("patchgan_targets", out, want, n);
        free(out);
        free(want);
    } else if (strcmp(func, "pix2pix_g_loss") == 0) {
        int ns = case_i(c, "ns");
        int ch = case_i(c, "c");
        int h = case_i(c, "h");
        int w = case_i(c, "w");
        int n = ch * h * w;
        double *scores = alloc_doubles(need(c, "scores"), ns);
        double *fake = alloc_doubles(need(c, "fake"), n);
        double *real = alloc_doubles(need(c, "real"), n);
        double loss = 0.0;
        int rc = gv_pix2pix_g_loss(scores, ns, fake, real, ch, h, w,
                                   case_d(c, "lambda_l1"), &loss);
        report_scalar("pix2pix_g_loss 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
        report_scalar("pix2pix_g_loss", loss, case_d(c, "expected"));
        free(scores);
        free(fake);
        free(real);
    } else if (strcmp(func, "r1_penalty") == 0 || strcmp(func, "r2_penalty") == 0) {
        int ns = case_i(c, "n_samples");
        int dim = case_i(c, "grad_dim");
        int n = ns * dim;
        double *grads = alloc_doubles(need(c, "grads"), n);
        double got = (func[1] == '1')
                         ? gv_r1_penalty(grads, ns, dim, case_d(c, "lam"))
                         : gv_r2_penalty(grads, ns, dim, case_d(c, "lam"));
        report_scalar(func, got, case_d(c, "expected"));
        free(grads);
    } else {
        fprintf(stderr, "compare_gan_variants: 未知 func %s\n", func);
        exit(2);
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "gan_variants_reference.json";
    if (argc > 2) {
        g_tol = atof(argv[2]);
    }
    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/gan_variants_reference.py %s\n", path);
        return 2;
    }
    const JsonValue *cases = json_get(root, "cases");
    int nc = json_array_len(cases);
    if (nc <= 0) {
        fprintf(stderr, "compare_gan_variants: 参考数据里没有用例\n");
        json_free(root);
        return 2;
    }
    printf("GAN 变体辅助函数对拍：%d 个用例，容差 %.1e\n", nc, g_tol);
    for (int i = 0; i < nc; i++) {
        const JsonValue *c = json_at(cases, i);
        printf("  用例 %d: %s\n", i + 1, case_func(c));
        run_case(c);
    }
    printf("----------------------------------------\n");
    printf("共 %d 项检查：逐位一致 %d，失败 %d\n", g_checks, g_exact, g_fail);
    json_free(root);
    return g_fail > 0 ? 1 : 0;
}
