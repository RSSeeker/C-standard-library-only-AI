/*
 * compare_model_merge.c —— 模型合并（soup / SLERP / TIES / DARE）与 AI.py 的对拍
 *
 * 用法：./tests/compare_model_merge [model_merge_reference.json]
 *
 * C 侧用扁平张量表达：weights[j*n_elem + i]。参考 JSON 同时记录 shape
 * （[rows, cols] 或 [n]）与扁平输入/输出，本程序会校验张量长度与 shape 一致。
 * 容差来自参考 JSON 的 "tol" 字段：0.0 要求逐位相同，否则 |差| <= tol。
 * 错误用例只要求 C 侧返回 -1（Python 侧在导出参考值时已确认会 raise）；
 * 「形状不一致」类错误在 C 扁平入参下无法表达，未纳入对拍。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../model_merge.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;
static char g_worst_label[128];

static void die(const char *msg)
{
    fprintf(stderr, "[compare_model_merge] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key,
                             const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_model_merge] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/model_merge_reference.py\n",
                who, key);
        exit(2);
    }
    return v;
}

static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static long jint(const JsonValue *v) { return json_int(v, 0); }

static int bits_equal(double a, double b)
{
    return memcmp(&a, &b, sizeof(double)) == 0;
}

static void check_num(const char *what, double got, double want, double tol)
{
    double d = fabs(got - want);
    int ok = (d <= tol);
    g_checks++;
    if (bits_equal(got, want))
        g_exact++;
    else if (d > g_worst) {
        g_worst = d;
        snprintf(g_worst_label, sizeof g_worst_label, "%s", what);
    }
    if (!ok) {
        printf("  %-36s C=%.17g py=%.17g 差=%.3e [超出容差]\n",
               what, got, want, d);
        g_fail++;
    }
}

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want)
        g_exact++;
    else {
        printf("  %-36s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

static double *seq_read(const JsonValue *v, int *out_n)
{
    int n = json_array_len(v);
    double *p = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!p)
        die("out of memory");
    for (int i = 0; i < n; i++)
        p[i] = jnum(json_at(v, i));
    *out_n = n;
    return p;
}

/* 读 weights：若干等长扁平张量（JSON 里是数组的数组）。 */
static double *tensors_read(const JsonValue *v, int *out_n_tensors,
                            int *out_n_elem)
{
    int n = json_array_len(v);
    int n_elem = n > 0 ? json_array_len(json_at(v, 0)) : 0;
    double *p = (double *)calloc(
        (size_t)((n > 0 ? n : 1) * (n_elem > 0 ? n_elem : 1)), sizeof(double));
    if (!p)
        die("out of memory");
    for (int j = 0; j < n; j++) {
        const JsonValue *row = json_at(v, j);
        if (json_array_len(row) != n_elem)
            die("参考 weights 的各张量长度不一致");
        for (int i = 0; i < n_elem; i++)
            p[(size_t)j * (size_t)n_elem + (size_t)i] = jnum(json_at(row, i));
    }
    *out_n_tensors = n;
    *out_n_elem = n_elem;
    return p;
}

/* shape 的维度乘积（错误用例可能没有 shape 字段，返回 0 表示未知）。 */
static int shape_product(const JsonValue *cs)
{
    const JsonValue *sh = json_get(cs, "shape");
    int prod = 1;
    if (!sh)
        return 0;
    for (int i = 0; i < json_array_len(sh); i++)
        prod *= json_int(json_at(sh, i), 0);
    return prod;
}

static double case_tol(const JsonValue *cs)
{
    const JsonValue *tv = json_get(cs, "tol");
    return tv ? jnum(tv) : 0.0;
}

static double *out_alloc(int n)
{
    double *p = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!p)
        die("out of memory");
    return p;
}

static void check_vec(const char *prefix, const double *got, int got_n,
                      const double *want, int want_n, double tol)
{
    char label[128];
    snprintf(label, sizeof label, "%s.len", prefix);
    check_int(label, got_n, want_n);
    for (int i = 0; i < want_n; i++) {
        double g = 0.0;
        if (i < got_n)
            g = got[i];
        if (want_n == 1)
            snprintf(label, sizeof label, "%s[0]", prefix);
        else
            snprintf(label, sizeof label, "%s[%d]", prefix, i);
        check_num(label, g, want[i], tol);
    }
}

static void run_soup(const JsonValue *cs, const char *name, double tol)
{
    int n_tensors = 0, n_elem = 0;
    double *w = tensors_read(need(cs, "weights", name), &n_tensors, &n_elem);
    int want_n = 0;
    double *want = seq_read(need(cs, "out", name), &want_n);
    double *out = out_alloc(n_elem);
    int prod = shape_product(cs);
    int rc;

    if (prod)
        check_int("shape->n_elem", n_elem, prod);
    rc = ai_model_soup(w, n_tensors, n_elem, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_vec("out", out, n_elem, want, want_n, tol);
    free(out);
    free(w);
    free(want);
}

static void run_slerp(const JsonValue *cs, const char *name, double tol)
{
    int a_n = 0, b_n = 0, want_n = 0;
    double *a = seq_read(need(cs, "a", name), &a_n);
    double *b = seq_read(need(cs, "b", name), &b_n);
    double t = jnum(need(cs, "t", name));
    double eps = jnum(need(cs, "eps", name));
    double *want = seq_read(need(cs, "out", name), &want_n);
    double *out = out_alloc(a_n);
    int prod = shape_product(cs);
    int rc;

    if (prod)
        check_int("shape->len", a_n, prod);
    rc = ai_slerp_merge(a, b, a_n, t, eps, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_vec("out", out, a_n, want, want_n, tol);
    free(out);
    free(a);
    free(b);
    free(want);
}

static void run_ties(const JsonValue *cs, const char *name, double tol)
{
    int n_tensors = 0, n_elem = 0;
    double *w = tensors_read(need(cs, "weights", name), &n_tensors, &n_elem);
    const JsonValue *bv = json_get(cs, "base");
    double *base = NULL;
    int base_n = 0;
    double trim = jnum(need(cs, "trim_fraction", name));
    double density = jnum(need(cs, "density", name));
    int want_n = 0;
    double *want = seq_read(need(cs, "out", name), &want_n);
    double *out = out_alloc(n_elem);
    int prod = shape_product(cs);
    int rc;

    if (bv) {
        base = seq_read(bv, &base_n);
        check_int("base.len", base_n, n_elem);
    }
    if (prod)
        check_int("shape->n_elem", n_elem, prod);
    rc = ai_ties_merge(w, n_tensors, n_elem, base, trim, density, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_vec("out", out, n_elem, want, want_n, tol);
    free(out);
    free(w);
    free(base);
    free(want);
}

static void run_dare(const JsonValue *cs, const char *name, double tol)
{
    int n_tensors = 0, n_elem = 0;
    double *w = tensors_read(need(cs, "weights", name), &n_tensors, &n_elem);
    const JsonValue *bv = json_get(cs, "base");
    double *base = NULL;
    int base_n = 0;
    double drop = jnum(need(cs, "drop_rate", name));
    int rescale = (int)jint(need(cs, "rescale", name));
    unsigned long long seed =
        (unsigned long long)jnum(need(cs, "seed", name));
    int want_n = 0;
    double *want = seq_read(need(cs, "out", name), &want_n);
    double *out = out_alloc(n_elem);
    int prod = shape_product(cs);
    int rc;

    if (bv) {
        base = seq_read(bv, &base_n);
        check_int("base.len", base_n, n_elem);
    }
    if (prod)
        check_int("shape->n_elem", n_elem, prod);
    rc = ai_dare_merge(w, n_tensors, n_elem, base, drop, rescale, seed, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_vec("out", out, n_elem, want, want_n, tol);
    free(out);
    free(w);
    free(base);
    free(want);
}

/* 错误用例：按 op 构造参数，只要求 C 侧返回 -1。 */
static void run_error(const JsonValue *cs, const char *name)
{
    const JsonValue *args = need(cs, "args", name);
    const char *op = json_str(need(cs, "op", name), "");
    int rc = 0;

    if (strcmp(op, "model_soup") == 0) {
        int n_tensors = 0, n_elem = 0;
        double *w = tensors_read(need(args, "weights", name), &n_tensors,
                                 &n_elem);
        double *out = out_alloc(n_elem);
        rc = ai_model_soup(w, n_tensors, n_elem, out);
        free(out);
        free(w);
    } else if (strcmp(op, "slerp") == 0) {
        int a_n = 0, b_n = 0;
        double *a = seq_read(need(args, "a", name), &a_n);
        double *b = seq_read(need(args, "b", name), &b_n);
        double t = jnum(json_get(args, "t"));
        double eps = jnum(json_get(args, "eps"));
        double *out = out_alloc(a_n);
        rc = ai_slerp_merge(a, b, a_n, t, eps, out);
        free(out);
        free(a);
        free(b);
    } else if (strcmp(op, "ties") == 0) {
        int n_tensors = 0, n_elem = 0;
        double *w = tensors_read(need(args, "weights", name), &n_tensors,
                                 &n_elem);
        const JsonValue *bv = json_get(args, "base");
        double *base = NULL;
        int base_n = 0;
        double trim = jnum(json_get(args, "trim_fraction"));
        double density = jnum(json_get(args, "density"));
        double *out = out_alloc(n_elem);
        if (bv) {
            base = seq_read(bv, &base_n);
        }
        rc = ai_ties_merge(w, n_tensors, n_elem, base, trim, density, out);
        free(out);
        free(w);
        free(base);
    } else if (strcmp(op, "dare") == 0) {
        int n_tensors = 0, n_elem = 0;
        double *w = tensors_read(need(args, "weights", name), &n_tensors,
                                 &n_elem);
        const JsonValue *bv = json_get(args, "base");
        double *base = NULL;
        int base_n = 0;
        double drop = jnum(json_get(args, "drop_rate"));
        int rescale = (int)jint(json_get(args, "rescale"));
        unsigned long long seed =
            (unsigned long long)jnum(json_get(args, "seed"));
        double *out = out_alloc(n_elem);
        if (bv) {
            base = seq_read(bv, &base_n);
        }
        rc = ai_dare_merge(w, n_tensors, n_elem, base, drop, rescale, seed,
                           out);
        free(out);
        free(w);
        free(base);
    } else {
        fprintf(stderr, "[compare_model_merge] 未知 error op: %s\n", op);
        exit(2);
    }
    check_int("rc(-1)", rc, -1);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "model_merge_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    const JsonValue *cases;
    if (!root) {
        fprintf(stderr, "[compare_model_merge] 读取 %s 失败: %s\n", path, err);
        return 2;
    }
    cases = json_get(root, "cases");
    if (!cases || json_array_len(cases) <= 0) {
        fprintf(stderr, "[compare_model_merge] 参考数据缺少 cases 数组\n");
        json_free(root);
        return 2;
    }

    for (int i = 0; i < json_array_len(cases); i++) {
        const JsonValue *cs = json_at(cases, i);
        const char *kind = json_str(json_get(cs, "kind"), "");
        const char *name = json_str(json_get(cs, "name"), "(unnamed)");
        double tol = case_tol(cs);

        if (strcmp(kind, "model_soup") == 0)
            run_soup(cs, name, tol);
        else if (strcmp(kind, "slerp") == 0)
            run_slerp(cs, name, tol);
        else if (strcmp(kind, "ties") == 0)
            run_ties(cs, name, tol);
        else if (strcmp(kind, "dare") == 0)
            run_dare(cs, name, tol);
        else if (strcmp(kind, "error") == 0)
            run_error(cs, name);
        else {
            fprintf(stderr, "[compare_model_merge] 未知 kind: %s\n", kind);
            json_free(root);
            return 2;
        }
    }

    json_free(root);
    printf("=== compare_model_merge：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    if (g_fail && g_worst_label[0])
        printf("    最大差异出现在 %s\n", g_worst_label);
    return g_fail ? 1 : 0;
}
