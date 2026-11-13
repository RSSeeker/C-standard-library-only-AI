/*
 * compare_linear_attn.c —— 线性注意力 / 长卷积与 AI.py 的对拍
 *
 * 用法：./tests/compare_linear_attn [linear_attn_reference.json]
 *
 * 覆盖 linear_attention_naive / lightning_attention / gla_forward /
 * deltanet_forward / hyena_long_conv / hyena_forward。每个用例的容差来自参考
 * JSON 的 "tol" 字段：0.0 表示要求 C 侧浮点逐位相同；否则允许 |差| <= tol。
 * 错误用例只要求 C 侧返回 -1（Python 侧在导出参考值时已确认会 raise）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../linear_attn.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;
static char g_worst_label[128];

static void die(const char *msg)
{
    fprintf(stderr, "[compare_linear_attn] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key,
                             const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_linear_attn] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/linear_attn_reference.py\n",
                who, key);
        exit(2);
    }
    return v;
}

static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

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

/* 二维数组（行主序）读入并校验每行等长；允许空矩阵（rows=0）。 */
typedef struct {
    int rows;
    int cols;
    double *v;
} Mat;

static Mat mat_read(const JsonValue *v, const char *who)
{
    Mat m;
    int n = json_array_len(v);
    m.rows = n;
    m.cols = 0;
    m.v = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!m.v)
        die("out of memory");
    if (n == 0)
        return m;
    m.cols = json_array_len(json_at(v, 0));
    free(m.v);
    m.v = (double *)calloc(
        (size_t)((size_t)n * (size_t)(m.cols > 0 ? m.cols : 1)), sizeof(double));
    if (!m.v)
        die("out of memory");
    for (int r = 0; r < n; r++) {
        const JsonValue *row = json_at(v, r);
        if (json_array_len(row) != m.cols)
            die("参考矩阵各行长度不一致");
        for (int c = 0; c < m.cols; c++)
            m.v[(size_t)r * (size_t)m.cols + (size_t)c] =
                jnum(json_at(row, c));
    }
    (void)who;
    return m;
}

static void mat_free(Mat *m)
{
    free(m->v);
    m->v = NULL;
}

/* 一维数组读入；允许空数组（返回非 NULL 指针，len=0）。 */
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

static double case_tol(const JsonValue *cs)
{
    const JsonValue *tv = json_get(cs, "tol");
    return tv ? jnum(tv) : 0.0;
}

static void check_mat(const char *prefix, const double *got, int got_rows,
                      int got_cols, const Mat *want, double tol)
{
    char label[128];
    snprintf(label, sizeof label, "%s.rows", prefix);
    check_int(label, got_rows, want->rows);
    snprintf(label, sizeof label, "%s.cols", prefix);
    check_int(label, got_cols, want->cols);
    for (int r = 0; r < want->rows; r++) {
        for (int c = 0; c < want->cols; c++) {
            double g = 0.0;
            if (r < got_rows && c < got_cols)
                g = got[(size_t)r * (size_t)got_cols + (size_t)c];
            snprintf(label, sizeof label, "%s[%d][%d]", prefix, r, c);
            check_num(label, g,
                      want->v[(size_t)r * (size_t)want->cols + (size_t)c], tol);
        }
    }
}

static double *out_alloc(int t_len, int width)
{
    size_t n = (size_t)(t_len > 0 ? t_len : 1) *
               (size_t)(width > 0 ? width : 1);
    double *p = (double *)calloc(n, sizeof(double));
    if (!p)
        die("out of memory");
    return p;
}

static void run_naive(const JsonValue *cs, const char *name, double tol)
{
    Mat q = mat_read(need(cs, "q", name), name);
    Mat k = mat_read(need(cs, "k", name), name);
    Mat v = mat_read(need(cs, "v", name), name);
    double decay = jnum(need(cs, "decay", name));
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(q.rows, v.cols);
    int rc = ai_linear_attention_naive(q.v, k.v, v.v, q.rows, q.cols, v.cols,
                                       decay, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, q.rows, v.cols, &want, tol);
    free(out);
    mat_free(&q);
    mat_free(&k);
    mat_free(&v);
    mat_free(&want);
}

static void run_lightning(const JsonValue *cs, const char *name, double tol)
{
    Mat q = mat_read(need(cs, "q", name), name);
    Mat k = mat_read(need(cs, "k", name), name);
    Mat v = mat_read(need(cs, "v", name), name);
    double decay = jnum(need(cs, "decay", name));
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(q.rows, v.cols);
    int rc = ai_lightning_attention(q.v, k.v, v.v, q.rows, q.cols, v.cols,
                                    decay, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, q.rows, v.cols, &want, tol);
    free(out);
    mat_free(&q);
    mat_free(&k);
    mat_free(&v);
    mat_free(&want);
}

static void run_gla(const JsonValue *cs, const char *name, double tol)
{
    Mat q = mat_read(need(cs, "q", name), name);
    Mat k = mat_read(need(cs, "k", name), name);
    Mat v = mat_read(need(cs, "v", name), name);
    int gates_n = 0;
    double *gates = seq_read(need(cs, "gates", name), &gates_n);
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(q.rows, v.cols);
    int rc;
    check_int("gates.len", gates_n, q.rows);
    rc = ai_gla_forward(q.v, k.v, v.v, gates, q.rows, q.cols, v.cols, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, q.rows, v.cols, &want, tol);
    free(gates);
    free(out);
    mat_free(&q);
    mat_free(&k);
    mat_free(&v);
    mat_free(&want);
}

static void run_deltanet(const JsonValue *cs, const char *name, double tol)
{
    Mat q = mat_read(need(cs, "q", name), name);
    Mat k = mat_read(need(cs, "k", name), name);
    Mat v = mat_read(need(cs, "v", name), name);
    int betas_n = 0;
    double *betas = seq_read(need(cs, "betas", name), &betas_n);
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(q.rows, v.cols);
    int rc;
    check_int("betas.len", betas_n, q.rows);
    rc = ai_deltanet_forward(q.v, k.v, v.v, betas, q.rows, q.cols, v.cols,
                             out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, q.rows, v.cols, &want, tol);
    free(betas);
    free(out);
    mat_free(&q);
    mat_free(&k);
    mat_free(&v);
    mat_free(&want);
}

static void run_hyena_conv(const JsonValue *cs, const char *name, double tol)
{
    Mat u = mat_read(need(cs, "u", name), name);
    int k_len = 0;
    double *kernel = seq_read(need(cs, "kernel", name), &k_len);
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(u.rows, u.cols);
    int rc = ai_hyena_long_conv(u.v, u.rows, u.cols, kernel, k_len, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, u.rows, u.cols, &want, tol);
    free(kernel);
    free(out);
    mat_free(&u);
    mat_free(&want);
}

static void run_hyena(const JsonValue *cs, const char *name, double tol)
{
    Mat u = mat_read(need(cs, "u", name), name);
    Mat v = mat_read(need(cs, "v", name), name);
    int k_len = 0;
    double *kernel = seq_read(need(cs, "kernel", name), &k_len);
    Mat want = mat_read(need(cs, "out", name), name);
    double *out = out_alloc(u.rows, u.cols);
    int rc;
    check_int("v.rows", v.rows, u.rows);
    check_int("v.cols", v.cols, u.cols);
    rc = ai_hyena_forward(u.v, v.v, u.rows, u.cols, kernel, k_len, out);
    check_int("rc", rc, 0);
    if (rc == 0)
        check_mat("out", out, u.rows, u.cols, &want, tol);
    free(kernel);
    free(out);
    mat_free(&u);
    mat_free(&v);
    mat_free(&want);
}

/* 错误用例：按 op 构造参数，只要求 C 侧返回 -1。 */
static void run_error(const JsonValue *cs, const char *name)
{
    const JsonValue *args = need(cs, "args", name);
    const char *op = json_str(need(cs, "op", name), "");
    int rc = 0;

    if (strcmp(op, "naive") == 0 || strcmp(op, "lightning") == 0 ||
        strcmp(op, "gla") == 0 || strcmp(op, "deltanet") == 0) {
        Mat q = mat_read(need(args, "q", name), name);
        Mat k = mat_read(need(args, "k", name), name);
        Mat v = mat_read(need(args, "v", name), name);
        double *out = out_alloc(q.rows, v.cols);
        if (strcmp(op, "naive") == 0) {
            double decay = jnum(json_get(args, "decay"));
            rc = ai_linear_attention_naive(q.v, k.v, v.v, q.rows, q.cols,
                                           v.cols, decay, out);
        } else if (strcmp(op, "lightning") == 0) {
            double decay = jnum(json_get(args, "decay"));
            rc = ai_lightning_attention(q.v, k.v, v.v, q.rows, q.cols, v.cols,
                                        decay, out);
        } else if (strcmp(op, "gla") == 0) {
            int gates_n = 0;
            double *gates = seq_read(need(args, "gates", name), &gates_n);
            rc = ai_gla_forward(q.v, k.v, v.v, gates, q.rows, q.cols, v.cols,
                                out);
            free(gates);
        } else {
            int betas_n = 0;
            double *betas = seq_read(need(args, "betas", name), &betas_n);
            rc = ai_deltanet_forward(q.v, k.v, v.v, betas, q.rows, q.cols,
                                     v.cols, out);
            free(betas);
        }
        free(out);
        mat_free(&q);
        mat_free(&k);
        mat_free(&v);
    } else if (strcmp(op, "hyena_conv") == 0) {
        Mat u = mat_read(need(args, "u", name), name);
        int k_len = 0;
        double *kernel = seq_read(need(args, "kernel", name), &k_len);
        double *out = out_alloc(u.rows, u.cols);
        rc = ai_hyena_long_conv(u.v, u.rows, u.cols, kernel, k_len, out);
        free(kernel);
        free(out);
        mat_free(&u);
    } else if (strcmp(op, "hyena") == 0) {
        Mat u = mat_read(need(args, "u", name), name);
        Mat v = mat_read(need(args, "v", name), name);
        int k_len = 0;
        double *kernel = seq_read(need(args, "kernel", name), &k_len);
        double *out = out_alloc(u.rows, u.cols);
        rc = ai_hyena_forward(u.v, v.v, u.rows, u.cols, kernel, k_len, out);
        free(kernel);
        free(out);
        mat_free(&u);
        mat_free(&v);
    } else {
        fprintf(stderr, "[compare_linear_attn] 未知 error op: %s\n", op);
        exit(2);
    }
    check_int("rc(-1)", rc, -1);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "linear_attn_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    const JsonValue *cases;
    if (!root) {
        fprintf(stderr, "[compare_linear_attn] 读取 %s 失败: %s\n", path, err);
        return 2;
    }
    cases = json_get(root, "cases");
    if (!cases || json_array_len(cases) <= 0) {
        fprintf(stderr, "[compare_linear_attn] 参考数据缺少 cases 数组\n");
        json_free(root);
        return 2;
    }

    for (int i = 0; i < json_array_len(cases); i++) {
        const JsonValue *cs = json_at(cases, i);
        const char *kind = json_str(json_get(cs, "kind"), "");
        const char *name = json_str(json_get(cs, "name"), "(unnamed)");
        double tol = case_tol(cs);

        if (strcmp(kind, "naive") == 0)
            run_naive(cs, name, tol);
        else if (strcmp(kind, "lightning") == 0)
            run_lightning(cs, name, tol);
        else if (strcmp(kind, "gla") == 0)
            run_gla(cs, name, tol);
        else if (strcmp(kind, "deltanet") == 0)
            run_deltanet(cs, name, tol);
        else if (strcmp(kind, "hyena_conv") == 0)
            run_hyena_conv(cs, name, tol);
        else if (strcmp(kind, "hyena") == 0)
            run_hyena(cs, name, tol);
        else if (strcmp(kind, "error") == 0)
            run_error(cs, name);
        else {
            fprintf(stderr, "[compare_linear_attn] 未知 kind: %s\n", kind);
            json_free(root);
            return 2;
        }
    }

    json_free(root);
    printf("=== compare_linear_attn：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    if (g_fail && g_worst_label[0])
        printf("    最大差异出现在 %s\n", g_worst_label);
    return g_fail ? 1 : 0;
}
