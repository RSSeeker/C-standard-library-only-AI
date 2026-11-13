/*
 * compare_rope_ext.c —— RoPE 扩展（ABF / LongRoPE / 2D / 3D）与 AI.py 的对拍
 *
 * 用法：./tests/compare_rope_ext [rope_ext_reference.json]
 *
 * 覆盖 abf_inv_freqs / longrope_inv_freqs / rope_2d / rope_2d_inverse /
 * rope_3d / rope_3d_inverse。每个用例的容差来自参考 JSON 的 "tol" 字段：
 *   tol = 0.0  要求 C 侧与 Python 浮点逐位相同（memcmp 相同）；
 *   tol > 0    允许 |差| <= tol（但逐位相同仍单独统计）。
 * 错误用例只要求 C 侧返回 -1（Python 侧在导出参考值时已确认会 raise，
 * 见 c/tools/rope_ext_reference.py）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../rope_ext.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;
static char g_worst_label[128];

static void die(const char *msg)
{
    fprintf(stderr, "[compare_rope_ext] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key,
                             const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_rope_ext] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/rope_ext_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
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
        printf("  %-34s C=%.17g py=%.17g 差=%.3e [超出容差]\n",
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
        printf("  %-34s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

/* 二维数组（行主序）读入并校验每行等长。 */
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
    if (m.cols == 0)
        die("参考矩阵行宽为 0");
    free(m.v);
    m.v = (double *)calloc((size_t)n * (size_t)m.cols, sizeof(double));
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

static void check_mat(const char *prefix, const Mat *got, const Mat *want,
                      double tol)
{
    char label[128];
    snprintf(label, sizeof label, "%s.rows", prefix);
    check_int(label, got->rows, want->rows);
    snprintf(label, sizeof label, "%s.cols", prefix);
    check_int(label, got->cols, want->cols);
    for (int r = 0; r < want->rows; r++) {
        for (int c = 0; c < want->cols; c++) {
            double g = 0.0;
            if (r < got->rows && c < got->cols)
                g = got->v[(size_t)r * (size_t)got->cols + (size_t)c];
            snprintf(label, sizeof label, "%s[%d][%d]", prefix, r, c);
            check_num(label, g,
                      want->v[(size_t)r * (size_t)want->cols + (size_t)c], tol);
        }
    }
}

static void check_vec(const char *prefix, const double *got, int got_n,
                      const JsonValue *want, double tol)
{
    int want_n = json_array_len(want);
    char label[128];
    snprintf(label, sizeof label, "%s.len", prefix);
    check_int(label, got_n, want_n);
    for (int i = 0; i < want_n; i++) {
        snprintf(label, sizeof label, "%s[%d]", prefix, i);
        check_num(label, (i < got_n) ? got[i] : NAN, jnum(json_at(want, i)), tol);
    }
}

static double case_tol(const JsonValue *cs)
{
    const JsonValue *tv = json_get(cs, "tol");
    return tv ? jnum(tv) : 0.0;
}

static Mat mat_arg(const JsonValue *cs, const char *key, const char *name)
{
    return mat_read(need(cs, key, name), name);
}

static void run_abf(const JsonValue *cs, const char *name, double tol)
{
    int d_k = jint(need(cs, "d_k", name));
    double base = jnum(need(cs, "base", name));
    double factor = jnum(need(cs, "factor", name));
    int n = (d_k > 0) ? d_k / 2 : 0;
    int got_n = -1;
    double *out = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!out)
        die("out of memory");
    check_int("rc", ai_abf_inv_freqs(d_k, base, factor, out, &got_n), 0);
    check_vec("out", out, got_n, need(cs, "out", name), tol);
    free(out);
}

static void run_longrope(const JsonValue *cs, const char *name, double tol)
{
    int d_k = jint(need(cs, "d_k", name));
    double base = jnum(need(cs, "base", name));
    double factor = jnum(need(cs, "factor", name));
    int original_max = jint(need(cs, "original_max", name));
    double low_fraction = jnum(need(cs, "low_fraction", name));
    int n = (d_k > 0) ? d_k / 2 : 0;
    int got_n = -1;
    double *out = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!out)
        die("out of memory");
    check_int("rc", ai_longrope_inv_freqs(d_k, base, factor, original_max,
                                          low_fraction, out, &got_n), 0);
    check_vec("out", out, got_n, need(cs, "out", name), tol);
    free(out);
}

static void run_rope_2d(const JsonValue *cs, const char *name, double tol,
                        int inverse)
{
    Mat in = mat_arg(cs, "rows", name);
    int grid_h = jint(need(cs, "grid_h", name));
    int grid_w = jint(need(cs, "grid_w", name));
    double base = jnum(need(cs, "base", name));
    Mat out;
    int rc;

    out.rows = in.rows;
    out.cols = in.cols;
    out.v = (double *)calloc((size_t)(in.rows * in.cols > 0
                                          ? in.rows * in.cols
                                          : 1), sizeof(double));
    if (!out.v)
        die("out of memory");
    rc = inverse
             ? ai_rope_2d_inverse(in.v, in.rows, in.cols, grid_h, grid_w,
                                  base, out.v)
             : ai_rope_2d(in.v, in.rows, in.cols, grid_h, grid_w, base, out.v);
    check_int("rc", rc, 0);
    if (rc == 0) {
        Mat want = mat_arg(cs, "out", name);
        check_mat("out", &out, &want, tol);
        mat_free(&want);
    }
    mat_free(&in);
    mat_free(&out);
}

static void run_rope_3d(const JsonValue *cs, const char *name, double tol,
                        int inverse)
{
    Mat in = mat_arg(cs, "rows", name);
    int grid_t = jint(need(cs, "grid_t", name));
    int grid_h = jint(need(cs, "grid_h", name));
    int grid_w = jint(need(cs, "grid_w", name));
    double base = jnum(need(cs, "base", name));
    Mat out;
    int rc;

    out.rows = in.rows;
    out.cols = in.cols;
    out.v = (double *)calloc((size_t)(in.rows * in.cols > 0
                                          ? in.rows * in.cols
                                          : 1), sizeof(double));
    if (!out.v)
        die("out of memory");
    rc = inverse
             ? ai_rope_3d_inverse(in.v, in.rows, in.cols, grid_t, grid_h,
                                  grid_w, base, out.v)
             : ai_rope_3d(in.v, in.rows, in.cols, grid_t, grid_h, grid_w,
                          base, out.v);
    check_int("rc", rc, 0);
    if (rc == 0) {
        Mat want = mat_arg(cs, "out", name);
        check_mat("out", &out, &want, tol);
        mat_free(&want);
    }
    mat_free(&in);
    mat_free(&out);
}

/* 错误用例：按 op 构造参数，只要求 C 侧返回 -1。 */
static void run_error(const JsonValue *cs, const char *name)
{
    const JsonValue *args = need(cs, "args", name);
    const char *op = json_str(need(cs, "op", name), "");
    int rc = 0;

    if (strcmp(op, "abf") == 0) {
        double out[1];
        int n = -1;
        rc = ai_abf_inv_freqs(jint(need(args, "d_k", name)),
                              jnum(need(args, "base", name)),
                              jnum(need(args, "factor", name)), out, &n);
    } else if (strcmp(op, "longrope") == 0) {
        double out[1];
        int n = -1;
        rc = ai_longrope_inv_freqs(jint(need(args, "d_k", name)),
                                   jnum(need(args, "base", name)),
                                   jnum(need(args, "factor", name)),
                                   jint(need(args, "original_max", name)),
                                   jnum(need(args, "low_fraction", name)),
                                   out, &n);
    } else if (strcmp(op, "rope_2d") == 0 ||
               strcmp(op, "rope_2d_inverse") == 0 ||
               strcmp(op, "rope_3d") == 0 ||
               strcmp(op, "rope_3d_inverse") == 0) {
        Mat in = mat_arg(args, "rows", name);
        Mat out;
        out.rows = in.rows;
        out.cols = in.cols;
        out.v = (double *)calloc((size_t)(in.rows * in.cols > 0
                                              ? in.rows * in.cols
                                              : 1), sizeof(double));
        if (!out.v)
            die("out of memory");
        if (strcmp(op, "rope_2d") == 0)
            rc = ai_rope_2d(in.v, in.rows, in.cols,
                            jint(need(args, "grid_h", name)),
                            jint(need(args, "grid_w", name)),
                            jnum(need(args, "base", name)), out.v);
        else if (strcmp(op, "rope_2d_inverse") == 0)
            rc = ai_rope_2d_inverse(in.v, in.rows, in.cols,
                                    jint(need(args, "grid_h", name)),
                                    jint(need(args, "grid_w", name)),
                                    jnum(need(args, "base", name)), out.v);
        else if (strcmp(op, "rope_3d") == 0)
            rc = ai_rope_3d(in.v, in.rows, in.cols,
                            jint(need(args, "grid_t", name)),
                            jint(need(args, "grid_h", name)),
                            jint(need(args, "grid_w", name)),
                            jnum(need(args, "base", name)), out.v);
        else
            rc = ai_rope_3d_inverse(in.v, in.rows, in.cols,
                                    jint(need(args, "grid_t", name)),
                                    jint(need(args, "grid_h", name)),
                                    jint(need(args, "grid_w", name)),
                                    jnum(need(args, "base", name)), out.v);
        mat_free(&in);
        mat_free(&out);
    } else {
        fprintf(stderr, "[compare_rope_ext] 未知 error op: %s\n", op);
        exit(2);
    }
    check_int("rc(-1)", rc, -1);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "rope_ext_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    const JsonValue *cases;
    if (!root) {
        fprintf(stderr, "[compare_rope_ext] 读取 %s 失败: %s\n", path, err);
        return 2;
    }
    cases = json_get(root, "cases");
    if (!cases || json_array_len(cases) <= 0) {
        fprintf(stderr, "[compare_rope_ext] 参考数据缺少 cases 数组\n");
        json_free(root);
        return 2;
    }

    for (int i = 0; i < json_array_len(cases); i++) {
        const JsonValue *cs = json_at(cases, i);
        const char *kind = json_str(json_get(cs, "kind"), "");
        const char *name = json_str(json_get(cs, "name"), "(unnamed)");
        double tol = case_tol(cs);

        if (strcmp(kind, "abf") == 0)
            run_abf(cs, name, tol);
        else if (strcmp(kind, "longrope") == 0)
            run_longrope(cs, name, tol);
        else if (strcmp(kind, "rope_2d") == 0)
            run_rope_2d(cs, name, tol, 0);
        else if (strcmp(kind, "rope_2d_inverse") == 0)
            run_rope_2d(cs, name, tol, 1);
        else if (strcmp(kind, "rope_3d") == 0)
            run_rope_3d(cs, name, tol, 0);
        else if (strcmp(kind, "rope_3d_inverse") == 0)
            run_rope_3d(cs, name, tol, 1);
        else if (strcmp(kind, "error") == 0)
            run_error(cs, name);
        else {
            fprintf(stderr, "[compare_rope_ext] 未知 kind: %s\n", kind);
            json_free(root);
            return 2;
        }
    }

    json_free(root);
    printf("=== compare_rope_ext：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    if (g_fail && g_worst_label[0])
        printf("    最大差异出现在 %s\n", g_worst_label);
    return g_fail ? 1 : 0;
}
