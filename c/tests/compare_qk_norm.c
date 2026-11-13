/*
 * compare_qk_norm.c —— QK-Norm / DynamicTanh 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_qk_norm [qk_norm_reference.json]
 *
 * 覆盖 qk_norm / qk_norm_backward / apply_qk_norm 与 DynamicTanh
 * （forward / backward / sgd_step / param_count）以及参数错误路径。
 *
 * 浮点比较策略与其它 compare_*.c 一致：先按位比较（memcmp），
 * 不一致再看绝对差是否 <= 参考用例里的 tol（1e-12）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../qk_norm.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_qk_norm] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_qk_norm] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/qk_norm_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

/* 二维矩阵：参考 JSON 里是 [[...], [...]]，C 侧展平成 rows*cols。 */
typedef struct {
    int rows;
    int cols;
    double *v;
} Mat;

static Mat mat_read(const JsonValue *j, const char *who)
{
    Mat m;
    int i, k;
    if (json_array_len(j) > 0) {
        m.rows = json_array_len(j);
        m.cols = json_array_len(json_at(j, 0));
    } else {
        m.rows = 0;
        m.cols = 0;
    }
    m.v = (double *)malloc(sizeof(double) * (size_t)(m.rows * m.cols + 1));
    if (!m.v) die("out of memory");
    for (i = 0; i < m.rows; i++) {
        const JsonValue *row = json_at(j, i);
        if (json_array_len(row) != m.cols) {
            fprintf(stderr, "[compare_qk_norm] %s 的矩阵各行长度不一致\n", who);
            exit(2);
        }
        for (k = 0; k < m.cols; k++)
            m.v[(size_t)i * (size_t)m.cols + (size_t)k] = jnum(json_at(row, k));
    }
    return m;
}

static void mat_free(Mat *m)
{
    if (!m) return;
    free(m->v);
    m->v = NULL;
    m->rows = m->cols = 0;
}

/* 一维向量；空向量也返回非 NULL（C 侧要求指针非空）。 */
static double *vec_read(const JsonValue *j, int *n_out)
{
    const int n = json_array_len(j);
    double *v = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    int i;
    if (!v) die("out of memory");
    for (i = 0; i < n; i++) v[i] = jnum(json_at(j, i));
    *n_out = n;
    return v;
}

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) {
        g_exact++;
    } else {
        printf("  %-30s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double d = fabs(got - want);
        if (d > g_worst) g_worst = d;
        if (d <= tol) return;
        printf("  %-30s C=%.17g py=%.17g [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_vec(const char *label, const double *got, int got_n,
                      const double *want, int want_n, double tol)
{
    char buf[128];
    int i;
    snprintf(buf, sizeof buf, "%s.n", label);
    check_int(buf, got_n, want_n);
    for (i = 0; i < want_n; i++) {
        const double g = (got && i < got_n) ? got[i] : NAN;
        snprintf(buf, sizeof buf, "%s[%d]", label, i);
        check_num(buf, g, want[i], tol);
    }
}

static void check_mat(const char *label, const double *got, int got_rows,
                      int got_cols, const Mat *want, double tol)
{
    char buf[128];
    int i, k;
    snprintf(buf, sizeof buf, "%s.rows", label);
    check_int(buf, got_rows, want->rows);
    snprintf(buf, sizeof buf, "%s.cols", label);
    check_int(buf, got_cols, want->cols);
    for (i = 0; i < want->rows; i++) {
        for (k = 0; k < want->cols; k++) {
            const double g = (got && i < got_rows && k < got_cols)
                                 ? got[(size_t)i * (size_t)got_cols + (size_t)k]
                                 : NAN;
            snprintf(buf, sizeof buf, "%s[%d][%d]", label, i, k);
            check_num(buf, g,
                      want->v[(size_t)i * (size_t)want->cols + (size_t)k], tol);
        }
    }
}

/* ------------------------------ 成功用例 ------------------------------ */

static void case_qk_norm(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *jrows = need(cs, "rows", name);
    const JsonValue *jgamma = need(cs, "gamma", name);
    const double eps = jnum(need(cs, "eps", name));
    Mat rows = mat_read(jrows, name);
    Mat want_out = mat_read(need(cs, "out", name), name);
    Mat want_xhat = mat_read(need(cs, "xhat", name), name);
    double *gamma;
    double *want_rms;
    int gamma_n = 0;
    int want_rms_n = 0;
    int rms_n = 0;
    AiQkNormResult res;

    gamma = vec_read(jgamma, &gamma_n);
    want_rms = vec_read(need(cs, "rms", name), &want_rms_n);
    if (gamma_n != rows.cols)
        die("参考文件里 gamma 长度与 rows 宽度不一致");

    if (ai_qk_norm(rows.v, rows.rows, rows.cols, gamma, eps, &res) != 0)
        die("ai_qk_norm 失败");
    check_mat("out", res.out, res.n_rows, res.width, &want_out, tol);
    check_mat("xhat", res.xhat, res.n_rows, res.width, &want_xhat, tol);
    rms_n = res.n_rows;
    check_vec("rms", res.rms, rms_n, want_rms, want_rms_n, tol);

    if (json_get(cs, "backward")) {
        const JsonValue *b = json_get(cs, "backward");
        Mat dout = mat_read(need(b, "dout", name), name);
        Mat want_ig = mat_read(need(b, "input_grad", name), name);
        int want_gg_n = 0;
        double *want_gg = vec_read(need(b, "gamma_grad", name), &want_gg_n);
        double *gg = (double *)malloc(sizeof(double) * (size_t)(res.width + 1));
        double *ig = (double *)malloc(
            sizeof(double) * (size_t)(res.n_rows * res.width + 1));
        if (!gg || !ig) die("out of memory");
        if (dout.rows != res.n_rows || dout.cols != res.width)
            die("参考文件里 dout 形状与 rows 不一致");
        if (ai_qk_norm_backward(dout.v, dout.rows, dout.cols, &res, gg, ig) != 0)
            die("ai_qk_norm_backward 失败");
        check_vec("gamma_grad", gg, res.width, want_gg, want_gg_n, tol);
        check_mat("input_grad", ig, dout.rows, dout.cols, &want_ig, tol);
        free(gg);
        free(ig);
        free(want_gg);
        mat_free(&dout);
        mat_free(&want_ig);
    }

    ai_qk_norm_free(&res);
    free(gamma);
    free(want_rms);
    mat_free(&rows);
    mat_free(&want_out);
    mat_free(&want_xhat);
}

static void case_apply(const JsonValue *cs, const char *name, double tol)
{
    const double eps = jnum(need(cs, "eps", name));
    Mat q_rows = mat_read(need(cs, "q_rows", name), name);
    Mat k_rows = mat_read(need(cs, "k_rows", name), name);
    Mat want_q = mat_read(need(cs, "q", name), name);
    Mat want_k = mat_read(need(cs, "k", name), name);
    int qg_n = 0, kg_n = 0;
    double *q_gamma = vec_read(need(cs, "q_gamma", name), &qg_n);
    double *k_gamma = vec_read(need(cs, "k_gamma", name), &kg_n);
    AiQkNormResult q_out, k_out;

    if (qg_n != q_rows.cols || kg_n != k_rows.cols)
        die("参考文件里 gamma 长度与 rows 宽度不一致");
    if (ai_apply_qk_norm(q_rows.v, q_rows.rows, q_rows.cols, q_gamma,
                         k_rows.v, k_rows.rows, k_rows.cols, k_gamma, eps,
                         &q_out, &k_out) != 0)
        die("ai_apply_qk_norm 失败");
    check_mat("q", q_out.out, q_out.n_rows, q_out.width, &want_q, tol);
    check_mat("k", k_out.out, k_out.n_rows, k_out.width, &want_k, tol);

    ai_qk_norm_free(&q_out);
    ai_qk_norm_free(&k_out);
    free(q_gamma);
    free(k_gamma);
    mat_free(&q_rows);
    mat_free(&k_rows);
    mat_free(&want_q);
    mat_free(&want_k);
}

static void case_dyt(const JsonValue *cs, const char *name, double tol)
{
    const int d_model = jint(need(cs, "d_model", name));
    const double alpha = jnum(need(cs, "alpha", name));
    const JsonValue *forwards = need(cs, "forwards", name);
    const int n_fwd = json_array_len(forwards);
    AiDynamicTanh model;
    int f;

    if (ai_dyt_create(&model, d_model, alpha) != 0) die("ai_dyt_create 失败");
    check_int("param_count", ai_dyt_param_count(&model),
              jint(need(cs, "param_count", name)));

    for (f = 0; f < n_fwd; f++) {
        const JsonValue *fs = json_at(forwards, f);
        Mat rows = mat_read(need(fs, "rows", name), name);
        Mat want = mat_read(need(fs, "out", name), name);
        double *out = (double *)malloc(
            sizeof(double) * (size_t)(rows.rows * rows.cols + 1));
        if (!out) die("out of memory");
        if (ai_dyt_forward(&model, rows.v, rows.rows, rows.cols, out) != 0)
            die("ai_dyt_forward 失败");
        check_mat("dyt.out", out, rows.rows, rows.cols, &want, tol);
        free(out);
        mat_free(&rows);
        mat_free(&want);
    }

    if (json_get(cs, "backward")) {
        const JsonValue *b = json_get(cs, "backward");
        Mat dout = mat_read(need(b, "dout", name), name);
        Mat want_ig = mat_read(need(b, "input_grad", name), name);
        int gg_n = 0, gb_n = 0;
        double *want_gg = vec_read(need(b, "grad_gamma", name), &gg_n);
        double *want_gb = vec_read(need(b, "grad_beta", name), &gb_n);
        double *ig = (double *)malloc(
            sizeof(double) * (size_t)(dout.rows * dout.cols + 1));
        if (!ig) die("out of memory");
        if (ai_dyt_backward(&model, dout.v, dout.rows, dout.cols, ig) != 0)
            die("ai_dyt_backward 失败");
        check_mat("dyt.input_grad", ig, dout.rows, dout.cols, &want_ig, tol);
        check_num("dyt.grad_alpha", model.grad_alpha,
                  jnum(need(b, "grad_alpha", name)), tol);
        check_vec("dyt.grad_gamma", model.grad_gamma, d_model, want_gg, gg_n,
                  tol);
        check_vec("dyt.grad_beta", model.grad_beta, d_model, want_gb, gb_n,
                  tol);
        free(ig);
        free(want_gg);
        free(want_gb);
        mat_free(&dout);
        mat_free(&want_ig);
    }

    if (json_get(cs, "sgd")) {
        const JsonValue *s = json_get(cs, "sgd");
        const double lr = jnum(need(s, "lr", name));
        int g_n = 0, b_n = 0, gg_n = 0, gb_n = 0;
        double *want_gamma = vec_read(need(s, "gamma", name), &g_n);
        double *want_beta = vec_read(need(s, "beta", name), &b_n);
        double *want_gg = vec_read(need(s, "grad_gamma", name), &gg_n);
        double *want_gb = vec_read(need(s, "grad_beta", name), &gb_n);
        if (ai_dyt_sgd_step(&model, lr) != 0) die("ai_dyt_sgd_step 失败");
        check_num("dyt.alpha(after sgd)", model.alpha,
                  jnum(need(s, "alpha", name)), tol);
        check_vec("dyt.gamma(after sgd)", model.gamma, d_model, want_gamma, g_n,
                  tol);
        check_vec("dyt.beta(after sgd)", model.beta, d_model, want_beta, b_n,
                  tol);
        check_num("dyt.grad_alpha(after sgd)", model.grad_alpha,
                  jnum(need(s, "grad_alpha", name)), tol);
        check_vec("dyt.grad_gamma(after sgd)", model.grad_gamma, d_model,
                  want_gg, gg_n, tol);
        check_vec("dyt.grad_beta(after sgd)", model.grad_beta, d_model,
                  want_gb, gb_n, tol);
        free(want_gamma);
        free(want_beta);
        free(want_gg);
        free(want_gb);
    }

    ai_dyt_free(&model);
}

/* ------------------------------ 错误用例 ------------------------------ */

static void expect_rc(const char *what, int rc)
{
    check_int(what, rc, -1);
}

static void case_error(const JsonValue *cs, const char *name)
{
    const char *op = json_str(need(cs, "op", name), "?");
    const JsonValue *a = need(cs, "args", name);

    if (strcmp(op, "qk_norm") == 0) {
        Mat rows = mat_read(need(a, "rows", name), name);
        int gn = 0;
        double *gamma = vec_read(need(a, "gamma", name), &gn);
        AiQkNormResult res;
        const int rc = ai_qk_norm(rows.v, rows.rows, rows.cols, gamma,
                                  jnum(need(a, "eps", name)), &res);
        if (rc == 0) ai_qk_norm_free(&res);
        expect_rc("qk_norm.rc", rc);
        free(gamma);
        mat_free(&rows);
    } else if (strcmp(op, "qk_norm_backward") == 0) {
        Mat rows = mat_read(need(a, "rows", name), name);
        Mat dout = mat_read(need(a, "dout", name), name);
        int gn = 0;
        double *gamma = vec_read(need(a, "gamma", name), &gn);
        AiQkNormResult res;
        const int frc = ai_qk_norm(rows.v, rows.rows, rows.cols, gamma,
                                   jnum(need(a, "eps", name)), &res);
        check_int("qk_norm_backward.forward_rc", frc, 0);
        if (frc == 0) {
            double *gg = (double *)malloc(
                sizeof(double) * (size_t)(res.width + 1));
            double *ig = (double *)malloc(
                sizeof(double) * (size_t)(res.n_rows * res.width + 1));
            int rc;
            if (!gg || !ig) die("out of memory");
            rc = ai_qk_norm_backward(dout.v, dout.rows, dout.cols, &res, gg,
                                     ig);
            expect_rc("qk_norm_backward.rc", rc);
            free(gg);
            free(ig);
            ai_qk_norm_free(&res);
        }
        free(gamma);
        mat_free(&rows);
        mat_free(&dout);
    } else if (strcmp(op, "apply_qk_norm") == 0) {
        Mat q_rows = mat_read(need(a, "q_rows", name), name);
        Mat k_rows = mat_read(need(a, "k_rows", name), name);
        int qg_n = 0, kg_n = 0;
        double *q_gamma = vec_read(need(a, "q_gamma", name), &qg_n);
        double *k_gamma = vec_read(need(a, "k_gamma", name), &kg_n);
        AiQkNormResult q_out, k_out;
        const int rc = ai_apply_qk_norm(
            q_rows.v, q_rows.rows, q_rows.cols, q_gamma, k_rows.v, k_rows.rows,
            k_rows.cols, k_gamma, jnum(need(a, "eps", name)), &q_out, &k_out);
        if (rc == 0) {
            ai_qk_norm_free(&q_out);
            ai_qk_norm_free(&k_out);
        }
        expect_rc("apply_qk_norm.rc", rc);
        free(q_gamma);
        free(k_gamma);
        mat_free(&q_rows);
        mat_free(&k_rows);
    } else if (strcmp(op, "dyt_create") == 0) {
        AiDynamicTanh model;
        const int rc = ai_dyt_create(&model, jint(need(a, "d_model", name)),
                                     jnum(need(a, "alpha", name)));
        if (rc == 0) ai_dyt_free(&model);
        expect_rc("dyt_create.rc", rc);
    } else if (strcmp(op, "dyt_forward") == 0) {
        AiDynamicTanh model;
        Mat rows = mat_read(need(a, "rows", name), name);
        int crc = ai_dyt_create(&model, jint(need(a, "d_model", name)),
                                jnum(need(a, "alpha", name)));
        check_int("dyt_forward.create_rc", crc, 0);
        if (crc == 0) {
            double *out = (double *)malloc(
                sizeof(double) * (size_t)(rows.rows * rows.cols + 1));
            int rc;
            if (!out) die("out of memory");
            rc = ai_dyt_forward(&model, rows.v, rows.rows, rows.cols, out);
            expect_rc("dyt_forward.rc", rc);
            free(out);
            ai_dyt_free(&model);
        }
        mat_free(&rows);
    } else if (strcmp(op, "dyt_backward") == 0) {
        const JsonValue *fwd = json_get(a, "forward_rows");
        AiDynamicTanh model;
        Mat dout = mat_read(need(a, "dout", name), name);
        int crc = ai_dyt_create(&model, jint(need(a, "d_model", name)),
                                jnum(need(a, "alpha", name)));
        check_int("dyt_backward.create_rc", crc, 0);
        if (crc == 0) {
            if (fwd && fwd->kind != JSON_NULL) {
                Mat fwd_rows = mat_read(fwd, name);
                double *fout = (double *)malloc(
                    sizeof(double) * (size_t)(fwd_rows.rows * fwd_rows.cols + 1));
                int frc;
                if (!fout) die("out of memory");
                frc = ai_dyt_forward(&model, fwd_rows.v, fwd_rows.rows,
                                     fwd_rows.cols, fout);
                check_int("dyt_backward.forward_rc", frc, 0);
                free(fout);
                mat_free(&fwd_rows);
            }
            {
                double *ig = (double *)malloc(
                    sizeof(double) * (size_t)(dout.rows * dout.cols + 1));
                int rc;
                if (!ig) die("out of memory");
                rc = ai_dyt_backward(&model, dout.v, dout.rows, dout.cols, ig);
                expect_rc("dyt_backward.rc", rc);
                free(ig);
            }
            ai_dyt_free(&model);
        }
        mat_free(&dout);
    } else if (strcmp(op, "dyt_sgd_step") == 0) {
        AiDynamicTanh model;
        Mat rows = mat_read(need(a, "rows", name), name);
        Mat dout = mat_read(need(a, "dout", name), name);
        int crc = ai_dyt_create(&model, jint(need(a, "d_model", name)),
                                jnum(need(a, "alpha", name)));
        check_int("dyt_sgd_step.create_rc", crc, 0);
        if (crc == 0) {
            double *out = (double *)malloc(
                sizeof(double) * (size_t)(rows.rows * rows.cols + 1));
            double *ig = (double *)malloc(
                sizeof(double) * (size_t)(dout.rows * dout.cols + 1));
            int frc;
            if (!out || !ig) die("out of memory");
            frc = ai_dyt_forward(&model, rows.v, rows.rows, rows.cols, out);
            check_int("dyt_sgd_step.forward_rc", frc, 0);
            if (frc == 0) {
                int brc = ai_dyt_backward(&model, dout.v, dout.rows, dout.cols,
                                          ig);
                check_int("dyt_sgd_step.backward_rc", brc, 0);
                if (brc == 0) {
                    const int rc = ai_dyt_sgd_step(
                        &model, jnum(need(a, "lr", name)));
                    expect_rc("dyt_sgd_step.rc", rc);
                }
            }
            free(out);
            free(ig);
            ai_dyt_free(&model);
        }
        mat_free(&rows);
        mat_free(&dout);
    } else {
        fprintf(stderr, "[compare_qk_norm] 未知 error op \"%s\"\n", op);
        exit(2);
    }
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "qk_norm_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    const JsonValue *cases;
    int n_cases;
    int c;

    if (!root) {
        fprintf(stderr, "[compare_qk_norm] 读不了 %s：%s\n"
                        "请先运行 python tools/qk_norm_reference.py\n",
                path, err);
        return 2;
    }
    cases = need(root, "cases", "root");
    n_cases = json_array_len(cases);
    printf("[compare_qk_norm] %s：%d 个用例\n", path, n_cases);

    for (c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 1e-12);
        printf("\n用例 %-22s kind=%s\n", name, kind);
        if (strcmp(kind, "qk_norm") == 0) case_qk_norm(cs, name, tol);
        else if (strcmp(kind, "apply_qk_norm") == 0) case_apply(cs, name, tol);
        else if (strcmp(kind, "dyt") == 0) case_dyt(cs, name, tol);
        else if (strcmp(kind, "error") == 0) case_error(cs, name);
        else die("未知 kind");
    }
    printf("\n=== compare_qk_norm：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
