/*
 * compare_egnn.c —— C 侧 EGNN 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_egnn [egnn_reference.json]
 *
 * 参考用例（tools/egnn_reference.py 生成）提供：
 *   - 模型结构 + 每层 phi_e/phi_x/phi_h 与 readout 初始权重
 *   - 一张确定性图（h、x、target）
 *   - 前向输出、loss、一次 sgd_step 后的输出、param_count
 * 比较：输出/loss（相对容差 1e-10）、参数量；另检查旋转/平移/置换等变。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ai.h"
#include "../egnn.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_egnn] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_egnn] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/egnn_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static double *darr(const JsonValue *v, int n)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    return out;
}

/* n×cols 的二维数组扁平化为行主序 */
static double *dmat(const JsonValue *v, int n, int cols)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)((n * cols) ? (n * cols) : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) {
        const JsonValue *row = json_at(v, i);
        for (int j = 0; j < cols; j++)
            out[(size_t)i * cols + j] = jnum(json_at(row, j));
    }
    return out;
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
        const double d = fabs(got - want) / scale;
        if (fabs(got - want) > g_worst) g_worst = fabs(got - want);
        if (d <= tol) return;
        printf("  %-30s C=%.17g py=%.17g 相对差=%.3e [不等]\n", what, got, want, d);
        g_fail++;
    }
}

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-30s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

static void load_mlp(MLP *m, const JsonValue *layers, const char *who)
{
    const int n = json_array_len(layers);
    if (n != m->n_layers) {
        fprintf(stderr, "[%s] 层数不一致：C=%d JSON=%d\n", who, m->n_layers, n);
        exit(2);
    }
    for (int i = 0; i < n; i++) {
        const JsonValue *lv = json_at(layers, i);
        const JsonValue *wv = need(lv, "w", who);
        const JsonValue *bv = need(lv, "b", who);
        Layer *l = &m->layers[i];
        if (json_array_len(wv) != l->fan_out
            || json_array_len(json_at(wv, 0)) != l->fan_in
            || json_array_len(bv) != l->fan_out) {
            fprintf(stderr, "[%s] 第 %d 层形状不一致\n", who, i);
            exit(2);
        }
        for (int r = 0; r < l->fan_out; r++) {
            const JsonValue *row = json_at(wv, r);
            for (int c = 0; c < l->fan_in; c++)
                l->W[(size_t)r * l->fan_in + c] = jnum(json_at(row, c));
        }
        for (int r = 0; r < l->fan_out; r++)
            l->b[r] = jnum(json_at(bv, r));
    }
}

static void check_grads(MLP *m, const JsonValue *g, const char *who, double tol)
{
    const int nl = json_array_len(g);
    if (nl != m->n_layers) die("梯度层数不一致");
    char name[96];
    for (int i = 0; i < nl; i++) {
        const JsonValue *lv = json_at(g, i);
        const JsonValue *wv = need(lv, "w", who);
        const JsonValue *bv = need(lv, "b", who);
        Layer *l = &m->layers[i];
        const int nw = l->fan_out * l->fan_in;
        for (int k = 0; k < nw; k++) {
            snprintf(name, sizeof name, "%s.%d.gw[%d]", who, i, k);
            check_num(name, l->gW[k], jnum(json_at(wv, k)), tol);
        }
        for (int k = 0; k < l->fan_out; k++) {
            snprintf(name, sizeof name, "%s.%d.gb[%d]", who, i, k);
            check_num(name, l->gb[k], jnum(json_at(bv, k)), tol);
        }
    }
}

static void load_model(EgnnModel *m, const JsonValue *root)
{
    const JsonValue *layers = need(root, "layers", "root");
    const int nl = json_array_len(layers);
    for (int i = 0; i < nl; i++) {
        const JsonValue *lv = json_at(layers, i);
        EgnnLayer *l = egnn_model_layer_mut(m, i);
        if (!l) die("egnn_model_layer_mut 失败");
        load_mlp(egnn_layer_phi_e(l), need(lv, "phi_e", "layer"), "phi_e");
        load_mlp(egnn_layer_phi_x(l), need(lv, "phi_x", "layer"), "phi_x");
        load_mlp(egnn_layer_phi_h(l), need(lv, "phi_h", "layer"), "phi_h");
    }
    load_mlp(egnn_model_readout_mut(m), need(root, "readout", "root"), "readout");
}

static void check_forward(EgnnModel *m, const double *h, const double *x, int n,
                          const JsonValue *want, const char *tag, double tol)
{
    double *h_out = (double *)malloc(sizeof(double) * (size_t)n * (size_t)egnn_model_d_model(m));
    double *x_out = (double *)malloc(sizeof(double) * (size_t)n * 3);
    double scalar = 0.0;
    if (!h_out || !x_out) die("out of memory");
    if (egnn_model_forward(m, h, x, n, h_out, x_out, &scalar) != 0)
        die("egnn_model_forward 失败");

    char name[64];
    const JsonValue *wh = need(want, "h_out", tag);
    const JsonValue *wx = need(want, "x_out", tag);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < egnn_model_d_model(m); k++) {
            snprintf(name, sizeof name, "%s.h_out[%d][%d]", tag, i, k);
            check_num(name, h_out[(size_t)i * egnn_model_d_model(m) + k],
                      jnum(json_at(json_at(wh, i), k)), tol);
        }
        for (int k = 0; k < 3; k++) {
            snprintf(name, sizeof name, "%s.x_out[%d][%d]", tag, i, k);
            check_num(name, x_out[(size_t)i * 3 + k],
                      jnum(json_at(json_at(wx, i), k)), tol);
        }
    }
    snprintf(name, sizeof name, "%s.scalar", tag);
    check_num(name, scalar, jnum(need(want, "scalar", tag)), tol);
    free(h_out);
    free(x_out);
}

/* 旋转/平移等变 + 置换等变 */
static void check_equivariance(EgnnModel *m, const double *h, const double *x, int n,
                               const JsonValue *after, double tol)
{
    const double c = 0.8660254037844387; /* cos(30°) */
    const double s = 0.5;                /* sin(30°) */
    const double t[3] = {0.7, -0.3, 1.1};

    double *x_rot = (double *)malloc(sizeof(double) * (size_t)n * 3);
    double *x_base = (double *)malloc(sizeof(double) * (size_t)n * 3);
    double *h_out = (double *)malloc(sizeof(double) * (size_t)n * (size_t)egnn_model_d_model(m));
    double *x_out = (double *)malloc(sizeof(double) * (size_t)n * 3);
    double *h_out2 = (double *)malloc(sizeof(double) * (size_t)n * (size_t)egnn_model_d_model(m));
    double *x_out2 = (double *)malloc(sizeof(double) * (size_t)n * 3);
    if (!x_rot || !x_base || !h_out || !x_out || !h_out2 || !x_out2)
        die("out of memory");

    for (int i = 0; i < n; i++) {
        const double xx = x[(size_t)i * 3 + 0];
        const double yy = x[(size_t)i * 3 + 1];
        const double zz = x[(size_t)i * 3 + 2];
        x_rot[(size_t)i * 3 + 0] = c * xx - s * yy + t[0];
        x_rot[(size_t)i * 3 + 1] = s * xx + c * yy + t[1];
        x_rot[(size_t)i * 3 + 2] = zz + t[2];
        x_base[(size_t)i * 3 + 0] = c * xx - s * yy;
        x_base[(size_t)i * 3 + 1] = s * xx + c * yy;
        x_base[(size_t)i * 3 + 2] = zz;
    }

    double scalar = 0.0, scalar_rot = 0.0;
    if (egnn_model_forward(m, h, x, n, h_out, x_out, &scalar) != 0) die("forward 失败");
    if (egnn_model_forward(m, h, x_rot, n, h_out2, x_out2, &scalar_rot) != 0)
        die("forward 失败");

    char name[80];
    check_num("rot.scalar", scalar_rot, scalar, tol);

    /* h' 对旋转/平移不变 */
    for (int i = 0; i < n; i++)
        for (int k = 0; k < egnn_model_d_model(m); k++) {
            snprintf(name, sizeof name, "rot.h_out[%d][%d]", i, k);
            check_num(name, h_out2[(size_t)i * egnn_model_d_model(m) + k],
                      h_out[(size_t)i * egnn_model_d_model(m) + k], tol);
        }

    /* x' 随坐标旋转/平移 */
    for (int i = 0; i < n; i++) {
        const double bx = x_out[(size_t)i * 3 + 0];
        const double by = x_out[(size_t)i * 3 + 1];
        const double bz = x_out[(size_t)i * 3 + 2];
        snprintf(name, sizeof name, "rot.x_out[%d][0]", i);
        check_num(name, x_out2[(size_t)i * 3 + 0], c * bx - s * by + t[0], tol);
        snprintf(name, sizeof name, "rot.x_out[%d][1]", i);
        check_num(name, x_out2[(size_t)i * 3 + 1], s * bx + c * by + t[1], tol);
        snprintf(name, sizeof name, "rot.x_out[%d][2]", i);
        check_num(name, x_out2[(size_t)i * 3 + 2], bz + t[2], tol);
    }

    /* 置换等变：节点顺序 [2,0,1] */
    {
        const int perm[3] = {2, 0, 1};
        double *h_p = (double *)malloc(sizeof(double) * (size_t)n * egnn_model_d_model(m));
        double *x_p = (double *)malloc(sizeof(double) * (size_t)n * 3);
        if (!h_p || !x_p) die("out of memory");
        for (int i = 0; i < n; i++) {
            const int src = perm[i];
            for (int k = 0; k < egnn_model_d_model(m); k++)
                h_p[(size_t)i * egnn_model_d_model(m) + k] = h[(size_t)src * egnn_model_d_model(m) + k];
            for (int k = 0; k < 3; k++)
                x_p[(size_t)i * 3 + k] = x[(size_t)src * 3 + k];
        }
        double scalar_p = 0.0;
        if (egnn_model_forward(m, h_p, x_p, n, h_out2, x_out2, &scalar_p) != 0)
            die("forward 失败");
        check_num("perm.scalar", scalar_p, scalar, tol);
        for (int i = 0; i < n; i++) {
            const int src = perm[i];
            for (int k = 0; k < egnn_model_d_model(m); k++) {
                snprintf(name, sizeof name, "perm.h_out[%d][%d]", i, k);
                check_num(name, h_out2[(size_t)i * egnn_model_d_model(m) + k],
                          h_out[(size_t)src * egnn_model_d_model(m) + k], tol);
            }
            for (int k = 0; k < 3; k++) {
                snprintf(name, sizeof name, "perm.x_out[%d][%d]", i, k);
                check_num(name, x_out2[(size_t)i * 3 + k],
                          x_out[(size_t)src * 3 + k], tol);
            }
        }
        free(h_p);
        free(x_p);
    }

    (void)after;
    free(x_rot); free(x_base); free(h_out); free(x_out); free(h_out2); free(x_out2);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "egnn_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_egnn] 读不了 %s：%s\n"
                        "请先运行 python tools/egnn_reference.py\n", path, err);
        return 2;
    }

    const int d_model = jint(need(root, "d_model", "root"));
    const int d_ff = jint(need(root, "d_ff", "root"));
    const int nl = jint(need(root, "num_layers", "root"));
    const double tol = 1e-10;

    EgnnModel *m = egnn_model_create(d_model, nl, d_ff);
    if (!m) die("egnn_model_create 失败");
    load_model(m, root);

    const JsonValue *hv = need(root, "h", "root");
    const JsonValue *xv = need(root, "x", "root");
    const int n = json_array_len(hv);
    double *h = dmat(hv, n, d_model);
    double *x = dmat(xv, n, 3);

    printf("[compare_egnn] %s：n=%d d_model=%d d_ff=%d layers=%d\n",
           path, n, d_model, d_ff, nl);

    check_int("param_count", egnn_model_param_count(m),
              (long)jnum(need(root, "param_count", "root")));

    check_forward(m, h, x, n, need(root, "forward", "root"), "forward", tol);

    egnn_model_zero_grad(m);
    const double loss = egnn_model_loss_and_backward(
        m, h, x, n, jnum(need(root, "target", "root")));
    check_num("loss", loss, jnum(need(root, "loss", "root")), tol);

    {
        const JsonValue *grads = need(root, "grads", "root");
        const JsonValue *gl = need(grads, "layers", "grads");
        for (int i = 0; i < nl; i++) {
            const JsonValue *lv = json_at(gl, i);
            EgnnLayer *l = egnn_model_layer_mut(m, i);
            check_grads(egnn_layer_phi_e(l), need(lv, "phi_e", "grad layer"), "ge", tol);
            check_grads(egnn_layer_phi_x(l), need(lv, "phi_x", "grad layer"), "gx", tol);
            check_grads(egnn_layer_phi_h(l), need(lv, "phi_h", "grad layer"), "gh", tol);
        }
        check_grads(egnn_model_readout_mut(m), need(grads, "readout", "grads"), "gr", tol);
    }

    egnn_model_sgd_step(m, jnum(need(root, "lr", "root")));
    check_forward(m, h, x, n, need(root, "after_step", "root"), "after", tol);

    check_equivariance(m, h, x, n, need(root, "after_step", "root"), tol);

    json_free(root);
    egnn_model_free(m);
    free(h);
    free(x);

    printf("\n[compare_egnn] %d 项检查：逐位一致 %d，失败 %d，最大绝对差 %.3e\n",
           g_checks, g_exact, g_fail, g_worst);
    return g_fail == 0 ? 0 : 1;
}
