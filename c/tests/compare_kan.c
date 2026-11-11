/*
 * compare_kan.c —— KAN（Kolmogorov-Arnold Network）与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_kan [kan_reference.json]
 *
 * 参考值由 tools/kan_reference.py 从 AI.py 现场导出（参数由公式注入，
 * 不涉及随机数）；本程序用相同参数跑 c/kan.c，逐项比较前向输出、
 * 反向梯度、SGD 一步后的参数与参数量。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../kan.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_kan] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key,
                             const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_kan] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/kan_reference.py\n", who, key);
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
        printf("  %-28s C=%.17g py=%.17g 相对差=%.3e [不等]\n", what, got,
               want, d);
        g_fail++;
    }
}

static void check_doubles(const char *what, const double *got,
                          const double *want, int n, double tol)
{
    for (int i = 0; i < n; i++) {
        char label[160];
        snprintf(label, sizeof label, "%s[%d]", what, i);
        check_num(label, got[i], want[i], tol);
    }
}

/* 把 reference 中的 params 注入 C 层；返回 0 表示成功。 */
static int load_layer_params(AiKANLayer *layer, const JsonValue *params,
                             const char *who)
{
    const int in = layer->in_dim;
    const int outd = layer->out_dim;
    const int nb = layer->num_basis;
    double *coeff = darr(need(params, "coeff", who), outd * in * nb);
    double *biases = darr(need(params, "biases", who), outd);
    memcpy(layer->coeff, coeff, sizeof(double) * (size_t)(outd * in * nb));
    memcpy(layer->biases, biases, sizeof(double) * (size_t)outd);
    free(coeff);
    free(biases);
    return 0;
}

static void case_layer(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *params = need(cs, "params", name);
    const JsonValue *grads = need(cs, "grads", name);
    const JsonValue *post = need(cs, "post", name);
    const int in = jint(need(params, "in_dim", name));
    const int outd = jint(need(params, "out_dim", name));
    const int num_grid = jint(need(params, "num_grid", name));
    const int order = jint(need(params, "spline_order", name));
    const double grid_range = jnum(need(params, "grid_range", name));
    const int nb = jint(need(params, "num_basis", name));
    const int n = jint(need(cs, "n", name));
    AiKANLayer *layer = NULL;

    double *xs = darr(need(cs, "xs", name), n * in);
    double *dout = darr(need(cs, "dout", name), n * outd);
    double *want_out = darr(need(cs, "out", name), n * outd);
    double *want_dx = darr(need(cs, "dx", name), n * in);
    double *want_gc = darr(need(grads, "grad_coeff", name), outd * in * nb);
    double *want_gb = darr(need(grads, "grad_biases", name), outd);
    double *want_pc = darr(need(post, "coeff", name), outd * in * nb);
    double *want_pb = darr(need(post, "biases", name), outd);
    double *out = (double *)malloc(sizeof(double) * (size_t)(n * outd));
    double *dx = (double *)malloc(sizeof(double) * (size_t)(n * in));
    if (!out || !dx) die("out of memory");

    if (ai_kan_layer_create(&layer, in, outd, num_grid, order, grid_range) != 0)
        die("ai_kan_layer_create 失败");
    if (load_layer_params(layer, params, name) != 0) die("load params 失败");
    check_num("param_count", (double)ai_kan_layer_param_count(layer),
              (double)jint(need(cs, "param_count", name)), 0.0);

    if (ai_kan_layer_forward(layer, xs, n, out) != 0) die("forward 失败");
    check_doubles("out", out, want_out, n * outd, tol);

    if (ai_kan_layer_backward(layer, dout, n, dx) != 0) die("backward 失败");
    check_doubles("dx", dx, want_dx, n * in, tol);
    check_doubles("grad_coeff", layer->grad_coeff, want_gc, outd * in * nb,
                  tol);
    check_doubles("grad_biases", layer->grad_biases, want_gb, outd, tol);

    if (ai_kan_layer_sgd_step(layer, 0.05) != 0) die("sgd_step 失败");
    check_doubles("post_coeff", layer->coeff, want_pc, outd * in * nb, tol);
    check_doubles("post_biases", layer->biases, want_pb, outd, tol);

    ai_kan_layer_free(layer);
    free(xs); free(dout); free(want_out); free(want_dx);
    free(want_gc); free(want_gb); free(want_pc); free(want_pb);
    free(out); free(dx);
}

static void case_model(const JsonValue *cs, const char *name, double tol)
{
    const JsonValue *sizes_v = need(cs, "sizes", name);
    const int n_sizes = json_array_len(sizes_v);
    const int num_grid = jint(need(cs, "num_grid", name));
    const int order = jint(need(cs, "spline_order", name));
    const double grid_range = jnum(need(cs, "grid_range", name));
    const int n = jint(need(cs, "n", name));
    const JsonValue *layers_params = need(cs, "layers_params", name);
    const JsonValue *grads_v = need(cs, "grads", name);
    const JsonValue *post_v = need(cs, "post", name);
    int *sizes = (int *)malloc(sizeof(int) * (size_t)n_sizes);
    AiKAN *kan = NULL;
    const int in_dim = jint(json_at(sizes_v, 0));
    const int out_dim = jint(json_at(sizes_v, n_sizes - 1));
    double *xs;
    double *ys;
    double *want_pred;
    double *want_dx;
    double loss = 0.0;
    double *pred;
    double *dx;

    if (!sizes) die("out of memory");
    for (int i = 0; i < n_sizes; i++) sizes[i] = jint(json_at(sizes_v, i));
    xs = darr(need(cs, "xs", name), n * in_dim);
    ys = darr(need(cs, "ys", name), n * out_dim);
    want_pred = darr(need(cs, "pred", name), n * out_dim);
    want_dx = darr(need(cs, "dx", name), n * in_dim);
    pred = (double *)malloc(sizeof(double) * (size_t)(n * out_dim));
    dx = (double *)malloc(sizeof(double) * (size_t)(n * in_dim));
    if (!pred || !dx) die("out of memory");

    if (ai_kan_create(&kan, sizes, n_sizes, num_grid, order, grid_range) != 0)
        die("ai_kan_create 失败");
    for (int idx = 0; idx < kan->n_layers; idx++)
        load_layer_params(kan->layers[idx], json_at(layers_params, idx), name);
    check_num("param_count", (double)ai_kan_param_count(kan),
              (double)jint(need(cs, "param_count", name)), 0.0);

    if (ai_kan_forward_batch(kan, xs, n, pred) != 0) die("forward_batch 失败");
    check_doubles("pred", pred, want_pred, n * out_dim, tol);

    ai_kan_zero_grad(kan);
    if (ai_kan_loss_and_backward(kan, xs, ys, n, &loss, dx) != 0)
        die("loss_and_backward 失败");
    check_num("loss", loss, jnum(need(cs, "loss", name)), tol);
    check_doubles("dx", dx, want_dx, n * in_dim, tol);
    for (int idx = 0; idx < kan->n_layers; idx++) {
        const AiKANLayer *layer = kan->layers[idx];
        const JsonValue *g = json_at(grads_v, idx);
        const int nb = layer->num_basis;
        const int cnt = layer->out_dim * layer->in_dim * nb;
        char label[96];
        snprintf(label, sizeof label, "grad_coeff_l%d", idx);
        check_doubles(label, layer->grad_coeff,
                      darr(need(g, "grad_coeff", name), cnt), cnt, tol);
        snprintf(label, sizeof label, "grad_biases_l%d", idx);
        check_doubles(label, layer->grad_biases,
                      darr(need(g, "grad_biases", name), layer->out_dim),
                      layer->out_dim, tol);
    }

    if (ai_kan_sgd_step(kan, 0.02) != 0) die("sgd_step 失败");
    for (int idx = 0; idx < kan->n_layers; idx++) {
        const AiKANLayer *layer = kan->layers[idx];
        const JsonValue *p = json_at(post_v, idx);
        const int nb = layer->num_basis;
        const int cnt = layer->out_dim * layer->in_dim * nb;
        char label[96];
        snprintf(label, sizeof label, "post_coeff_l%d", idx);
        check_doubles(label, layer->coeff,
                      darr(need(p, "coeff", name), cnt), cnt, tol);
        snprintf(label, sizeof label, "post_biases_l%d", idx);
        check_doubles(label, layer->biases,
                      darr(need(p, "biases", name), layer->out_dim),
                      layer->out_dim, tol);
    }

    ai_kan_free(kan);
    free(sizes); free(xs); free(ys); free(want_pred); free(want_dx);
    free(pred); free(dx);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "kan_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_kan] 读不了 %s：%s\n"
                        "请先运行 python tools/kan_reference.py\n", path, err);
        return 2;
    }
    {
        const JsonValue *cases = need(root, "cases", "root");
        const int n_cases = json_array_len(cases);
        printf("[compare_kan] %s：%d 个用例\n", path, n_cases);
        for (int c = 0; c < n_cases; c++) {
            const JsonValue *cs = json_at(cases, c);
            const char *name = json_str(json_get(cs, "name"), "?");
            const char *kind = json_str(json_get(cs, "kind"), "?");
            const double tol = json_num(json_get(cs, "tol"), 0.0);
            printf("\n用例 %-20s kind=%s\n", name, kind);
            if (strcmp(kind, "layer") == 0) case_layer(cs, name, tol);
            else if (strcmp(kind, "model") == 0) case_model(cs, name, tol);
            else die("未知 kind");
        }
    }
    printf("\n=== compare_kan：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
