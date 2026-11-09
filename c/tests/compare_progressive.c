/*
 * compare_progressive.c —— 读取 tools/progressive_reference.py 生成的参考数据，
 * 用 C 的 ProgressiveGenerator 重算前向（含 grow 与 fade），并逐项比对。
 *
 * 用法: compare_progressive [progressive_reference.json] [容差]
 */
#include "../json.h"
#include "../progressive.h"

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
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/progressive_reference.py）\n", key);
        exit(2);
    }
    return v;
}

static double *alloc_doubles(const JsonValue *arr, int n)
{
    if (!arr || json_array_len(arr) != n) {
        fprintf(stderr, "compare_progressive: 数组长度不符合预期\n");
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_progressive: 内存不足\n");
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
        printf("    %-22s 最大差异 %.3e [超出容差] index %d C=%.17g py=%.17g\n",
               what, worst, at, got[at], want[at]);
    } else {
        printf("    %-22s 最大差异 %.3e [OK]%s\n", what, worst, exact ? " (逐位相同)" : "");
    }
}

static void report_scalar(const char *what, double got, double want)
{
    report(what, &got, &want, 1);
}

static void load_layer(Layer *layer, const JsonValue *j)
{
    int fin = json_int(need(j, "fan_in"), 0);
    int fout = json_int(need(j, "fan_out"), 0);
    double *w = alloc_doubles(need(j, "w"), fin * fout);
    double *b = alloc_doubles(need(j, "b"), fout);
    memcpy(layer->W, w, (size_t)fin * fout * sizeof(double));
    memcpy(layer->b, b, (size_t)fout * sizeof(double));
    free(w);
    free(b);
}

static void load_convt(ConvTranspose2d *c, const JsonValue *j)
{
    int in = json_int(need(j, "in_channels"), 0);
    int out = json_int(need(j, "out_channels"), 0);
    int k = json_int(need(j, "k"), 0);
    int groups = json_int(need(j, "groups"), 1);
    double *w = alloc_doubles(need(j, "w"), in * out * k * k);
    double *b = alloc_doubles(need(j, "b"), out);
    (void)groups;
    convt2d_set_weights(c, w, b);
    free(w);
    free(b);
}

static void load_conv(Conv2d *c, const JsonValue *j)
{
    int in = json_int(need(j, "in_channels"), 0);
    int out = json_int(need(j, "out_channels"), 0);
    int k = json_int(need(j, "k"), 0);
    double *w = alloc_doubles(need(j, "w"), out * in * k * k);
    double *b = alloc_doubles(need(j, "b"), out);
    conv2d_set_weights(c, w, b);
    free(w);
    free(b);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "progressive_reference.json";
    if (argc > 2) {
        g_tol = atof(argv[2]);
    }
    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/progressive_reference.py %s\n", path);
        return 2;
    }
    const JsonValue *model = need(root, "model");
    int latent = json_int(need(model, "latent_dim"), 0);
    int base = json_int(need(model, "base"), 0);
    int oc = json_int(need(model, "out_channels"), 0);
    const JsonValue *chj = need(model, "channels");
    int nch = json_array_len(chj);
    int *channels = (int *)malloc((size_t)nch * sizeof(int));
    for (int i = 0; i < nch; i++) {
        channels[i] = json_int(json_at(chj, i), 0);
    }
    int pc_base = json_int(need(model, "param_count_base"), 0);

    ProgressiveGenerator *g = pg_create(latent, channels, nch, base, oc);
    if (!g) {
        fprintf(stderr, "pg_create 失败\n");
        json_free(root);
        free(channels);
        return 2;
    }
    int c_out = 0;
    int side = 0;
    pg_out_shape(g, &c_out, &side);
    report_scalar("out_channels", (double)c_out, (double)oc);
    report_scalar("side(base)", (double)side, (double)(base << (nch - 1)));
    report_scalar("param_count(base)", (double)pg_param_count(g), (double)pc_base);

    const JsonValue *b = need(root, "base");
    load_layer(&g->fc->layers[0], json_at(need(b, "fc"), 0));
    const JsonValue *convs = need(b, "convs");
    for (int i = 0; i < json_array_len(convs); i++) {
        load_convt(g->convs[i], json_at(convs, i));
    }
    const JsonValue *heads = need(b, "rgb_heads");
    for (int i = 0; i < json_array_len(heads); i++) {
        load_conv(g->rgb_heads[i], json_at(heads, i));
    }

    const JsonValue *fwd = need(root, "base_forward");
    double *z = alloc_doubles(need(fwd, "z"), latent);
    int out_n = oc * side * side;
    double *got = (double *)calloc((size_t)out_n, sizeof(double));
    double *want = alloc_doubles(need(fwd, "alpha1"), out_n);
    report_scalar("forward(alpha=1) 返回码", pg_forward(g, z, 1.0, got) == 0 ? 0.0 : 1.0, 0.0);
    report("forward alpha=1", got, want, out_n);
    free(want);

    want = alloc_doubles(need(fwd, "alpha05"), out_n);
    report_scalar("forward(alpha=0.5) 返回码", pg_forward(g, z, 0.5, got) == 0 ? 0.0 : 1.0, 0.0);
    report("forward alpha=0.5", got, want, out_n);
    free(want);

    const JsonValue *grown = need(root, "grown");
    int new_side = pg_grow(g);
    int gside = json_int(need(grown, "out_shape") ? json_at(need(grown, "out_shape"), 1) : NULL, 0);
    report_scalar("grow side", (double)new_side, (double)gside);
    report_scalar("param_count(grown)", (double)pg_param_count(g),
                  (double)json_int(need(grown, "param_count"), 0));
    load_convt(g->convs[g->n_stages - 2], json_at(need(grown, "convs"), 0));
    load_conv(g->rgb_heads[g->n_stages - 1], json_at(need(grown, "rgb_heads"), 0));

    double alpha = json_num(need(grown, "alpha"), 0.0);
    int out_n2 = oc * new_side * new_side;
    double *want2 = alloc_doubles(need(grown, "y"), out_n2);
    double *got2 = (double *)calloc((size_t)out_n2, sizeof(double));
    report_scalar("forward(grown) 返回码", pg_forward(g, z, alpha, got2) == 0 ? 0.0 : 1.0, 0.0);
    report("forward grown alpha=0.6", got2, want2, out_n2);

    free(got);
    free(got2);
    free(want2);
    free(z);
    free(channels);
    pg_free(g);
    printf("----------------------------------------\n");
    printf("共 %d 项检查：逐位一致 %d，失败 %d\n", g_checks, g_exact, g_fail);
    json_free(root);
    return g_fail > 0 ? 1 : 0;
}
