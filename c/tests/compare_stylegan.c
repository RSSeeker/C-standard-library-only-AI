/*
 * compare_stylegan.c —— 读取 tools/stylegan_reference.py 生成的参考数据，
 * 用 C 的 StyleGanGenerator 重算前向 / 平均 w / 参数量并逐项比对。
 *
 * 用法: compare_stylegan [stylegan_reference.json] [容差]
 */
#include "../json.h"
#include "../stylegan.h"

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
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/stylegan_reference.py）\n", key);
        exit(2);
    }
    return v;
}

static double *alloc_doubles(const JsonValue *arr, int n)
{
    if (!arr || json_array_len(arr) != n) {
        fprintf(stderr, "compare_stylegan: 数组长度不符合预期\n");
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_stylegan: 内存不足\n");
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

static void load_mlp(MLP *m, const JsonValue *layers)
{
    int n = json_array_len(layers);
    if (n != m->n_layers) {
        fprintf(stderr, "compare_stylegan: MLP 层数不一致 %d != %d\n", n, m->n_layers);
        exit(2);
    }
    for (int i = 0; i < n; i++) {
        const JsonValue *layer = json_at(layers, i);
        int fin = json_int(need(layer, "fan_in"), 0);
        int fout = json_int(need(layer, "fan_out"), 0);
        if (fin != m->layers[i].fan_in || fout != m->layers[i].fan_out) {
            fprintf(stderr, "compare_stylegan: 第 %d 层形状不一致\n", i);
            exit(2);
        }
        double *w = alloc_doubles(need(layer, "w"), fin * fout);
        double *b = alloc_doubles(need(layer, "b"), fout);
        memcpy(m->layers[i].W, w, (size_t)fin * fout * sizeof(double));
        memcpy(m->layers[i].b, b, (size_t)fout * sizeof(double));
        free(w);
        free(b);
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "stylegan_reference.json";
    if (argc > 2) {
        g_tol = atof(argv[2]);
    }
    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/stylegan_reference.py %s\n", path);
        return 2;
    }
    const JsonValue *model = need(root, "model");
    int latent = json_int(need(model, "latent_dim"), 0);
    int d_model = json_int(need(model, "d_model"), 0);
    int base = json_int(need(model, "base"), 0);
    int oc = json_int(need(model, "out_channels"), 0);
    int pc = json_int(need(model, "param_count"), 0);

    StyleGanGenerator *g = sg_create(latent, d_model, base, oc);
    if (!g) {
        fprintf(stderr, "sg_create 失败\n");
        json_free(root);
        return 2;
    }
    report_scalar("param_count", (double)sg_param_count(g), (double)pc);

    double *cst = alloc_doubles(need(root, "const"), base * 16);
    memcpy(g->const_feat, cst, (size_t)base * 16 * sizeof(double));
    free(cst);

    load_mlp(g->mapping, need(root, "mapping"));
    load_mlp(g->style1, need(root, "style1"));
    load_mlp(g->style2, need(root, "style2"));

    {
        const JsonValue *convs[3] = {need(root, "conv1"), need(root, "conv2"),
                                     need(root, "to_rgb")};
        Conv2d *layers[3] = {g->conv1, g->conv2, g->to_rgb};
        for (int i = 0; i < 3; i++) {
            int in = json_int(need(convs[i], "in_channels"), 0);
            int out = json_int(need(convs[i], "out_channels"), 0);
            int kh = json_int(need(convs[i], "kh"), 0);
            int kw = json_int(need(convs[i], "kw"), 0);
            double *w = alloc_doubles(need(convs[i], "w"), out * in * kh * kw);
            double *b = alloc_doubles(need(convs[i], "b"), out);
            conv2d_set_weights(layers[i], w, b);
            free(w);
            free(b);
        }
    }

    const JsonValue *fwd = need(root, "forward");
    double *z = alloc_doubles(need(fwd, "z"), latent);
    int out_n = oc * 16 * 16;
    double *want_y = alloc_doubles(need(fwd, "y"), out_n);
    double *got_y = (double *)calloc((size_t)out_n, sizeof(double));
    int rc = sg_forward(g, z, got_y);
    report_scalar("forward 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
    report("forward y", got_y, want_y, out_n);

    const JsonValue *avg = need(root, "average_w");
    const JsonValue *zs_j = need(avg, "zs");
    int nz = json_array_len(zs_j);
    double *zs = (double *)calloc((size_t)latent * nz, sizeof(double));
    for (int i = 0; i < nz; i++) {
        const JsonValue *zi = json_at(zs_j, i);
        for (int k = 0; k < latent; k++) {
            zs[(size_t)i * latent + k] = json_num(json_at(zi, k), 0.0);
        }
    }
    double *got_w = (double *)calloc((size_t)d_model, sizeof(double));
    rc = sg_average_w(g, zs, nz, got_w);
    report_scalar("average_w 返回码", rc == 0 ? 0.0 : 1.0, 0.0);
    double *want_w = alloc_doubles(need(avg, "w"), d_model);
    report("average_w", got_w, want_w, d_model);

    free(z);
    free(want_y);
    free(got_y);
    free(zs);
    free(got_w);
    free(want_w);
    sg_free(g);
    printf("----------------------------------------\n");
    printf("共 %d 项检查：逐位一致 %d，失败 %d\n", g_checks, g_exact, g_fail);
    json_free(root);
    return g_fail > 0 ? 1 : 0;
}
