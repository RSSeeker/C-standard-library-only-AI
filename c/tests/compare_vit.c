/*
 * compare_vit.c —— 读取 tools/vit_reference.py 生成的参考数据，
 * 用 C 的 ViT 前向重算 logits / 参数量并逐项比对。
 *
 * 用法: compare_vit [vit_reference.json] [容差]
 */
#include "../json.h"
#include "../vit.h"

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
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/vit_reference.py）\n", key);
        exit(2);
    }
    return v;
}

static const JsonValue *find_weight(const JsonValue *weights, const char *name)
{
    int n = json_array_len(weights);
    for (int i = 0; i < n; i++) {
        const JsonValue *b = json_at(weights, i);
        const char *nm = json_str(json_get(b, "name"), "");
        if (nm && strcmp(nm, name) == 0) return b;
    }
    fprintf(stderr, "参考权重缺少 %s\n", name);
    exit(2);
}

static double *alloc_doubles_exact(const JsonValue *block, int expect, const char *who)
{
    const JsonValue *data = need(block, "data");
    int n = json_array_len(data);
    if (n != expect) {
        fprintf(stderr, "权重 %s 长度 %d，期望 %d\n", who, n, expect);
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_vit: 内存不足\n");
        exit(2);
    }
    for (int i = 0; i < n; i++) p[i] = json_num(json_at(data, i), 0.0);
    return p;
}

static void load_weight(const JsonValue *weights, const char *name, double *dst, int expect)
{
    const JsonValue *b = find_weight(weights, name);
    double *p = alloc_doubles_exact(b, expect, name);
    memcpy(dst, p, (size_t)expect * sizeof(double));
    free(p);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    g_checks++;
    double worst = 0.0;
    int at = -1;
    int exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    if (exact) g_exact++;
    if (worst > g_tol) {
        g_fail++;
        printf("    %-22s 最大差异 %.3e [超出容差] index %d C=%.17g py=%.17g\n",
               what, worst, at, at >= 0 ? got[at] : 0.0, at >= 0 ? want[at] : 0.0);
    } else {
        printf("    %-22s 最大差异 %.3e [OK]%s\n", what, worst, exact ? " (逐位相同)" : "");
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "vit_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/vit_reference.py %s\n", path);
        return 2;
    }

    const JsonValue *cfg = need(root, "config");
    int in_c = json_int(need(cfg, "in_channels"), 0);
    int ih = json_int(need(cfg, "image_h"), 0);
    int iw = json_int(need(cfg, "image_w"), 0);
    int patch = json_int(need(cfg, "patch_size"), 0);
    int d_model = json_int(need(cfg, "d_model"), 0);
    int heads = json_int(need(cfg, "num_heads"), 0);
    int d_ff = json_int(need(cfg, "d_ff"), 0);
    int layers = json_int(need(cfg, "num_layers"), 0);
    int classes = json_int(need(cfg, "num_classes"), 0);
    int want_params = json_int(need(cfg, "param_count"), 0);

    VitModel *m = vit_create(in_c, ih, iw, patch, d_model, heads, d_ff, layers,
                             classes, 20240501ULL);
    const JsonValue *weights = need(root, "weights");

    double *Wc = (double *)malloc((size_t)d_model * in_c * patch * patch * sizeof(double));
    double *bc = (double *)malloc((size_t)d_model * sizeof(double));
    load_weight(weights, "patch.weight", Wc, d_model * in_c * patch * patch);
    load_weight(weights, "patch.bias", bc, d_model);
    conv2d_set_weights(vit_patch_conv(m), Wc, bc);
    free(Wc);
    free(bc);

    int P = (ih / patch) * (iw / patch);
    double *cls = (double *)malloc((size_t)d_model * sizeof(double));
    double *pos = (double *)malloc((size_t)(P + 1) * d_model * sizeof(double));
    load_weight(weights, "cls", cls, d_model);
    load_weight(weights, "pos", pos, (P + 1) * d_model);
    vit_set_cls_pos(m, cls, pos);
    free(cls);
    free(pos);

    for (int li = 0; li < layers; li++) {
        TransformerEncoderLayer *layer = m->encoder->layers[li];
        char name[128];
        double *Wq = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wk = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wv = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wo = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *bq = (double *)malloc((size_t)d_model * sizeof(double));
        double *bk = (double *)malloc((size_t)d_model * sizeof(double));
        double *bv = (double *)malloc((size_t)d_model * sizeof(double));
        double *bo = (double *)malloc((size_t)d_model * sizeof(double));
        if (!Wq || !Wk || !Wv || !Wo || !bq || !bk || !bv || !bo) exit(2);

#define LW(field, dst, count) \
        do { snprintf(name, sizeof(name), "enc.L%d.%s", li, field); \
             load_weight(weights, name, (dst), (count)); } while (0)

        LW("attn.W_q", Wq, d_model * d_model);
        LW("attn.W_k", Wk, d_model * d_model);
        LW("attn.W_v", Wv, d_model * d_model);
        LW("attn.W_o", Wo, d_model * d_model);
        LW("attn.b_q", bq, d_model);
        LW("attn.b_k", bk, d_model);
        LW("attn.b_v", bv, d_model);
        LW("attn.b_o", bo, d_model);
        mha_set_params(layer->self_attn, Wq, bq, Wk, bk, Wv, bv, Wo, bo);
        free(Wq); free(Wk); free(Wv); free(Wo);
        free(bq); free(bk); free(bv); free(bo);

        double *W1 = (double *)malloc((size_t)d_ff * d_model * sizeof(double));
        double *W2 = (double *)malloc((size_t)d_model * d_ff * sizeof(double));
        double *b1 = (double *)malloc((size_t)d_ff * sizeof(double));
        double *b2 = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ffn.W1", W1, d_ff * d_model);
        LW("ffn.W2", W2, d_model * d_ff);
        LW("ffn.b1", b1, d_ff);
        LW("ffn.b2", b2, d_model);
        ff_set_params(layer->ffn, W1, b1, W2, b2);
        free(W1); free(W2); free(b1); free(b2);

        double *g1 = (double *)malloc((size_t)d_model * sizeof(double));
        double *b1n = (double *)malloc((size_t)d_model * sizeof(double));
        double *g2 = (double *)malloc((size_t)d_model * sizeof(double));
        double *b2n = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ln1.gamma", g1, d_model);
        LW("ln1.beta", b1n, d_model);
        LW("ln2.gamma", g2, d_model);
        LW("ln2.beta", b2n, d_model);
        memcpy(layer->ln1.gamma, g1, (size_t)d_model * sizeof(double));
        memcpy(layer->ln1.beta, b1n, (size_t)d_model * sizeof(double));
        memcpy(layer->ln2.gamma, g2, (size_t)d_model * sizeof(double));
        memcpy(layer->ln2.beta, b2n, (size_t)d_model * sizeof(double));
        free(g1); free(b1n); free(g2); free(b2n);
#undef LW
    }

    double *ng = (double *)malloc((size_t)d_model * sizeof(double));
    double *nb = (double *)malloc((size_t)d_model * sizeof(double));
    load_weight(weights, "norm.gamma", ng, d_model);
    load_weight(weights, "norm.beta", nb, d_model);
    layernorm_set_params(vit_norm(m), ng, nb);
    free(ng);
    free(nb);

    double *Wh = (double *)malloc((size_t)classes * d_model * sizeof(double));
    double *bh = (double *)malloc((size_t)classes * sizeof(double));
    load_weight(weights, "head.W", Wh, classes * d_model);
    load_weight(weights, "head.b", bh, classes);
    vit_set_head(m, Wh, bh);
    free(Wh);
    free(bh);

    printf("config: in=%d image=%dx%d patch=%d d_model=%d heads=%d layers=%d classes=%d\n",
           in_c, ih, iw, patch, d_model, heads, layers, classes);

    int got_params = vit_param_count(m);
    g_checks++;
    if (got_params == want_params) {
        g_exact++;
        printf("    %-22s %d [OK] (逐位相同)\n", "param_count", got_params);
    } else {
        g_fail++;
        printf("    %-22s C=%d py=%d [不一致]\n", "param_count", got_params, want_params);
    }

    const JsonValue *c = need(root, "case");
    const JsonValue *img_v = need(c, "image");
    int img_n = json_array_len(img_v);
    double *image = (double *)malloc((size_t)img_n * sizeof(double));
    for (int i = 0; i < img_n; i++) image[i] = json_num(json_at(img_v, i), 0.0);

    double *got = (double *)malloc((size_t)classes * sizeof(double));
    double *want = (double *)malloc((size_t)classes * sizeof(double));
    const JsonValue *want_v = need(c, "logits");
    for (int i = 0; i < classes; i++) want[i] = json_num(json_at(want_v, i), 0.0);

    vit_forward(m, image, got);
    report("logits", got, want, classes);

    printf("=== vit: %d 项检查, 失败 %d, 逐位相同 %d ===\n", g_checks, g_fail, g_exact);
    free(image);
    free(got);
    free(want);
    json_free(root);
    vit_free(m);
    return g_fail == 0 ? 0 : 1;
}
