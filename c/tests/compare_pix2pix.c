/*
 * compare_pix2pix.c —— 读取 tools/pix2pix_reference.py 生成的参考数据，
 * 用 C 实现重算 Pix2Pix 前向、一步训练后的损失与权重，并逐项比对。
 *
 * 用法: compare_pix2pix [pix2pix_reference.json] [容差]
 */
#include "../json.h"
#include "../layers_conv.h"
#include "../pix2pix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static double g_tol = 1e-9;

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/pix2pix_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static void read_doubles(const JsonValue *arr, double *dst, int n)
{
    if (json_array_len(arr) != n) {
        fprintf(stderr, "compare_pix2pix: 数组长度 %d != %d\n", json_array_len(arr), n);
        exit(2);
    }
    for (int i = 0; i < n; i++) dst[i] = json_num(json_at(arr, i), 0.0);
}

static double *alloc_read(const JsonValue *arr, int n)
{
    double *p = (double *)malloc((size_t)(n ? n : 1) * sizeof(double));
    if (!p) { fprintf(stderr, "compare_pix2pix: 内存不足\n"); exit(2); }
    read_doubles(arr, p, n);
    return p;
}

static void report(const char *what, const double *got, const double *want, int n)
{
    double worst = 0.0;
    int at = -1, exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    int ok = (worst <= g_tol);
    if (!ok) g_fail++;
    printf("    %-26s 最大差异 %.3e  %s%s\n", what, worst,
           ok ? "[OK]" : "[超出容差]", exact ? "  (逐位相同)" : "");
    if (!ok && at >= 0) printf("        index %d: C=%.17g py=%.17g\n", at, got[at], want[at]);
}

static void report_grid(const char *what, const double *got, const JsonValue *want)
{
    int n = json_int(need(want, "rows", what), 0) * json_int(need(want, "cols", what), 0);
    double *w = alloc_read(need(want, "data", what), n);
    report(what, got, w, n);
    free(w);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "pix2pix_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/pix2pix_reference.py %s\n", path);
        return 1;
    }
    printf("参考数据: %s（容差 %.1e）\n\n", path, g_tol);

    const JsonValue *param = need(root, "param", "root");
    const JsonValue *input = need(root, "input", "root");
    const JsonValue *init = need(root, "init", "root");
    const JsonValue *fwd = need(root, "forward", "root");
    const JsonValue *step = need(root, "step", "root");

    int size = json_int(need(input, "size", "input"), 8);
    int n_img = size * size;

    Pix2PixGenerator *g = p2p_gen_create(1, 1, 4);
    PatchDiscriminator *d = p2p_disc_create(1, 4);

    printf("参数个数: gen=%d (py %d), disc=%d (py %d)\n",
           p2p_gen_param_count(g), json_int(need(param, "gen", "param"), 0),
           p2p_disc_param_count(d), json_int(need(param, "disc", "param"), 0));
    if (p2p_gen_param_count(g) != json_int(need(param, "gen", "param"), 0)) g_fail++;
    if (p2p_disc_param_count(d) != json_int(need(param, "disc", "param"), 0)) g_fail++;

    const JsonValue *gi = need(init, "gen", "init");
    const JsonValue *di = need(init, "disc", "init");

    /* enc1: in=1,out=4,k=3 -> 36 w / 4 b；enc2: in=4,out=8,k=3 -> 288 / 8
     * dec1: in=8,out=4,k=4 -> 512 / 4；dec2: in=4,out=1,k=4 -> 64 / 1 */
    {
        const JsonValue *c = need(gi, "enc1", "gen");
        double *w = alloc_read(need(c, "w", "enc1"), 36);
        double *b = alloc_read(need(c, "b", "enc1"), 4);
        conv2d_set_weights((Conv2d *)g->enc1, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(gi, "enc2", "gen");
        double *w = alloc_read(need(c, "w", "enc2"), 288);
        double *b = alloc_read(need(c, "b", "enc2"), 8);
        conv2d_set_weights((Conv2d *)g->enc2, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(gi, "dec1", "gen");
        double *w = alloc_read(need(c, "w", "dec1"), 512);
        double *b = alloc_read(need(c, "b", "dec1"), 4);
        convt2d_set_weights((ConvTranspose2d *)g->dec1, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(gi, "dec2", "gen");
        double *w = alloc_read(need(c, "w", "dec2"), 64);
        double *b = alloc_read(need(c, "b", "dec2"), 1);
        convt2d_set_weights((ConvTranspose2d *)g->dec2, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(di, "conv1", "disc");
        double *w = alloc_read(need(c, "w", "conv1"), 36);
        double *b = alloc_read(need(c, "b", "conv1"), 4);
        conv2d_set_weights((Conv2d *)d->conv1, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(di, "conv2", "disc");
        double *w = alloc_read(need(c, "w", "conv2"), 288);
        double *b = alloc_read(need(c, "b", "conv2"), 8);
        conv2d_set_weights((Conv2d *)d->conv2, w, b);
        free(w); free(b);
    }
    {
        const JsonValue *c = need(di, "conv3", "disc");
        double *w = alloc_read(need(c, "w", "conv3"), 72);
        double *b = alloc_read(need(c, "b", "conv3"), 1);
        conv2d_set_weights((Conv2d *)d->conv3, w, b);
        free(w); free(b);
    }

    double *source = alloc_read(need(input, "source", "input"), n_img);
    double *target = alloc_read(need(input, "target", "input"), n_img);

    /* ---- 前向 ---- */
    printf("\n[前向]\n");
    double *fake = 0; int fh = 0, fw = 0;
    if (p2p_gen_forward(g, source, 1, size, size, &fake, &fh, &fw) != 0) return 1;
    double *want_fake = alloc_read(need(fwd, "fake", "forward"), fh * fw);
    report("fake", fake, want_fake, fh * fw);
    free(want_fake);

    double *scores = 0; int sh = 0, sw = 0;
    if (p2p_disc_forward(d, target, 1, size, size, &scores, &sh, &sw) != 0) return 1;
    report_grid("real_scores", scores, need(fwd, "real_scores", "forward"));
    double *want_scores = alloc_read(need(need(fwd, "fake_scores", "forward"), "data", "fake_scores"),
                                     json_int(need(need(fwd, "fake_scores", "forward"), "rows", "f"), 0) *
                                     json_int(need(need(fwd, "fake_scores", "forward"), "cols", "f"), 0));
    if (p2p_disc_forward(d, fake, 1, fh, fw, &scores, &sh, &sw) != 0) return 1;
    report("fake_scores", scores, want_scores, sh * sw);
    free(want_scores);

    /* ---- 一步训练 ---- */
    printf("\n[训练一步]\n");
    double d_loss = 0, g_loss = 0;
    const JsonValue *sj = step;
    double lambda_l1 = json_num(need(sj, "lambda_l1", "step"), 10.0);
    double lr_g = json_num(need(sj, "lr_g", "step"), 0.01);
    double lr_d = json_num(need(sj, "lr_d", "step"), 0.01);
    if (p2p_train_step(g, d, source, 1, size, size, target, 1,
                       lambda_l1, lr_g, lr_d, &d_loss, &g_loss) != 0) return 1;
    double want_d = json_num(need(sj, "d_loss", "step"), 0.0);
    double want_g = json_num(need(sj, "g_loss", "step"), 0.0);
    report("d_loss", &d_loss, &want_d, 1);
    report("g_loss", &g_loss, &want_g, 1);

    /* 更新后权重 */
    {
        const JsonValue *ga = need(sj, "gen_after", "step");
        const JsonValue *da = need(sj, "disc_after", "step");
        const char *g_names[4] = { "enc1", "enc2", "dec1", "dec2" };
        const int g_nw[4] = { 36, 288, 512, 64 };
        const int g_nb[4] = { 4, 8, 4, 1 };
        const double *g_W[4] = { ((Conv2d *)g->enc1)->W, ((Conv2d *)g->enc2)->W,
                                 ((ConvTranspose2d *)g->dec1)->W, ((ConvTranspose2d *)g->dec2)->W };
        const double *g_b[4] = { ((Conv2d *)g->enc1)->b, ((Conv2d *)g->enc2)->b,
                                 ((ConvTranspose2d *)g->dec1)->b, ((ConvTranspose2d *)g->dec2)->b };
        for (int i = 0; i < 4; i++) {
            const JsonValue *c = need(ga, g_names[i], "gen_after");
            double *ww = alloc_read(need(c, "w", g_names[i]), g_nw[i]);
            double *bb = alloc_read(need(c, "b", g_names[i]), g_nb[i]);
            char tag[64];
            snprintf(tag, sizeof(tag), "gen.%s.w", g_names[i]);
            report(tag, g_W[i], ww, g_nw[i]);
            snprintf(tag, sizeof(tag), "gen.%s.b", g_names[i]);
            report(tag, g_b[i], bb, g_nb[i]);
            free(ww); free(bb);
        }

        const char *d_names[3] = { "conv1", "conv2", "conv3" };
        const int d_nw[3] = { 36, 288, 72 };
        const int d_nb[3] = { 4, 8, 1 };
        const double *d_W[3] = { ((Conv2d *)d->conv1)->W, ((Conv2d *)d->conv2)->W, ((Conv2d *)d->conv3)->W };
        const double *d_b[3] = { ((Conv2d *)d->conv1)->b, ((Conv2d *)d->conv2)->b, ((Conv2d *)d->conv3)->b };
        for (int i = 0; i < 3; i++) {
            const JsonValue *c = need(da, d_names[i], "disc_after");
            double *ww = alloc_read(need(c, "w", d_names[i]), d_nw[i]);
            double *bb = alloc_read(need(c, "b", d_names[i]), d_nb[i]);
            char tag[64];
            snprintf(tag, sizeof(tag), "disc.%s.w", d_names[i]);
            report(tag, d_W[i], ww, d_nw[i]);
            snprintf(tag, sizeof(tag), "disc.%s.b", d_names[i]);
            report(tag, d_b[i], bb, d_nb[i]);
            free(ww); free(bb);
        }

        /* 更新后再做一次前向 */
        double *fake2 = 0; int fh2 = 0, fw2 = 0;
        if (p2p_gen_forward(g, source, 1, size, size, &fake2, &fh2, &fw2) != 0) return 1;
        double *wf = alloc_read(need(sj, "fake_after", "step"), fh2 * fw2);
        report("fake_after", fake2, wf, fh2 * fw2);
        free(wf);
        /* fake2 指向 g->y 内部缓存，不能 free */
    }
    free(source); free(target);
    /* fake / scores 指向生成器/判别器内部缓存，不能单独 free */

    printf("\n%s（失败 %d 项）\n", g_fail ? "对拍失败" : "对拍通过", g_fail);
    p2p_gen_free(g);
    p2p_disc_free(d);
    json_free(root);
    return g_fail ? 1 : 0;
}
