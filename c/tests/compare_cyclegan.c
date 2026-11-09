/*
 * compare_cyclegan.c —— 读取 tools/cyclegan_reference.py 生成的参考数据，
 * 用 C 实现重算 CycleGAN 一步训练（损失、更新后权重、更新后输出）并逐项比对。
 *
 * 用法: compare_cyclegan [cyclegan_reference.json] [容差]
 */
#include "../cyclegan.h"
#include "../json.h"
#include "../layers_conv.h"
#include "../pix2pix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_fail = 0;
static int g_exact = 0;
static double g_tol = 1e-9;

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/cyclegan_reference.py）\n",
                who, key);
        exit(2);
    }
    return v;
}

static double *alloc_read(const JsonValue *arr, int n)
{
    if (!arr || json_array_len(arr) != n) {
        fprintf(stderr, "compare_cyclegan: 数组长度不符合预期\n");
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_cyclegan: 内存不足\n");
        exit(2);
    }
    for (int i = 0; i < n; i++) p[i] = json_num(json_at(arr, i), 0.0);
    return p;
}

static void report(const char *what, const double *got, const double *want, int n)
{
    g_checks++;
    double worst = 0.0;
    int at = -1, exact = 1;
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
        printf("    %-28s 最大差异 %.3e [超出容差] index %d C=%.17g py=%.17g\n",
               what, worst, at, got[at], want[at]);
    } else {
        printf("    %-28s 最大差异 %.3e [OK]%s\n", what, worst,
               exact ? " (逐位相同)" : "");
    }
}

static void report_scalar(const char *what, double got, double want)
{
    report(what, &got, &want, 1);
}

/* ---- 权重读写 ---- */

static void load_conv_block(Conv2d *c, const JsonValue *j, int in, int out, int k)
{
    double *w = alloc_read(need(j, "w", "conv"), out * in * k * k);
    double *b = alloc_read(need(j, "b", "conv"), out);
    conv2d_set_weights(c, w, b);
    free(w);
    free(b);
}

static void load_convt_block(ConvTranspose2d *c, const JsonValue *j, int in, int out,
                             int k)
{
    double *w = alloc_read(need(j, "w", "convt"), in * out * k * k);
    double *b = alloc_read(need(j, "b", "convt"), out);
    convt2d_set_weights(c, w, b);
    free(w);
    free(b);
}

static void load_gen(Pix2PixGenerator *g, const JsonValue *j)
{
    load_conv_block((Conv2d *)g->enc1, need(j, "enc1", "gen"), g->in_channels, g->base, 3);
    load_conv_block((Conv2d *)g->enc2, need(j, "enc2", "gen"), g->base, 2 * g->base, 3);
    load_convt_block((ConvTranspose2d *)g->dec1, need(j, "dec1", "gen"), 2 * g->base,
                     g->base, 4);
    load_convt_block((ConvTranspose2d *)g->dec2, need(j, "dec2", "gen"), g->base,
                     g->out_channels, 4);
}

static void load_disc(PatchDiscriminator *d, const JsonValue *j)
{
    load_conv_block((Conv2d *)d->conv1, need(j, "conv1", "disc"), d->in_channels,
                    d->base, 3);
    load_conv_block((Conv2d *)d->conv2, need(j, "conv2", "disc"), d->base, 2 * d->base, 3);
    load_conv_block((Conv2d *)d->conv3, need(j, "conv3", "disc"), 2 * d->base, 1, 3);
}

static void compare_gen(const char *tag, Pix2PixGenerator *g, const JsonValue *j)
{
    char name[96];
    const double *W[4] = {
        ((Conv2d *)g->enc1)->W,
        ((Conv2d *)g->enc2)->W,
        ((ConvTranspose2d *)g->dec1)->W,
        ((ConvTranspose2d *)g->dec2)->W,
    };
    const double *B[4] = {
        ((Conv2d *)g->enc1)->b,
        ((Conv2d *)g->enc2)->b,
        ((ConvTranspose2d *)g->dec1)->b,
        ((ConvTranspose2d *)g->dec2)->b,
    };
    const int nW[4] = {
        g->in_channels * g->base * 9,
        g->base * 2 * g->base * 9,
        2 * g->base * g->base * 16,
        g->base * g->out_channels * 16,
    };
    const int nB[4] = {g->base, 2 * g->base, g->base, g->out_channels};
    const char *keys[4] = {"enc1", "enc2", "dec1", "dec2"};
    for (int i = 0; i < 4; i++) {
        const JsonValue *part = need(j, keys[i], tag);
        double *want = alloc_read(need(part, "w", keys[i]), nW[i]);
        snprintf(name, sizeof(name), "%s.%s.w", tag, keys[i]);
        report(name, W[i], want, nW[i]);
        free(want);
        want = alloc_read(need(part, "b", keys[i]), nB[i]);
        snprintf(name, sizeof(name), "%s.%s.b", tag, keys[i]);
        report(name, B[i], want, nB[i]);
        free(want);
    }
}

static void compare_disc(const char *tag, PatchDiscriminator *d, const JsonValue *j)
{
    char name[96];
    const double *W[3] = {((Conv2d *)d->conv1)->W, ((Conv2d *)d->conv2)->W,
                          ((Conv2d *)d->conv3)->W};
    const double *B[3] = {((Conv2d *)d->conv1)->b, ((Conv2d *)d->conv2)->b,
                          ((Conv2d *)d->conv3)->b};
    const int nW[3] = {
        d->in_channels * d->base * 9,
        d->base * 2 * d->base * 9,
        2 * d->base * 1 * 9,
    };
    const int nB[3] = {d->base, 2 * d->base, 1};
    const char *keys[3] = {"conv1", "conv2", "conv3"};
    for (int i = 0; i < 3; i++) {
        const JsonValue *part = need(j, keys[i], tag);
        double *want = alloc_read(need(part, "w", keys[i]), nW[i]);
        snprintf(name, sizeof(name), "%s.%s.w", tag, keys[i]);
        report(name, W[i], want, nW[i]);
        free(want);
        want = alloc_read(need(part, "b", keys[i]), nB[i]);
        snprintf(name, sizeof(name), "%s.%s.b", tag, keys[i]);
        report(name, B[i], want, nB[i]);
        free(want);
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "cyclegan_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/cyclegan_reference.py %s\n", path);
        return 2;
    }

    const JsonValue *model = need(root, "model", "root");
    int C = json_int(need(model, "channels", "model"), 1);
    int H = json_int(need(model, "height", "model"), 8);
    int W = json_int(need(model, "width", "model"), 8);
    int base = json_int(need(model, "base", "model"), 4);

    Pix2PixGenerator *g_ab = p2p_gen_create(C, C, base);
    Pix2PixGenerator *g_ba = p2p_gen_create(C, C, base);
    PatchDiscriminator *d_a = p2p_disc_create(C, base);
    PatchDiscriminator *d_b = p2p_disc_create(C, base);
    if (!g_ab || !g_ba || !d_a || !d_b) {
        fprintf(stderr, "网络创建失败\n");
        return 2;
    }

    const JsonValue *before = need(root, "before", "root");
    load_gen(g_ab, need(before, "g_ab", "before"));
    load_gen(g_ba, need(before, "g_ba", "before"));
    load_disc(d_a, need(before, "d_a", "before"));
    load_disc(d_b, need(before, "d_b", "before"));

    const JsonValue *real = need(root, "real", "root");
    int n_img = C * H * W;
    double *real_a = alloc_read(need(real, "a", "real"), n_img);
    double *real_b = alloc_read(need(real, "b", "real"), n_img);

    CycleGanStep st;
    int rc = cyclegan_train_step(g_ab, g_ba, d_a, d_b, real_a, real_b, C, H, W,
                                 10.0, 5.0, 0.01, 0.01, &st);
    report_scalar("train_step 返回码", rc == 0 ? 0.0 : 1.0, 0.0);

    const JsonValue *losses = need(root, "losses", "root");
    report_scalar("d_loss", st.d_loss, json_num(need(losses, "d_loss", "losses"), 0.0));
    report_scalar("g_loss", st.g_loss, json_num(need(losses, "g_loss", "losses"), 0.0));
    report_scalar("adv_loss", st.adv_loss, json_num(need(losses, "adv_loss", "losses"), 0.0));
    report_scalar("cycle_loss", st.cycle_loss,
                  json_num(need(losses, "cycle_loss", "losses"), 0.0));
    report_scalar("identity_loss", st.identity_loss,
                  json_num(need(losses, "identity_loss", "losses"), 0.0));

    const JsonValue *after = need(root, "after", "root");
    compare_gen("after.g_ab", g_ab, need(after, "g_ab", "after"));
    compare_gen("after.g_ba", g_ba, need(after, "g_ba", "after"));
    compare_disc("after.d_a", d_a, need(after, "d_a", "after"));
    compare_disc("after.d_b", d_b, need(after, "d_b", "after"));

    const JsonValue *fa = need(root, "fake_after", "root");
    double *y = NULL;
    int yh = 0, yw = 0;
    if (p2p_gen_forward(g_ab, real_a, C, H, W, &y, &yh, &yw) != 0 ||
        yh * yw * C != n_img) {
        fprintf(stderr, "g_ab 前向失败\n");
        return 2;
    }
    double *want = alloc_read(need(fa, "fake_b", "fake_after"), n_img);
    report("fake_b(after)", y, want, n_img);
    free(want);
    y = NULL;

    if (p2p_gen_forward(g_ba, real_b, C, H, W, &y, &yh, &yw) != 0 ||
        yh * yw * C != n_img) {
        fprintf(stderr, "g_ba 前向失败\n");
        return 2;
    }
    want = alloc_read(need(fa, "fake_a", "fake_after"), n_img);
    report("fake_a(after)", y, want, n_img);
    free(want);

    report_scalar("param_count(g_ab)", (double)p2p_gen_param_count(g_ab), 917.0);
    report_scalar("param_count(d_a)", (double)p2p_disc_param_count(d_a), 409.0);

    printf("----------------------------------------\n");
    printf("共 %d 项检查：逐位一致 %d，失败 %d\n", g_checks, g_exact, g_fail);

    free(real_a);
    free(real_b);
    p2p_gen_free(g_ab);
    p2p_gen_free(g_ba);
    p2p_disc_free(d_a);
    p2p_disc_free(d_b);
    json_free(root);
    return g_fail > 0 ? 1 : 0;
}
