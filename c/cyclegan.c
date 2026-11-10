/*
 * cyclegan.c —— CycleGAN 一步训练（双向循环一致性 + identity 约束）
 *
 * 复用 Pix2Pix 模块的 PatchDiscriminator / Pix2PixGenerator。
 * 对应 Python AI.py:11397 的 train_cyclegan_step。
 *
 * 注意：p2p_disc_forward / p2p_gen_forward 返回的是对方模块的**内部缓存指针**
 * （下一次 forward 会覆盖），本文件一律复制一份再跨网络使用。
 */
#include "cyclegan.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double cg_sigmoid(double v)
{
    if (v >= 0.0) {
        return 1.0 / (1.0 + exp(-v));
    }
    double e = exp(v);
    return e / (1.0 + e);
}

/* 裸 logit 网格上的 BCE：返回平均损失；ds 可为 NULL。 */
static double cg_bce(const double *scores, int n, double target, double *ds)
{
    if (n <= 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double p = cg_sigmoid(scores[i]);
        double pc = p;
        if (pc < 1e-12) pc = 1e-12;
        if (pc > 1.0 - 1e-12) pc = 1.0 - 1e-12;
        total -= target * log(pc) + (1.0 - target) * log(1.0 - pc);
        if (ds) ds[i] = (p - target) / (double)n;
    }
    return total / (double)n;
}

/* 加权 L1：返回 weight * mean(|fake-real|)；d 可为 NULL。 */
static double cg_l1(const double *fake, const double *real, int n, double weight,
                    double *d)
{
    if (n <= 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fake[i] - real[i];
        total += fabs(diff);
        if (d) {
            d[i] = weight * (diff > 0.0 ? 1.0 : (diff < 0.0 ? -1.0 : 0.0)) /
                   (double)n;
        }
    }
    return weight * total / (double)n;
}

static int cg_add_inplace(double *dst, const double *src, int n)
{
    if (!dst || !src || n <= 0) return -1;
    for (int i = 0; i < n; i++) dst[i] += src[i];
    return 0;
}

/* 判别器一次「前向 + BCE + 反向」，返回 BCE 损失。 */
static double cg_disc_step(PatchDiscriminator *d, const double *image,
                           int C, int H, int W, double target)
{
    double *scores_buf = NULL;
    int sh = 0, sw = 0;
    if (p2p_disc_forward(d, image, C, H, W, &scores_buf, &sh, &sw) != 0) {
        fprintf(stderr, "cyclegan: 判别器前向失败\n");
        return -1.0;
    }
    int n = sh * sw;
    double *scores = (double *)malloc((size_t)n * sizeof(double));
    if (!scores) return -1.0;
    memcpy(scores, scores_buf, (size_t)n * sizeof(double));
    double *ds = (double *)malloc((size_t)n * sizeof(double));
    if (!ds) {
        free(scores);
        return -1.0;
    }
    double loss = cg_bce(scores, n, target, ds);
    int rc = p2p_disc_backward(d, ds, sh, sw);
    free(scores);
    free(ds);
    if (rc != 0) {
        fprintf(stderr, "cyclegan: 判别器反向失败\n");
        return -1.0;
    }
    return loss;
}

/* 生成器一次纯前向，返回 malloc 的输出图像（内部缓存副本）。 */
static int cg_gen_forward(Pix2PixGenerator *g, const double *x,
                          int C, int H, int W,
                          double **y, int *yc, int *yh, int *yw)
{
    double *buf = NULL;
    int bh = 0, bw = 0;
    if (p2p_gen_forward(g, x, C, H, W, &buf, &bh, &bw) != 0) return -1;
    int oc = g->out_channels;
    int n = oc * bh * bw;
    double *copy = (double *)malloc((size_t)n * sizeof(double));
    if (!copy) return -1;
    memcpy(copy, buf, (size_t)n * sizeof(double));
    *y = copy;
    *yc = oc;
    *yh = bh;
    *yw = bw;
    return 0;
}

/* 生成器一次「前向 + 加权 L1 + 反向」，返回 L1 损失；可选取输入梯度。 */
static double cg_gen_l1_step(Pix2PixGenerator *g, const double *x,
                             int C, int H, int W,
                             const double *target, double weight,
                             double **dx_out, int *dx_n)
{
    double *y = NULL;
    int yc = 0, yh = 0, yw = 0;
    if (cg_gen_forward(g, x, C, H, W, &y, &yc, &yh, &yw) != 0) {
        fprintf(stderr, "cyclegan: 生成器前向失败\n");
        return -1.0;
    }
    int n = yc * yh * yw;
    double *d_rec = (double *)malloc((size_t)n * sizeof(double));
    if (!d_rec) {
        free(y);
        return -1.0;
    }
    double loss = cg_l1(y, target, n, weight, d_rec);
    int rc = p2p_gen_backward(g, d_rec, yc, yh, yw);
    free(y);
    free(d_rec);
    if (rc != 0) {
        fprintf(stderr, "cyclegan: 生成器反向失败\n");
        return -1.0;
    }
    if (dx_out) {
        if (p2p_gen_input_grad(g, dx_out, dx_n) != 0) return -1.0;
    }
    return loss;
}

int cyclegan_train_step(Pix2PixGenerator *g_ab, Pix2PixGenerator *g_ba,
                        PatchDiscriminator *d_a, PatchDiscriminator *d_b,
                        const double *real_a, const double *real_b,
                        int channels, int height, int width,
                        double lambda_cycle, double lambda_identity,
                        double lr_g, double lr_d,
                        CycleGanStep *out)
{
    if (!g_ab || !g_ba || !d_a || !d_b || !real_a || !real_b || !out) return -1;
    if (channels <= 0 || height <= 0 || width <= 0) return -1;
    if (lambda_cycle < 0.0 || lambda_identity < 0.0) return -1;
    if (lr_g <= 0.0 || lr_d <= 0.0) return -1;

    const int C = channels, H = height, W = width;
    memset(out, 0, sizeof(*out));

    double *fake_b = NULL, *fake_a = NULL;
    int fbc = 0, fbh = 0, fbw = 0;
    int fac = 0, fah = 0, faw = 0;
    double *d_fake_b = NULL, *d_fake_a = NULL;
    int ndfb = 0, ndfa = 0;
    double *tmp = NULL;
    int tc = 0, th = 0, tw = 0;
    double loss_d_real_a = 0.0, loss_d_fake_a = 0.0;
    double loss_d_real_b = 0.0, loss_d_fake_b = 0.0;
    double loss_adv_a = 0.0, loss_adv_b = 0.0;
    double loss_cyc_a = 0.0, loss_cyc_b = 0.0;
    double loss_id_a = 0.0, loss_id_b = 0.0;
    int ret = -1;

    if (cg_gen_forward(g_ab, real_a, C, H, W, &fake_b, &fbc, &fbh, &fbw) != 0)
        goto done;
    if (cg_gen_forward(g_ba, real_b, C, H, W, &fake_a, &fac, &fah, &faw) != 0)
        goto done;

    /* ---------- 1) 更新判别器 ---------- */
    p2p_disc_zero_grad(d_a);
    loss_d_real_a = cg_disc_step(d_a, real_a, C, H, W, 1.0);
    if (loss_d_real_a < 0.0) goto done;
    loss_d_fake_a = cg_disc_step(d_a, fake_a, fac, fah, faw, 0.0);
    if (loss_d_fake_a < 0.0) goto done;
    p2p_disc_sgd_step(d_a, lr_d);

    p2p_disc_zero_grad(d_b);
    loss_d_real_b = cg_disc_step(d_b, real_b, C, H, W, 1.0);
    if (loss_d_real_b < 0.0) goto done;
    loss_d_fake_b = cg_disc_step(d_b, fake_b, fbc, fbh, fbw, 0.0);
    if (loss_d_fake_b < 0.0) goto done;
    p2p_disc_sgd_step(d_b, lr_d);

    /* ---------- 2) 更新生成器 ---------- */
    p2p_gen_zero_grad(g_ab);
    p2p_gen_zero_grad(g_ba);

    free(fake_b);
    fake_b = NULL;
    free(fake_a);
    fake_a = NULL;
    if (cg_gen_forward(g_ab, real_a, C, H, W, &fake_b, &fbc, &fbh, &fbw) != 0)
        goto done;
    if (cg_gen_forward(g_ba, real_b, C, H, W, &fake_a, &fac, &fah, &faw) != 0)
        goto done;

    /* 对抗损失：D_B(fake_b) 目标 1，D_A(fake_a) 目标 1 */
    p2p_disc_zero_grad(d_b);
    loss_adv_b = cg_disc_step(d_b, fake_b, fbc, fbh, fbw, 1.0);
    if (loss_adv_b < 0.0) goto done;
    if (p2p_disc_input_grad(d_b, &d_fake_b, &ndfb) != 0) goto done;

    p2p_disc_zero_grad(d_a);
    loss_adv_a = cg_disc_step(d_a, fake_a, fac, fah, faw, 1.0);
    if (loss_adv_a < 0.0) goto done;
    if (p2p_disc_input_grad(d_a, &d_fake_a, &ndfa) != 0) goto done;

    /* 循环一致：G_BA(G_AB(a)) ≈ a  ->  d 加回 d_fake_b */
    loss_cyc_a = cg_gen_l1_step(g_ba, fake_b, fbc, fbh, fbw, real_a,
                                lambda_cycle, &tmp, &tc);
    if (loss_cyc_a < 0.0 || tmp == NULL) goto done;
    if (cg_add_inplace(d_fake_b, tmp, ndfb < tc ? ndfb : tc) != 0) goto done;
    free(tmp);
    tmp = NULL;

    /* 循环一致：G_AB(G_BA(b)) ≈ b  ->  d 加回 d_fake_a */
    loss_cyc_b = cg_gen_l1_step(g_ab, fake_a, fac, fah, faw, real_b,
                                lambda_cycle, &tmp, &tc);
    if (loss_cyc_b < 0.0 || tmp == NULL) goto done;
    if (cg_add_inplace(d_fake_a, tmp, ndfa < tc ? ndfa : tc) != 0) goto done;
    free(tmp);
    tmp = NULL;

    /* identity：G_AB(real_b) ≈ real_b、G_BA(real_a) ≈ real_a
     * 只累积参数梯度，不需要输入梯度。 */
    loss_id_b = cg_gen_l1_step(g_ab, real_b, C, H, W, real_b,
                               lambda_identity, NULL, NULL);
    if (loss_id_b < 0.0) goto done;
    loss_id_a = cg_gen_l1_step(g_ba, real_a, C, H, W, real_a,
                               lambda_identity, NULL, NULL);
    if (loss_id_a < 0.0) goto done;

    /* 用最终梯度再反传一次（对应 Python 最后重新 forward 再 backward） */
    free(tmp);
    tmp = NULL;
    if (cg_gen_forward(g_ab, fake_b, fbc, fbh, fbw, &tmp, &tc, &th, &tw) != 0)
        goto done;
    free(tmp);
    tmp = NULL;
    if (p2p_gen_backward(g_ab, d_fake_b, fbc, fbh, fbw) != 0) goto done;

    if (cg_gen_forward(g_ba, fake_a, fac, fah, faw, &tmp, &tc, &th, &tw) != 0)
        goto done;
    free(tmp);
    tmp = NULL;
    if (p2p_gen_backward(g_ba, d_fake_a, fac, fah, faw) != 0) goto done;

    p2p_gen_sgd_step(g_ab, lr_g);
    p2p_gen_sgd_step(g_ba, lr_g);

    p2p_disc_zero_grad(d_a);
    p2p_disc_zero_grad(d_b);

    out->d_loss = 0.5 * (loss_d_real_a + loss_d_fake_a + loss_d_real_b +
                         loss_d_fake_b);
    out->g_loss = loss_adv_a + loss_adv_b + loss_cyc_a + loss_cyc_b + loss_id_a +
                  loss_id_b;
    out->adv_loss = loss_adv_a + loss_adv_b;
    out->cycle_loss = loss_cyc_a + loss_cyc_b;
    out->identity_loss = loss_id_a + loss_id_b;
    ret = 0;

done:
    free(fake_b);
    free(fake_a);
    free(d_fake_b);
    free(d_fake_a);
    free(tmp);
    return ret;
}
