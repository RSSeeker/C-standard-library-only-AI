/*
 * pix2pix.c —— 小型 Pix2Pix（PatchGAN 判别器 + 卷积/转置卷积生成器）
 *
 * 逐行对齐 Python AI.py 中 PatchDiscriminator / Pix2PixGenerator /
 * _p2p_bce / _p2p_l1_and_grad / train_pix2pix_step 的运算顺序：
 *   - 卷积前向/反向复用 layers_conv.c（其循环顺序已与 Python 对齐）
 *   - 激活在容器层手写，与前向缓存一起服务反向
 *   - 判别器 backward 返回输入图像梯度（供生成器反向链使用）
 */
#include "pix2pix.h"

#include "layers_conv.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define P2P_LEAK 0.2

static Conv2d *as_conv(void *p) { return (Conv2d *)p; }
static ConvTranspose2d *as_convt(void *p) { return (ConvTranspose2d *)p; }

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "pix2pix: 内存分配失败\n");
        exit(1);
    }
    return p;
}

static double *alloc_double(size_t n)
{
    return (double *)xmalloc(n * sizeof(double));
}

static double p2p_sigmoid(double v)
{
    if (v >= 0.0) {
        double e = exp(-v);
        return 1.0 / (1.0 + e);
    }
    double e = exp(v);
    return e / (1.0 + e);
}

static Tensor *tensor3(int c, int h, int w)
{
    int shape[3] = { c, h, w };
    return tensor_new(3, shape);
}

/* 从 (C,H,W) 图像拷贝到 Tensor；返回 Tensor（调用方 tensor_free）。 */
static Tensor *tensor3_from(const double *image, int c, int h, int w)
{
    Tensor *t = tensor3(c, h, w);
    memcpy(t->data, image, (size_t)c * (size_t)h * (size_t)w * sizeof(double));
    return t;
}

/* o = conv(x) 的前向 + 激活。mask 保存 z>0 的 0/1 掩码；a 为激活后的图像。 */
static void conv_forward_act(Conv2d *layer, const Tensor *x, Tensor *o,
                             double *mask, double *a, double slope)
{
    conv2d_forward(layer, x, o);
    int n = o->size;
    for (int i = 0; i < n; i++) {
        double v = o->data[i];
        if (v > 0.0) {
            mask[i] = 1.0;
            a[i] = v;
        } else {
            mask[i] = 0.0;
            a[i] = slope * v;
        }
    }
}

/* 同上，但底层是转置卷积。 */
static void convt_forward_act(ConvTranspose2d *layer, const Tensor *x, Tensor *o,
                              double *mask, double *a, double slope)
{
    convt2d_forward(layer, x, o);
    int n = o->size;
    for (int i = 0; i < n; i++) {
        double v = o->data[i];
        if (v > 0.0) {
            mask[i] = 1.0;
            a[i] = v;
        } else {
            mask[i] = 0.0;
            a[i] = slope * v;
        }
    }
}

/* ============================ 判别器 ============================ */

PatchDiscriminator *p2p_disc_create(int in_channels, int base)
{
    if (in_channels < 1 || base < 1) {
        fprintf(stderr, "p2p_disc_create: in_channels/base 必须是正整数\n");
        exit(1);
    }
    PatchDiscriminator *d = (PatchDiscriminator *)xmalloc(sizeof(*d));
    memset(d, 0, sizeof(*d));
    d->in_channels = in_channels;
    d->base = base;
    d->conv1 = conv2d_create(in_channels, base, 3, 2, 1);
    d->conv2 = conv2d_create(base, 2 * base, 3, 2, 1);
    d->conv3 = conv2d_create(2 * base, 1, 3, 1, 1);
    return d;
}

void p2p_disc_free(PatchDiscriminator *d)
{
    if (!d) return;
    conv2d_free(as_conv(d->conv1));
    conv2d_free(as_conv(d->conv2));
    conv2d_free(as_conv(d->conv3));
    free(d->m1);
    free(d->m2);
    free(d->scores);
    free(d->dx);
    free(d);
}

void p2p_disc_zero_grad(PatchDiscriminator *d)
{
    conv2d_zero_grad(as_conv(d->conv1));
    conv2d_zero_grad(as_conv(d->conv2));
    conv2d_zero_grad(as_conv(d->conv3));
}

int p2p_disc_param_count(const PatchDiscriminator *d)
{
    return conv2d_param_count(as_conv(d->conv1))
         + conv2d_param_count(as_conv(d->conv2))
         + conv2d_param_count(as_conv(d->conv3));
}

static void sgd_conv2d(Conv2d *c, double lr)
{
    int nW = c->out_channels * c->cg * c->kh * c->kw;
    for (int i = 0; i < nW; i++) c->W[i] -= lr * c->gW[i];
    for (int i = 0; i < c->out_channels; i++) c->b[i] -= lr * c->gb[i];
}

static void sgd_convt(ConvTranspose2d *c, double lr)
{
    int nW = c->in_channels * c->og * c->kh * c->kw;
    for (int i = 0; i < nW; i++) c->W[i] -= lr * c->gW[i];
    for (int i = 0; i < c->out_channels; i++) c->b[i] -= lr * c->gb[i];
}

void p2p_disc_sgd_step(PatchDiscriminator *d, double lr)
{
    sgd_conv2d(as_conv(d->conv1), lr);
    sgd_conv2d(as_conv(d->conv2), lr);
    sgd_conv2d(as_conv(d->conv3), lr);
}

int p2p_disc_forward(PatchDiscriminator *d, const double *image,
                     int C, int H, int W, double **scores, int *sh, int *sw)
{
    if (!d || !image || !scores || !sh || !sw) return -1;
    if (C != d->in_channels || H < 4 || W < 4 || H % 2 || W % 2) {
        fprintf(stderr, "p2p_disc_forward: 需要 C=%d、H/W>=4 的偶数尺寸\n", d->in_channels);
        return -1;
    }
    Conv2d *c1 = as_conv(d->conv1), *c2 = as_conv(d->conv2), *c3 = as_conv(d->conv3);
    const int h1 = (H - 1) / 2 + 1, w1 = (W - 1) / 2 + 1;
    const int h2 = (h1 - 1) / 2 + 1, w2 = (w1 - 1) / 2 + 1;

    Tensor *x = tensor3_from(image, C, H, W);
    Tensor *o1 = tensor3(d->base, h1, w1), *a1 = tensor3(d->base, h1, w1);
    Tensor *o2 = tensor3(2 * d->base, h2, w2), *a2 = tensor3(2 * d->base, h2, w2);
    Tensor *o3 = tensor3(1, h2, w2);

    free(d->m1); d->m1 = alloc_double((size_t)d->base * h1 * w1);
    free(d->m2); d->m2 = alloc_double((size_t)2 * d->base * h2 * w2);
    d->h1 = h1; d->w1 = w1; d->h2 = h2; d->w2 = w2;
    d->in_h = H; d->in_w = W;

    conv_forward_act(c1, x, o1, d->m1, a1->data, P2P_LEAK);
    conv_forward_act(c2, a1, o2, d->m2, a2->data, P2P_LEAK);
    conv2d_forward(c3, a2, o3);

    free(d->scores);
    d->scores = alloc_double((size_t)h2 * w2);
    memcpy(d->scores, o3->data, (size_t)h2 * w2 * sizeof(double));
    d->sh = h2; d->sw = w2;
    *scores = d->scores;
    *sh = h2; *sw = w2;

    tensor_free(x); tensor_free(o1); tensor_free(a1);
    tensor_free(o2); tensor_free(a2); tensor_free(o3);
    return 0;
}

int p2p_disc_backward(PatchDiscriminator *d, const double *dscores, int sh, int sw)
{
    if (!d || !dscores || !d->scores) return -1;
    if (sh != d->sh || sw != d->sw) {
        fprintf(stderr, "p2p_disc_backward: dscore 形状与分数网格不一致\n");
        return -1;
    }
    Conv2d *c1 = as_conv(d->conv1), *c2 = as_conv(d->conv2), *c3 = as_conv(d->conv3);
    const int h1 = d->h1, w1 = d->w1, h2 = d->h2, w2 = d->w2;
    Tensor *d_o3 = tensor3_from(dscores, 1, h2, w2);
    Tensor *d_a2 = tensor3(2 * d->base, h2, w2);
    Tensor *d_o2 = tensor3(2 * d->base, h2, w2);
    Tensor *d_a1 = tensor3(d->base, h1, w1);
    Tensor *d_o1 = tensor3(d->base, h1, w1);
    Tensor *dx = tensor3(d->in_channels, d->in_h, d->in_w);

    conv2d_backward(c3, d_o3, d_a2);
    int n2 = d_a2->size;
    for (int i = 0; i < n2; i++)
        d_o2->data[i] = d_a2->data[i] * (d->m2[i] + P2P_LEAK * (1.0 - d->m2[i]));
    conv2d_backward(c2, d_o2, d_a1);
    int n1 = d_a1->size;
    for (int i = 0; i < n1; i++)
        d_o1->data[i] = d_a1->data[i] * (d->m1[i] + P2P_LEAK * (1.0 - d->m1[i]));
    conv2d_backward(c1, d_o1, dx);

    free(d->dx);
    d->dx = alloc_double((size_t)dx->size);
    memcpy(d->dx, dx->data, (size_t)dx->size * sizeof(double));
    d->dx_n = dx->size;
    d->dx_c = d->in_channels; d->dx_h = d->in_h; d->dx_w = d->in_w;

    tensor_free(d_o3); tensor_free(d_a2); tensor_free(d_o2);
    tensor_free(d_a1); tensor_free(d_o1); tensor_free(dx);
    return 0;
}

int p2p_disc_input_grad(const PatchDiscriminator *d, double **dx, int *n)
{
    if (!d || !d->dx || !dx) return -1;
    *dx = d->dx;
    if (n) *n = d->dx_n;
    return 0;
}

int p2p_disc_set_weights(PatchDiscriminator *d,
                         const double *w1, const double *b1,
                         const double *w2, const double *b2,
                         const double *w3, const double *b3)
{
    if (!d) return -1;
    Conv2d *c1 = as_conv(d->conv1), *c2 = as_conv(d->conv2), *c3 = as_conv(d->conv3);
    conv2d_set_weights(c1, w1, b1);
    conv2d_set_weights(c2, w2, b2);
    conv2d_set_weights(c3, w3, b3);
    return 0;
}

/* ============================ 生成器 ============================ */

Pix2PixGenerator *p2p_gen_create(int in_channels, int out_channels, int base)
{
    if (in_channels < 1 || out_channels < 1 || base < 1) {
        fprintf(stderr, "p2p_gen_create: in/out/base 必须是正整数\n");
        exit(1);
    }
    Pix2PixGenerator *g = (Pix2PixGenerator *)xmalloc(sizeof(*g));
    memset(g, 0, sizeof(*g));
    g->in_channels = in_channels;
    g->out_channels = out_channels;
    g->base = base;
    g->enc1 = conv2d_create(in_channels, base, 3, 2, 1);
    g->enc2 = conv2d_create(base, 2 * base, 3, 2, 1);
    g->dec1 = convt2d_create(2 * base, base, 4, 2, 1, 0, 1);
    g->dec2 = convt2d_create(base, out_channels, 4, 2, 1, 0, 1);
    return g;
}

void p2p_gen_free(Pix2PixGenerator *g)
{
    if (!g) return;
    conv2d_free(as_conv(g->enc1));
    conv2d_free(as_conv(g->enc2));
    convt2d_free(as_convt(g->dec1));
    convt2d_free(as_convt(g->dec2));
    free(g->m1);
    free(g->m2);
    free(g->m3);
    free(g->y);
    free(g);
}

void p2p_gen_zero_grad(Pix2PixGenerator *g)
{
    conv2d_zero_grad(as_conv(g->enc1));
    conv2d_zero_grad(as_conv(g->enc2));
    convt2d_zero_grad(as_convt(g->dec1));
    convt2d_zero_grad(as_convt(g->dec2));
}

int p2p_gen_param_count(const Pix2PixGenerator *g)
{
    return conv2d_param_count(as_conv(g->enc1))
         + conv2d_param_count(as_conv(g->enc2))
         + convt2d_param_count(as_convt(g->dec1))
         + convt2d_param_count(as_convt(g->dec2));
}

void p2p_gen_sgd_step(Pix2PixGenerator *g, double lr)
{
    sgd_conv2d(as_conv(g->enc1), lr);
    sgd_conv2d(as_conv(g->enc2), lr);
    sgd_convt(as_convt(g->dec1), lr);
    sgd_convt(as_convt(g->dec2), lr);
}

int p2p_gen_forward(Pix2PixGenerator *g, const double *image,
                    int C, int H, int W, double **y, int *yh, int *yw)
{
    if (!g || !image || !y || !yh || !yw) return -1;
    if (C != g->in_channels || H < 4 || W < 4 || H % 4 || W % 4) {
        fprintf(stderr, "p2p_gen_forward: 需要 C=%d、H/W 为 >=4 且能被 4 整除\n", g->in_channels);
        return -1;
    }
    int h1 = (H - 1) / 2 + 1, w1 = (W - 1) / 2 + 1;
    int h2 = (h1 - 1) / 2 + 1, w2 = (w1 - 1) / 2 + 1;
    int h3 = (h2 - 1) * 2 - 2 + 4, w3 = (w2 - 1) * 2 - 2 + 4;
    int ho = (h3 - 1) * 2 - 2 + 4, wo = (w3 - 1) * 2 - 2 + 4;

    Tensor *x = tensor3_from(image, C, H, W);
    Tensor *o1 = tensor3(g->base, h1, w1), *a1 = tensor3(g->base, h1, w1);
    Tensor *o2 = tensor3(2 * g->base, h2, w2), *a2 = tensor3(2 * g->base, h2, w2);
    Tensor *o3 = tensor3(g->base, h3, w3), *a3 = tensor3(g->base, h3, w3);
    Tensor *z = tensor3(g->out_channels, ho, wo);

    free(g->m1); g->m1 = alloc_double((size_t)g->base * h1 * w1);
    free(g->m2); g->m2 = alloc_double((size_t)2 * g->base * h2 * w2);
    free(g->m3); g->m3 = alloc_double((size_t)g->base * h3 * w3);
    g->h1 = h1; g->w1 = w1; g->h2 = h2; g->w2 = w2; g->h3 = h3; g->w3 = w3;
    g->in_h = H; g->in_w = W;

    conv_forward_act(as_conv(g->enc1), x, o1, g->m1, a1->data, 0.0);
    conv_forward_act(as_conv(g->enc2), a1, o2, g->m2, a2->data, 0.0);
    convt_forward_act(as_convt(g->dec1), a2, o3, g->m3, a3->data, 0.0);
    convt2d_forward(as_convt(g->dec2), a3, z);

    free(g->y);
    g->y = alloc_double((size_t)z->size);
    for (int i = 0; i < z->size; i++) g->y[i] = tanh(z->data[i]);
    g->yh = ho; g->yw = wo;
    *y = g->y; *yh = ho; *yw = wo;

    tensor_free(x); tensor_free(o1); tensor_free(a1);
    tensor_free(o2); tensor_free(a2); tensor_free(o3);
    tensor_free(a3); tensor_free(z);
    return 0;
}

int p2p_gen_backward(Pix2PixGenerator *g, const double *dy, int C, int H, int W)
{
    if (!g || !dy || !g->y) return -1;
    if (C != g->out_channels) return -1;
    if (g->yh != H || g->yw != W) {
        fprintf(stderr, "p2p_gen_backward: dy 形状与输出不一致\n");
        return -1;
    }
    int h1 = g->h1, w1 = g->w1, h2 = g->h2, w2 = g->w2, h3 = g->h3, w3 = g->w3;
    int n_out = g->out_channels * H * W;

    Tensor *d_z = tensor3(C, H, W);
    for (int i = 0; i < n_out; i++) {
        double yy = g->y[i];
        d_z->data[i] = dy[i] * (1.0 - yy * yy);
    }
    Tensor *d_a3 = tensor3(g->base, h3, w3);
    Tensor *d_o3 = tensor3(g->base, h3, w3);
    Tensor *d_a2 = tensor3(2 * g->base, h2, w2);
    Tensor *d_o2 = tensor3(2 * g->base, h2, w2);
    Tensor *d_a1 = tensor3(g->base, h1, w1);
    Tensor *d_o1 = tensor3(g->base, h1, w1);
    Tensor *dx = tensor3(g->in_channels, g->in_h, g->in_w);

    convt2d_backward(as_convt(g->dec2), d_z, d_a3);
    for (int i = 0; i < d_a3->size; i++) d_o3->data[i] = d_a3->data[i] * g->m3[i];
    convt2d_backward(as_convt(g->dec1), d_o3, d_a2);
    for (int i = 0; i < d_a2->size; i++) d_o2->data[i] = d_a2->data[i] * g->m2[i];
    conv2d_backward(as_conv(g->enc2), d_o2, d_a1);
    for (int i = 0; i < d_a1->size; i++) d_o1->data[i] = d_a1->data[i] * g->m1[i];
    conv2d_backward(as_conv(g->enc1), d_o1, dx);

    tensor_free(d_z); tensor_free(d_a3); tensor_free(d_o3);
    tensor_free(d_a2); tensor_free(d_o2); tensor_free(d_a1);
    tensor_free(d_o1); tensor_free(dx);
    return 0;
}

int p2p_gen_set_weights(Pix2PixGenerator *g,
                        const double *we1, const double *be1,
                        const double *we2, const double *be2,
                        const double *wd1, const double *bd1,
                        const double *wd2, const double *bd2)
{
    if (!g) return -1;
    conv2d_set_weights(as_conv(g->enc1), we1, be1);
    conv2d_set_weights(as_conv(g->enc2), we2, be2);
    convt2d_set_weights(as_convt(g->dec1), wd1, bd1);
    convt2d_set_weights(as_convt(g->dec2), wd2, bd2);
    return 0;
}

/* ============================ 损失与训练步 ============================ */

static double p2p_bce(const double *scores, int n, double target, double *dscore)
{
    if (n <= 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double p = p2p_sigmoid(scores[i]);
        double pc = p;
        if (pc < 1e-12) pc = 1e-12;
        if (pc > 1.0 - 1e-12) pc = 1.0 - 1e-12;
        total -= target * log(pc) + (1.0 - target) * log(1.0 - pc);
        if (dscore) dscore[i] = (p - target) / (double)n;
    }
    return total / (double)n;
}

static double p2p_l1(const double *fake, const double *real, int n,
                     double weight, double *dout)
{
    if (n <= 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fake[i] - real[i];
        total += fabs(diff);
        if (dout) {
            double s = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
            dout[i] = weight * s / (double)n;
        }
    }
    return weight * total / (double)n;
}

int p2p_train_step(Pix2PixGenerator *g, PatchDiscriminator *d,
                   const double *source, int sc, int sh, int sw,
                   const double *target, int tc,
                   double lambda_l1, double lr_g, double lr_d,
                   double *d_loss, double *g_loss)
{
    if (!g || !d || !source || !target) return -1;
    if (sc != g->in_channels || tc != g->out_channels || sh != sw || sh % 4) {
        fprintf(stderr, "p2p_train_step: 通道/尺寸不合法（需要 H=W 且能被 4 整除）\n");
        return -1;
    }
    double *fake = 0, *scores = 0, *dscore = 0, *d_l1 = 0, *d_fake = 0;
    int fh = 0, fw = 0, ssh = 0, ssw = 0;

    /* 1) 判别器：先真图后假图，累积两次梯度后一次性 sgd_step */
    if (p2p_gen_forward(g, source, sc, sh, sw, &fake, &fh, &fw) != 0) return -1;
    p2p_disc_zero_grad(d);

    if (p2p_disc_forward(d, target, tc, sh, sw, &scores, &ssh, &ssw) != 0) goto fail;
    dscore = alloc_double((size_t)ssh * ssw);
    double loss_d_real = p2p_bce(scores, ssh * ssw, 1.0, dscore);
    if (p2p_disc_backward(d, dscore, ssh, ssw) != 0) goto fail;

    if (p2p_disc_forward(d, fake, g->out_channels, fh, fw, &scores, &ssh, &ssw) != 0) goto fail;
    double loss_d_fake = p2p_bce(scores, ssh * ssw, 0.0, dscore);
    if (p2p_disc_backward(d, dscore, ssh, ssw) != 0) goto fail;
    p2p_disc_sgd_step(d, lr_d);

    /* 2) 生成器：BCE(D(fake), 1) + λ·L1(fake, target) */
    p2p_gen_zero_grad(g);
    p2p_disc_zero_grad(d);
    if (p2p_gen_forward(g, source, sc, sh, sw, &fake, &fh, &fw) != 0) goto fail;
    if (p2p_disc_forward(d, fake, g->out_channels, fh, fw, &scores, &ssh, &ssw) != 0) goto fail;
    double loss_bce = p2p_bce(scores, ssh * ssw, 1.0, dscore);
    d_l1 = alloc_double((size_t)g->out_channels * fh * fw);
    double loss_l1 = p2p_l1(fake, target, g->out_channels * fh * fw, lambda_l1, d_l1);
    if (p2p_disc_backward(d, dscore, ssh, ssw) != 0) goto fail;

    double *dx = 0;
    int dxn = 0;
    if (p2p_disc_input_grad(d, &dx, &dxn) != 0) goto fail;
    int nf = g->out_channels * fh * fw;
    if (dxn != nf) {
        fprintf(stderr, "p2p_train_step: 判别器输入梯度大小 %d != %d\n", dxn, nf);
        goto fail;
    }
    d_fake = alloc_double((size_t)nf);
    for (int i = 0; i < nf; i++) d_fake[i] = dx[i] + d_l1[i];
    if (p2p_gen_backward(g, d_fake, g->out_channels, fh, fw) != 0) goto fail;
    p2p_gen_sgd_step(g, lr_g);

#ifdef P2P_DEBUG
            loss_d_real, loss_d_fake, loss_bce, loss_l1);
#endif
    if (d_loss) *d_loss = 0.5 * (loss_d_real + loss_d_fake);
    if (g_loss) *g_loss = loss_bce + loss_l1;
    free(dscore); free(d_l1); free(d_fake);
    return 0;

fail:
    free(dscore); free(d_l1); free(d_fake);
    return -1;
}
