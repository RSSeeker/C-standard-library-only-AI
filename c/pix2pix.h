#ifndef PIX2PIX_H
#define PIX2PIX_H

/* ======================================================================
 *  Pix2Pix 可训练基础设施（小型图像到图像）
 *
 *  对应 Python AI.py 的 PatchDiscriminator / Pix2PixGenerator /
 *  train_pix2pix_step：
 *    - 判别器：三层 PatchGAN（卷积 stride-2、stride-2、stride-1），LeakyReLU(0.2)
 *    - 生成器：两次 stride-2 卷积编码 + 两次转置卷积解码，ReLU + tanh 输出
 *    - 一步训练：先更新判别器（真图目标 1 / 假图目标 0），再更新生成器
 *      （BCE(D(fake), 1) + lambda_l1 * L1(fake, target)）
 *  像素约定：通道主序，image[k*H*W + r*W + c]。
 * ========================================================================== */

typedef struct {
    int in_channels;
    int base;

    void *conv1;
    void *conv2;
    void *conv3;

    /* 前向缓存 */
    double *m1;
    double *m2;
    int h1, w1;
    int h2, w2;
    double *scores;
    int sh, sw;

    /* 输入尺寸与输入梯度（backward 时保存，供生成器反向链读取） */
    int in_h, in_w;
    double *dx;
    int dx_n, dx_c, dx_h, dx_w;
} PatchDiscriminator;

typedef struct {
    int in_channels;
    int out_channels;
    int base;

    void *enc1;
    void *enc2;
    void *dec1;
    void *dec2;

    /* 前向缓存 */
    double *m1;
    double *m2;
    double *m3;
    int h1, w1;
    int h2, w2;
    int h3, w3;
    double *y;
    int yh, yw;

    /* 输入尺寸（backward 反推 dx 尺寸用） */
    int in_h, in_w;
} Pix2PixGenerator;

PatchDiscriminator *p2p_disc_create(int in_channels, int base);
void p2p_disc_free(PatchDiscriminator *d);
void p2p_disc_zero_grad(PatchDiscriminator *d);
void p2p_disc_sgd_step(PatchDiscriminator *d, double lr);
int  p2p_disc_param_count(const PatchDiscriminator *d);

/* 返回 malloc 的分数网格（sh x sw），调用方 free。成功 0。 */
int p2p_disc_forward(PatchDiscriminator *d, const double *image,
                     int C, int H, int W, double **scores, int *sh, int *sw);
/* dscores: sh x sw；反向累积各层梯度。成功 0。 */
int p2p_disc_backward(PatchDiscriminator *d, const double *dscores,
                      int sh, int sw);
/* 取输入图像梯度（反向之后调用；调用方 free） */
int p2p_disc_input_grad(const PatchDiscriminator *d, double **dx, int *n);

Pix2PixGenerator *p2p_gen_create(int in_channels, int out_channels, int base);
void p2p_gen_free(Pix2PixGenerator *g);
void p2p_gen_zero_grad(Pix2PixGenerator *g);
void p2p_gen_sgd_step(Pix2PixGenerator *g, double lr);
int  p2p_gen_param_count(const Pix2PixGenerator *g);

/* 返回 malloc 的输出图像（out_channels x yh x yw），调用方 free。成功 0。 */
int p2p_gen_forward(Pix2PixGenerator *g, const double *image,
                    int C, int H, int W, double **y, int *yh, int *yw);
/* dy: out_channels x yh x yw；反向累积各层梯度。成功 0。 */
int p2p_gen_backward(Pix2PixGenerator *g, const double *dy,
                     int C, int H, int W);

/* 从外部灌入权重（跨语言对拍用）。
 * disc W1/W2/W3 按 [oc][ic][kh][kw] 展平；gen enc 同 Conv2d 布局，
 * dec 按 ConvTranspose2d 的 [cin][out][kh][kw] 布局。 */
int p2p_disc_set_weights(PatchDiscriminator *d,
                         const double *w1, const double *b1,
                         const double *w2, const double *b2,
                         const double *w3, const double *b3);
int p2p_gen_set_weights(Pix2PixGenerator *g,
                        const double *we1, const double *be1,
                        const double *we2, const double *be2,
                        const double *wd1, const double *bd1,
                        const double *wd2, const double *bd2);

/* 一步训练：先 D 后 G，返回两项损失。成功 0。 */
int p2p_train_step(Pix2PixGenerator *g, PatchDiscriminator *d,
                   const double *source, int sc, int sh, int sw,
                   const double *target, int tc,
                   double lambda_l1, double lr_g, double lr_d,
                   double *d_loss, double *g_loss);

#endif /* PIX2PIX_H */
