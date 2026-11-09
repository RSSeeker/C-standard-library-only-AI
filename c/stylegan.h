#ifndef STYLEGAN_H
#define STYLEGAN_H

#include "ai.h"
#include "layers_conv.h"
#include "tensor.h"

/* 简化 StyleGAN 生成器（只做前向推理，与 Python AI.StyleGANGenerator 对齐）
 *
 * 流程：z -> Mapping MLP -> w；常数 4x4 特征 -> conv1 -> AdaIN(style1) -> ReLU
 *       -> 最近邻上采样 8x8 -> conv2 -> AdaIN(style2) -> ReLU -> 上采样 16x16
 *       -> to_rgb -> tanh。
 *
 * 所有字段公开，便于对拍脚本直接灌入权重（conv 用 conv2d_set_weights，
 * 全连接层直接写 layer->W / layer->b）。
 */
typedef struct {
    int latent_dim;
    int d_model;
    int base;
    int out_channels;

    MLP *mapping;      /* [latent_dim, d_model, d_model]，relu + linear */
    MLP *style1;       /* [d_model, 2*base]，linear */
    MLP *style2;       /* [d_model, 2*base]，linear */
    Conv2d *conv1;     /* base -> base, k=3, pad=1 */
    Conv2d *conv2;     /* base -> base, k=3, pad=1 */
    Conv2d *to_rgb;    /* base -> out_channels, k=3, pad=1 */

    double *const_feat; /* base * 4 * 4 */

    /* 工作缓冲（通道主序） */
    double *p1;   /* base*4*4 */
    double *a1;   /* base*4*4 */
    double *u1;   /* base*8*8 */
    double *p2;   /* base*8*8 */
    double *a2;   /* base*8*8 */
    double *u2;   /* base*16*16 */
    double *rgb;  /* out_channels*16*16 */
    double *y;    /* out_channels*16*16 */

    double *w;          /* d_model */
    double *style1_out; /* 2*base */
    double *style2_out; /* 2*base */

    Tensor *t_const;
    Tensor *t_p1;
    Tensor *t_a1;
    Tensor *t_u1;
    Tensor *t_p2;
    Tensor *t_a2;
    Tensor *t_u2;
    Tensor *t_rgb;
    Tensor *t_y;
} StyleGanGenerator;

StyleGanGenerator *sg_create(int latent_dim, int d_model, int base, int out_channels);
void sg_free(StyleGanGenerator *g);

int sg_param_count(const StyleGanGenerator *g);

/* z 长度 latent_dim；out 长度 out_channels*16*16（+ tanh 后的输出）。 */
int sg_forward(StyleGanGenerator *g, const double *z, double *out);

/* 对 n 个 z（n*latent_dim 扁平）的 mapping 输出取平均，out_w 长度 d_model。 */
int sg_average_w(StyleGanGenerator *g, const double *zs, int n, double *out_w);

/* 取最近一次前向的 w（长度 d_model），供 truncate 使用。 */
const double *sg_last_w(const StyleGanGenerator *g);

#endif /* STYLEGAN_H */
