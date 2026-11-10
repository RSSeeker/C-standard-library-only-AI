/* ViT 式视觉 Transformer 前向（patch embedding → CLS + 位置编码 → TransformerEncoder
 * → LayerNorm(CLS) → 线性分类头）。对应 Python 的 AI.ViT（只移植前向）。
 *
 * 设计约定：
 *   - 图像输入 (C, H, W) 展平成 double 数组，行主序 c*H*W + r*W + col；
 *   - patch embedding 就是一个 kernel=stride=patch_size 的 Conv2d；
 *   - 权重通过访问器灌入（conv2d_set_weights / mha_set_params / ff_set_params /
 *     直接写 TransformerEncoderLayer.ln1/ln2 与 LayerNorm 的 gamma/beta）。
 */
#ifndef AI_VIT_H
#define AI_VIT_H

#include "attention.h"
#include "layers_conv.h"
#include "layers_norm.h"
#include "tensor.h"

typedef struct {
    int in_channels;
    int image_h;
    int image_w;
    int patch_size;
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    int num_classes;
    int grid_h;
    int grid_w;
    int num_patches;

    Conv2d *patch_conv;          /* (d_model, in_channels, p, p) */
    TransformerEncoder *encoder;
    LayerNorm *norm;             /* 对 CLS 的 d_model 向量做 LayerNorm */

    double *cls;                 /* (d_model) */
    double *pos;                 /* (num_patches+1, d_model) */
    double *head_W;              /* (num_classes, d_model) */
    double *head_b;              /* (num_classes) */

    Tensor *t_img;               /* (C, H, W) 输入占位 */
    Tensor *t_patch;             /* (d_model, gh, gw) 卷积输出 */
    Tensor *t_seq;               /* (P+1, d_model) */
    Tensor *t_enc;               /* (P+1, d_model) */
    Tensor *t_h;                 /* (d_model) 归一化输入 */
    Tensor *t_norm;              /* (d_model) */
} VitModel;

VitModel *vit_create(int in_channels, int image_h, int image_w, int patch_size,
                     int d_model, int num_heads, int d_ff, int num_layers,
                     int num_classes, unsigned long long seed);
void vit_free(VitModel *m);

int vit_param_count(const VitModel *m);

/* forward：image 长度 C*H*W；logits 长度 num_classes。 */
void vit_forward(VitModel *m, const double *image, double *logits);

/* 权重访问器（对拍时逐块灌入）。 */
Conv2d *vit_patch_conv(VitModel *m);
TransformerEncoder *vit_encoder(VitModel *m);
LayerNorm *vit_norm(VitModel *m);
void vit_set_cls_pos(VitModel *m, const double *cls, const double *pos);
void vit_set_head(VitModel *m, const double *W, const double *b);

#endif /* AI_VIT_H */
