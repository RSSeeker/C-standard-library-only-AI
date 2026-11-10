/* ViT 式视觉 Transformer 前向实现，见 vit.h。 */
#include "vit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void vit_die(const char *msg) {
    fprintf(stderr, "ViT: %s\n", msg);
    abort();
}

static double vit_u01(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (double)((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
}

static double vit_gauss(unsigned long long *s) {
    double u1 = vit_u01(s), u2 = vit_u01(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * u2);
}

VitModel *vit_create(int in_channels, int image_h, int image_w, int patch_size,
                     int d_model, int num_heads, int d_ff, int num_layers,
                     int num_classes, unsigned long long seed) {
    if (in_channels < 1 || image_h < 1 || image_w < 1 || patch_size < 1 ||
        d_model < 1 || num_heads < 1 || d_ff < 1 || num_layers < 1 || num_classes < 1) {
        vit_die("create: 参数必须是正整数");
    }
    if (image_h % patch_size != 0 || image_w % patch_size != 0) {
        vit_die("create: image_size 必须能被 patch_size 整除");
    }
    if (d_model % num_heads != 0) vit_die("create: d_model 必须能被 num_heads 整除");

    VitModel *m = (VitModel *)calloc(1, sizeof(VitModel));
    if (!m) vit_die("create: 内存不足");
    m->in_channels = in_channels;
    m->image_h = image_h;
    m->image_w = image_w;
    m->patch_size = patch_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->num_classes = num_classes;
    m->grid_h = image_h / patch_size;
    m->grid_w = image_w / patch_size;
    m->num_patches = m->grid_h * m->grid_w;
    int P = m->num_patches;
    int seq_len = P + 1;

    m->patch_conv = conv2d_create(in_channels, d_model, patch_size, patch_size, 0);
    m->encoder = te_create(d_model, num_heads, d_ff, num_layers, seq_len, 0.0, seed);
    te_set_train(m->encoder, 0);
    m->norm = layernorm_create(d_model, 1e-5);

    m->cls = (double *)calloc((size_t)d_model, sizeof(double));
    m->pos = (double *)calloc((size_t)seq_len * (size_t)d_model, sizeof(double));
    m->head_W = (double *)calloc((size_t)num_classes * (size_t)d_model, sizeof(double));
    m->head_b = (double *)calloc((size_t)num_classes, sizeof(double));
    if (!m->cls || !m->pos || !m->head_W || !m->head_b) vit_die("create: 内存不足");

    unsigned long long st = seed ? seed : 0x20240501ULL;
    double std = sqrt(1.0 / (double)d_model);
    for (int i = 0; i < d_model; i++) m->cls[i] = vit_gauss(&st) * std;
    for (int i = 0; i < seq_len * d_model; i++) m->pos[i] = vit_gauss(&st) * std;
    for (int i = 0; i < num_classes * d_model; i++) m->head_W[i] = vit_gauss(&st) * std;

    int img_shape[3] = { in_channels, image_h, image_w };
    int patch_shape[3] = { d_model, m->grid_h, m->grid_w };
    int seq_shape[2] = { seq_len, d_model };
    int vec_shape[1] = { d_model };
    m->t_img = tensor_new(3, img_shape);
    m->t_patch = tensor_new(3, patch_shape);
    m->t_seq = tensor_new(2, seq_shape);
    m->t_enc = tensor_new(2, seq_shape);
    m->t_h = tensor_new(1, vec_shape);
    m->t_norm = tensor_new(1, vec_shape);
    return m;
}

void vit_free(VitModel *m) {
    if (!m) return;
    conv2d_free(m->patch_conv);
    te_free(m->encoder);
    layernorm_free(m->norm);
    free(m->cls);
    free(m->pos);
    free(m->head_W);
    free(m->head_b);
    tensor_free(m->t_img);
    tensor_free(m->t_patch);
    tensor_free(m->t_seq);
    tensor_free(m->t_enc);
    tensor_free(m->t_h);
    tensor_free(m->t_norm);
    free(m);
}

int vit_param_count(const VitModel *m) {
    int n = conv2d_param_count(m->patch_conv);
    n += te_param_count(m->encoder);
    n += m->d_model * 2;                      /* norm gamma/beta */
    n += m->d_model;                          /* cls */
    n += (m->num_patches + 1) * m->d_model;   /* pos */
    n += m->num_classes * m->d_model + m->num_classes;
    return n;
}

Conv2d *vit_patch_conv(VitModel *m) { return m->patch_conv; }
TransformerEncoder *vit_encoder(VitModel *m) { return m->encoder; }
LayerNorm *vit_norm(VitModel *m) { return m->norm; }

void vit_set_cls_pos(VitModel *m, const double *cls, const double *pos) {
    memcpy(m->cls, cls, (size_t)m->d_model * sizeof(double));
    memcpy(m->pos, pos, (size_t)(m->num_patches + 1) * (size_t)m->d_model * sizeof(double));
}

void vit_set_head(VitModel *m, const double *W, const double *b) {
    memcpy(m->head_W, W, (size_t)m->num_classes * (size_t)m->d_model * sizeof(double));
    memcpy(m->head_b, b, (size_t)m->num_classes * sizeof(double));
}

void vit_forward(VitModel *m, const double *image, double *logits) {
    if (!image || !logits) vit_die("forward: 输入/输出不能为空");
    memcpy(m->t_img->data, image,
           (size_t)m->in_channels * (size_t)m->image_h * (size_t)m->image_w * sizeof(double));
    conv2d_forward(m->patch_conv, m->t_img, m->t_patch);

    const int d = m->d_model;
    const int gh = m->grid_h, gw = m->grid_w;
    double *seq = m->t_seq->data;
    memcpy(seq, m->cls, (size_t)d * sizeof(double));
    for (int j = 0; j < d; j++) seq[j] += m->pos[j];
    for (int r = 0; r < gh; r++) {
        for (int c = 0; c < gw; c++) {
            int tk = 1 + r * gw + c;
            double *row = seq + (size_t)tk * d;
            for (int j = 0; j < d; j++) {
                row[j] = m->t_patch->data[((size_t)j * gh + r) * gw + c] +
                         m->pos[(size_t)tk * d + j];
            }
        }
    }

    te_forward(m->encoder, m->t_seq, m->t_enc);

    memcpy(m->t_h->data, m->t_enc->data, (size_t)d * sizeof(double));
    layernorm_forward(m->norm, m->t_h, m->t_norm);
    for (int c = 0; c < m->num_classes; c++) {
        const double *row = m->head_W + (size_t)c * d;
        double sum = 0.0;
        for (int j = 0; j < d; j++) sum += row[j] * m->t_norm->data[j];
        logits[c] = sum + m->head_b[c];
    }
}
