#include "stylegan.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static Tensor *sg_tensor(int c, int h, int w)
{
    int shape[3] = {c, h, w};
    return tensor_new(3, shape);
}

static void sg_adain(const double *feat, double *out, const double *style,
                     int c, int h, int w, double eps)
{
    int n = h * w;
    for (int k = 0; k < c; k++) {
        const double *src = feat + (size_t)k * n;
        double *dst = out + (size_t)k * n;
        double mean = 0.0;
        for (int i = 0; i < n; i++) {
            mean += src[i];
        }
        mean /= (double)n;
        double var = 0.0;
        for (int i = 0; i < n; i++) {
            double d = src[i] - mean;
            var += d * d;
        }
        var /= (double)n;
        double std = sqrt(var + eps);
        double scale = style[k];
        double bias = style[c + k];
        for (int i = 0; i < n; i++) {
            dst[i] = (src[i] - mean) / std * scale + bias;
        }
    }
}

static void sg_relu_inplace(double *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0) {
            x[i] = 0.0;
        }
    }
}

static void sg_upsample(const double *x, int c, int h, int w, double *out)
{
    int hh = h * 2;
    int ww = w * 2;
    for (int k = 0; k < c; k++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                double v = x[((size_t)k * h + i) * w + j];
                int oi = i * 2;
                int oj = j * 2;
                size_t base = ((size_t)k * hh + oi) * ww + oj;
                out[base] = v;
                out[base + 1] = v;
                out[base + ww] = v;
                out[base + ww + 1] = v;
            }
        }
    }
}

StyleGanGenerator *sg_create(int latent_dim, int d_model, int base, int out_channels)
{
    if (latent_dim <= 0 || d_model <= 0 || base <= 0 || out_channels <= 0) {
        return NULL;
    }
    StyleGanGenerator *g = (StyleGanGenerator *)calloc(1, sizeof(*g));
    if (!g) {
        return NULL;
    }
    g->latent_dim = latent_dim;
    g->d_model = d_model;
    g->base = base;
    g->out_channels = out_channels;

    int map_sizes[3] = {latent_dim, d_model, d_model};
    Activation map_acts[2] = {ACT_RELU, ACT_LINEAR};
    int style_sizes[2] = {d_model, 2 * base};
    Activation style_acts[1] = {ACT_LINEAR};

    g->mapping = mlp_create(map_sizes, map_acts, 3);
    g->style1 = mlp_create(style_sizes, style_acts, 2);
    g->style2 = mlp_create(style_sizes, style_acts, 2);
    g->conv1 = conv2d_create(base, base, 3, 1, 1);
    g->conv2 = conv2d_create(base, base, 3, 1, 1);
    g->to_rgb = conv2d_create(base, out_channels, 3, 1, 1);

    g->const_feat = (double *)calloc((size_t)base * 16, sizeof(double));
    g->p1 = (double *)calloc((size_t)base * 16, sizeof(double));
    g->a1 = (double *)calloc((size_t)base * 16, sizeof(double));
    g->u1 = (double *)calloc((size_t)base * 64, sizeof(double));
    g->p2 = (double *)calloc((size_t)base * 64, sizeof(double));
    g->a2 = (double *)calloc((size_t)base * 64, sizeof(double));
    g->u2 = (double *)calloc((size_t)base * 256, sizeof(double));
    g->rgb = (double *)calloc((size_t)out_channels * 256, sizeof(double));
    g->y = (double *)calloc((size_t)out_channels * 256, sizeof(double));
    g->w = (double *)calloc((size_t)d_model, sizeof(double));
    g->style1_out = (double *)calloc((size_t)2 * base, sizeof(double));
    g->style2_out = (double *)calloc((size_t)2 * base, sizeof(double));

    g->t_const = sg_tensor(base, 4, 4);
    g->t_p1 = sg_tensor(base, 4, 4);
    g->t_a1 = sg_tensor(base, 4, 4);
    g->t_u1 = sg_tensor(base, 8, 8);
    g->t_p2 = sg_tensor(base, 8, 8);
    g->t_a2 = sg_tensor(base, 8, 8);
    g->t_u2 = sg_tensor(base, 16, 16);
    g->t_rgb = sg_tensor(out_channels, 16, 16);
    g->t_y = sg_tensor(out_channels, 16, 16);

    if (!g->mapping || !g->style1 || !g->style2 || !g->conv1 || !g->conv2 ||
        !g->to_rgb || !g->const_feat || !g->p1 || !g->a1 || !g->u1 || !g->p2 ||
        !g->a2 || !g->u2 || !g->rgb || !g->y || !g->w || !g->style1_out ||
        !g->style2_out || !g->t_const || !g->t_p1 || !g->t_a1 || !g->t_u1 ||
        !g->t_p2 || !g->t_a2 || !g->t_u2 || !g->t_rgb || !g->t_y) {
        sg_free(g);
        return NULL;
    }
    return g;
}

void sg_free(StyleGanGenerator *g)
{
    if (!g) {
        return;
    }
    if (g->mapping) {
        mlp_free(g->mapping);
    }
    if (g->style1) {
        mlp_free(g->style1);
    }
    if (g->style2) {
        mlp_free(g->style2);
    }
    if (g->conv1) {
        conv2d_free(g->conv1);
    }
    if (g->conv2) {
        conv2d_free(g->conv2);
    }
    if (g->to_rgb) {
        conv2d_free(g->to_rgb);
    }
    free(g->const_feat);
    free(g->p1);
    free(g->a1);
    free(g->u1);
    free(g->p2);
    free(g->a2);
    free(g->u2);
    free(g->rgb);
    free(g->y);
    free(g->w);
    free(g->style1_out);
    free(g->style2_out);
    tensor_free(g->t_const);
    tensor_free(g->t_p1);
    tensor_free(g->t_a1);
    tensor_free(g->t_u1);
    tensor_free(g->t_p2);
    tensor_free(g->t_a2);
    tensor_free(g->t_u2);
    tensor_free(g->t_rgb);
    tensor_free(g->t_y);
    free(g);
}

static int sg_mlp_param_count(const MLP *m)
{
    int total = 0;
    for (int i = 0; i < m->n_layers; i++) {
        total += m->layers[i].fan_in * m->layers[i].fan_out + m->layers[i].fan_out;
    }
    return total;
}

int sg_param_count(const StyleGanGenerator *g)
{
    if (!g) {
        return 0;
    }
    return g->base * 16 + conv2d_param_count(g->conv1) +
           conv2d_param_count(g->conv2) + conv2d_param_count(g->to_rgb) +
           sg_mlp_param_count(g->style1) + sg_mlp_param_count(g->style2) +
           sg_mlp_param_count(g->mapping);
}

int sg_forward(StyleGanGenerator *g, const double *z, double *out)
{
    if (!g || !z || !out) {
        return -1;
    }
    const int H = 4;
    const int W = 4;
    const int B = g->base;
    const int OC = g->out_channels;
    const double eps = 1e-5;

    /* 常数特征 -> conv1 */
    memcpy(g->t_const->data, g->const_feat, (size_t)B * H * W * sizeof(double));
    conv2d_forward(g->conv1, g->t_const, g->t_p1);
    memcpy(g->p1, g->t_p1->data, (size_t)B * H * W * sizeof(double));

    /* style1 / adain / relu / upsample */
    mlp_forward(g->mapping, z, g->w);
    mlp_forward(g->style1, g->w, g->style1_out);
    sg_adain(g->p1, g->a1, g->style1_out, B, H, W, eps);
    sg_relu_inplace(g->a1, B * H * W);
    sg_upsample(g->a1, B, H, W, g->u1);
    memcpy(g->t_u1->data, g->u1, (size_t)B * 64 * sizeof(double));

    /* conv2 -> style2 / adain / relu / upsample */
    conv2d_forward(g->conv2, g->t_u1, g->t_p2);
    memcpy(g->p2, g->t_p2->data, (size_t)B * 64 * sizeof(double));
    mlp_forward(g->style2, g->w, g->style2_out);
    sg_adain(g->p2, g->a2, g->style2_out, B, 8, 8, eps);
    sg_relu_inplace(g->a2, B * 64);
    sg_upsample(g->a2, B, 8, 8, g->u2);
    memcpy(g->t_u2->data, g->u2, (size_t)B * 256 * sizeof(double));

    /* to_rgb -> tanh */
    conv2d_forward(g->to_rgb, g->t_u2, g->t_rgb);
    memcpy(g->rgb, g->t_rgb->data, (size_t)OC * 256 * sizeof(double));
    for (int i = 0; i < OC * 256; i++) {
        g->y[i] = tanh(g->rgb[i]);
    }
    memcpy(out, g->y, (size_t)OC * 256 * sizeof(double));
    return 0;
}

int sg_average_w(StyleGanGenerator *g, const double *zs, int n, double *out_w)
{
    if (!g || !zs || n <= 0 || !out_w) {
        return -1;
    }
    memset(out_w, 0, (size_t)g->d_model * sizeof(double));
    for (int s = 0; s < n; s++) {
        mlp_forward(g->mapping, zs + (size_t)s * g->latent_dim, g->w);
        for (int k = 0; k < g->d_model; k++) {
            out_w[k] += g->w[k];
        }
    }
    for (int k = 0; k < g->d_model; k++) {
        out_w[k] /= (double)n;
    }
    return 0;
}

const double *sg_last_w(const StyleGanGenerator *g)
{
    return g ? g->w : NULL;
}
