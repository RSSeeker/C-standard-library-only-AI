#include "progressive.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int pg_mlp_param_count(const MLP *m)
{
    int total = 0;
    for (int i = 0; i < m->n_layers; i++) {
        total += m->layers[i].fan_in * m->layers[i].fan_out + m->layers[i].fan_out;
    }
    return total;
}

static void pg_relu_inplace(double *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0) {
            x[i] = 0.0;
        }
    }
}

static int pg_check_channels(const int *channels, int n)
{
    if (!channels || n < 1) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (channels[i] < 1) {
            return 0;
        }
    }
    return 1;
}

ProgressiveGenerator *pg_create(int latent_dim, const int *channels, int n_stages,
                                int base, int out_channels)
{
    if (latent_dim < 1 || base < 1 || out_channels < 1 ||
        !pg_check_channels(channels, n_stages)) {
        return NULL;
    }
    ProgressiveGenerator *g = (ProgressiveGenerator *)calloc(1, sizeof(*g));
    if (!g) {
        return NULL;
    }
    g->latent_dim = latent_dim;
    g->base = base;
    g->out_channels = out_channels;
    g->n_stages = n_stages;
    g->channels = (int *)calloc((size_t)n_stages, sizeof(int));
    g->convs = (ConvTranspose2d **)calloc((size_t)(n_stages - 1 > 0 ? n_stages - 1 : 1),
                                         sizeof(ConvTranspose2d *));
    g->rgb_heads = (Conv2d **)calloc((size_t)n_stages, sizeof(Conv2d *));
    if (!g->channels || !g->convs || !g->rgb_heads) {
        pg_free(g);
        return NULL;
    }
    memcpy(g->channels, channels, (size_t)n_stages * sizeof(int));
    int feat = channels[0] * base * base;
    int fc_sizes[2] = {latent_dim, feat};
    Activation fc_act[1] = {ACT_RELU};
    g->fc = mlp_create(fc_sizes, fc_act, 2);
    g->fc_out = (double *)calloc((size_t)feat, sizeof(double));
    if (!g->fc || !g->fc_out) {
        pg_free(g);
        return NULL;
    }
    for (int i = 1; i < n_stages; i++) {
        g->convs[i - 1] = convt2d_create(channels[i - 1], channels[i], 4, 2, 1, 0, 1);
        if (!g->convs[i - 1]) {
            pg_free(g);
            return NULL;
        }
    }
    for (int i = 0; i < n_stages; i++) {
        g->rgb_heads[i] = conv2d_create(channels[i], out_channels, 1, 1, 0);
        if (!g->rgb_heads[i]) {
            pg_free(g);
            return NULL;
        }
    }
    return g;
}

void pg_free(ProgressiveGenerator *g)
{
    if (!g) {
        return;
    }
    if (g->convs) {
        for (int i = 0; i < g->n_stages - 1; i++) {
            convt2d_free(g->convs[i]);
        }
        free(g->convs);
    }
    if (g->rgb_heads) {
        for (int i = 0; i < g->n_stages; i++) {
            conv2d_free(g->rgb_heads[i]);
        }
        free(g->rgb_heads);
    }
    if (g->fc) {
        mlp_free(g->fc);
    }
    free(g->channels);
    free(g->fc_out);
    free(g);
}

void pg_out_shape(const ProgressiveGenerator *g, int *c, int *side)
{
    if (!g) {
        if (c) {
            *c = 0;
        }
        if (side) {
            *side = 0;
        }
        return;
    }
    if (c) {
        *c = g->out_channels;
    }
    if (side) {
        *side = g->base << (g->n_stages - 1);
    }
}

int pg_grow(ProgressiveGenerator *g)
{
    if (!g) {
        return -1;
    }
    int last = g->channels[g->n_stages - 1];
    int *new_ch = (int *)realloc(g->channels, (size_t)(g->n_stages + 1) * sizeof(int));
    if (!new_ch) {
        return -1;
    }
    g->channels = new_ch;
    g->channels[g->n_stages] = last;

    ConvTranspose2d **new_convs = (ConvTranspose2d **)realloc(
        g->convs, (size_t)g->n_stages * sizeof(ConvTranspose2d *));
    if (!new_convs) {
        return -1;
    }
    g->convs = new_convs;
    Conv2d **new_heads = (Conv2d **)realloc(
        g->rgb_heads, (size_t)(g->n_stages + 1) * sizeof(Conv2d *));
    if (!new_heads) {
        return -1;
    }
    g->rgb_heads = new_heads;

    ConvTranspose2d *conv = convt2d_create(last, last, 4, 2, 1, 0, 1);
    Conv2d *head = conv2d_create(last, g->out_channels, 1, 1, 0);
    if (!conv || !head) {
        convt2d_free(conv);
        conv2d_free(head);
        return -1;
    }
    g->convs[g->n_stages - 1] = conv;
    g->rgb_heads[g->n_stages] = head;
    g->n_stages += 1;
    return g->base << (g->n_stages - 1);
}

int pg_param_count(const ProgressiveGenerator *g)
{
    if (!g) {
        return 0;
    }
    int total = pg_mlp_param_count(g->fc);
    for (int i = 0; i < g->n_stages - 1; i++) {
        total += convt2d_param_count(g->convs[i]);
    }
    for (int i = 0; i < g->n_stages; i++) {
        total += conv2d_param_count(g->rgb_heads[i]);
    }
    return total;
}

int pg_forward(ProgressiveGenerator *g, const double *z, double alpha, double *out)
{
    if (!g || !z || !out || !(alpha >= 0.0 && alpha <= 1.0)) {
        return -1;
    }
    const int oc = g->out_channels;
    const int c0 = g->channels[0];
    int h = g->base;
    mlp_forward(g->fc, z, g->fc_out);

    int shape[3] = {c0, h, h};
    Tensor *xf = tensor_new(3, shape);
    if (!xf) {
        return -1;
    }
    memcpy(xf->data, g->fc_out, (size_t)c0 * h * h * sizeof(double));

    int maxside = g->base << (g->n_stages - 1);
    double *prev = (double *)malloc((size_t)(oc * h * h) * sizeof(double));
    double *rgb_cur = (double *)malloc((size_t)oc * maxside * maxside * sizeof(double));
    if (!prev || !rgb_cur) {
        tensor_free(xf);
        free(prev);
        free(rgb_cur);
        return -1;
    }

    /* 阶段 0：1x1 RGB 头 */
    {
        int rshape[3] = {oc, h, h};
        Tensor *rgb = tensor_new(3, rshape);
        if (!rgb) {
            tensor_free(xf);
            free(prev);
            free(rgb_cur);
            return -1;
        }
        conv2d_bind_shape(g->rgb_heads[0], c0, h, h);
        conv2d_forward(g->rgb_heads[0], xf, rgb);
        memcpy(prev, rgb->data, (size_t)oc * h * h * sizeof(double));
        memcpy(rgb_cur, rgb->data, (size_t)oc * h * h * sizeof(double));
        tensor_free(rgb);
    }
    int ph = h;

    for (int stage = 1; stage < g->n_stages; stage++) {
        int hi = ph * 2;
        int ci = g->channels[stage];
        int ishape[3] = {ci, hi, hi};
        Tensor *xo = tensor_new(3, ishape);
        if (!xo) {
            tensor_free(xf);
            free(prev);
            free(rgb_cur);
            return -1;
        }
        convt2d_bind_shape(g->convs[stage - 1], g->channels[stage - 1], ph, ph);
        convt2d_forward(g->convs[stage - 1], xf, xo);
        tensor_free(xf);
        xf = xo;
        pg_relu_inplace(xf->data, ci * hi * hi);

        int rshape[3] = {oc, hi, hi};
        Tensor *rgb = tensor_new(3, rshape);
        if (!rgb) {
            tensor_free(xf);
            free(prev);
            free(rgb_cur);
            return -1;
        }
        conv2d_bind_shape(g->rgb_heads[stage], ci, hi, hi);
        conv2d_forward(g->rgb_heads[stage], xf, rgb);
        memcpy(rgb_cur, rgb->data, (size_t)oc * hi * hi * sizeof(double));

        int last = (stage == g->n_stages - 1);
        if (last) {
            if (alpha < 1.0) {
                for (int k = 0; k < oc; k++) {
                    for (int i = 0; i < hi; i++) {
                        for (int j = 0; j < hi; j++) {
                            double lo = prev[((size_t)k * ph + i / 2) * ph + j / 2];
                            double hi_v = rgb_cur[((size_t)k * hi + i) * hi + j];
                            out[((size_t)k * hi + i) * hi + j] =
                                tanh((1.0 - alpha) * lo + alpha * hi_v);
                        }
                    }
                }
            } else {
                int n = oc * hi * hi;
                for (int i = 0; i < n; i++) {
                    out[i] = tanh(rgb_cur[i]);
                }
            }
        } else {
            free(prev);
            prev = (double *)malloc((size_t)oc * hi * hi * sizeof(double));
            if (!prev) {
                tensor_free(rgb);
                tensor_free(xf);
                free(rgb_cur);
                return -1;
            }
            memcpy(prev, rgb_cur, (size_t)oc * hi * hi * sizeof(double));
        }
        tensor_free(rgb);
        ph = hi;
    }

    if (g->n_stages == 1) {
        int n = oc * h * h;
        for (int i = 0; i < n; i++) {
            out[i] = tanh(rgb_cur[i]);
        }
    }

    tensor_free(xf);
    free(prev);
    free(rgb_cur);
    return 0;
}
