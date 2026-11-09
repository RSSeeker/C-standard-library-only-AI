/*
 * visual_metrics.c —— LPIPS / FVD 轻量代理的 C 实现
 *
 * 逐项对齐 Python 版 AI.py 的 lpips_lite / fvd_lite：相同的通道平均、池化、
 * 梯度特征顺序、单位化与求和顺序，保证同一组输入下的数值一致（由
 * tools/visual_metrics_reference.py + tests/compare_visual_metrics 验证）。
 */
#include "visual_metrics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 形状校验 ---------- */

static int vm_valid_image(const VmImage *img)
{
    if (!img || !img->pixels) return 0;
    if (img->channels <= 0 || img->height <= 0 || img->width <= 0) return 0;
    return 1;
}

static int vm_same_shape(const VmImage *a, const VmImage *b)
{
    return a->channels == b->channels && a->height == b->height && a->width == b->width;
}

/* ---------- 灰度 / 池化 / 梯度特征 / 单位化 ---------- */

static void vm_gray(const VmImage *img, double *out)
{
    const int c = img->channels;
    const int h = img->height;
    const int w = img->width;
    for (int r = 0; r < h; r++) {
        for (int col = 0; col < w; col++) {
            double acc = 0.0;
            for (int k = 0; k < c; k++) {
                acc += img->pixels[(size_t)k * (size_t)h * (size_t)w
                                   + (size_t)r * (size_t)w + (size_t)col];
            }
            out[(size_t)r * (size_t)w + (size_t)col] = acc / (double)c;
        }
    }
}

/* 平均池化：返回 0 成功；尺寸不足返回 -1。 */
static int vm_pool(const double *gray, int h, int w, int scale, double *out,
                   int *out_h, int *out_w)
{
    if (scale <= 1) {
        memcpy(out, gray, sizeof(double) * (size_t)h * (size_t)w);
        *out_h = h;
        *out_w = w;
        return 0;
    }
    const int oh = h / scale;
    const int ow = w / scale;
    if (oh < 1 || ow < 1) return -1;
    for (int r = 0; r < oh; r++) {
        for (int c = 0; c < ow; c++) {
            double acc = 0.0;
            for (int dr = 0; dr < scale; dr++) {
                for (int dc = 0; dc < scale; dc++) {
                    acc += gray[(size_t)(r * scale + dr) * (size_t)w
                                + (size_t)(c * scale + dc)];
                }
            }
            out[(size_t)r * (size_t)ow + (size_t)c] = acc / (double)(scale * scale);
        }
    }
    *out_h = oh;
    *out_w = ow;
    return 0;
}

/* 特征：每个位置依次追加灰度值、水平梯度（末尾列 0）、垂直梯度（末行 0）。 */
static void vm_gradient_feature(const double *gray, int h, int w, double *out)
{
    size_t pos = 0;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            const double g = gray[(size_t)r * (size_t)w + (size_t)c];
            const double gx = (c + 1 < w) ? gray[(size_t)r * (size_t)w + (size_t)(c + 1)] - g : 0.0;
            const double gy = (r + 1 < h) ? gray[(size_t)(r + 1) * (size_t)w + (size_t)c] - g : 0.0;
            out[pos++] = g;
            out[pos++] = gx;
            out[pos++] = gy;
        }
    }
}

/* 单位化：norm<=eps 时全零；否则逐元素除以范数。 */
static void vm_unit(double *vec, int n, double eps)
{
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += vec[i] * vec[i];
    const double norm = sqrt(acc);
    if (norm <= eps) {
        for (int i = 0; i < n; i++) vec[i] = 0.0;
        return;
    }
    for (int i = 0; i < n; i++) vec[i] = vec[i] / norm;
}

/* 单尺度特征（灰度→池化→梯度→单位化），返回特征维度；失败返回 -1。 */
static int vm_scale_feature(const VmImage *img, int scale, double eps,
                            double **out_vec)
{
    *out_vec = NULL;
    const int h = img->height;
    const int w = img->width;
    const int n_gray = h * w;
    double *gray = (double *)malloc(sizeof(double) * (size_t)n_gray);
    double *pooled = (double *)malloc(sizeof(double) * (size_t)n_gray);
    if (!gray || !pooled) {
        free(gray);
        free(pooled);
        return -1;
    }
    vm_gray(img, gray);
    int oh = 0, ow = 0;
    if (vm_pool(gray, h, w, scale, pooled, &oh, &ow) != 0) {
        free(gray);
        free(pooled);
        return -1;
    }
    const int dim = 3 * oh * ow;
    double *feat = (double *)malloc(sizeof(double) * (size_t)dim);
    if (!feat) {
        free(gray);
        free(pooled);
        return -1;
    }
    vm_gradient_feature(pooled, oh, ow, feat);
    vm_unit(feat, dim, eps);
    free(gray);
    free(pooled);
    *out_vec = feat;
    return dim;
}

/* ---------- LPIPS ---------- */

double vm_lpips_lite(const VmImage *a, const VmImage *b,
                     const int *scales, int n_scales,
                     const double *weights, double eps)
{
    if (!vm_valid_image(a) || !vm_valid_image(b)) return NAN;
    if (!vm_same_shape(a, b)) return NAN;
    if (!scales || n_scales <= 0) return NAN;
    for (int i = 0; i < n_scales; i++) {
        if (scales[i] < 1) return NAN;
    }
    double *w = (double *)malloc(sizeof(double) * (size_t)n_scales);
    if (!w) return NAN;
    if (!weights) {
        for (int i = 0; i < n_scales; i++) w[i] = 1.0 / (double)n_scales;
    } else {
        double tot = 0.0;
        for (int i = 0; i < n_scales; i++) {
            if (!isfinite(weights[i]) || weights[i] < 0.0) {
                free(w);
                return NAN;
            }
            tot += weights[i];
        }
        if (tot <= 0.0) {
            free(w);
            return NAN;
        }
        for (int i = 0; i < n_scales; i++) w[i] = weights[i] / tot;
    }

    double total = 0.0;
    for (int i = 0; i < n_scales; i++) {
        double *fa = NULL;
        double *fb = NULL;
        const int da = vm_scale_feature(a, scales[i], eps, &fa);
        const int db = vm_scale_feature(b, scales[i], eps, &fb);
        if (da < 0 || db < 0 || !fa || !fb || da != db) {
            free(fa);
            free(fb);
            free(w);
            return NAN;
        }
        double mse = 0.0;
        for (int j = 0; j < da; j++) {
            const double d = fa[j] - fb[j];
            mse += d * d;
        }
        mse /= (double)da;
        total += w[i] * sqrt(mse);
        free(fa);
        free(fb);
    }
    free(w);
    return total;
}

/* ---------- FVD ---------- */

int vm_fvd_embed_frame(const VmImage *frame, double *out, int out_cap)
{
    if (!vm_valid_image(frame) || !out) return -1;
    const int h = frame->height;
    const int w = frame->width;
    const int oh = h / 2;
    const int ow = w / 2;
    if (oh < 1 || ow < 1) return -1;
    const int dim = 3 * oh * ow;
    if (out_cap < dim) return -1;
    double *gray = (double *)malloc(sizeof(double) * (size_t)h * (size_t)w);
    double *pooled = (double *)malloc(sizeof(double) * (size_t)h * (size_t)w);
    if (!gray || !pooled) {
        free(gray);
        free(pooled);
        return -1;
    }
    vm_gray(frame, gray);
    int ph = 0, pw = 0;
    if (vm_pool(gray, h, w, 2, pooled, &ph, &pw) != 0 || 3 * ph * pw != dim) {
        free(gray);
        free(pooled);
        return -1;
    }
    vm_gradient_feature(pooled, ph, pw, out);
    vm_unit(out, dim, 1e-12); /* Python _pm_unit 的默认 eps */
    free(gray);
    free(pooled);
    return dim;
}

double vm_fvd_lite(const VmImage *frames_a, int n_a,
                   const VmImage *frames_b, int n_b, double eps)
{
    if (!frames_a || !frames_b || n_a < 2 || n_b < 2) return NAN;
    if (!vm_valid_image(&frames_a[0]) || !vm_valid_image(&frames_b[0])) return NAN;
    if (!vm_same_shape(&frames_a[0], &frames_b[0])) return NAN;

    const int h = frames_a[0].height;
    const int w = frames_a[0].width;
    const int oh = h / 2;
    const int ow = w / 2;
    if (oh < 1 || ow < 1) return NAN;
    const int dim = 3 * oh * ow;

    double *fa = (double *)malloc(sizeof(double) * (size_t)n_a * (size_t)dim);
    double *fb = (double *)malloc(sizeof(double) * (size_t)n_b * (size_t)dim);
    if (!fa || !fb) {
        free(fa);
        free(fb);
        return NAN;
    }
    for (int t = 0; t < n_a; t++) {
        if (!vm_valid_image(&frames_a[t]) ||
            vm_fvd_embed_frame(&frames_a[t], fa + (size_t)t * (size_t)dim, dim) < 0) {
            free(fa);
            free(fb);
            return NAN;
        }
    }
    for (int t = 0; t < n_b; t++) {
        if (!vm_valid_image(&frames_b[t]) ||
            vm_fvd_embed_frame(&frames_b[t], fb + (size_t)t * (size_t)dim, dim) < 0) {
            free(fa);
            free(fb);
            return NAN;
        }
    }

    double *mu_a = (double *)malloc(sizeof(double) * (size_t)dim);
    double *mu_b = (double *)malloc(sizeof(double) * (size_t)dim);
    double *var_a = (double *)malloc(sizeof(double) * (size_t)dim);
    double *var_b = (double *)malloc(sizeof(double) * (size_t)dim);
    if (!mu_a || !mu_b || !var_a || !var_b) {
        free(fa);
        free(fb);
        free(mu_a);
        free(mu_b);
        free(var_a);
        free(var_b);
        return NAN;
    }
    for (int i = 0; i < dim; i++) {
        double sum_a = 0.0, sum_b = 0.0;
        for (int t = 0; t < n_a; t++) sum_a += fa[(size_t)t * (size_t)dim + (size_t)i];
        for (int t = 0; t < n_b; t++) sum_b += fb[(size_t)t * (size_t)dim + (size_t)i];
        mu_a[i] = sum_a / (double)n_a;
        mu_b[i] = sum_b / (double)n_b;
        double va = 0.0, vb = 0.0;
        for (int t = 0; t < n_a; t++) {
            const double d = fa[(size_t)t * (size_t)dim + (size_t)i] - mu_a[i];
            va += d * d;
        }
        for (int t = 0; t < n_b; t++) {
            const double d = fb[(size_t)t * (size_t)dim + (size_t)i] - mu_b[i];
            vb += d * d;
        }
        var_a[i] = va / (double)n_a;
        var_b[i] = vb / (double)n_b;
    }
    /* 顺序对齐 Python：先算 mean_term（逐维平方和），再算 var_term，最后相加。 */
    double mean_term = 0.0;
    for (int i = 0; i < dim; i++) {
        const double dm = mu_a[i] - mu_b[i];
        mean_term += dm * dm;
    }
    double var_term = 0.0;
    for (int i = 0; i < dim; i++) {
        const double ds = sqrt(var_a[i] + eps) - sqrt(var_b[i] + eps);
        var_term += ds * ds;
    }
    free(fa);
    free(fb);
    free(mu_a);
    free(mu_b);
    free(var_a);
    free(var_b);
    const double total = mean_term + var_term;
    return sqrt(total > 0.0 ? total : 0.0);
}
