#include "gan_variants.h"

#include <math.h>
#include <stddef.h>

static int gv_dims_ok(int c, int h, int w)
{
    return c > 0 && h > 0 && w > 0;
}

static double gv_sigmoid(double v)
{
    if (v >= 0.0) {
        return 1.0 / (1.0 + exp(-v));
    }
    double e = exp(v);
    return e / (1.0 + e);
}

int gv_fade_in(const double *low, const double *high, int c, int lh, int lw,
               double alpha, double *out)
{
    if (!low || !high || !out || !gv_dims_ok(c, lh, lw)) {
        return -1;
    }
    if (!isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
        return -1;
    }
    int hh = lh * 2;
    int hw = lw * 2;
    for (int k = 0; k < c; k++) {
        for (int i = 0; i < lh; i++) {
            for (int j = 0; j < lw; j++) {
                double v = low[(size_t)k * lh * lw + (size_t)i * lw + j];
                int hi = i * 2;
                int hj = j * 2;
                for (int di = 0; di < 2; di++) {
                    for (int dj = 0; dj < 2; dj++) {
                        size_t oi = (size_t)(hi + di);
                        size_t oj = (size_t)(hj + dj);
                        size_t idx = ((size_t)k * hh + oi) * hw + oj;
                        out[idx] = (1.0 - alpha) * v + alpha * high[idx];
                    }
                }
            }
        }
    }
    return 0;
}

int gv_progressive_alpha(int step, int total, double *alpha)
{
    if (!alpha || step < 0 || total < 1) {
        return -1;
    }
    double a = (double)step / (double)total;
    if (a < 0.0) {
        a = 0.0;
    }
    if (a > 1.0) {
        a = 1.0;
    }
    *alpha = a;
    return 0;
}

int gv_progressive_stages(const int *resolutions, int n, int fade_steps,
                          int *low_out, int *high_out, double *alpha_out,
                          int cap)
{
    if (!resolutions || !low_out || !high_out || !alpha_out || n < 2) {
        return -1;
    }
    if (fade_steps < 1) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (resolutions[i] < 1) {
            return -1;
        }
        if (i > 0 && resolutions[i] <= resolutions[i - 1]) {
            return -1;
        }
    }
    int idx = 0;
    for (int i = 1; i < n; i++) {
        for (int s = 0; s < fade_steps; s++) {
            if (idx >= cap) {
                return -1;
            }
            low_out[idx] = resolutions[i - 1];
            high_out[idx] = resolutions[i];
            alpha_out[idx] = (double)s / (double)fade_steps;
            idx++;
        }
        if (idx >= cap) {
            return -1;
        }
        low_out[idx] = resolutions[i - 1];
        high_out[idx] = resolutions[i];
        alpha_out[idx] = 1.0;
        idx++;
    }
    return idx;
}

int gv_adain_style(const double *features, int c, int h, int w,
                   const double *style, double eps,
                   double *out, double *stats_mean, double *stats_std)
{
    if (!features || !style || !out || !gv_dims_ok(c, h, w)) {
        return -1;
    }
    if (!isfinite(eps) || eps <= 0.0) {
        return -1;
    }
    size_t n = (size_t)h * (size_t)w;
    for (int k = 0; k < c; k++) {
        const double *src = features + (size_t)k * n;
        double mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            mean += src[i];
        }
        mean /= (double)n;
        double var = 0.0;
        for (size_t i = 0; i < n; i++) {
            double d = src[i] - mean;
            var += d * d;
        }
        var /= (double)n;
        double std = sqrt(var + eps);
        double scale = style[k];
        double bias = style[c + k];
        double *dst = out + (size_t)k * n;
        for (size_t i = 0; i < n; i++) {
            dst[i] = (src[i] - mean) / std * scale + bias;
        }
        if (stats_mean) {
            stats_mean[k] = mean;
        }
        if (stats_std) {
            stats_std[k] = std;
        }
    }
    return 0;
}

int gv_truncation_trick(const double *w, const double *w_avg, int n,
                        double psi, double *out)
{
    if (!w || !w_avg || !out || n <= 0) {
        return -1;
    }
    if (!isfinite(psi) || psi < 0.0 || psi > 1.0) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        out[i] = w_avg[i] + psi * (w[i] - w_avg[i]);
    }
    return 0;
}

int gv_cycle_consistency_loss(const double *a, const double *a_rec,
                              const double *b, const double *b_rec,
                              int c, int h, int w, double lambda_cycle,
                              double *loss, double *grad_a, double *grad_b)
{
    if (!a || !a_rec || !b || !b_rec || !loss || !gv_dims_ok(c, h, w)) {
        return -1;
    }
    if (!isfinite(lambda_cycle) || lambda_cycle < 0.0) {
        return -1;
    }
    size_t n = (size_t)c * (size_t)h * (size_t)w;
    double total = 0.0;
    double coef = lambda_cycle / (double)n;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - a_rec[i];
        total += fabs(diff);
        if (grad_a) {
            grad_a[i] = diff > 0.0 ? coef : (diff < 0.0 ? -coef : 0.0);
        }
    }
    for (size_t i = 0; i < n; i++) {
        double diff = b[i] - b_rec[i];
        total += fabs(diff);
        if (grad_b) {
            grad_b[i] = diff > 0.0 ? coef : (diff < 0.0 ? -coef : 0.0);
        }
    }
    *loss = lambda_cycle * total;
    return 0;
}

int gv_identity_mapping_loss(const double *a, const double *g_ba,
                             const double *b, const double *g_ab,
                             int c, int h, int w, double lambda_identity,
                             double *loss)
{
    if (!a || !g_ba || !b || !g_ab || !loss || !gv_dims_ok(c, h, w)) {
        return -1;
    }
    if (!isfinite(lambda_identity) || lambda_identity < 0.0) {
        return -1;
    }
    size_t n = (size_t)c * (size_t)h * (size_t)w;
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += fabs(a[i] - g_ba[i]);
    }
    for (size_t i = 0; i < n; i++) {
        total += fabs(b[i] - g_ab[i]);
    }
    *loss = lambda_identity * total;
    return 0;
}

int gv_patchgan_targets(int h, int w, int real, double *out)
{
    if (!out || h <= 0 || w <= 0) {
        return -1;
    }
    double v = real ? 1.0 : 0.0;
    size_t n = (size_t)h * (size_t)w;
    for (size_t i = 0; i < n; i++) {
        out[i] = v;
    }
    return 0;
}

int gv_pix2pix_g_loss(const double *scores, int ns, const double *fake,
                      const double *real, int c, int h, int w,
                      double lambda_l1, double *loss)
{
    if (!scores || !fake || !real || !loss || ns <= 0 || !gv_dims_ok(c, h, w)) {
        return -1;
    }
    if (!isfinite(lambda_l1) || lambda_l1 < 0.0) {
        return -1;
    }
    double bce = 0.0;
    for (int i = 0; i < ns; i++) {
        double p = gv_sigmoid(scores[i]);
        if (p < 1e-12) {
            p = 1e-12;
        }
        if (p > 1.0 - 1e-12) {
            p = 1.0 - 1e-12;
        }
        bce -= log(p);
    }
    bce /= (double)ns;
    size_t n = (size_t)c * (size_t)h * (size_t)w;
    double l1 = 0.0;
    for (size_t i = 0; i < n; i++) {
        l1 += fabs(fake[i] - real[i]);
    }
    l1 /= (double)n;
    *loss = bce + lambda_l1 * l1;
    return 0;
}

static double gv_r_penalty(const double *grads, int n_samples, int grad_dim,
                           double lam)
{
    if (!grads || n_samples <= 0 || grad_dim <= 0) {
        return NAN;
    }
    if (!isfinite(lam) || lam < 0.0) {
        return NAN;
    }
    double total = 0.0;
    size_t n = (size_t)n_samples * (size_t)grad_dim;
    for (size_t i = 0; i < n; i++) {
        total += grads[i] * grads[i];
    }
    return 0.5 * lam * total / (double)n_samples;
}

double gv_r1_penalty(const double *grads, int n_samples, int grad_dim, double lam)
{
    return gv_r_penalty(grads, n_samples, grad_dim, lam);
}

double gv_r2_penalty(const double *grads, int n_samples, int grad_dim, double lam)
{
    return gv_r_penalty(grads, n_samples, grad_dim, lam);
}
