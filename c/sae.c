/*
 * sae.c —— 稀疏自编码器 C 实现
 *
 * 与 Python 版 AI.py 的 SAE 逐项对齐：encode / decode / forward /
 * loss_and_backward（重建 MSE + L1）/ sgd_step / feature_stats / param_count。
 */
#include "sae.h"

#include <math.h>
#include <stdlib.h>

AiSAE *ai_sae_create(int d_in, int d_hidden, double l1_coeff)
{
    AiSAE *sae;
    if (d_in < 1 || d_hidden < 1 || !(l1_coeff >= 0.0)) return NULL;
    sae = (AiSAE *)calloc(1, sizeof(AiSAE));
    if (!sae) return NULL;
    sae->d_in = d_in;
    sae->d_hidden = d_hidden;
    sae->l1_coeff = l1_coeff;
    sae->W_enc = (double *)calloc((size_t)d_hidden * (size_t)d_in, sizeof(double));
    sae->b_enc = (double *)calloc((size_t)d_hidden, sizeof(double));
    sae->W_dec = (double *)calloc((size_t)d_in * (size_t)d_hidden, sizeof(double));
    sae->b_dec = (double *)calloc((size_t)d_in, sizeof(double));
    sae->gW_enc = (double *)calloc((size_t)d_hidden * (size_t)d_in, sizeof(double));
    sae->gb_enc = (double *)calloc((size_t)d_hidden, sizeof(double));
    sae->gW_dec = (double *)calloc((size_t)d_in * (size_t)d_hidden, sizeof(double));
    sae->gb_dec = (double *)calloc((size_t)d_in, sizeof(double));
    if (!sae->W_enc || !sae->b_enc || !sae->W_dec || !sae->b_dec ||
        !sae->gW_enc || !sae->gb_enc || !sae->gW_dec || !sae->gb_dec) {
        ai_sae_free(sae);
        return NULL;
    }
    return sae;
}

void ai_sae_free(AiSAE *sae)
{
    if (!sae) return;
    free(sae->W_enc);
    free(sae->b_enc);
    free(sae->W_dec);
    free(sae->b_dec);
    free(sae->gW_enc);
    free(sae->gb_enc);
    free(sae->gW_dec);
    free(sae->gb_dec);
    free(sae);
}

int ai_sae_load_params(AiSAE *sae, const double *W_enc, const double *b_enc,
                       const double *W_dec, const double *b_dec)
{
    int i;
    if (!sae || !W_enc || !b_enc || !W_dec || !b_dec) return -1;
    for (i = 0; i < sae->d_hidden * sae->d_in; i++) sae->W_enc[i] = W_enc[i];
    for (i = 0; i < sae->d_hidden; i++) sae->b_enc[i] = b_enc[i];
    for (i = 0; i < sae->d_in * sae->d_hidden; i++) sae->W_dec[i] = W_dec[i];
    for (i = 0; i < sae->d_in; i++) sae->b_dec[i] = b_dec[i];
    ai_sae_zero_grad(sae);
    return 0;
}

void ai_sae_zero_grad(AiSAE *sae)
{
    int i;
    if (!sae) return;
    for (i = 0; i < sae->d_hidden * sae->d_in; i++) sae->gW_enc[i] = 0.0;
    for (i = 0; i < sae->d_hidden; i++) sae->gb_enc[i] = 0.0;
    for (i = 0; i < sae->d_in * sae->d_hidden; i++) sae->gW_dec[i] = 0.0;
    for (i = 0; i < sae->d_in; i++) sae->gb_dec[i] = 0.0;
}

void ai_sae_encode(const AiSAE *sae, const double *x, double *features)
{
    int i, j;
    if (!sae || !x || !features) return;
    for (i = 0; i < sae->d_hidden; i++) {
        double pre = sae->b_enc[i];
        const double *row = sae->W_enc + (size_t)i * (size_t)sae->d_in;
        for (j = 0; j < sae->d_in; j++) pre += row[j] * x[j];
        features[i] = pre > 0.0 ? pre : 0.0;
    }
}

void ai_sae_decode(const AiSAE *sae, const double *features, double *recon)
{
    int i, j;
    if (!sae || !features || !recon) return;
    for (i = 0; i < sae->d_in; i++) {
        double out = sae->b_dec[i];
        const double *row = sae->W_dec + (size_t)i * (size_t)sae->d_hidden;
        for (j = 0; j < sae->d_hidden; j++) out += row[j] * features[j];
        recon[i] = out;
    }
}

int ai_sae_forward(const AiSAE *sae, const double *xs, int n, double *recon,
                   double *features, double *pre)
{
    int k, i, j;
    if (!sae || !xs || n < 1 || !recon || !features) return -1;
    for (k = 0; k < n; k++) {
        const double *x = xs + (size_t)k * (size_t)sae->d_in;
        double *f = features + (size_t)k * (size_t)sae->d_hidden;
        double *r = recon + (size_t)k * (size_t)sae->d_in;
        double *p = pre ? pre + (size_t)k * (size_t)sae->d_hidden : NULL;
        for (i = 0; i < sae->d_hidden; i++) {
            double v = sae->b_enc[i];
            const double *row = sae->W_enc + (size_t)i * (size_t)sae->d_in;
            for (j = 0; j < sae->d_in; j++) v += row[j] * x[j];
            if (p) p[i] = v;
            f[i] = v > 0.0 ? v : 0.0;
        }
        for (i = 0; i < sae->d_in; i++) {
            double v = sae->b_dec[i];
            const double *row = sae->W_dec + (size_t)i * (size_t)sae->d_hidden;
            for (j = 0; j < sae->d_hidden; j++) v += row[j] * f[j];
            r[i] = v;
        }
    }
    return 0;
}

int ai_sae_loss_and_backward(AiSAE *sae, const double *xs, int n,
                             double *out_loss, double *out_mse,
                             double *out_l1)
{
    double mse_total = 0.0, l1_total = 0.0;
    double *pre, *f, *r, *d_r, *d_f, *d_pre;
    int k, i, j;
    if (!sae || !xs || n < 1) return -1;
    pre = (double *)malloc(sizeof(double) * (size_t)sae->d_hidden);
    f = (double *)malloc(sizeof(double) * (size_t)sae->d_hidden);
    d_f = (double *)malloc(sizeof(double) * (size_t)sae->d_hidden);
    d_pre = (double *)malloc(sizeof(double) * (size_t)sae->d_hidden);
    r = (double *)malloc(sizeof(double) * (size_t)sae->d_in);
    d_r = (double *)malloc(sizeof(double) * (size_t)sae->d_in);
    if (!pre || !f || !d_f || !d_pre || !r || !d_r) {
        free(pre); free(f); free(d_f); free(d_pre); free(r); free(d_r);
        return -1;
    }
    for (k = 0; k < n; k++) {
        const double *x = xs + (size_t)k * (size_t)sae->d_in;
        for (i = 0; i < sae->d_hidden; i++) {
            double v = sae->b_enc[i];
            const double *row = sae->W_enc + (size_t)i * (size_t)sae->d_in;
            for (j = 0; j < sae->d_in; j++) v += row[j] * x[j];
            pre[i] = v;
            f[i] = v > 0.0 ? v : 0.0;
            l1_total += fabs(f[i]);
        }
        for (i = 0; i < sae->d_in; i++) {
            double v = sae->b_dec[i];
            const double *row = sae->W_dec + (size_t)i * (size_t)sae->d_hidden;
            for (j = 0; j < sae->d_hidden; j++) v += row[j] * f[j];
            r[i] = v;
            mse_total += (v - x[i]) * (v - x[i]);
            d_r[i] = 2.0 * (v - x[i]) / (double)sae->d_in;
        }
        for (i = 0; i < sae->d_in; i++) {
            sae->gb_dec[i] += d_r[i] / (double)n;
            for (j = 0; j < sae->d_hidden; j++) {
                if (f[j] != 0.0) {
                    sae->gW_dec[(size_t)i * (size_t)sae->d_hidden + j] +=
                        d_r[i] * f[j] / (double)n;
                }
            }
        }
        for (j = 0; j < sae->d_hidden; j++) {
            double acc = 0.0;
            for (i = 0; i < sae->d_in; i++) {
                acc += sae->W_dec[(size_t)i * (size_t)sae->d_hidden + j] *
                       d_r[i];
            }
            if (f[j] > 0.0) acc += sae->l1_coeff / (double)sae->d_hidden;
            else if (f[j] < 0.0) acc -= sae->l1_coeff / (double)sae->d_hidden;
            d_f[j] = acc;
            d_pre[j] = pre[j] > 0.0 ? d_f[j] : 0.0;
        }
        for (i = 0; i < sae->d_hidden; i++) {
            sae->gb_enc[i] += d_pre[i] / (double)n;
            for (j = 0; j < sae->d_in; j++) {
                sae->gW_enc[(size_t)i * (size_t)sae->d_in + j] +=
                    d_pre[i] * x[j] / (double)n;
            }
        }
    }
    {
        const double mse = mse_total / ((double)n * (double)sae->d_in);
        const double l1 = sae->l1_coeff * l1_total /
                          ((double)n * (double)sae->d_hidden);
        if (out_mse) *out_mse = mse;
        if (out_l1) *out_l1 = l1;
        if (out_loss) *out_loss = mse + l1;
    }
    free(pre); free(f); free(d_f); free(d_pre); free(r); free(d_r);
    return 0;
}

int ai_sae_sgd_step(AiSAE *sae, double lr)
{
    int i, j;
    if (!sae || !(lr > 0.0) || !isfinite(lr)) return -1;
    for (i = 0; i < sae->d_hidden; i++) {
        for (j = 0; j < sae->d_in; j++) {
            sae->W_enc[(size_t)i * (size_t)sae->d_in + j] -=
                lr * sae->gW_enc[(size_t)i * (size_t)sae->d_in + j];
        }
        sae->b_enc[i] -= lr * sae->gb_enc[i];
    }
    for (i = 0; i < sae->d_in; i++) {
        for (j = 0; j < sae->d_hidden; j++) {
            sae->W_dec[(size_t)i * (size_t)sae->d_hidden + j] -=
                lr * sae->gW_dec[(size_t)i * (size_t)sae->d_hidden + j];
        }
        sae->b_dec[i] -= lr * sae->gb_dec[i];
    }
    ai_sae_zero_grad(sae);
    return 0;
}

int ai_sae_feature_stats(const AiSAE *sae, const double *xs, int n,
                         double *active_fraction, int *out_dead,
                         double *out_mean_active)
{
    int *counts;
    int i, j, k, dead = 0, total = 0;
    if (!sae || !xs || n < 1) return -1;
    counts = (int *)calloc((size_t)sae->d_hidden, sizeof(int));
    if (!counts) return -1;
    for (k = 0; k < n; k++) {
        const double *x = xs + (size_t)k * (size_t)sae->d_in;
        for (i = 0; i < sae->d_hidden; i++) {
            double v = sae->b_enc[i];
            const double *row = sae->W_enc + (size_t)i * (size_t)sae->d_in;
            for (j = 0; j < sae->d_in; j++) v += row[j] * x[j];
            if (v > 0.0) counts[i]++;
        }
    }
    for (i = 0; i < sae->d_hidden; i++) {
        if (active_fraction) active_fraction[i] = (double)counts[i] / (double)n;
        if (counts[i] == 0) dead++;
        total += counts[i];
    }
    if (out_dead) *out_dead = dead;
    if (out_mean_active) {
        *out_mean_active = (double)total /
                           ((double)n * (double)sae->d_hidden);
    }
    free(counts);
    return 0;
}

int ai_sae_param_count(const AiSAE *sae)
{
    if (!sae) return 0;
    return sae->d_hidden * sae->d_in + sae->d_hidden +
           sae->d_in * sae->d_hidden + sae->d_in;
}
