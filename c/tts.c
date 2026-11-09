#include "tts.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TTS_PI 3.14159265358979323846

/* ============================ MT19937 (Python 兼容) ============================ */

static void mt_init_genrand(unsigned int *mt, unsigned int seed)
{
    int i;
    mt[0] = seed;
    for (i = 1; i < 624; ++i) {
        mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (unsigned int)i;
    }
}

static void mt_init_by_array(unsigned int *mt, const unsigned int *key,
                             int key_length)
{
    int i = 1;
    int j = 0;
    int k;
    mt_init_genrand(mt, 19650218U);
    k = (624 > key_length) ? 624 : key_length;
    for (; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U))
                + key[j] + (unsigned int)j;
        ++i;
        ++j;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = 623; k; --k) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U))
                - (unsigned int)i;
        ++i;
        if (i >= 624) {
            mt[0] = mt[623];
            i = 1;
        }
    }
    mt[0] = 0x80000000U;
}

static unsigned int mt_next(TtsRng *rng)
{
    unsigned int y;
    static const unsigned int mag01[2] = {0x0U, 0x9908b0dfU};
    if (rng->index >= 624) {
        int i;
        for (i = 0; i < 624; ++i) {
            unsigned int y2 = (rng->mt[i] & 0x80000000U)
                            | (rng->mt[(i + 1) % 624] & 0x7fffffffU);
            rng->mt[i] = rng->mt[(i + 397) % 624] ^ (y2 >> 1) ^ mag01[y2 & 1U];
        }
        rng->index = 0;
    }
    y = rng->mt[rng->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

void tts_rng_init(TtsRng *rng, unsigned int seed)
{
    unsigned int key[1];
    key[0] = seed;
    mt_init_by_array(rng->mt, key, 1);
    rng->index = 624;
}

double tts_rng_random(TtsRng *rng)
{
    unsigned int a = mt_next(rng) >> 5;
    unsigned int b = mt_next(rng) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

/* ================================ 基础 DSP ================================ */

void tts_hann(int n, double *window)
{
    int i;
    if (n <= 0 || window == NULL) {
        return;
    }
    if (n == 1) {
        window[0] = 1.0;
        return;
    }
    for (i = 0; i < n; ++i) {
        window[i] = 0.5 - 0.5 * cos(2.0 * TTS_PI * (double)i / (double)(n - 1));
    }
}

static void dft_half(const double *frame, int n, double *re, double *im)
{
    int k;
    int t;
    for (k = 0; k <= n / 2; ++k) {
        double sr = 0.0;
        double si = 0.0;
        for (t = 0; t < n; ++t) {
            double ang = 2.0 * TTS_PI * (double)k * (double)t / (double)n;
            sr += frame[t] * cos(ang);
            si -= frame[t] * sin(ang);
        }
        re[k] = sr;
        im[k] = si;
    }
}

static void idft_real(const double *mag, const double *phase, int n,
                      double *out, double *xr, double *xi)
{
    int half = n / 2 + 1;
    int k;
    int t;
    for (k = 0; k < half; ++k) {
        xr[k] = mag[k] * cos(phase[k]);
        xi[k] = mag[k] * sin(phase[k]);
    }
    for (k = 1; k < n - half + 1; ++k) {
        xr[n - k] = xr[k];
        xi[n - k] = -xi[k];
    }
    for (t = 0; t < n; ++t) {
        double total = 0.0;
        for (k = 0; k < n; ++k) {
            double ang = 2.0 * TTS_PI * (double)k * (double)t / (double)n;
            total += xr[k] * cos(ang) - xi[k] * sin(ang);
        }
        out[t] = total / (double)n;
    }
}

/* ============================== Griffin-Lim ============================== */

int tts_griffin_lim(const double *magnitude, int n_frames, int n_bins,
                    int n_fft, int hop_length, int iterations,
                    unsigned int seed, const double *window,
                    double **out, int *out_len)
{
    double *win = NULL;
    double *phase = NULL;
    double *acc = NULL;
    double *norm = NULL;
    double *wave = NULL;
    double *seg = NULL;
    double *xr = NULL;
    double *xi = NULL;
    double *re = NULL;
    double *im = NULL;
    double *frame = NULL;
    TtsRng rng;
    int out_length;
    int t;
    int k;
    int it;
    int rc = 0;

    if (out != NULL) {
        *out = NULL;
    }
    if (out_len != NULL) {
        *out_len = 0;
    }
    if (magnitude == NULL || out == NULL || out_len == NULL) {
        return -1;
    }
    if (n_frames < 1 || n_bins < 2) {
        return -1;
    }
    if (n_fft != 2 * (n_bins - 1) || n_fft < 2) {
        return -1;
    }
    if (hop_length < 1 || iterations < 1) {
        return -1;
    }
    for (t = 0; t < n_frames * n_bins; ++t) {
        if (!(magnitude[t] >= 0.0) || !isfinite(magnitude[t])) {
            return -1;
        }
    }

    out_length = (n_frames - 1) * hop_length + n_fft;
    win = (double *)malloc(sizeof(double) * (size_t)n_fft);
    phase = (double *)malloc(sizeof(double) * (size_t)n_frames * (size_t)n_bins);
    acc = (double *)malloc(sizeof(double) * (size_t)out_length);
    norm = (double *)malloc(sizeof(double) * (size_t)out_length);
    wave = (double *)malloc(sizeof(double) * (size_t)out_length);
    seg = (double *)malloc(sizeof(double) * (size_t)n_fft);
    xr = (double *)malloc(sizeof(double) * (size_t)n_fft);
    xi = (double *)malloc(sizeof(double) * (size_t)n_fft);
    re = (double *)malloc(sizeof(double) * (size_t)n_bins);
    im = (double *)malloc(sizeof(double) * (size_t)n_bins);
    frame = (double *)malloc(sizeof(double) * (size_t)n_fft);
    if (win == NULL || phase == NULL || acc == NULL || norm == NULL
            || wave == NULL || seg == NULL || xr == NULL || xi == NULL
            || re == NULL || im == NULL || frame == NULL) {
        rc = -1;
        goto done;
    }

    if (window == NULL) {
        tts_hann(n_fft, win);
    } else {
        memcpy(win, window, sizeof(double) * (size_t)n_fft);
    }

    tts_rng_init(&rng, seed);
    for (t = 0; t < n_frames; ++t) {
        for (k = 0; k < n_bins; ++k) {
            phase[(size_t)t * (size_t)n_bins + (size_t)k] =
                -TTS_PI + 2.0 * TTS_PI * tts_rng_random(&rng);
        }
    }

    for (it = 0; it < iterations; ++it) {
        memset(acc, 0, sizeof(double) * (size_t)out_length);
        memset(norm, 0, sizeof(double) * (size_t)out_length);
        for (t = 0; t < n_frames; ++t) {
            int start = t * hop_length;
            idft_real(magnitude + (size_t)t * (size_t)n_bins,
                      phase + (size_t)t * (size_t)n_bins, n_fft, seg, xr, xi);
            for (k = 0; k < n_fft; ++k) {
                double w = win[k];
                acc[start + k] += seg[k] * w;
                norm[start + k] += w * w;
            }
        }
        for (k = 0; k < out_length; ++k) {
            wave[k] = (norm[k] > 1e-12) ? acc[k] / norm[k] : 0.0;
        }
        for (t = 0; t < n_frames; ++t) {
            int start = t * hop_length;
            for (k = 0; k < n_fft; ++k) {
                frame[k] = wave[start + k] * win[k];
            }
            dft_half(frame, n_fft, re, im);
            for (k = 0; k < n_bins; ++k) {
                phase[(size_t)t * (size_t)n_bins + (size_t)k] =
                    atan2(im[k], re[k]);
            }
        }
    }

    *out = wave;
    *out_len = out_length;
    wave = NULL;

done:
    free(win);
    free(phase);
    free(acc);
    free(norm);
    free(wave);
    free(seg);
    free(xr);
    free(xi);
    free(re);
    free(im);
    free(frame);
    return rc;
}

void tts_wave_free(double *wave)
{
    free(wave);
}

/* ============================ Mel -> 线性幅度 ============================ */

int tts_mel_to_linear(const double *mel, int n_frames, int n_mels,
                      const double *mel_fb, int n_bins, double eps,
                      double *out)
{
    double *gram = NULL;
    double *aug = NULL;
    double *sol = NULL;
    int i;
    int j;
    int k;
    int t;
    int rc = 0;

    if (mel == NULL || mel_fb == NULL || out == NULL) {
        return -1;
    }
    if (n_frames < 1 || n_mels < 1 || n_bins < 1 || eps < 0.0) {
        return -1;
    }
    gram = (double *)malloc(sizeof(double) * (size_t)n_mels * (size_t)n_mels);
    aug = (double *)malloc(sizeof(double) * (size_t)n_mels * (size_t)(n_mels + 1));
    sol = (double *)malloc(sizeof(double) * (size_t)n_mels);
    if (gram == NULL || aug == NULL || sol == NULL) {
        rc = -1;
        goto done;
    }
    for (i = 0; i < n_mels; ++i) {
        for (j = 0; j < n_mels; ++j) {
            double total = 0.0;
            for (k = 0; k < n_bins; ++k) {
                total += mel_fb[(size_t)i * (size_t)n_bins + (size_t)k]
                       * mel_fb[(size_t)j * (size_t)n_bins + (size_t)k];
            }
            gram[(size_t)i * (size_t)n_mels + (size_t)j] = total;
        }
        gram[(size_t)i * (size_t)n_mels + (size_t)i] += eps;
    }
    for (t = 0; t < n_frames; ++t) {
        /* 增广矩阵 [gram | b] */
        for (i = 0; i < n_mels; ++i) {
            for (j = 0; j < n_mels; ++j) {
                aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)j] =
                    gram[(size_t)i * (size_t)n_mels + (size_t)j];
            }
            aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)n_mels] =
                mel[(size_t)t * (size_t)n_mels + (size_t)i];
        }
        for (k = 0; k < n_mels; ++k) {
            int pivot = k;
            double pv;
            for (i = k + 1; i < n_mels; ++i) {
                if (fabs(aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)k])
                        > fabs(aug[(size_t)pivot * (size_t)(n_mels + 1) + (size_t)k])) {
                    pivot = i;
                }
            }
            if (fabs(aug[(size_t)pivot * (size_t)(n_mels + 1) + (size_t)k]) < 1e-12) {
                rc = -1;
                goto done;
            }
            if (pivot != k) {
                for (j = 0; j <= n_mels; ++j) {
                    double tmp = aug[(size_t)k * (size_t)(n_mels + 1) + (size_t)j];
                    aug[(size_t)k * (size_t)(n_mels + 1) + (size_t)j] =
                        aug[(size_t)pivot * (size_t)(n_mels + 1) + (size_t)j];
                    aug[(size_t)pivot * (size_t)(n_mels + 1) + (size_t)j] = tmp;
                }
            }
            pv = aug[(size_t)k * (size_t)(n_mels + 1) + (size_t)k];
            for (j = k; j <= n_mels; ++j) {
                aug[(size_t)k * (size_t)(n_mels + 1) + (size_t)j] /= pv;
            }
            for (i = 0; i < n_mels; ++i) {
                double factor;
                if (i == k) {
                    continue;
                }
                factor = aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)k];
                if (factor == 0.0) {
                    continue;
                }
                for (j = k; j <= n_mels; ++j) {
                    aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)j] -=
                        factor * aug[(size_t)k * (size_t)(n_mels + 1) + (size_t)j];
                }
            }
        }
        for (i = 0; i < n_mels; ++i) {
            sol[i] = aug[(size_t)i * (size_t)(n_mels + 1) + (size_t)n_mels];
        }
        for (j = 0; j < n_bins; ++j) {
            double total = 0.0;
            for (i = 0; i < n_mels; ++i) {
                total += mel_fb[(size_t)i * (size_t)n_bins + (size_t)j] * sol[i];
            }
            out[(size_t)t * (size_t)n_bins + (size_t)j] = total > 0.0 ? total : 0.0;
        }
    }

done:
    free(gram);
    free(aug);
    free(sol);
    return rc;
}

/* ============================== TextToMel ============================== */

TextToMel *text_to_mel_create(int vocab_size, int d_model, int n_mels,
                              int kernel_size)
{
    TextToMel *m;
    if (vocab_size < 1 || d_model < 1 || n_mels < 1 || kernel_size < 1
            || kernel_size % 2 != 1) {
        return NULL;
    }
    m = (TextToMel *)calloc(1, sizeof(TextToMel));
    if (m == NULL) {
        return NULL;
    }
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->n_mels = n_mels;
    m->kernel_size = kernel_size;
    m->emb = (double *)calloc((size_t)vocab_size * (size_t)d_model, sizeof(double));
    m->c1w = (double *)calloc((size_t)d_model * (size_t)d_model * (size_t)kernel_size, sizeof(double));
    m->c1b = (double *)calloc((size_t)d_model, sizeof(double));
    m->c2w = (double *)calloc((size_t)d_model * (size_t)d_model * (size_t)kernel_size, sizeof(double));
    m->c2b = (double *)calloc((size_t)d_model, sizeof(double));
    m->ow = (double *)calloc((size_t)n_mels * (size_t)d_model, sizeof(double));
    m->ob = (double *)calloc((size_t)n_mels, sizeof(double));
    if (m->emb == NULL || m->c1w == NULL || m->c1b == NULL || m->c2w == NULL
            || m->c2b == NULL || m->ow == NULL || m->ob == NULL) {
        text_to_mel_free(m);
        return NULL;
    }
    return m;
}

void text_to_mel_free(TextToMel *m)
{
    if (m == NULL) {
        return;
    }
    free(m->emb);
    free(m->c1w);
    free(m->c1b);
    free(m->c2w);
    free(m->c2b);
    free(m->ow);
    free(m->ob);
    free(m->emb_rows);
    free(m->frames);
    free(m->c1);
    free(m->a1);
    free(m->c2);
    free(m->a2);
    free(m);
}

int text_to_mel_param_count(const TextToMel *m)
{
    if (m == NULL) {
        return 0;
    }
    return m->vocab_size * m->d_model
         + (m->d_model * m->d_model * m->kernel_size + m->d_model) * 2
         + m->n_mels * m->d_model + m->n_mels;
}

int text_to_mel_set_weights(TextToMel *m,
                            const double *emb,
                            const double *c1w, const double *c1b,
                            const double *c2w, const double *c2b,
                            const double *ow, const double *ob)
{
    if (m == NULL) {
        return -1;
    }
    if (emb != NULL) {
        memcpy(m->emb, emb, sizeof(double) * (size_t)m->vocab_size * (size_t)m->d_model);
    }
    if (c1w != NULL) {
        memcpy(m->c1w, c1w, sizeof(double) * (size_t)m->d_model * (size_t)m->d_model
               * (size_t)m->kernel_size);
    }
    if (c1b != NULL) {
        memcpy(m->c1b, c1b, sizeof(double) * (size_t)m->d_model);
    }
    if (c2w != NULL) {
        memcpy(m->c2w, c2w, sizeof(double) * (size_t)m->d_model * (size_t)m->d_model
               * (size_t)m->kernel_size);
    }
    if (c2b != NULL) {
        memcpy(m->c2b, c2b, sizeof(double) * (size_t)m->d_model);
    }
    if (ow != NULL) {
        memcpy(m->ow, ow, sizeof(double) * (size_t)m->n_mels * (size_t)m->d_model);
    }
    if (ob != NULL) {
        memcpy(m->ob, ob, sizeof(double) * (size_t)m->n_mels);
    }
    return 0;
}

static void conv1d_same(const double *x, int cin, int clen,
                        const double *w, const double *b, int cout, int k,
                        int pad, double *out)
{
    int oc;
    int l;
    int ic;
    int kh;
    for (oc = 0; oc < cout; ++oc) {
        for (l = 0; l < clen; ++l) {
            double total = b[oc];
            for (ic = 0; ic < cin; ++ic) {
                for (kh = 0; kh < k; ++kh) {
                    int idx = l + kh - pad;
                    if (idx < 0 || idx >= clen) {
                        continue;
                    }
                    total += w[((size_t)oc * (size_t)cin + (size_t)ic) * (size_t)k
                               + (size_t)kh]
                           * x[(size_t)ic * (size_t)clen + (size_t)idx];
                }
            }
            out[(size_t)oc * (size_t)clen + (size_t)l] = total;
        }
    }
}

int text_to_mel_forward(const TextToMel *m, const int *ids,
                        const int *durations, int n_tokens,
                        double **mel, int *n_frames)
{
    int total = 0;
    int t;
    int i;
    int j;
    int l;
    int pad;
    double *expanded = NULL;
    double *frames = NULL;
    double *c1 = NULL;
    double *a1 = NULL;
    double *c2 = NULL;
    double *a2 = NULL;
    double *out = NULL;
    int rc = -1;

    if (mel != NULL) {
        *mel = NULL;
    }
    if (n_frames != NULL) {
        *n_frames = 0;
    }
    if (m == NULL || ids == NULL || durations == NULL || mel == NULL
            || n_frames == NULL || n_tokens < 1) {
        return -1;
    }
    for (t = 0; t < n_tokens; ++t) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size || durations[t] < 1) {
            return -1;
        }
        total += durations[t];
    }
    pad = m->kernel_size / 2;
    expanded = (double *)malloc(sizeof(double) * (size_t)total * (size_t)m->d_model);
    frames = (double *)malloc(sizeof(double) * (size_t)m->d_model * (size_t)total);
    c1 = (double *)malloc(sizeof(double) * (size_t)m->d_model * (size_t)total);
    a1 = (double *)malloc(sizeof(double) * (size_t)m->d_model * (size_t)total);
    c2 = (double *)malloc(sizeof(double) * (size_t)m->d_model * (size_t)total);
    a2 = (double *)malloc(sizeof(double) * (size_t)m->d_model * (size_t)total);
    out = (double *)malloc(sizeof(double) * (size_t)total * (size_t)m->n_mels);
    if (expanded == NULL || frames == NULL || c1 == NULL || a1 == NULL
            || c2 == NULL || a2 == NULL || out == NULL) {
        goto done;
    }

    l = 0;
    for (t = 0; t < n_tokens; ++t) {
        int rep;
        for (rep = 0; rep < durations[t]; ++rep) {
            for (i = 0; i < m->d_model; ++i) {
                expanded[(size_t)l * (size_t)m->d_model + (size_t)i] =
                    m->emb[(size_t)ids[t] * (size_t)m->d_model + (size_t)i];
            }
            ++l;
        }
    }
    for (l = 0; l < total; ++l) {
        for (i = 0; i < m->d_model; ++i) {
            frames[(size_t)i * (size_t)total + (size_t)l] =
                expanded[(size_t)l * (size_t)m->d_model + (size_t)i];
        }
    }

    conv1d_same(frames, m->d_model, total, m->c1w, m->c1b, m->d_model,
                m->kernel_size, pad, c1);
    for (l = 0; l < m->d_model * total; ++l) {
        a1[l] = c1[l] > 0.0 ? c1[l] : 0.0;
    }
    conv1d_same(a1, m->d_model, total, m->c2w, m->c2b, m->d_model,
                m->kernel_size, pad, c2);
    for (l = 0; l < m->d_model * total; ++l) {
        a2[l] = c2[l] > 0.0 ? c2[l] : 0.0;
    }
    for (t = 0; t < total; ++t) {
        for (j = 0; j < m->n_mels; ++j) {
            double totalv = m->ob[j];
            for (i = 0; i < m->d_model; ++i) {
                totalv += m->ow[(size_t)j * (size_t)m->d_model + (size_t)i]
                        * a2[(size_t)i * (size_t)total + (size_t)t];
            }
            out[(size_t)t * (size_t)m->n_mels + (size_t)j] = totalv;
        }
    }

    *mel = out;
    *n_frames = total;
    out = NULL;
    rc = 0;

done:
    free(expanded);
    free(frames);
    free(c1);
    free(a1);
    free(c2);
    free(a2);
    free(out);
    return rc;
}
