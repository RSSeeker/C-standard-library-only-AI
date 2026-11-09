/*
 * audio_metrics.c —— STOI / PESQ 轻量代理的 C 实现
 *
 * 逐位对齐 Python 版 AI.py 的 stoi_score / pesq_score：相同的分帧、去均值
 * DFT、梅尔滤波器组、包络归一化与相关系数计算顺序，保证同一组输入下输出
 * 双精度逐位一致（由 tools/audio_metrics_reference.py + tests/compare_audio_metrics
 * 验证）。
 */
#include "audio_metrics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define AM_PI 3.14159265358979323846

/* ---------- 基础刻度 ---------- */

static double am_hz_to_mel(double f)
{
    return 2595.0 * log10(1.0 + f / 700.0);
}

static double am_mel_to_hz(double m)
{
    return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
}

/* 参数校验：返回 1 表示合法。语义对齐 Python 的 _au_check_params。 */
static int am_check_params(int sample_rate, int frame_length, int hop_length,
                           int n_bands, double fmin, double fmax)
{
    if (frame_length < 32) return 0;
    if (hop_length <= 0 || hop_length > frame_length) return 0;
    if (sample_rate <= 0) return 0;
    if (n_bands <= 0 || n_bands > frame_length / 2 + 1) return 0;
    if (!isfinite(fmin)) return 0;
    const double lo = fmin;
    const double hi = (fmax < 0.0) ? (double)sample_rate / 2.0 : fmax;
    if (!isfinite(hi)) return 0;
    if (lo < 0.0) return 0;
    if (hi <= lo) return 0;
    if (hi > (double)sample_rate / 2.0 + 1e-9) return 0;
    return 1;
}

/* ---------- 分帧 / DFT / 滤波器组 ---------- */

/* 与 Python frame_signal(pad=False) 相同：只保留完整帧。 */
static double *am_frame_signal(const double *x, int n, int frame_length,
                               int hop_length, int *out_count)
{
    *out_count = 0;
    if (n < frame_length) return NULL;
    const int count = (n - frame_length) / hop_length + 1;
    double *frames = (double *)malloc(sizeof(double) * (size_t)count * (size_t)frame_length);
    if (!frames) return NULL;
    for (int f = 0; f < count; f++) {
        memcpy(frames + (size_t)f * (size_t)frame_length,
               x + (size_t)f * (size_t)hop_length,
               sizeof(double) * (size_t)frame_length);
    }
    *out_count = count;
    return frames;
}

/* 朴素 DFT 幅度谱（n/2+1 个频点），循环顺序与 Python dft_power 一致。 */
static double *am_dft_power(const double *frame, int n)
{
    const int bins = n / 2 + 1;
    double *out = (double *)malloc(sizeof(double) * (size_t)bins);
    if (!out) return NULL;
    for (int k = 0; k < bins; k++) {
        double re = 0.0;
        double im = 0.0;
        for (int t = 0; t < n; t++) {
            const double ang = -2.0 * AM_PI * (double)k * (double)t / (double)n;
            re += frame[t] * cos(ang);
            im += frame[t] * sin(ang);
        }
        out[k] = sqrt(re * re + im * im);
    }
    return out;
}

/* 三角梅尔滤波器组，返回 n_mels × bins 行主序；bins 由 out_bins 带回。 */
static double *am_mel_filters(int sample_rate, int n_fft, int n_mels,
                              double fmin, double fmax, int *out_bins)
{
    *out_bins = 0;
    const double hi = (fmax < 0.0) ? (double)sample_rate / 2.0 : fmax;
    if (n_mels <= 0 || n_fft <= 0) return NULL;
    if (!(hi > fmin)) return NULL;
    const double m_lo = am_hz_to_mel(fmin);
    const double m_hi = am_hz_to_mel(hi);
    double *hz = (double *)malloc(sizeof(double) * (size_t)(n_mels + 2));
    if (!hz) return NULL;
    for (int i = 0; i < n_mels + 2; i++) {
        const double m = m_lo + (m_hi - m_lo) * (double)i / (double)(n_mels + 1);
        hz[i] = am_mel_to_hz(m);
    }
    const int bins = n_fft / 2 + 1;
    double *filters = (double *)malloc(sizeof(double) * (size_t)n_mels * (size_t)bins);
    if (!filters) {
        free(hz);
        return NULL;
    }
    for (int m = 0; m < n_mels; m++) {
        const double left = hz[m];
        const double center = hz[m + 1];
        const double right = hz[m + 2];
        double best = 0.0;
        for (int i = 0; i < bins; i++) {
            const double f = (double)i * (double)sample_rate / (double)n_fft;
            double v;
            if (f <= left || f >= right) {
                v = 0.0;
            } else if (f <= center) {
                v = (f - left) / (center - left);
            } else {
                v = (right - f) / (right - center);
            }
            filters[(size_t)m * (size_t)bins + (size_t)i] = v;
            if (v > best) best = v;
        }
        if (best <= 0.0) {
            free(filters);
            free(hz);
            return NULL;
        }
    }
    free(hz);
    *out_bins = bins;
    return filters;
}

/* 每帧 n_bands 个频带能量：去均值 DFT 幅度谱 → 滤波器加权平方和开方。 */
static double *am_band_energies(const double *x, int n, int sample_rate,
                                int frame_length, int hop_length, int n_bands,
                                double fmin, double fmax, int *out_frames)
{
    *out_frames = 0;
    int frames_count = 0;
    double *frames = am_frame_signal(x, n, frame_length, hop_length, &frames_count);
    if (!frames) return NULL;
    int bins = 0;
    double *filters = am_mel_filters(sample_rate, frame_length, n_bands, fmin, fmax, &bins);
    if (!filters) {
        free(frames);
        return NULL;
    }
    double *out = (double *)malloc(sizeof(double) * (size_t)frames_count * (size_t)n_bands);
    double *power = (double *)malloc(sizeof(double) * (size_t)bins);
    double *centered = (double *)malloc(sizeof(double) * (size_t)frame_length);
    if (!out || !power || !centered) {
        free(frames);
        free(filters);
        free(out);
        free(power);
        free(centered);
        return NULL;
    }
    for (int f = 0; f < frames_count; f++) {
        const double *fr = frames + (size_t)f * (size_t)frame_length;
        double mean = 0.0;
        for (int i = 0; i < frame_length; i++) mean += fr[i];
        mean /= (double)frame_length;
        for (int i = 0; i < frame_length; i++) centered[i] = fr[i] - mean;
        double *pw = am_dft_power(centered, frame_length);
        if (!pw) {
            free(frames);
            free(filters);
            free(out);
            free(power);
            free(centered);
            return NULL;
        }
        for (int i = 0; i < bins; i++) power[i] = pw[i];
        free(pw);
        for (int m = 0; m < n_bands; m++) {
            double acc = 0.0;
            const double *row = filters + (size_t)m * (size_t)bins;
            for (int i = 0; i < bins; i++) {
                acc += row[i] * power[i] * power[i];
            }
            out[(size_t)f * (size_t)n_bands + (size_t)m] = sqrt(acc);
        }
    }
    free(frames);
    free(filters);
    free(power);
    free(centered);
    *out_frames = frames_count;
    return out;
}

/* 两条波形按较短长度对齐；长度不足帧长返回 0（调用方返回 NAN）。 */
static int am_trim(int n_a, int n_b, int frame_length, int *out_n)
{
    const int n = (n_a < n_b) ? n_a : n_b;
    if (n < frame_length) return 0;
    *out_n = n;
    return 1;
}

static int am_same_samples(const double *a, const double *b, int n)
{
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

/* ---------- STOI ---------- */

double am_stoi_score(const double *clean, int n_clean,
                     const double *degraded, int n_degraded,
                     int sample_rate, int frame_length, int hop_length,
                     int n_bands, double fmin, double fmax, double eps)
{
    if (!clean || !degraded || n_clean <= 0 || n_degraded <= 0) return NAN;
    if (!(eps > 0.0) || !isfinite(eps)) return NAN;
    if (!am_check_params(sample_rate, frame_length, hop_length, n_bands, fmin, fmax))
        return NAN;
    int n = 0;
    if (!am_trim(n_clean, n_degraded, frame_length, &n)) return NAN;

    int fa_count = 0, fb_count = 0;
    double *ea = am_band_energies(clean, n, sample_rate, frame_length, hop_length,
                                  n_bands, fmin, fmax, &fa_count);
    double *eb = am_band_energies(degraded, n, sample_rate, frame_length, hop_length,
                                  n_bands, fmin, fmax, &fb_count);
    if (!ea || !eb) {
        free(ea);
        free(eb);
        return NAN;
    }
    const int t_frames = fa_count;
    if (t_frames < 2 || fb_count != t_frames) {
        free(ea);
        free(eb);
        return NAN;
    }

    double *mean_ea = (double *)malloc(sizeof(double) * (size_t)n_bands);
    double *xc = (double *)malloc(sizeof(double) * (size_t)t_frames);
    double *xd = (double *)malloc(sizeof(double) * (size_t)t_frames);
    double *nc = (double *)malloc(sizeof(double) * (size_t)t_frames);
    double *nd = (double *)malloc(sizeof(double) * (size_t)t_frames);
    if (!mean_ea || !xc || !xd || !nc || !nd) {
        free(ea);
        free(eb);
        free(mean_ea);
        free(xc);
        free(xd);
        free(nc);
        free(nd);
        return NAN;
    }
    for (int m = 0; m < n_bands; m++) {
        double s = 0.0;
        for (int t = 0; t < t_frames; t++) s += ea[(size_t)t * (size_t)n_bands + (size_t)m];
        mean_ea[m] = s / (double)t_frames;
    }
    double peak = 0.0;
    for (int m = 0; m < n_bands; m++) {
        if (mean_ea[m] > peak) peak = mean_ea[m];
    }

    double corr_sum = 0.0;
    int corr_count = 0;
    for (int m = 0; m < n_bands; m++) {
        if (peak > 0.0 && mean_ea[m] < 1e-3 * peak) continue; /* 静音频带 */
        for (int t = 0; t < t_frames; t++) {
            xc[t] = ea[(size_t)t * (size_t)n_bands + (size_t)m];
            xd[t] = eb[(size_t)t * (size_t)n_bands + (size_t)m];
        }
        double mc = 0.0, md = 0.0;
        for (int t = 0; t < t_frames; t++) {
            mc += xc[t];
            md += xd[t];
        }
        mc /= (double)t_frames;
        md /= (double)t_frames;
        double vc = 0.0, vd = 0.0;
        for (int t = 0; t < t_frames; t++) {
            const double dc = xc[t] - mc;
            const double dd = xd[t] - md;
            vc += dc * dc;
            vd += dd * dd;
        }
        vc /= (double)t_frames;
        vd /= (double)t_frames;
        if (vc <= eps || vd <= eps) continue;
        const double sc = sqrt(vc);
        const double sd = sqrt(vd);
        for (int t = 0; t < t_frames; t++) {
            double a = (xc[t] - mc) / sc;
            double b = (xd[t] - md) / sd;
            if (a > 3.0) a = 3.0;
            if (a < -3.0) a = -3.0;
            if (b > 3.0) b = 3.0;
            if (b < -3.0) b = -3.0;
            nc[t] = a;
            nd[t] = b;
        }
        double num = 0.0, snc = 0.0, snd = 0.0;
        for (int t = 0; t < t_frames; t++) {
            num += nc[t] * nd[t];
            snc += nc[t] * nc[t];
            snd += nd[t] * nd[t];
        }
        const double den = sqrt(snc * snd);
        if (den <= eps) continue;
        double corr = num / den;
        if (corr > 1.0) corr = 1.0;
        if (corr < -1.0) corr = -1.0;
        corr_sum += corr;
        corr_count++;
    }

    free(ea);
    free(eb);
    free(mean_ea);
    free(xc);
    free(xd);
    free(nc);
    free(nd);

    if (corr_count == 0) {
        return am_same_samples(clean, degraded, n) ? 1.0 : 0.0;
    }
    double score = corr_sum / (double)corr_count;
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return score;
}

/* ---------- PESQ ---------- */

double am_pesq_score(const double *clean, int n_clean,
                     const double *degraded, int n_degraded,
                     int sample_rate, int frame_length, int hop_length,
                     int n_bands, double fmin, double fmax, double eps)
{
    if (!clean || !degraded || n_clean <= 0 || n_degraded <= 0) return NAN;
    if (!(eps > 0.0) || !isfinite(eps)) return NAN;
    if (!am_check_params(sample_rate, frame_length, hop_length, n_bands, fmin, fmax))
        return NAN;
    int n = 0;
    if (!am_trim(n_clean, n_degraded, frame_length, &n)) return NAN;

    int fa_count = 0, fb_count = 0;
    double *ea = am_band_energies(clean, n, sample_rate, frame_length, hop_length,
                                  n_bands, fmin, fmax, &fa_count);
    double *eb = am_band_energies(degraded, n, sample_rate, frame_length, hop_length,
                                  n_bands, fmin, fmax, &fb_count);
    if (!ea || !eb || fa_count <= 0 || fb_count != fa_count) {
        free(ea);
        free(eb);
        return NAN;
    }
    const int t_frames = fa_count;

    double *mean_ea = (double *)malloc(sizeof(double) * (size_t)n_bands);
    int *active = (int *)malloc(sizeof(int) * (size_t)n_bands);
    double *env_a = (double *)malloc(sizeof(double) * (size_t)t_frames);
    double *env_b = (double *)malloc(sizeof(double) * (size_t)t_frames);
    if (!mean_ea || !active || !env_a || !env_b) {
        free(ea);
        free(eb);
        free(mean_ea);
        free(active);
        free(env_a);
        free(env_b);
        return NAN;
    }
    for (int m = 0; m < n_bands; m++) {
        double s = 0.0;
        for (int t = 0; t < t_frames; t++) s += ea[(size_t)t * (size_t)n_bands + (size_t)m];
        mean_ea[m] = s / (double)t_frames;
    }
    double peak = 0.0;
    for (int m = 0; m < n_bands; m++) {
        if (mean_ea[m] > peak) peak = mean_ea[m];
    }
    int n_active = 0;
    for (int m = 0; m < n_bands; m++) {
        if (peak > 0.0 && mean_ea[m] >= 1e-3 * peak) active[n_active++] = m;
    }
    if (n_active == 0) {
        free(ea);
        free(eb);
        free(mean_ea);
        free(active);
        free(env_a);
        free(env_b);
        return 4.5;
    }

    double total = 0.0;
    for (int t = 0; t < t_frames; t++) {
        double row = 0.0;
        for (int k = 0; k < n_active; k++) {
            const int m = active[k];
            const double lc = log(ea[(size_t)t * (size_t)n_bands + (size_t)m] + eps);
            const double ld = log(eb[(size_t)t * (size_t)n_bands + (size_t)m] + eps);
            const double d = lc - ld;
            row += d * d;
        }
        total += row / (double)n_active;
    }
    const double dist = total / (double)t_frames;

    for (int t = 0; t < t_frames; t++) {
        double sa = 0.0, sb = 0.0;
        for (int k = 0; k < n_active; k++) {
            const int m = active[k];
            sa += ea[(size_t)t * (size_t)n_bands + (size_t)m];
            sb += eb[(size_t)t * (size_t)n_bands + (size_t)m];
        }
        env_a[t] = sa;
        env_b[t] = sb;
    }
    double ma = 0.0, mb = 0.0;
    for (int t = 0; t < t_frames; t++) {
        ma += env_a[t];
        mb += env_b[t];
    }
    ma /= (double)t_frames;
    mb /= (double)t_frames;
    double va = 0.0, vb = 0.0;
    for (int t = 0; t < t_frames; t++) {
        const double da = env_a[t] - ma;
        const double db = env_b[t] - mb;
        va += da * da;
        vb += db * db;
    }
    va /= (double)t_frames;
    vb /= (double)t_frames;
    double env_corr;
    if (va <= eps || vb <= eps) {
        env_corr = am_same_samples(env_a, env_b, t_frames) ? 1.0 : 0.0;
    } else {
        double num = 0.0;
        for (int t = 0; t < t_frames; t++) {
            num += (env_a[t] - ma) * (env_b[t] - mb);
        }
        env_corr = num / sqrt(va * vb * (double)t_frames * (double)t_frames);
        if (env_corr > 1.0) env_corr = 1.0;
        if (env_corr < -1.0) env_corr = -1.0;
    }

    free(ea);
    free(eb);
    free(mean_ea);
    free(active);
    free(env_a);
    free(env_b);

    double sq = sqrt(dist);
    if (sq > 1.0) sq = 1.0;
    double score = 4.5 - 2.5 * sq - 1.5 * (1.0 - env_corr);
    if (score < 1.0) score = 1.0;
    if (score > 4.5) score = 4.5;
    return score;
}
