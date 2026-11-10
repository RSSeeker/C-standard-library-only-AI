#ifndef TTS_H
#define TTS_H

#include <stddef.h>

/* ===================== 跨语言 RNG（Python random.Random 兼容） =====================
 *
 * Python 的 random.Random(seed) 对整数 seed 使用 MT19937 + init_by_array 一个
 * 32 位 key，因此下面实现与该 RNG 的前 53 位浮点输出逐位一致。
 */
typedef struct {
    unsigned int mt[624];
    int index;
} TtsRng;

void   tts_rng_init(TtsRng *rng, unsigned int seed);
double tts_rng_random(TtsRng *rng);

/* ============================ Griffin-Lim 声码器 ============================ */

/* 长度 n 的 Hann 窗 */
void tts_hann(int n, double *window);

/* 由幅度谱迭代估计相位并还原波形。
 * magnitude: n_frames x n_bins 行主序，非负
 * n_fft: 必须等于 2*(n_bins-1)；window: 长度 n_fft（NULL 用 Hann）
 * 成功返回 0，并把 malloc 的波形写入 *out、长度写入 *out_len；
 * 调用方用 tts_wave_free 释放。参数非法返回 -1。 */
int tts_griffin_lim(const double *magnitude, int n_frames, int n_bins,
                    int n_fft, int hop_length, int iterations,
                    unsigned int seed, const double *window,
                    double **out, int *out_len);
void tts_wave_free(double *wave);

/* =========================== Mel -> 线性幅度（最小二乘） ===========================
 *
 * mel: n_frames x n_mels；mel_fb: n_mels x n_bins；out: n_frames x n_bins
 * 成功返回 0，参数非法/方程奇异返回 -1。 */
int tts_mel_to_linear(const double *mel, int n_frames, int n_mels,
                      const double *mel_fb, int n_bins, double eps,
                      double *out);

/* ============================ TextToMel 声学模型（推理） ============================
 *
 * 只提供前向推理（Embedding -> 两次 same-padding Conv1d + ReLU -> 线性输出），
 * 不实现训练反向；参数计数与 Python 的 TextToMel.param_count 一致。
 */
typedef struct {
    int vocab_size;
    int d_model;
    int n_mels;
    int kernel_size;

    double *emb;   /* vocab_size x d_model */
    double *c1w;   /* d_model x d_model x kernel_size, 布局 [oc][ic][k] */
    double *c1b;   /* d_model */
    double *c2w;   /* d_model x d_model x kernel_size */
    double *c2b;   /* d_model */
    double *ow;    /* n_mels x d_model, 布局 [j][i] */
    double *ob;    /* n_mels */

    /* 前向临时缓冲，create 时按输入长度增长 */
    int cap_frames;
    double *emb_rows;  /* cap_frames x d_model */
    double *frames;    /* d_model x cap_frames */
    double *c1;        /* d_model x cap_frames */
    double *a1;
    double *c2;
    double *a2;
} TextToMel;

TextToMel *text_to_mel_create(int vocab_size, int d_model, int n_mels,
                              int kernel_size);
void text_to_mel_free(TextToMel *m);

int text_to_mel_param_count(const TextToMel *m);

/* 传入全部权重：emb(v*d)、c1w/c1b、c2w/c2b、ow(n_mels*d)、ob(n_mels)。
 * 返回 0 成功，-1 参数非法。 */
int text_to_mel_set_weights(TextToMel *m,
                            const double *emb,
                            const double *c1w, const double *c1b,
                            const double *c2w, const double *c2b,
                            const double *ow, const double *ob);

/* ids/durations: n_tokens；成功返回 0，并把 malloc 的 mel（n_frames x n_mels）
 * 写入 *mel、帧数写入 *n_frames。调用方 free(*mel)。 */
int text_to_mel_forward(const TextToMel *m, const int *ids,
                        const int *durations, int n_tokens,
                        double **mel, int *n_frames);

#endif /* TTS_H */
