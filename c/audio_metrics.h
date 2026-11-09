/*
 * audio_metrics.h —— 音频质量指标（STOI / PESQ 轻量代理）的 C 实现
 *
 * 与 Python 版 AI.py 的 stoi_score / pesq_score 逐位对拍：
 *   - STOI：切帧 → 去均值 DFT 幅度谱 → 梅尔滤波器组能量 → 逐频带零均值单位
 *     方差归一化（裁剪 [-3,3]）→ 逐频带皮尔逊相关 → 有效频带平均，返回 [0,1]。
 *   - PESQ：频带对数谱失真 + 帧总能量包络相关，返回 [1.0,4.5]。
 *
 * 边界与 Python 版一致：
 *   - 两条波形先按较短长度对齐；帧数为 0 / 参数非法时返回 NAN（调用方用
 *     isnan 判断，而不是拿 NAN 去比较）。
 *   - 这是无对齐、无心理声学模型的轻量代理，只适合同一对信号下的相对比较，
 *     不是 ITU-T P.862 的实现。
 */
#ifndef AUDIO_METRICS_H
#define AUDIO_METRICS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 短时客观可懂度（简化版），返回 [0,1]；参数非法或帧太少返回 NAN。
 *
 * clean / degraded：1 维采样（double），长度可不同，内部按较短者截断；
 * sample_rate、frame_length、hop_length、n_bands 为正整数；
 * fmin >= 0，fmax 为负时表示取奈奎斯特频率（与 Python 的 fmax=None 对应，
 * 传入 -1.0 即用默认值）；eps 必须是正数。 */
double am_stoi_score(const double *clean, int n_clean,
                     const double *degraded, int n_degraded,
                     int sample_rate, int frame_length, int hop_length,
                     int n_bands, double fmin, double fmax, double eps);

/* PESQ 轻量代理，返回 [1.0,4.5]；参数非法或帧太少返回 NAN。 */
double am_pesq_score(const double *clean, int n_clean,
                     const double *degraded, int n_degraded,
                     int sample_rate, int frame_length, int hop_length,
                     int n_bands, double fmin, double fmax, double eps);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_METRICS_H */
