/*
 * visual_metrics.h —— 感知 / 视频分布指标（LPIPS / FVD 轻量代理）的 C 实现
 *
 * 与 Python 版 AI.py 的 lpips_lite / fvd_lite 对拍：
 *   - LPIPS：多尺度平均池化灰度图 + 水平/垂直梯度特征，逐尺度单位化后求
 *     均方根误差，按权重相加。相同图像为 0，越小越相似。
 *   - FVD：默认把每帧压成 1/2 池化梯度特征，用对角协方差近似的 Fréchet
 *     距离比较两组帧分布。相同分布为 0。
 *
 * 边界：
 *   - 没有预训练 VGG/AlexNet/I3D 权重，不是真实感知分数或 FVD，只适合
 *     同一组信号下的相对比较；
 *   - 图像像素按通道存储：pixels[k*height*width + r*width + c]；
 *   - 非法尺寸 / 参数返回 NAN。
 */
#ifndef VISUAL_METRICS_H
#define VISUAL_METRICS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const double *pixels; /* 通道主序：k*H*W + r*W + c */
    int channels;
    int height;
    int width;
} VmImage;

/* LPIPS 轻量代理：scales 为每尺度池化因子（>=1），n_scales>=1；
 * weights 为 NULL 时各尺度等权，否则长度 n_scales、非负且和为正。 */
double vm_lpips_lite(const VmImage *a, const VmImage *b,
                     const int *scales, int n_scales,
                     const double *weights, double eps);

/* FVD 轻量代理（默认 1/2 池化梯度特征 + 对角协方差 Fréchet 距离）。
 * 两组各至少 2 帧，所有帧形状与特征维度必须一致。 */
double vm_fvd_lite(const VmImage *frames_a, int n_a,
                   const VmImage *frames_b, int n_b, double eps);

/* 默认帧表征（1/2 池化 + 梯度特征 + 单位化），out 容量至少
 * 3*(height/2)*(width/2)。供调用方做自定义分布比较时复用。 */
int vm_fvd_embed_frame(const VmImage *frame, double *out, int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* VISUAL_METRICS_H */
