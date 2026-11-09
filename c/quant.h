/*
 * quant.h —— 低精度数值模拟：fp16 / bf16 / fp8 舍入与对称伪量化
 *
 * 对应 Python AI.py 的 fp16_round / bf16_round / fp8_round / fake_quantize
 * （对扁平 double 数组逐元素）。返回值：0 成功，-1 非法参数。
 */
#ifndef AI_QUANT_H
#define AI_QUANT_H

/* fp16（IEEE binary16）最近偶数舍入；溢出返回 ±INFINITY。 */
int quant_fp16_round(const double *v, int n, double *out);

/* bf16（8 位指数 + 7 位尾数，round-half-to-even）。 */
int quant_bf16_round(const double *v, int n, double *out);

/* fp8 舍入；fmt=0 为 e4m3（最大 448），fmt=1 为 e5m2（最大 57344）。
 * 超最大饱和，低于最小次正规舍入到 0，NaN/inf 原样返回。 */
int quant_fp8_round(const double *v, int n, int fmt, double *out);

/* 对称均匀伪量化：q = clamp(round(v/scale), -qmax, qmax)，输出 q*scale。
 * has_scale=0 时按峰值取 scale = peak/qmax（peak<=0 时 scale=1）；bits 须 >= 2。 */
int quant_fake_quantize(const double *v, int n, int bits, double scale,
                        int has_scale, double *out);

#endif
