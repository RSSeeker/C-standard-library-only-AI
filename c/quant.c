/*
 * quant.c —— 低精度舍入与伪量化实现
 */
#include "quant.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int q_check(const double *v, int n, double *out)
{
    if (!v || !out || n < 0) return -1;
    for (int i = 0; i < n; i++) {
        if (isnan(v[i])) return -1;
    }
    return 0;
}

/* 舍入到最近的二进制小数点后 p 位（ties-to-even）。 */
static double q_round_step(double a, int mant_bits, int exp)
{
    /* Python 算法：step = 2 ** (exp - mant_bits)；q = round(a/step)*step。
     * round() 是 ties-to-even；nearbyint 默认舍入模式即最近偶数。 */
    double step = ldexp(1.0, exp - mant_bits);
    return nearbyint(a / step) * step;
}

static double q_fp16_one(double value)
{
    double v = value;
    if (isnan(v) || isinf(v) || v == 0.0) return v;
    double sign = v < 0.0 ? -1.0 : 1.0;
    double a = fabs(v);
    if (a > 65519.0) {
        return sign * INFINITY;
    }
    if (a < ldexp(1.0, -14)) {
        /* 次正规区：步长 2^-24。 */
        double step = ldexp(1.0, -24);
        double q = nearbyint(a / step) * step;
        if (q >= ldexp(1.0, -14)) return sign * ldexp(1.0, -14);
        return sign * q;
    }
    int exp = 0;
    (void)frexp(a, &exp); /* a = m * 2^exp, 0.5 <= m < 1 */
    exp -= 1;             /* a 在 [2^exp, 2^(exp+1)) */
    double step = ldexp(1.0, exp - 10); /* 10 位尾数（含隐含位） */
    double q = nearbyint(a / step) * step;
    if (q > 65504.0) return sign * INFINITY;
    return sign * q;
}

static double q_bf16_one(double value)
{
    double v = value;
    if (isnan(v) || isinf(v) || v == 0.0) return v;
    float f = (float)v;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t bias = 0x7FFFu + lsb;
    bits = (bits + bias) & 0xFFFF0000u;
    memcpy(&f, &bits, sizeof(f));
    return (double)f;
}

static double q_fp8_one(double value, int fmt)
{
    double v = value;
    if (isnan(v) || isinf(v) || v == 0.0) return v;
    int mant_bits;
    int emin;
    double maxval;
    if (fmt == 0) { /* e4m3 */
        mant_bits = 3;
        emin = -6;
        maxval = 448.0;
    } else if (fmt == 1) { /* e5m2 */
        mant_bits = 2;
        emin = -14;
        maxval = 57344.0;
    } else {
        return NAN;
    }
    double sign = v < 0.0 ? -1.0 : 1.0;
    double a = fabs(v);
    if (a >= maxval) return sign * maxval;
    if (a < ldexp(1.0, emin)) {
        double step = ldexp(1.0, emin - mant_bits);
        return sign * nearbyint(a / step) * step;
    }
    int exp = 0;
    (void)frexp(a, &exp);
    exp -= 1;
    double q = q_round_step(a, mant_bits, exp);
    if (q > maxval) q = maxval;
    return sign * q;
}

int quant_fp16_round(const double *v, int n, double *out)
{
    if (q_check(v, n, out) != 0) return -1;
    for (int i = 0; i < n; i++) out[i] = q_fp16_one(v[i]);
    return 0;
}

int quant_bf16_round(const double *v, int n, double *out)
{
    if (q_check(v, n, out) != 0) return -1;
    for (int i = 0; i < n; i++) out[i] = q_bf16_one(v[i]);
    return 0;
}

int quant_fp8_round(const double *v, int n, int fmt, double *out)
{
    if (fmt != 0 && fmt != 1) return -1;
    if (q_check(v, n, out) != 0) return -1;
    for (int i = 0; i < n; i++) out[i] = q_fp8_one(v[i], fmt);
    return 0;
}

int quant_fake_quantize(const double *v, int n, int bits, double scale,
                        int has_scale, double *out)
{
    if (q_check(v, n, out) != 0 || bits < 2) return -1;
    double qmax = (double)((1 << (bits - 1)) - 1);
    double s = scale;
    if (!has_scale) {
        double peak = 0.0;
        for (int i = 0; i < n; i++) {
            double a = fabs(v[i]);
            if (a > peak) peak = a;
        }
        s = peak <= 0.0 ? 1.0 : peak / qmax;
    } else if (!(s > 0.0) || !isfinite(s)) {
        return -1;
    }
    for (int i = 0; i < n; i++) {
        double q = nearbyint(v[i] / s);
        if (q > qmax) q = qmax;
        else if (q < -qmax) q = -qmax;
        out[i] = q * s;
    }
    return 0;
}
