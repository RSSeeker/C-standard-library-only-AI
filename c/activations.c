/*
 * activations.c —— SiLU / Mish / ELU / SELU 的实现
 *
 * 每个函数都逐行对照 AI.py:82-140 的标量版本；运算顺序、分段判据、
 * 常数都不做任何"等价变形"（改写会改变舍入）。
 */
#include "activations.h"

#include <math.h>

double softplus_scalar(double z)
{
    /* Python: if z > 30.0: return z; return math.log1p(math.exp(z)) */
    if (z > 30.0) return z;
    return log1p(exp(z));
}

double silu_scalar(double z)
{
    /* Python: z >= 0 时 z / (1.0 + exp(-z))；否则 e = exp(z); z*e/(1.0+e) */
    if (z >= 0.0) return z / (1.0 + exp(-z));
    const double e = exp(z);
    return z * e / (1.0 + e);
}

double dsilu_scalar(double z)
{
    /* Python: s = sigmoid(z)（分段求法），返回 s * (1.0 + z * (1.0 - s)) */
    double s;
    if (z >= 0.0) {
        s = 1.0 / (1.0 + exp(-z));
    } else {
        const double e = exp(z);
        s = e / (1.0 + e);
    }
    return s * (1.0 + z * (1.0 - s));
}

double mish_scalar(double z)
{
    /* Python: z * math.tanh(_softplus_scalar(z)) */
    return z * tanh(softplus_scalar(z));
}

double dmish_scalar(double z)
{
    /* Python: sp=softplus(z); t=tanh(sp); s=sigmoid(z); t + z*(1-t*t)*s */
    const double sp = softplus_scalar(z);
    const double t = tanh(sp);
    double s;
    if (z >= 0.0) {
        s = 1.0 / (1.0 + exp(-z));
    } else {
        const double e = exp(z);
        s = e / (1.0 + e);
    }
    return t + z * (1.0 - t * t) * s;
}

double elu_scalar(double z)
{
    /* Python: return z if z > 0.0 else _ELU_ALPHA * (math.exp(z) - 1.0) */
    if (z > 0.0) return z;
    return ACT_ELU_ALPHA * (exp(z) - 1.0);
}

double delu_scalar(double z)
{
    /* Python: return 1.0 if z > 0.0 else _ELU_ALPHA * math.exp(z) */
    if (z > 0.0) return 1.0;
    return ACT_ELU_ALPHA * exp(z);
}

double selu_scalar(double z)
{
    /* Python: z > 0 -> _SELU_LAMBDA*z；否则 _SELU_LAMBDA*_SELU_ALPHA*(exp(z)-1) */
    if (z > 0.0) return ACT_SELU_LAMBDA * z;
    return ACT_SELU_LAMBDA * ACT_SELU_ALPHA * (exp(z) - 1.0);
}

double dselu_scalar(double z)
{
    /* Python: z > 0 -> _SELU_LAMBDA；否则 _SELU_LAMBDA*_SELU_ALPHA*exp(z) */
    if (z > 0.0) return ACT_SELU_LAMBDA;
    return ACT_SELU_LAMBDA * ACT_SELU_ALPHA * exp(z);
}

/* ---------- 向量版本 ---------- */

void softplus_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = softplus_scalar(z[i]);
}

void silu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = silu_scalar(z[i]);
}

void dsilu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = dsilu_scalar(z[i]);
}

void mish_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = mish_scalar(z[i]);
}

void dmish_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = dmish_scalar(z[i]);
}

void elu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = elu_scalar(z[i]);
}

void delu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = delu_scalar(z[i]);
}

void selu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = selu_scalar(z[i]);
}

void dselu_vec(const double *z, double *out, int n)
{
    for (int i = 0; i < n; i++) out[i] = dselu_scalar(z[i]);
}
