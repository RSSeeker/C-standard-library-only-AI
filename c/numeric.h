/*
 * numeric.h —— 与 Python 版数值行为对齐的基础原语
 *
 * 关键点：CPython 3.12+ 的 sum() 对浮点使用 Neumaier 补偿求和，不是朴素顺序累加。
 * Python 版里所有 sum(...) / sum(map(mul, ...)) 的位置（Layer.forward 的点积、
 * LayerNorm 的均值方差、注意力里的 scores/dQ/dK/dV、Conv2d 的 grad_w 点积……）
 * 在 C 侧都应使用这里的函数，否则会差 1 ULP 量级并随层数累积。
 *
 * 用法：任何需要"和 Python 的 sum() 对齐"的地方都用 sum_neumaier / dot_neumaier；
 * 而 Python 侧本来就是朴素 += 累加的地方（例如梯度累加、优化器更新），
 * 直接用普通 + 即可，不要用这里。
 */
#ifndef AI_NUMERIC_H
#define AI_NUMERIC_H

#include <math.h>
#include <stddef.h>

/* 与 CPython builtin_sum 的浮点快路径同一算法 */
static inline double sum_neumaier(const double *v, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double x = v[i];
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* Σ a[i]*b[i]，对齐 Python 的 sum(map(operator.mul, a, b)) */
static inline double dot_neumaier(const double *a, const double *b, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a[i] * b[i];
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* 逐元素 a += b（朴素累加，对应 Python 的梯度累加） */
static inline void axpy_add(double *a, const double *b, int n)
{
    for (int i = 0; i < n; i++) a[i] += b[i];
}

#endif /* AI_NUMERIC_H */
