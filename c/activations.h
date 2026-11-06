/*
 * activations.h —— SiLU / Mish / ELU / SELU 的标量与向量版本
 *
 * 逐字对应 AI.py 的标量函数（AI.py:82-140）：
 *   _silu_scalar / _dsilu_scalar / _softplus_scalar / _mish_scalar / _dmish_scalar
 *   _elu_scalar / _delu_scalar / _selu_scalar / _dselu_scalar
 *
 * 为什么单独放一个文件：ai.c 里的 act_apply/act_deriv 是"按数组跑"的开关，
 * 而这几个激活的分段判据、常数与运算顺序都要逐条对照 Python；集中在这里
 * 既有标量版本（好测、好对拍），也有向量版本（供 MLP 直接调用）。
 *
 * 数值约定（与 Python 逐位对齐的关键）：
 *   1. silu 的正半轴是 z / (1 + exp(-z))，是**除法**，不是 z * sigmoid——
 *      写成乘法会差最后一位。
 *   2. 分段一律按 Python 的严格/非严格不等号：silu 用 z >= 0，elu/selu 用
 *      z > 0（所以 z == 0 走 exp 分支：elu(0) = expm1(0) = 0，selu(0) = 0）。
 *   3. elu 是 _ELU_ALPHA * (exp(z) - 1)，不是 expm1(z)：两者在 z 很小时
 *      差得比 1 ULP 多，必须照抄。
 *   4. mish 的软阈值是 z > 30.0，与 Python 的 _softplus_scalar 完全一致。
 *
 * libm 差异（实测，见 tools/activations_reference.py 的 note）：
 *   本机 C 走 msvcrt.dll、Python 走 UCRT，exp 有约 0.4% 的输入差 1 ULP，
 *   log1p 约 15%，tanh 约 0.1%。因此 mish（log1p+tanh）与部分 elu/selu/silu
 *   用例无法保证逐位相同，对拍里对这些用例标 tol=1e-12 并注明原因。
 */
#ifndef AI_ACTIVATIONS_H
#define AI_ACTIVATIONS_H

#define ACT_SELU_LAMBDA 1.0507009873554805   /* AI.py 的 _SELU_LAMBDA */
#define ACT_SELU_ALPHA  1.6732632423543772   /* AI.py 的 _SELU_ALPHA  */
#define ACT_ELU_ALPHA   1.0                  /* AI.py 的 _ELU_ALPHA    */

/* ---------- 标量版本（逐字对应 Python 的 *_scalar） ---------- */
double softplus_scalar(double z);
double silu_scalar(double z);
double dsilu_scalar(double z);
double mish_scalar(double z);
double dmish_scalar(double z);
double elu_scalar(double z);
double delu_scalar(double z);
double selu_scalar(double z);
double dselu_scalar(double z);

/* ---------- 向量版本（out 可与 z 相同，逐元素独立） ---------- */
void softplus_vec(const double *z, double *out, int n);
void silu_vec(const double *z, double *out, int n);
void dsilu_vec(const double *z, double *out, int n);
void mish_vec(const double *z, double *out, int n);
void dmish_vec(const double *z, double *out, int n);
void elu_vec(const double *z, double *out, int n);
void delu_vec(const double *z, double *out, int n);
void selu_vec(const double *z, double *out, int n);
void dselu_vec(const double *z, double *out, int n);

#endif /* AI_ACTIVATIONS_H */
