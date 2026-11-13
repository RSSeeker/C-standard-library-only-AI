/*
 * rope_ext.h —— RoPE 外推扩展：ABF / LongRoPE 频率与 2D/3D RoPE
 *
 * 对应 Python 版 AI.py 的 B3 段（AI.py:8094 abf_inv_freqs、AI.py:8115
 * longrope_inv_freqs、AI.py:8148 rope_2d、AI.py:8173 rope_2d_inverse、
 * AI.py:8194 rope_3d、AI.py:8228 rope_3d_inverse）。
 *
 * 约定：
 *   - 二维行列表一律用扁平数组 rows（行主序）+ n_rows + width 表示；
 *   - 所有函数返回 0 表示成功、-1 表示参数非法（C 侧用 -1 对应 Python 的
 *     ValueError；Python 的「各行不等长」「元素非有限」等校验在 C 扁平入参
 *     下无法表达，故不在此模块内）；
 *   - 输出缓冲由调用者分配：inv-freqs 需 d_k/2 个 double，旋转类函数需
 *     n_rows*width 个 double。
 *
 * 这些函数都是纯函数、不持有状态，便于与 Python 逐位对拍。
 */
#ifndef AI_C_ROPE_EXT_H
#define AI_C_ROPE_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ABF（Adjusted Base Frequency）逐对逆频率：
 * base_eff = base * factor，
 * out[i] = 1 / base_eff ** (2*i / d_k)，i = 0 .. d_k/2-1。
 * 要求 d_k >= 2、base 有限且 > 0、factor 有限且 >= 1。
 */
int ai_abf_inv_freqs(int d_k, double base, double factor,
                     double *out, int *out_n);

/*
 * LongRoPE 风格非均匀频率缩放（简化版）：
 * freqs[i] = 1 / base ** (2*i / d_k)，然后把最低频的 k 个
 * （k = max(1, round(n * low_fraction))，n = d_k/2）除以 factor。
 * 要求 d_k >= 2、base 有限且 > 0、factor 有限且 >= 1、
 * original_max >= 1（仅校验，不参与计算）、low_fraction 在 (0,1) 内。
 */
int ai_longrope_inv_freqs(int d_k, double base, double factor,
                          int original_max, double low_fraction,
                          double *out, int *out_n);

/*
 * 2D RoPE：rows 按行主序 (r*w + c) 展开，行数必须等于 grid_h*grid_w，
 * width 必须被 4 整除；前半段按行位置、后半段按列位置旋转。
 * 返回旋转后的新行（不修改入参 rows）。
 */
int ai_rope_2d(const double *rows, int n_rows, int width,
               int grid_h, int grid_w, double base, double *out);

/* 2D RoPE 的逆旋转（角度取负，旋转顺序相反）。 */
int ai_rope_2d_inverse(const double *rows, int n_rows, int width,
                       int grid_h, int grid_w, double base, double *out);

/*
 * 3D RoPE：rows 按 (t,h,w) 行主序展开，行数必须等于 grid_t*grid_h*grid_w，
 * width 必须被 6 整除；时间/行/列各占 width/3 个维度。
 */
int ai_rope_3d(const double *rows, int n_rows, int width,
               int grid_t, int grid_h, int grid_w, double base, double *out);

/* 3D RoPE 的逆旋转。 */
int ai_rope_3d_inverse(const double *rows, int n_rows, int width,
                       int grid_t, int grid_h, int grid_w, double base, double *out);

#ifdef __cplusplus
}
#endif

#endif /* AI_C_ROPE_EXT_H */
