/* ============================================================================
 *  augment.h —— 数据增广内核的 C17 实现（增广族）
 *
 *  对齐 Python：`AI.augment_cutmix` / `augment_cutout` / `augment_randaugment`
 *  与 `AI.mixup`，以及它们的内部助手 `_aug_brightness` / `_aug_contrast` /
 *  `_aug_shift` / `_aug_rotate90`。
 *
 *  设计约定：**随机性一律从外部注入**。Python 侧用 `random.Random`（Mersenne
 *  Twister）与 `rng.gauss`，C 侧用 splitmix64，两者不可复现，因此本模块只实现
 *  「给定 lam / 裁剪框 / 洞位置 / 操作序列与强度」之后的确定性内核，跨语言对拍
 *  时由参考脚本把 Python 抽到的随机量写进 JSON。这与 `dropout_set_mask` /
 *  `droppath_set_keep` 的处理方式一致。
 *
 *  图像布局：`(C, H, W)` 行主序扁平数组，长度 `C*H*W`，元素为 double。
 *  `aug_rotate90` 会交换 H/W（元素个数不变），所以画布缓冲可以复用。
 * ==========================================================================*/
#ifndef AI_AUGMENT_H
#define AI_AUGMENT_H

/* 操作码（与 Python augment_randaugment 的默认池同序） */
#define AUG_OP_BRIGHTNESS 0
#define AUG_OP_CONTRAST   1
#define AUG_OP_TRANSLATE  2
#define AUG_OP_ROTATE90   3
#define AUG_OP_IDENTITY   4

/* ---- 单步内核（就地或写出到 dst；dst 与 src 不得重叠） ---- */

/* 亮度：out = min(1, max(0, v * factor))，就地；n = C*H*W */
void aug_brightness(double *img, int n, double factor);

/* 对比度：mean 先用「逐行 Neumaier、再对行和 Neumaier」求（照抄 Python 的
 * sum(sum(v for v in row) ...)），然后 out = clamp((v-mean)*factor + mean)。
 * img 就地；rows = C*H，row_len = W。 */
void aug_contrast(double *img, int rows, int row_len, double factor);

/* 平移：out[c][y][x] = src[c][y-dy][x-dx]，越界补 0 */
void aug_shift(const double *src, int C, int H, int W, int dy, int dx, double *dst);

/* 旋转 90 度 `turns` 次（turns 可为负，按模 4 处理）。
 * dst 的尺寸是 (C, W, H)（turns 为奇数）或 (C, H, W)；
 * *out_h / *out_w 回传实际尺寸。 */
void aug_rotate90(const double *src, int C, int H, int W, int turns,
                  double *dst, int *out_h, int *out_w);

/* Cutout：抹掉 num_holes 个 side×side 的正方形。
 * side = max(1, rint(min(H,W) * hole_size))（Python 用 round()，即四舍六入五成双）。
 * xs[i] / ys[i] 是第 i 个洞的左上角（由调用方抽样，Python 用 randrange）；
 * 实际写入会被夹到图像内。就地修改。 */
void aug_cutout(double *img, int C, int H, int W, int num_holes,
                double hole_size, double fill, const int *xs, const int *ys);

/* CutMix 的裁剪框：cut_w = rint(W*sqrt(1-lam))、cut_h = rint(H*sqrt(1-lam))，
 * 框为以 (cx, cy) 为中心、边长 cut_w/cut_h 的矩形并夹到图像内。
 * 返回 1 表示框非空（x0 / x1 / y0 / y1 为半开区间），返回 0 表示退化为空框。 */
int aug_cutmix_box(int W, int H, double lam, int cx, int cy,
                   int *x0, int *x1, int *y0, int *y1);

/* CutMix：把 xb 的框内像素贴到 xa 上，写出到 dst，返回**有效 lam**
 * （= 1 - 框面积/(H*W)；框为空时返回 1.0，且 dst 等于 xa）。
 * lam 的取值由调用方给出（Python 侧是 Beta(alpha,alpha) 采样）。 */
double aug_cutmix(const double *xa, const double *xb, int C, int H, int W,
                  double lam, int cx, int cy, double *dst);

/* Mixup：dst = lam*x1 + (1-lam)*x2（逐元素）。 */
void aug_mixup(const double *x1, const double *x2, int n, double lam, double *dst);

/* Mixup 的软标签：soft 先清零，soft[y1] += lam、soft[y2] += 1-lam。
 * 要求 0 <= y1,y2 < n_cls。 */
void aug_mixup_soft(int y1, int y2, int n_cls, double lam, double *soft);

/* RandAugment 驱动：按 ops[0..n-1] 依次应用，每步的随机量放在 params[2*i]、
 * params[2*i+1]（brightness/contrast 只用第一个当 factor，translate 用两个当
 * dy/dx，rotate90 用第一个当 turns，identity 不用）。
 * src 不被修改，结果写到 dst（容量至少 C*H*W 个 double），
 * out_h / out_w 回传最终尺寸。返回应用的步数（即 n）。 */
int aug_randaugment(const double *src, int C, int H, int W, int n,
                    const int *ops, const double *params,
                    double *dst, int *out_h, int *out_w);

/* 操作名（越界返回 "identity"） */
const char *aug_op_name(int op);

#endif /* AI_AUGMENT_H */
