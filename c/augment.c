/* ============================================================================
 *  augment.c —— 数据增广内核的 C17 实现
 *
 *  与 Python `AI.py` 的对应关系：
 *    aug_brightness            <- _aug_brightness
 *    aug_contrast              <- _aug_contrast
 *    aug_shift                 <- _aug_shift
 *    aug_rotate90              <- _aug_rotate90
 *    aug_cutout                <- augment_cutout
 *    aug_cutmix_box / aug_cutmix <- augment_cutmix
 *    aug_mixup / aug_mixup_soft  <- mixup
 *    aug_randaugment           <- augment_randaugment
 *
 *  逐位一致的关键点：
 *    1) Python 的 `round()` 是「四舍六入五成双」，C 的 `round()` 是「五入」，
 *       所以这里一律用 `rint()`（IEEE 默认舍入模式恰好就是 round-half-even）。
 *    2) `_aug_contrast` 的 mean 是 `sum(sum(v for v in row) for ch in img for row in ch)`：
 *       先对每一行做 `sum()`（Neumaier），再对所有行和做一次 `sum()`（Neumaier），
 *       不是把所有元素塞进一次求和。两个层次都必须补偿求和。
 *    3) Rotate90 的索引是 `out[c][y][x] = in[c][H-1-x][y]`（不是常见的
 *       `out[y][x] = in[x][W-1-y]`），照抄 Python 的三重列表推导。
 * ==========================================================================*/
#include "augment.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numeric.h"

static void die(const char *where, const char *msg) {
    fprintf(stderr, "%s: %s\n", where, msg);
    exit(1);
}

static void check_args(const char *who, int C, int H, int W) {
    if (C <= 0 || H <= 0 || W <= 0) die(who, "图像维度必须为正");
}

/* 逐行的 Neumaier 行和，再对行和做一次 Neumaier（对齐 Python 的两层 sum） */
static double row_sums_neumaier(const double *img, int rows, int row_len) {
    double *partial = (double *)malloc(sizeof(double) * (size_t)(rows > 0 ? rows : 1));
    double out;
    int r;
    if (partial == NULL) die("row_sums_neumaier", "内存不足");
    for (r = 0; r < rows; r++) partial[r] = sum_neumaier(img + (size_t)r * row_len, row_len);
    out = sum_neumaier(partial, rows);
    free(partial);
    return out;
}

void aug_brightness(double *img, int n, double factor) {
    int i;
    if (img == NULL) die("aug_brightness", "img 不能为 NULL");
    if (n < 0) die("aug_brightness", "n 不能为负");
    for (i = 0; i < n; i++) {
        double v = img[i] * factor;
        if (v > 1.0) v = 1.0;
        if (v < 0.0) v = 0.0;
        img[i] = v;
    }
}

void aug_contrast(double *img, int rows, int row_len, double factor) {
    double total, mean;
    int cnt, i;
    if (img == NULL) die("aug_contrast", "img 不能为 NULL");
    if (rows <= 0 || row_len <= 0) die("aug_contrast", "rows / row_len 必须为正");
    total = row_sums_neumaier(img, rows, row_len);
    cnt = rows * row_len;
    mean = total / (double)cnt;
    for (i = 0; i < cnt; i++) {
        double v = (img[i] - mean) * factor + mean;
        if (v > 1.0) v = 1.0;
        if (v < 0.0) v = 0.0;
        img[i] = v;
    }
}

void aug_shift(const double *src, int C, int H, int W, int dy, int dx, double *dst) {
    int c, y, x;
    check_args("aug_shift", C, H, W);
    if (src == NULL || dst == NULL) die("aug_shift", "src / dst 不能为 NULL");
    if (src == dst) die("aug_shift", "src 与 dst 不能是同一块内存");
    memset(dst, 0, sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
    for (c = 0; c < C; c++) {
        for (y = 0; y < H; y++) {
            int sy = y - dy;
            if (sy < 0 || sy >= H) continue;
            for (x = 0; x < W; x++) {
                int sx = x - dx;
                if (sx < 0 || sx >= W) continue;
                dst[((size_t)c * H + y) * W + x] = src[((size_t)c * H + sy) * W + sx];
            }
        }
    }
}

void aug_rotate90(const double *src, int C, int H, int W, int turns,
                  double *dst, int *out_h, int *out_w) {
    const double *cur_src = src;
    double *scratch0 = NULL, *scratch1 = NULL;
    int cur_h = H, cur_w = W, c, t, total;

    check_args("aug_rotate90", C, H, W);
    if (src == NULL || dst == NULL) die("aug_rotate90", "src / dst 不能为 NULL");
    if (src == dst) die("aug_rotate90", "src 与 dst 不能是同一块内存");

    total = ((turns % 4) + 4) % 4;
    if (total == 0) {
        memcpy(dst, src, sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
        if (out_h != NULL) *out_h = H;
        if (out_w != NULL) *out_w = W;
        return;
    }

    /* 每转一次 H/W 互换。转 1 次直接写 dst；转 2 次需要 1 个中转缓冲；
     * 转 3 次需要 2 个（否则第 2 步会 src == dst）。 */
    if (total > 1) {
        scratch0 = (double *)malloc(sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
        if (scratch0 == NULL) die("aug_rotate90", "内存不足");
    }
    if (total > 2) {
        scratch1 = (double *)malloc(sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
        if (scratch1 == NULL) die("aug_rotate90", "内存不足");
    }
    for (t = 0; t < total; t++) {
        double *next;
        int nh = cur_w, nw = cur_h;
        if (t == total - 1) {
            next = dst;
        } else {
            next = (t == 0) ? scratch0 : scratch1;
        }
        for (c = 0; c < C; c++) {
            int y, x;
            for (y = 0; y < nh; y++) {
                for (x = 0; x < nw; x++) {
                    /* Python: out[c][y][x] = in[c][H-1-x][y]（H 是**输入**的高度） */
                    next[((size_t)c * nh + y) * nw + x] =
                        cur_src[((size_t)c * cur_h + (cur_h - 1 - x)) * cur_w + y];
                }
            }
        }
        cur_src = next;
        cur_h = nh;
        cur_w = nw;
    }
    free(scratch0);
    free(scratch1);
    if (out_h != NULL) *out_h = cur_h;
    if (out_w != NULL) *out_w = cur_w;
}

void aug_cutout(double *img, int C, int H, int W, int num_holes,
                double hole_size, double fill, const int *xs, const int *ys) {
    int side, i, c, y, x, minhw;
    if (img == NULL) die("aug_cutout", "img 不能为 NULL");
    check_args("aug_cutout", C, H, W);
    if (num_holes < 0) die("aug_cutout", "num_holes 必须 >= 0");
    if (!(hole_size > 0.0 && hole_size < 1.0)) die("aug_cutout", "hole_size 必须在 (0,1)");
    if (num_holes > 0 && (xs == NULL || ys == NULL)) die("aug_cutout", "xs / ys 不能为 NULL");
    minhw = H < W ? H : W;
    side = (int)rint((double)minhw * hole_size);
    if (side < 1) side = 1;
    for (i = 0; i < num_holes; i++) {
        int x0 = xs[i], y0 = ys[i];
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        for (c = 0; c < C; c++) {
            for (y = y0; y < y0 + side && y < H; y++) {
                for (x = x0; x < x0 + side && x < W; x++) {
                    img[((size_t)c * H + y) * W + x] = fill;
                }
            }
        }
    }
}

int aug_cutmix_box(int W, int H, double lam, int cx, int cy,
                   int *x0, int *x1, int *y0, int *y1) {
    int cut_w, cut_h;
    if (W <= 0 || H <= 0) die("aug_cutmix_box", "W / H 必须为正");
    if (!(lam >= 0.0 && lam <= 1.0)) die("aug_cutmix_box", "lam 必须在 [0,1]");
    cut_w = (int)rint((double)W * sqrt(1.0 - lam));
    cut_h = (int)rint((double)H * sqrt(1.0 - lam));
    if (cut_w <= 0 || cut_h <= 0) return 0;
    if (x0 == NULL || x1 == NULL || y0 == NULL || y1 == NULL)
        die("aug_cutmix_box", "输出指针不能为 NULL");
    *x0 = cx - cut_w / 2;
    *x1 = cx + cut_w / 2;
    *y0 = cy - cut_h / 2;
    *y1 = cy + cut_h / 2;
    if (*x0 < 0) *x0 = 0;
    if (*x1 > W) *x1 = W;
    if (*y0 < 0) *y0 = 0;
    if (*y1 > H) *y1 = H;
    return (*x1 > *x0 && *y1 > *y0) ? 1 : 0;
}

double aug_cutmix(const double *xa, const double *xb, int C, int H, int W,
                  double lam, int cx, int cy, double *dst) {
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    int c, y, x;
    check_args("aug_cutmix", C, H, W);
    if (xa == NULL || xb == NULL || dst == NULL) die("aug_cutmix", "图像指针不能为 NULL");
    if (dst == xa || dst == xb) die("aug_cutmix", "dst 不能与输入重叠");
    memcpy(dst, xa, sizeof(double) * (size_t)C * (size_t)H * (size_t)W);

    if (!aug_cutmix_box(W, H, lam, cx, cy, &x0, &x1, &y0, &y1)) return 1.0;

    for (c = 0; c < C; c++) {
        for (y = y0; y < y1; y++) {
            for (x = x0; x < x1; x++) {
                dst[((size_t)c * H + y) * W + x] = xb[((size_t)c * H + y) * W + x];
            }
        }
    }
    /* Python: lam = 1.0 - ((x1_-x0)*(y1_-y0)) / float(H*W) */
    return 1.0 - (double)((x1 - x0) * (y1 - y0)) / (double)(H * W);
}

void aug_mixup(const double *x1, const double *x2, int n, double lam, double *dst) {
    int i;
    if (x1 == NULL || x2 == NULL || dst == NULL) die("aug_mixup", "指针不能为 NULL");
    if (n < 0) die("aug_mixup", "n 不能为负");
    for (i = 0; i < n; i++) dst[i] = lam * x1[i] + (1.0 - lam) * x2[i];
}

void aug_mixup_soft(int y1, int y2, int n_cls, double lam, double *soft) {
    if (soft == NULL) die("aug_mixup_soft", "soft 不能为 NULL");
    if (n_cls <= 0) die("aug_mixup_soft", "n_cls 必须为正");
    if (y1 < 0 || y1 >= n_cls || y2 < 0 || y2 >= n_cls)
        die("aug_mixup_soft", "类别下标越界");
    memset(soft, 0, sizeof(double) * (size_t)n_cls);
    soft[y1] += lam;         /* 顺序与 Python 一致：两次 +=，y1 == y2 时和为 1 */
    soft[y2] += 1.0 - lam;
}

const char *aug_op_name(int op) {
    switch (op) {
        case AUG_OP_BRIGHTNESS: return "brightness";
        case AUG_OP_CONTRAST: return "contrast";
        case AUG_OP_TRANSLATE: return "translate";
        case AUG_OP_ROTATE90: return "rotate90";
        default: return "identity";
    }
}

int aug_randaugment(const double *src, int C, int H, int W, int n,
                    const int *ops, const double *params,
                    double *dst, int *out_h, int *out_w) {
    double *a = NULL, *b = NULL, *cur, *next;
    int cur_h = H, cur_w = W, i;

    check_args("aug_randaugment", C, H, W);
    if (src == NULL || dst == NULL) die("aug_randaugment", "src / dst 不能为 NULL");
    if (n < 0) die("aug_randaugment", "n 必须 >= 0");
    if (n > 0 && (ops == NULL || params == NULL))
        die("aug_randaugment", "ops / params 不能为 NULL");

    if (n == 0) {
        memcpy(dst, src, sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
        if (out_h != NULL) *out_h = H;
        if (out_w != NULL) *out_w = W;
        return 0;
    }

    a = (double *)malloc(sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
    b = (double *)malloc(sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
    if (a == NULL || b == NULL) die("aug_randaugment", "内存不足");
    memcpy(a, src, sizeof(double) * (size_t)C * (size_t)H * (size_t)W);
    cur = a;
    next = b;

    for (i = 0; i < n; i++) {
        double p0 = params[2 * i];
        double p1 = params[2 * i + 1];
        int op = ops[i];
        if (op == AUG_OP_BRIGHTNESS) {
            memcpy(next, cur, sizeof(double) * (size_t)C * (size_t)cur_h * (size_t)cur_w);
            aug_brightness(next, C * cur_h * cur_w, p0);
        } else if (op == AUG_OP_CONTRAST) {
            memcpy(next, cur, sizeof(double) * (size_t)C * (size_t)cur_h * (size_t)cur_w);
            aug_contrast(next, C * cur_h, cur_w, p0);
        } else if (op == AUG_OP_TRANSLATE) {
            aug_shift(cur, C, cur_h, cur_w, (int)p0, (int)p1, next);
        } else if (op == AUG_OP_ROTATE90) {
            int nh = cur_h, nw = cur_w;
            aug_rotate90(cur, C, cur_h, cur_w, (int)p0, next, &nh, &nw);
            cur_h = nh;
            cur_w = nw;
        } else if (op == AUG_OP_IDENTITY) {
            memcpy(next, cur, sizeof(double) * (size_t)C * (size_t)cur_h * (size_t)cur_w);
        } else {
            free(a);
            free(b);
            die("aug_randaugment", "未知操作码");
        }
        {
            double *tmp = cur;
            cur = next;
            next = tmp;
        }
    }

    memcpy(dst, cur, sizeof(double) * (size_t)C * (size_t)cur_h * (size_t)cur_w);
    free(a);
    free(b);
    if (out_h != NULL) *out_h = cur_h;
    if (out_w != NULL) *out_w = cur_w;
    return n;
}
