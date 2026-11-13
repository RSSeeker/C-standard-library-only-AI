/*
 * rope_ext.c —— RoPE 外推扩展实现（ABF / LongRoPE / 2D / 3D）
 *
 * 逐行对齐 Python 版 AI.py 的实现与运算顺序：
 *   - _rope_rotate_section：angle = sign * pos * freq，cos/sin 逐对旋转；
 *   - _rope_inv_pairs：inv[j] = 1 / base ** (2*j/seg)；
 *   - 2D：先按行位置旋前半段，再按列位置旋后半段；
 *   - 3D：按时间/行/列依次旋转 width/3 段；
 *   - 逆旋转：段顺序与 sign 均取反。
 */
#include "rope_ext.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>

/* 段内位置类型（对应 Python 的 position_fn）。 */
typedef enum {
    ROPE_POS_ROW = 0, /* t // grid_w */
    ROPE_POS_COL,     /* t % grid_w  */
    ROPE_POS_T,       /* t // (grid_h*grid_w) */
    ROPE_POS_H,       /* (t // grid_w) % grid_h */
    ROPE_POS_W        /* t % grid_w  */
} RopePosKind;

typedef struct {
    int grid_h;
    int grid_w;
} RopeGrid;

static int rope_pos(RopePosKind kind, int t, const RopeGrid *g)
{
    switch (kind) {
    case ROPE_POS_ROW:
        return t / g->grid_w;
    case ROPE_POS_COL:
        return t % g->grid_w;
    case ROPE_POS_T:
        return t / (g->grid_h * g->grid_w);
    case ROPE_POS_H:
        return (t / g->grid_w) % g->grid_h;
    case ROPE_POS_W:
        return t % g->grid_w;
    default:
        return 0;
    }
}

static int rope_check_base(double base)
{
    return isfinite(base) && base > 0.0;
}

/* Python 的 seg % 2 != 0 校验：段维度必须是偶数。 */
static int rope_rotate_section(double *rows, int n_rows, int width,
                               int start, int seg, RopePosKind kind,
                               const RopeGrid *grid, double base, double sign)
{
    if (!rows || n_rows < 1 || width < 1 || seg < 2 || seg % 2 != 0)
        return -1;
    if (start < 0 || (long long)start + seg > width)
        return -1;

    for (int t = 0; t < n_rows; t++) {
        double pos = (double)rope_pos(kind, t, grid);
        double *row = rows + (size_t)t * (size_t)width;
        for (int j = 0; j < seg / 2; j++) {
            /* Python: freq = 1.0 / (base ** (2.0 * j / seg)) */
            double freq = 1.0 / pow(base, (2.0 * (double)j) / (double)seg);
            double angle = sign * pos * freq;
            double c = cos(angle);
            double s = sin(angle);
            int i0 = start + 2 * j;
            double x0 = row[i0];
            double x1 = row[i0 + 1];
            row[i0] = x0 * c - x1 * s;
            row[i0 + 1] = x0 * s + x1 * c;
        }
    }
    return 0;
}

static int rope_copy_rows(const double *rows, int n_rows, int width, double *out)
{
    size_t total;
    if (!rows || !out)
        return -1;
    total = (size_t)n_rows * (size_t)width;
    for (size_t i = 0; i < total; i++)
        out[i] = rows[i];
    return 0;
}

/* 把二维行数换成 long long 做溢出安全的乘积比较。 */
static int rope_rows_match(int n_rows, long long want)
{
    return (long long)n_rows == want;
}

int ai_abf_inv_freqs(int d_k, double base, double factor,
                     double *out, int *out_n)
{
    int n;
    if (!out || !out_n || d_k < 2)
        return -1;
    if (!rope_check_base(base) || !isfinite(factor) || factor < 1.0)
        return -1;
    n = d_k / 2;
    base = base * factor; /* Python: base_eff = base * factor */
    for (int i = 0; i < n; i++) {
        /* Python: 1.0 / (base_eff ** (2.0 * i / d_k)) */
        double e = (2.0 * (double)i) / (double)d_k;
        out[i] = 1.0 / pow(base, e);
    }
    *out_n = n;
    return 0;
}

int ai_longrope_inv_freqs(int d_k, double base, double factor,
                          int original_max, double low_fraction,
                          double *out, int *out_n)
{
    int n, k;
    if (!out || !out_n || d_k < 2 || original_max < 1)
        return -1;
    if (!rope_check_base(base) || !isfinite(factor) || factor < 1.0)
        return -1;
    if (!isfinite(low_fraction) || !(low_fraction > 0.0 && low_fraction < 1.0))
        return -1;
    n = d_k / 2;
    for (int i = 0; i < n; i++) {
        double e = (2.0 * (double)i) / (double)d_k;
        out[i] = 1.0 / pow(base, e);
    }
    /* Python: k = max(1, int(round(n * low_fraction))) */
    k = (int)rint((double)n * low_fraction);
    if (k < 1)
        k = 1;
    for (int i = n - k; i < n; i++)
        out[i] /= factor;
    *out_n = n;
    return 0;
}

int ai_rope_2d(const double *rows, int n_rows, int width,
               int grid_h, int grid_w, double base, double *out)
{
    RopeGrid grid;
    int half;
    if (!rows || !out || n_rows < 1 || width < 1)
        return -1;
    if (grid_h < 1 || grid_w < 1 || !rope_check_base(base))
        return -1;
    if (!rope_rows_match(n_rows, (long long)grid_h * (long long)grid_w))
        return -1;
    if (width % 4 != 0)
        return -1;
    half = width / 2;
    if (rope_copy_rows(rows, n_rows, width, out) != 0)
        return -1;
    grid.grid_h = grid_h;
    grid.grid_w = grid_w;
    /* Python: 先按行位置（t // grid_w）旋前半段，再按列位置旋后半段。 */
    if (rope_rotate_section(out, n_rows, width, 0, half, ROPE_POS_ROW,
                            &grid, base, 1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, half, half, ROPE_POS_COL,
                            &grid, base, 1.0) != 0)
        return -1;
    return 0;
}

int ai_rope_2d_inverse(const double *rows, int n_rows, int width,
                       int grid_h, int grid_w, double base, double *out)
{
    RopeGrid grid;
    int half;
    if (!rows || !out || n_rows < 1 || width < 1)
        return -1;
    if (grid_h < 1 || grid_w < 1 || !rope_check_base(base))
        return -1;
    if (!rope_rows_match(n_rows, (long long)grid_h * (long long)grid_w))
        return -1;
    if (width % 4 != 0)
        return -1;
    half = width / 2;
    if (rope_copy_rows(rows, n_rows, width, out) != 0)
        return -1;
    grid.grid_h = grid_h;
    grid.grid_w = grid_w;
    /* Python: 逆序 + sign=-1。 */
    if (rope_rotate_section(out, n_rows, width, half, half, ROPE_POS_COL,
                            &grid, base, -1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, 0, half, ROPE_POS_ROW,
                            &grid, base, -1.0) != 0)
        return -1;
    return 0;
}

int ai_rope_3d(const double *rows, int n_rows, int width,
               int grid_t, int grid_h, int grid_w, double base, double *out)
{
    RopeGrid grid;
    int seg;
    long long want;
    if (!rows || !out || n_rows < 1 || width < 1)
        return -1;
    if (grid_t < 1 || grid_h < 1 || grid_w < 1 || !rope_check_base(base))
        return -1;
    want = (long long)grid_t * (long long)grid_h * (long long)grid_w;
    if (!rope_rows_match(n_rows, want))
        return -1;
    if (width % 6 != 0)
        return -1;
    seg = width / 3;
    if (rope_copy_rows(rows, n_rows, width, out) != 0)
        return -1;
    grid.grid_h = grid_h;
    grid.grid_w = grid_w;
    if (rope_rotate_section(out, n_rows, width, 0, seg, ROPE_POS_T,
                            &grid, base, 1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, seg, seg, ROPE_POS_H,
                            &grid, base, 1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, 2 * seg, seg, ROPE_POS_W,
                            &grid, base, 1.0) != 0)
        return -1;
    return 0;
}

int ai_rope_3d_inverse(const double *rows, int n_rows, int width,
                       int grid_t, int grid_h, int grid_w, double base, double *out)
{
    RopeGrid grid;
    int seg;
    long long want;
    if (!rows || !out || n_rows < 1 || width < 1)
        return -1;
    if (grid_t < 1 || grid_h < 1 || grid_w < 1 || !rope_check_base(base))
        return -1;
    want = (long long)grid_t * (long long)grid_h * (long long)grid_w;
    if (!rope_rows_match(n_rows, want))
        return -1;
    if (width % 6 != 0)
        return -1;
    seg = width / 3;
    if (rope_copy_rows(rows, n_rows, width, out) != 0)
        return -1;
    grid.grid_h = grid_h;
    grid.grid_w = grid_w;
    if (rope_rotate_section(out, n_rows, width, 2 * seg, seg, ROPE_POS_W,
                            &grid, base, -1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, seg, seg, ROPE_POS_H,
                            &grid, base, -1.0) != 0)
        return -1;
    if (rope_rotate_section(out, n_rows, width, 0, seg, ROPE_POS_T,
                            &grid, base, -1.0) != 0)
        return -1;
    return 0;
}
