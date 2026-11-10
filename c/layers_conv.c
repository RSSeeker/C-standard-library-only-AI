/*
 * layers_conv.c —— Conv2d / ConvTranspose2d / Conv1d / MaxPool2d / Flatten 的 C17 实现
 *
 * 逐段对照 AI.py（当前版本）：
 *   Conv2d.__init__ / zero_grad / _img_shape / _pad / _im2col / _col2im
 *   / _flatten_weights / _unpad / forward / backward          —— 178~366 行
 *   MaxPool2d.__init__ / forward / backward                   —— 369~420 行
 *   Flatten.__init__ / forward / backward                     —— 423~450 行
 *
 * 数值对齐的三条铁律（不照抄运算顺序就会差 1 ULP 并随层数累积）：
 *   1) Python 里写 sum(...) 或 sum(map(_mul, ...)) 的地方 -> numeric.h 的
 *      sum_neumaier / dot_neumaier（CPython 3.12+ 对浮点用 Neumaier 补偿求和）。
 *      注意 Conv2d.forward 并不是朴素三重循环：它走 _matmul，而 _matmul 内部是
 *      sum(map(_mul, W_row, col_col))，所以前向点积同样是 Neumaier 求和。
 *   2) Python 里本来就是朴素 += 的地方（dcol 的 drow[i] += w*d[i]、_col2im 的
 *      dx[ih][iw] += ...、grad_w/grad_b 的累加）-> 普通 +，不要用补偿求和。
 *   3) 多重循环 + 朴素累加时"加法顺序"决定逐位结果，循环嵌套必须一致：
 *      _col2im 是 c/kh/kw/oh/ow，dcol 是 oc/row/i，MaxPool2d.backward 是 c/oh/ow。
 *
 * 另外两处 Python 语义上的坑（都在下面注释里标注了）：
 *   - _img_shape / MaxPool2d 用的是 // （向下取整），C 的 / 是向零取整；
 *     这里先要求 (H + 2*ph - kh) >= 0，此时两者一致（Python 在负数时会得到
 *     空输出或 <=0 的尺寸，C 的张量表示不出这种形状，直接报错退出）。
 *   - MaxPool2d.forward 从 -inf 开始且用严格 > 比较，所以并列时取窗口内
 *     第一个（kh,kw）位置；NaN 因为比较恒为假而永远不会被选中——与 Python 相同。
 */
#include "layers_conv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numeric.h"

/* 严格 C17 下 math.h 不保证提供 M_PI（那是 POSIX 扩展），自己定义 */
#define CONV_PI 3.14159265358979323846

/* ============================ 内存 ============================ */

/* 与 ai.c 一致的分配行为：失败即退出，避免到处判空。
 * 需要清零的缓冲用 xcalloc；xpad/col/dcol/dxpad/scratch/mask 这些每次前向都会被
 * 完整写入的缓冲用 xmalloc，省掉一次 memset */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "layers_conv: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "layers_conv: out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

/* 统一的报错出口：形状/参数非法一律立刻退出，不静默截断 */
static void die(const char *who, const char *msg)
{
    fprintf(stderr, "layers_conv: %s: %s\n", who, msg);
    exit(1);
}

/* ==================== 随机数（仅用于 He 初始化） ====================
 * 与 ai.c 一样用 xorshift + Box-Muller。Python 用的是 Mersenne Twister +
 * 极坐标法，两边不可能逐位一致；需要跨语言比对时用 conv2d_set_weights()
 * 从 Python 导出权重再灌进来。
 */
static unsigned long long rng_next(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(unsigned long long *s)
{
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

static double rng_gauss(unsigned long long *s)
{
    double u1 = rng_uniform(s);
    double u2 = rng_uniform(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * CONV_PI * u2);
}

/* ==================== Module 包装（供容器使用） ====================
 * 每个层结构体的第一个成员就是 Module，因此 &layer->mod == layer，
 * module_free() 里的 free(m) 正好释放整个层结构体（见 layers_conv.h 的
 * "所有权约定"）。vtable 里的 self 指向层结构体本身。
 */
#define MODULE_SELF(type, m) ((type *)((m)->self))

/* ============================== Conv2d ============================== */

/* 对应 Python 的 _img_shape：ho = (H + 2*ph - kh_eff)//sh + 1
 * Python 用 // 向下取整；先要求 (H + 2*ph - kh_eff) >= 0，此时 C 的截断除法与
 * // 结果相同。负数时 Python 会得到 <=0 的输出尺寸（空输出），C 的张量
 * 表示不出这种形状，直接报错退出 */
static void conv_img_shape(const Conv2d *self, int H, int W, int *ho, int *wo)
{
    int dh = H + 2 * self->ph - self->kh_eff;
    int dw = W + 2 * self->pw - self->kw_eff;
    if (dh < 0 || dw < 0)
        die("conv2d", "输入比有效卷积核还小（H + 2*padding < kh_eff），Python 版此时是空输出");
    *ho = dh / self->sh + 1;
    *wo = dw / self->sw + 1;
}

/* 释放层内部缓冲（不 free 结构体本身）：单独使用与容器路径共用 */
static void conv_release(Conv2d *self)
{
    free(self->W);  free(self->b);  free(self->gW); free(self->gb);
    free(self->mW); free(self->vW); free(self->mb); free(self->vb);
    free(self->xpad); free(self->col); free(self->dcol);
    free(self->dxpad); free(self->scratch);
    self->W = self->b = self->gW = self->gb = NULL;
    self->mW = self->vW = self->mb = self->vb = NULL;
    self->xpad = self->col = self->dcol = self->dxpad = self->scratch = NULL;
    self->bound = 0;
}

/* ---------- Conv2d 的 Module vtable（回调体就是下面的公开函数） ---------- */

static void conv_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    conv2d_forward(MODULE_SELF(Conv2d, m), x, out);
}

static void conv_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    conv2d_backward(MODULE_SELF(Conv2d, m), dout, dx);
}

static void conv_vt_zero_grad(Module *m)
{
    conv2d_zero_grad(MODULE_SELF(Conv2d, m));
}

static void conv_vt_free(Module *m)
{
    /* 容器走 module_free()：这里只释放内部缓冲，层结构体由 module_free() 里的
     * free(m) 释放（Module 是结构体首成员，两者地址相同） */
    conv_release(MODULE_SELF(Conv2d, m));
}

static int conv_vt_param_count(Module *m)
{
    return conv2d_param_count(MODULE_SELF(Conv2d, m));
}

static void conv_vt_to_string(Module *m, char *buf, size_t n)
{
    conv2d_to_string(MODULE_SELF(Conv2d, m), buf, n);
}

static const ModuleVT conv2d_vt = {
    "Conv2d",
    conv_vt_forward, conv_vt_backward, conv_vt_zero_grad,
    conv_vt_free, conv_vt_param_count, conv_vt_to_string
};

/* ---------- Conv2d 的实现 ---------- */

void conv2d_out_shape(const Conv2d *self, int H, int W, int *ho, int *wo)
{
    conv_img_shape(self, H, W, ho, wo);
}

void conv2d_bind_shape(Conv2d *self, int C, int H, int W)
{
    if (C != self->in_channels)
        die("conv2d_bind_shape", "通道数与 in_channels 不一致");
    if (H < 1 || W < 1)
        die("conv2d_bind_shape", "H/W 必须为正");

    int ho, wo;
    conv_img_shape(self, H, W, &ho, &wo);

    if (self->bound && C == self->in_c && H == self->in_h && W == self->in_w)
        return;     /* 同形状：空操作 */

    /* 换形状是冷路径：先把旧缓冲全部释放，再按新尺寸一次性分配 */
    free(self->xpad); free(self->col); free(self->dcol);
    free(self->dxpad); free(self->scratch);
    self->xpad = self->col = self->dcol = self->dxpad = self->scratch = NULL;

    self->in_c = C; self->in_h = H; self->in_w = W;
    self->out_h = ho; self->out_w = wo;
    self->pad_h = H + 2 * self->ph;
    self->pad_w = W + 2 * self->pw;
    /* col_rows 是"一组"的行数；col 里拼 groups 份，所以 col_n 仍等于
     * in_channels*kh*kw*P，与 groups=1 时的内存量完全相同 */
    self->col_rows = self->cg * self->kh * self->kw;
    self->col_cols = ho * wo;

    size_t pad_n = (size_t)self->cg * self->pad_h * self->pad_w;  /* 按组复用的 x_pad / dx_padded */
    size_t col_n = (size_t)C * self->kh * self->kw * self->col_cols;

    /* 这几个缓冲每次 forward/backward 都会先整体写满（memset 或逐元素赋值）再读，
     * 所以用 xmalloc 不做零初始化；权重/Adam 状态那种"必须从 0 开始"的才用 xcalloc */
    self->xpad    = xmalloc(pad_n * sizeof(double));
    self->col     = xmalloc(col_n * sizeof(double));
    self->dcol    = xmalloc(col_n * sizeof(double));
    self->dxpad   = xmalloc(pad_n * sizeof(double));
    self->scratch = xmalloc((size_t)self->col_rows * sizeof(double));  /* 前向点积暂存 */
    self->bound = 1;
}

Conv2d *conv2d_create_ex(int in_channels, int out_channels,
                         int kernel_size, int stride, int padding,
                         int groups, int dilation)
{
    if (in_channels < 1 || out_channels < 1)
        die("conv2d_create", "in_channels / out_channels 必须为正");
    if (kernel_size < 1) die("conv2d_create", "kernel_size 必须 >= 1");
    if (stride < 1)      die("conv2d_create", "stride 必须 >= 1");
    if (padding < 0)     die("conv2d_create", "padding 必须 >= 0");
    if (groups < 1)      die("conv2d_create", "groups 必须 >= 1");
    if (in_channels % groups != 0)
        die("conv2d_create", "in_channels 必须能被 groups 整除");
    if (out_channels % groups != 0)
        die("conv2d_create", "out_channels 必须能被 groups 整除");
    if (dilation < 1)    die("conv2d_create", "dilation 必须 >= 1");

    Conv2d *self = xcalloc(1, sizeof(Conv2d));
    self->in_channels = in_channels;
    self->out_channels = out_channels;
    self->groups = groups;
    self->cg = in_channels / groups;
    self->og = out_channels / groups;
    self->kh = self->kw = kernel_size;   /* 只支持方形核（Python 的 int 形式） */
    self->sh = self->sw = stride;
    self->ph = self->pw = padding;
    self->dh = self->dw = dilation;
    self->kh_eff = (self->kh - 1) * self->dh + 1;
    self->kw_eff = (self->kw - 1) * self->dw + 1;

    size_t w_n = (size_t)out_channels * self->cg * kernel_size * kernel_size;
    self->W  = xcalloc(w_n, sizeof(double));
    self->b  = xcalloc((size_t)out_channels, sizeof(double));
    self->gW = xcalloc(w_n, sizeof(double));
    self->gb = xcalloc((size_t)out_channels, sizeof(double));
    self->mW = xcalloc(w_n, sizeof(double));
    self->vW = xcalloc(w_n, sizeof(double));
    self->mb = xcalloc((size_t)out_channels, sizeof(double));
    self->vb = xcalloc((size_t)out_channels, sizeof(double));
    /* 权重零初始化（与 mlp_create 一致）；要 He 初始化请调 conv2d_init_he() */

    /* Module 包装：不额外分配，直接把结构体首成员填好 */
    self->mod.vt = &conv2d_vt;
    self->mod.self = self;
    return self;
}

Conv2d *conv2d_create(int in_channels, int out_channels,
                      int kernel_size, int stride, int padding)
{
    return conv2d_create_ex(in_channels, out_channels, kernel_size, stride, padding, 1, 1);
}

void conv2d_free(Conv2d *self)
{
    if (!self) return;
    conv_release(self);   /* 与容器路径共用同一份释放代码 */
    free(self);           /* Module 是首成员，这块内存就是层本身 */
}

void conv2d_zero_grad(Conv2d *self)
{
    /* 与 Python 的 zero_grad 一致：只清梯度，保留前向缓存 */
    memset(self->gW, 0, (size_t)self->out_channels * self->cg
                        * self->kh * self->kw * sizeof(double));
    memset(self->gb, 0, (size_t)self->out_channels * sizeof(double));
}

void conv2d_set_weights(Conv2d *self, const double *W, const double *b)
{
    size_t w_n = (size_t)self->out_channels * self->cg * self->kh * self->kw;
    if (W) memcpy(self->W, W, w_n * sizeof(double));
    if (b) memcpy(self->b, b, (size_t)self->out_channels * sizeof(double));
}

void conv2d_get_weights(const Conv2d *self, double *W, double *b)
{
    size_t w_n = (size_t)self->out_channels * self->cg * self->kh * self->kw;
    if (W) memcpy(W, self->W, w_n * sizeof(double));
    if (b) memcpy(b, self->b, (size_t)self->out_channels * sizeof(double));
}

void conv2d_init_he(Conv2d *self, unsigned long long *rng_state)
{
    /* 对应 Python __init__ 里的 He 初始化：
     *   fan_in = cg * kh * kw;  std = sqrt(2.0 / fan_in)
     * 生成顺序也和 Python 的嵌套列表推导一致：oc -> ic -> kh -> kw */
    int fan_in = self->cg * self->kh * self->kw;
    double std = sqrt(2.0 / (double)fan_in);
    size_t w_n = (size_t)self->out_channels * fan_in;
    for (size_t i = 0; i < w_n; i++) self->W[i] = rng_gauss(rng_state) * std;
    for (int oc = 0; oc < self->out_channels; oc++) self->b[oc] = 0.0;
}

void conv2d_forward(Conv2d *self, const Tensor *x, Tensor *out)
{
    /* ---- 形状校验（不匹配即退出，不静默截断） ---- */
    if (x->ndim != 3) die("conv2d_forward", "输入必须是 (C,H,W)");
    const int want_x[3] = { self->in_channels, x->shape[1], x->shape[2] };
    tensor_check_shape(x, 3, want_x, "conv2d_forward(x)");

    int C = x->shape[0], H = x->shape[1], W = x->shape[2];

    /* 懒绑定：第一次 forward 按输入形状一次性分配缓冲；之后形状变化必须显式重新绑定，
     * 以免在训练热路径里偷偷 malloc */
    if (!self->bound) {
        conv2d_bind_shape(self, C, H, W);
    } else if (C != self->in_c || H != self->in_h || W != self->in_w) {
        die("conv2d_forward", "输入形状与绑定时不一致；换尺寸请显式调用 conv2d_bind_shape()");
    }

    const int Ho = self->out_h, Wo = self->out_w;
    const int P = Ho * Wo;             /* = col_cols */
    const int Kg = self->col_rows;     /* cg*kh*kw：一组 im2col 的行数 */
    const int Hp = self->pad_h, Wp = self->pad_w;
    const int CG = self->cg, OG = self->og;

    const int want_out[3] = { self->out_channels, Ho, Wo };
    tensor_check_shape(out, 3, want_out, "conv2d_forward(out)");

    /* 每组独立完成 pad -> im2col -> matmul。groups == 1 时循环只跑一次，
     * 循环体内的顺序、索引表达式与旧版单组实现逐字相同（逐位不变） */
    for (int g = 0; g < self->groups; g++) {
        const int c0 = g * CG;          /* 本组第一个输入通道 */
        const int oc0 = g * OG;         /* 本组第一个输出通道 */
        double *xg_pad = self->xpad;    /* xpad 只按一组分配，组间复用 */
        double *xg_col = self->col + (size_t)g * Kg * P;

        /* ---- 1) 零填充：对应 _pad；无 padding 时 Python 直接把该组切片当 x_pad 用 ---- */
        memset(xg_pad, 0, (size_t)CG * Hp * Wp * sizeof(double));
        for (int c = 0; c < CG; c++) {
            for (int h = 0; h < H; h++) {
                memcpy(xg_pad + ((size_t)c * Hp + h + self->ph) * Wp + self->pw,
                       x->data + ((size_t)(c0 + c) * H + h) * W,
                       (size_t)W * sizeof(double));
            }
        }

        /* ---- 2) im2col：逐元素对应 _im2col，行号 = c*kh*kw + kh*kw + kw
         * 膨胀体现在读取位置 oh*sh + kh*dh（Python 的 _im2col 同样是 kh*dh） ---- */
        for (int c = 0; c < CG; c++) {
            for (int kh = 0; kh < self->kh; kh++) {
                for (int kw = 0; kw < self->kw; kw++) {
                    int row = c * self->kh * self->kw + kh * self->kw + kw;
                    double *dst = xg_col + (size_t)row * P;
                    for (int oh = 0; oh < Ho; oh++) {
                        for (int ow = 0; ow < Wo; ow++) {
                            dst[oh * Wo + ow] =
                                xg_pad[((size_t)c * Hp + oh * self->sh + kh * self->dh) * Wp
                                        + ow * self->sw + kw * self->dw];
                        }
                    }
                }
            }
        }

        /* ---- 3) 本组的输出通道做一次矩阵乘法：与 Python 的 _matmul(W_row, col) 同序 ----
         * _matmul 里是 sum(map(_mul, W_row, col_col))，即对 col 的每一列（长度 Kg）
         * 与扁平化权重按行号顺序做 Neumaier 点积。col 行主序、同一列的元素跨步，
         * 没法把列指针直接交给 dot_neumaier；先把 Kg 个乘积按同样顺序写进 scratch，
         * 再交给 sum_neumaier —— 乘法顺序与求和顺序都与 Python 完全一致（逐位相同） */
        for (int oc = oc0; oc < oc0 + OG; oc++) {
            const double *wrow = self->W + (size_t)oc * Kg;
            const double bias = self->b[oc];
            double *orow = out->data + (size_t)oc * P;
            for (int pos = 0; pos < P; pos++) {
                for (int i = 0; i < Kg; i++)
                    self->scratch[i] = wrow[i] * xg_col[(size_t)i * P + pos];
                orow[pos] = sum_neumaier(self->scratch, Kg) + bias;
            }
        }
    }
}

void conv2d_backward(Conv2d *self, const Tensor *dout, Tensor *dx)
{
    if (!self->bound) die("conv2d_backward", "还没有 forward 过，没有 im2col 缓存可用");

    const int C = self->in_c, H = self->in_h, W = self->in_w;
    const int Ho = self->out_h, Wo = self->out_w;
    const int P = Ho * Wo;
    const int K = self->col_rows;
    const int Hp = self->pad_h, Wp = self->pad_w;
    const int OC = self->out_channels;

    /* 形状校验：dout 必须是 (out_channels, Ho, Wo)，dx 必须是 (C,H,W) */
    const int want_dout[3] = { OC, Ho, Wo };
    tensor_check_shape(dout, 3, want_dout, "conv2d_backward(dout)");
    const int want_dx[3] = { C, H, W };
    tensor_check_shape(dx, 3, want_dx, "conv2d_backward(dx)");

    const int Kg = K;                  /* 一组 im2col 的行数（cg*kh*kw） */
    const int CG = self->cg, OG = self->og;

    /* Python 先把 dout 展平成 (OC, ho*wo)，而那正是 dout 的行主序布局，
     * 所以下面直接用 dout->data + oc*P，求和顺序不变。
     *
     * 每组独立回传：grad_w / grad_b 累加进组内通道，dcol -> col2im -> unpad 得到
     * 该组的输入梯度，写进 dx 对应的通道切片。groups == 1 时循环只跑一次，
     * 顺序与旧版单组实现逐字相同（逐位不变） */
    for (int g = 0; g < self->groups; g++) {
        const int c0 = g * CG;
        const int oc0 = g * OG;
        const double *xg_col = self->col + (size_t)g * Kg * P;

        /* ---- grad_w：对应
         *   grad_w[oc][ic][kh][kw] += sum(map(_mul, d, x_col[row]))
         * 求和轴是 ho*wo（Neumaier），累加进 grad_w 用 +=（跨 batch 累积） ---- */
        for (int oc = oc0; oc < oc0 + OG; oc++) {
            const double *d = dout->data + (size_t)oc * P;
            double *grow = self->gW + (size_t)oc * Kg;
            for (int ic = 0; ic < CG; ic++) {
                for (int kh = 0; kh < self->kh; kh++) {
                    for (int kw = 0; kw < self->kw; kw++) {
                        int row = ic * self->kh * self->kw + kh * self->kw + kw;
                        grow[row] += dot_neumaier(d, xg_col + (size_t)row * P, P);
                    }
                }
            }
        }

        /* ---- grad_b：对应 grad_b[oc] += sum(dout_flat[oc])（Neumaier） ---- */
        for (int oc = oc0; oc < oc0 + OG; oc++)
            self->gb[oc] += sum_neumaier(dout->data + (size_t)oc * P, P);

        /* ---- dcol = W_flat^T × dout_flat：对应
         *   for oc / for row / for i: drow[i] += w * d[i]
         * 朴素累加，且每个元素按 oc 升序累加（循环嵌套照抄 Python）。
         * dcol 只按一组分配，组间复用，所以每组开始前先清零 ---- */
        memset(self->dcol, 0, (size_t)Kg * P * sizeof(double));
        for (int oc = oc0; oc < oc0 + OG; oc++) {
            const double *d = dout->data + (size_t)oc * P;
            const double *wflat = self->W + (size_t)oc * Kg;
            for (int row = 0; row < Kg; row++) {
                const double w = wflat[row];
                double *drow = self->dcol + (size_t)row * P;
                for (int i = 0; i < P; i++) drow[i] += w * d[i];
            }
        }

        /* ---- col2im：对应 _col2im，循环嵌套必须是 c/kh/kw/oh/ow（朴素累加），
         * 写入位置 oh*sh + kh*dh / ow*sw + kw*dw（膨胀只在取值上） ---- */
        memset(self->dxpad, 0, (size_t)CG * Hp * Wp * sizeof(double));
        for (int c = 0; c < CG; c++) {
            for (int kh = 0; kh < self->kh; kh++) {
                for (int kw = 0; kw < self->kw; kw++) {
                    int row = c * self->kh * self->kw + kh * self->kw + kw;
                    const double *drow = self->dcol + (size_t)row * P;
                    for (int oh = 0; oh < Ho; oh++) {
                        for (int ow = 0; ow < Wo; ow++) {
                            self->dxpad[((size_t)c * Hp + oh * self->sh + kh * self->dh) * Wp
                                        + ow * self->sw + kw * self->dw] += drow[oh * Wo + ow];
                        }
                    }
                }
            }
        }

        /* ---- 去掉 padding：对应 _unpad（纯拷贝；dx 是"写入"不是累加） ---- */
        if (self->ph > 0 || self->pw > 0) {
            for (int c = 0; c < CG; c++) {
                for (int h = 0; h < H; h++) {
                    memcpy(dx->data + ((size_t)(c0 + c) * H + h) * W,
                           self->dxpad + ((size_t)c * Hp + h + self->ph) * Wp + self->pw,
                           (size_t)W * sizeof(double));
                }
            }
        } else {
            memcpy(dx->data + (size_t)c0 * H * W, self->dxpad,
                   (size_t)CG * H * W * sizeof(double));
        }
    }
}

int conv2d_param_count(Conv2d *self)
{
    if (!self) return 0;
    return self->out_channels * self->cg * self->kh * self->kw + self->out_channels;
}

void conv2d_to_string(const Conv2d *self, char *buf, size_t n)
{
    /* 默认配置（groups=1、无膨胀）保持原来的字符串，免得既有输出与测试要改 */
    if (self->groups == 1 && self->dh == 1 && self->dw == 1) {
        snprintf(buf, n, "Conv2d(in=%d, out=%d, k=%d, s=%d, p=%d)",
                 self->in_channels, self->out_channels, self->kh, self->sh, self->ph);
    } else {
        snprintf(buf, n, "Conv2d(in=%d, out=%d, k=%d, s=%d, p=%d, groups=%d, dilation=%d)",
                 self->in_channels, self->out_channels, self->kh, self->sh, self->ph,
                 self->groups, self->dh);
    }
}

Module *conv2d_as_module(Conv2d *self)
{
    /* create 时 vt / self 已经填好；这里只是取地址，不分配任何东西 */
    return &self->mod;
}

/* ============================ MaxPool2d ============================ */

/* 对应 Python 的 ho = (H - kh)//sh + 1，同样先要求 (H - kh) >= 0 */
static void pool_img_shape(const MaxPool2d *self, int H, int W, int *ho, int *wo)
{
    int dh = H - self->kh;
    int dw = W - self->kw;
    if (dh < 0 || dw < 0)
        die("maxpool2d", "输入比池化窗口还小，Python 版此时是空输出");
    *ho = dh / self->sh + 1;
    *wo = dw / self->sw + 1;
}

/* ---------- MaxPool2d 的 Module vtable ---------- */

static void pool_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    maxpool2d_forward(MODULE_SELF(MaxPool2d, m), x, out);
}

static void pool_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    maxpool2d_backward(MODULE_SELF(MaxPool2d, m), dout, dx);
}

static void pool_vt_zero_grad(Module *m)
{
    maxpool2d_zero_grad(MODULE_SELF(MaxPool2d, m));
}

static void pool_vt_free(Module *m)
{
    MaxPool2d *self = MODULE_SELF(MaxPool2d, m);
    free(self->mask);
    self->mask = NULL;
    self->bound = 0;
}

static int pool_vt_param_count(Module *m)
{
    return maxpool2d_param_count(MODULE_SELF(MaxPool2d, m));
}

static void pool_vt_to_string(Module *m, char *buf, size_t n)
{
    maxpool2d_to_string(MODULE_SELF(MaxPool2d, m), buf, n);
}

static const ModuleVT maxpool2d_vt = {
    "MaxPool2d",
    pool_vt_forward, pool_vt_backward, pool_vt_zero_grad,
    pool_vt_free, pool_vt_param_count, pool_vt_to_string
};

/* ---------- MaxPool2d 的实现 ---------- */

void maxpool2d_out_shape(const MaxPool2d *self, int H, int W, int *ho, int *wo)
{
    pool_img_shape(self, H, W, ho, wo);
}

MaxPool2d *maxpool2d_create(int kernel_size, int stride)
{
    if (kernel_size < 1) die("maxpool2d_create", "kernel_size 必须 >= 1");

    MaxPool2d *self = xcalloc(1, sizeof(MaxPool2d));
    self->kh = self->kw = kernel_size;
    /* Python 的 stride=None 表示与 kernel_size 相同；C 里用 <=0 表达这件事 */
    self->sh = self->sw = (stride <= 0) ? kernel_size : stride;

    self->mod.vt = &maxpool2d_vt;
    self->mod.self = self;
    return self;
}

void maxpool2d_bind_shape(MaxPool2d *self, int C, int H, int W)
{
    if (C < 1 || H < 1 || W < 1) die("maxpool2d_bind_shape", "C/H/W 必须为正");

    int ho, wo;
    pool_img_shape(self, H, W, &ho, &wo);

    if (self->bound && C == self->in_c && H == self->in_h && W == self->in_w) return;

    free(self->mask);
    self->mask = NULL;
    self->in_c = C; self->in_h = H; self->in_w = W;
    self->out_h = ho; self->out_w = wo;
    /* mask 存"窗口内偏移" kh*kw_dim + kw，与 Python 的 (kh,kw) 元组等价；
     * forward 会把每个输出位置都写一遍，所以不必预清零 */
    self->mask = xmalloc((size_t)C * ho * wo * sizeof(int));
    self->bound = 1;
}

void maxpool2d_forward(MaxPool2d *self, const Tensor *x, Tensor *out)
{
    if (x->ndim != 3) die("maxpool2d_forward", "输入必须是 (C,H,W)");
    int C = x->shape[0], H = x->shape[1], W = x->shape[2];

    if (!self->bound) {
        maxpool2d_bind_shape(self, C, H, W);
    } else if (C != self->in_c || H != self->in_h || W != self->in_w) {
        die("maxpool2d_forward", "输入形状与绑定时不一致；换尺寸请显式调用 maxpool2d_bind_shape()");
    }

    const int Ho = self->out_h, Wo = self->out_w, kw_dim = self->kw;
    const int want_out[3] = { C, Ho, Wo };
    tensor_check_shape(out, 3, want_out, "maxpool2d_forward(out)");

    /* 对应 Python MaxPool2d.forward：max_val 从 -inf 起、严格 > 比较，
     * 所以并列时保留窗口内第一个 (kh,kw)（行优先），NaN 永远不会被选中。
     * 索引顺序 c/oh/ow/kh/kw 与 Python 完全一致 */
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            for (int ow = 0; ow < Wo; ow++) {
                double max_val = -INFINITY;
                int max_pos = 0;
                for (int kh = 0; kh < self->kh; kh++) {
                    for (int kw = 0; kw < kw_dim; kw++) {
                        int ih = oh * self->sh + kh;
                        int iw = ow * self->sw + kw;
                        double v = x->data[((size_t)c * H + ih) * W + iw];
                        if (v > max_val) {
                            max_val = v;
                            max_pos = kh * kw_dim + kw;
                        }
                    }
                }
                size_t idx = ((size_t)c * Ho + oh) * Wo + ow;
                out->data[idx] = max_val;
                self->mask[idx] = max_pos;      /* 记录 argmax 供 backward 使用 */
            }
        }
    }
}

void maxpool2d_backward(MaxPool2d *self, const Tensor *dout, Tensor *dx)
{
    if (!self->bound) die("maxpool2d_backward", "还没有 forward 过，没有 argmax 掩码可用");

    const int C = self->in_c, H = self->in_h, W = self->in_w;
    const int Ho = self->out_h, Wo = self->out_w;

    const int want_dout[3] = { C, Ho, Wo };
    tensor_check_shape(dout, 3, want_dout, "maxpool2d_backward(dout)");
    const int want_dx[3] = { C, H, W };
    tensor_check_shape(dx, 3, want_dx, "maxpool2d_backward(dx)");

    /* Python 是新建零矩阵再累加；这里等价于"先清零再写"，dx 是写入不是累加。
     * 循环顺序 c/oh/ow 与 Python 一致：stride < kernel 时同一位置可能被多个窗口
     * 选中，加法顺序会影响逐位结果 */
    memset(dx->data, 0, (size_t)C * H * W * sizeof(double));
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < Ho; oh++) {
            for (int ow = 0; ow < Wo; ow++) {
                size_t idx = ((size_t)c * Ho + oh) * Wo + ow;
                int pos = self->mask[idx];
                int kh = pos / self->kw;
                int kw = pos % self->kw;
                int ih = oh * self->sh + kh;
                int iw = ow * self->sw + kw;
                dx->data[((size_t)c * H + ih) * W + iw] += dout->data[idx];
            }
        }
    }
}

int maxpool2d_param_count(MaxPool2d *self)
{
    (void)self;     /* 池化层没有参数 */
    return 0;
}

void maxpool2d_zero_grad(MaxPool2d *self)
{
    (void)self;     /* 池化层没有参数梯度 */
}

void maxpool2d_to_string(const MaxPool2d *self, char *buf, size_t n)
{
    snprintf(buf, n, "MaxPool2d(k=%d, s=%d)", self->kh, self->sh);
}

Module *maxpool2d_as_module(MaxPool2d *self)
{
    return &self->mod;
}

void maxpool2d_free(MaxPool2d *self)
{
    if (!self) return;
    pool_vt_free(&self->mod);
    free(self);
}

/* ============================= Flatten ============================= */

/* ---------- Flatten 的 Module vtable ---------- */

static void flat_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    flatten_forward(MODULE_SELF(Flatten, m), x, out);
}

static void flat_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    flatten_backward(MODULE_SELF(Flatten, m), dout, dx);
}

static void flat_vt_zero_grad(Module *m)
{
    flatten_zero_grad(MODULE_SELF(Flatten, m));
}

static void flat_vt_free(Module *m)
{
    Flatten *self = MODULE_SELF(Flatten, m);
    self->bound = 0;    /* 没有内部缓冲；结构体由 module_free() 释放 */
}

static int flat_vt_param_count(Module *m)
{
    return flatten_param_count(MODULE_SELF(Flatten, m));
}

static void flat_vt_to_string(Module *m, char *buf, size_t n)
{
    flatten_to_string(MODULE_SELF(Flatten, m), buf, n);
}

static const ModuleVT flatten_vt = {
    "Flatten",
    flat_vt_forward, flat_vt_backward, flat_vt_zero_grad,
    flat_vt_free, flat_vt_param_count, flat_vt_to_string
};

/* ---------- Flatten 的实现 ---------- */

Flatten *flatten_create(void)
{
    Flatten *self = xcalloc(1, sizeof(Flatten));
    self->mod.vt = &flatten_vt;
    self->mod.self = self;
    return self;
}

void flatten_forward(Flatten *self, const Tensor *x, Tensor *out)
{
    if (x->ndim != 3) die("flatten_forward", "输入必须是 (C,H,W)");
    int C = x->shape[0], H = x->shape[1], W = x->shape[2];

    const int want_out[1] = { C * H * W };
    tensor_check_shape(out, 1, want_out, "flatten_forward(out)");

    /* Python 是按 c/h/w 顺序 append，即行主序整块拷贝 */
    memcpy(out->data, x->data, (size_t)C * H * W * sizeof(double));

    self->in_c = C; self->in_h = H; self->in_w = W;   /* 对应 Python 的 self.x_shape */
    self->bound = 1;
}

void flatten_backward(Flatten *self, const Tensor *dout, Tensor *dx)
{
    if (!self->bound) die("flatten_backward", "还没有 forward 过，没有 x_shape 可用");

    const int C = self->in_c, H = self->in_h, W = self->in_w;
    const int want_dout[1] = { C * H * W };
    tensor_check_shape(dout, 1, want_dout, "flatten_backward(dout)");
    const int want_dx[3] = { C, H, W };
    tensor_check_shape(dx, 3, want_dx, "flatten_backward(dx)");

    /* Python 是 dx[c][h][w] = delta[idx]：赋值而不是累加，所以这里直接覆盖 */
    memcpy(dx->data, dout->data, (size_t)C * H * W * sizeof(double));
}

int flatten_param_count(Flatten *self)
{
    (void)self;
    return 0;
}

void flatten_zero_grad(Flatten *self)
{
    (void)self;
}

void flatten_to_string(const Flatten *self, char *buf, size_t n)
{
    if (self->bound) snprintf(buf, n, "Flatten(%d,%d,%d)", self->in_c, self->in_h, self->in_w);
    else             snprintf(buf, n, "Flatten(unbound)");
}

Module *flatten_as_module(Flatten *self)
{
    return &self->mod;
}

void flatten_free(Flatten *self)
{
    if (!self) return;
    flat_vt_free(&self->mod);
    free(self);
}

/* ===================== 参数个数（两个模块共用） ===================== */

/* Conv1d：W 是 [oc][ic][k]（ic 只到 cg） */
static size_t conv1d_weight_count(const Conv1d *self)
{
    return (size_t)self->out_channels * (size_t)self->cg * (size_t)self->k;
}

/* ConvTranspose2d：W 是 [cin][og][kh][kw]（与 PyTorch 的 (in, out/groups, kh, kw) 同序） */
static size_t convt2d_weight_count(const ConvTranspose2d *self)
{
    return (size_t)self->in_channels * (size_t)self->og
         * (size_t)self->kh * (size_t)self->kw;
}

/* ============================= Conv1d ============================= */

/* 与 conv_release 同样的理由：只释放内部缓冲，结构体本身由 conv1d_free / module_free 释放 */
static void conv1d_release(Conv1d *self)
{
    free(self->W);   free(self->b);
    free(self->gW);  free(self->gb);
    free(self->mW);  free(self->vW);
    free(self->mb);  free(self->vb);
    free(self->xbuf);
    self->W = self->b = self->gW = self->gb = NULL;
    self->mW = self->vW = self->mb = self->vb = NULL;
    self->xbuf = NULL;
}

static void conv1d_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    conv1d_forward(MODULE_SELF(Conv1d, m), x, out);
}

static void conv1d_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    conv1d_backward(MODULE_SELF(Conv1d, m), dout, dx);
}

static void conv1d_vt_zero_grad(Module *m)
{
    conv1d_zero_grad(MODULE_SELF(Conv1d, m));
}

static void conv1d_vt_free(Module *m)
{
    conv1d_release(MODULE_SELF(Conv1d, m));
}

static int conv1d_vt_param_count(Module *m)
{
    return conv1d_param_count(MODULE_SELF(Conv1d, m));
}

static void conv1d_vt_to_string(Module *m, char *buf, size_t n)
{
    conv1d_to_string(MODULE_SELF(Conv1d, m), buf, n);
}

static const ModuleVT conv1d_vt = {
    "Conv1d",
    conv1d_vt_forward, conv1d_vt_backward, conv1d_vt_zero_grad,
    conv1d_vt_free, conv1d_vt_param_count, conv1d_vt_to_string
};

Conv1d *conv1d_create(int in_channels, int out_channels, int kernel_size,
                      int stride, int padding, int groups)
{
    if (in_channels < 1 || out_channels < 1)
        die("conv1d_create", "输入/输出通道数必须 >= 1");
    if (kernel_size < 1) die("conv1d_create", "kernel_size 必须 >= 1");
    if (stride < 1)      die("conv1d_create", "stride 必须 >= 1");
    if (padding < 0)     die("conv1d_create", "padding 必须 >= 0");
    if (groups < 1)      die("conv1d_create", "groups 必须 >= 1");
    if (in_channels % groups != 0)
        die("conv1d_create", "in_channels 必须能被 groups 整除");
    if (out_channels % groups != 0)
        die("conv1d_create", "out_channels 必须能被 groups 整除");

    Conv1d *self = xcalloc(1, sizeof *self);
    self->mod.vt = &conv1d_vt;
    self->mod.self = self;

    self->in_channels = in_channels;
    self->out_channels = out_channels;
    self->k = kernel_size;
    self->s = stride;
    self->p = padding;
    self->groups = groups;
    self->cg = in_channels / groups;
    self->og = out_channels / groups;

    const size_t w_n = conv1d_weight_count(self);
    self->W  = xcalloc(w_n, sizeof(double));
    self->gW = xcalloc(w_n, sizeof(double));
    self->mW = xcalloc(w_n, sizeof(double));
    self->vW = xcalloc(w_n, sizeof(double));

    const size_t b_n = (size_t)out_channels;
    self->b  = xcalloc(b_n, sizeof(double));
    self->gb = xcalloc(b_n, sizeof(double));
    self->mb = xcalloc(b_n, sizeof(double));
    self->vb = xcalloc(b_n, sizeof(double));
    return self;
}

void conv1d_free(Conv1d *self)
{
    if (!self) return;
    conv1d_release(self);
    free(self);
}

void conv1d_out_len(const Conv1d *self, int L, int *lo)
{
    if (L < 1) die("conv1d_out_len", "输入长度必须 >= 1");
    /* Python：lo = (L + 2*p - k)//s + 1，且 lo <= 0 时报错。
     * 先要求 L + 2p - k >= 0，此时 C 的 / 与 Python 的 // 一致（都是非负整除） */
    const int span = L + 2 * self->p - self->k;
    if (span < 0)
        die("conv1d_out_len", "输入长度加 padding 之后还比核短，没有合法输出");
    const int out = span / self->s + 1;
    if (out < 1) die("conv1d_out_len", "输出长度 <= 0");
    *lo = out;
}

void conv1d_bind_shape(Conv1d *self, int C, int L)
{
    if (C != self->in_channels)
        die("conv1d_bind_shape", "通道数与 in_channels 不一致");
    int lo = 0;
    conv1d_out_len(self, L, &lo);
    if (!self->xbuf)
        self->xbuf = xcalloc((size_t)C * (size_t)L, sizeof(double));
    self->in_c = C;
    self->in_l = L;
    self->out_l = lo;
    self->bound = 1;
}

void conv1d_zero_grad(Conv1d *self)
{
    memset(self->gW, 0, conv1d_weight_count(self) * sizeof(double));
    memset(self->gb, 0, (size_t)self->out_channels * sizeof(double));
}

void conv1d_forward(Conv1d *self, const Tensor *x, Tensor *out)
{
    if (x->ndim != 2) die("conv1d_forward", "输入必须是 (C,L)");
    const int want_x[2] = { self->in_channels, x->shape[1] };
    tensor_check_shape(x, 2, want_x, "conv1d_forward(x)");

    const int C = x->shape[0];
    const int L = x->shape[1];
    if (!self->bound) conv1d_bind_shape(self, C, L);
    else if (L != self->in_l)
        die("conv1d_forward", "输入长度与绑定时不一致（需要显式重新绑定）");

    const int lo = self->out_l;
    const int want_out[2] = { self->out_channels, lo };
    tensor_check_shape(out, 2, want_out, "conv1d_forward(out)");

    /* Python 的 backward 直接读 self.x；这里在 forward 时留一份快照 */
    memcpy(self->xbuf, x->data, (size_t)C * (size_t)L * sizeof(double));

    /* Python 是朴素三重循环（不走 im2col）：
     *   total = biases[oc]; base = o*s - p
     *   for ic in range(cg): for kk in range(k):
     *       idx = base + kk; 0 <= idx < L 时 total += w[ic][kk] * x[ic0+ic][idx]
     * 加法顺序与乘法操作数顺序都照抄（w * x，不是 x * w） */
    for (int oc = 0; oc < self->out_channels; oc++) {
        const int ic0 = (oc / self->og) * self->cg;
        double *orow = out->data + (size_t)oc * (size_t)lo;
        for (int o = 0; o < lo; o++) {
            double total = self->b[oc];
            const int base = o * self->s - self->p;
            for (int ic = 0; ic < self->cg; ic++) {
                const double *wic = self->W + ((size_t)oc * (size_t)self->cg + (size_t)ic) * (size_t)self->k;
                const double *xc = self->xbuf + (size_t)(ic0 + ic) * (size_t)L;
                for (int kk = 0; kk < self->k; kk++) {
                    const int idx = base + kk;
                    if (idx < 0 || idx >= L) continue;
                    total += wic[kk] * xc[idx];
                }
            }
            orow[o] = total;
        }
    }
}

void conv1d_backward(Conv1d *self, const Tensor *dout, Tensor *dx)
{
    if (!self->bound) die("conv1d_backward", "还没有 forward 过，没有输入缓存可用");

    const int C = self->in_c;
    const int L = self->in_l;
    const int lo = self->out_l;
    const int want_dout[2] = { self->out_channels, lo };
    tensor_check_shape(dout, 2, want_dout, "conv1d_backward(dout)");
    const int want_dx[2] = { C, L };
    tensor_check_shape(dx, 2, want_dx, "conv1d_backward(dx)");

    /* Python 里 dx = [[0.0]*L for _ in range(c)] 然后累加 */
    memset(dx->data, 0, (size_t)C * (size_t)L * sizeof(double));

    for (int oc = 0; oc < self->out_channels; oc++) {
        const double *d = dout->data + (size_t)oc * (size_t)lo;
        /* Python：grad_b[oc] += sum(d) —— 走 sum_neumaier */
        self->gb[oc] += sum_neumaier(d, lo);

        const int ic0 = (oc / self->og) * self->cg;
        for (int o = 0; o < lo; o++) {
            const double dv = d[o];
            const int base = o * self->s - self->p;
            for (int ic = 0; ic < self->cg; ic++) {
                const size_t row = ((size_t)oc * (size_t)self->cg + (size_t)ic) * (size_t)self->k;
                const double *wic = self->W + row;
                double *gwic = self->gW + row;
                const double *xc = self->xbuf + (size_t)(ic0 + ic) * (size_t)L;
                double *dxi = dx->data + (size_t)(ic0 + ic) * (size_t)L;
                for (int kk = 0; kk < self->k; kk++) {
                    const int idx = base + kk;
                    if (idx < 0 || idx >= L) continue;
                    gwic[kk] += dv * xc[idx];   /* Python：gwic[k] += dv * xc[idx] */
                    dxi[idx] += dv * wic[kk];   /* Python：dxi[idx] += dv * wic[k] */
                }
            }
        }
    }
}

int conv1d_param_count(Conv1d *self)
{
    return (int)(conv1d_weight_count(self) + (size_t)self->out_channels);
}

void conv1d_set_weights(Conv1d *self, const double *W, const double *b)
{
    const size_t w_n = conv1d_weight_count(self);
    if (W) memcpy(self->W, W, w_n * sizeof(double));
    if (b) memcpy(self->b, b, (size_t)self->out_channels * sizeof(double));
}

void conv1d_get_weights(const Conv1d *self, double *W, double *b)
{
    const size_t w_n = conv1d_weight_count(self);
    if (W) memcpy(W, self->W, w_n * sizeof(double));
    if (b) memcpy(b, self->b, (size_t)self->out_channels * sizeof(double));
}

void conv1d_init_he(Conv1d *self, unsigned long long *rng_state)
{
    /* Python：fan_in = cg * k; std = sqrt(2.0 / fan_in)，生成顺序 oc -> ic -> kk */
    const int fan_in = self->cg * self->k;
    const double std = sqrt(2.0 / (double)fan_in);
    const size_t w_n = conv1d_weight_count(self);
    for (size_t i = 0; i < w_n; i++) self->W[i] = rng_gauss(rng_state) * std;
    for (int oc = 0; oc < self->out_channels; oc++) self->b[oc] = 0.0;
}

Module *conv1d_as_module(Conv1d *self)
{
    return &self->mod;
}

void conv1d_to_string(const Conv1d *self, char *buf, size_t n)
{
    if (self->groups == 1)
        snprintf(buf, n, "Conv1d(%d->%d, k=%d, s=%d, p=%d)",
                 self->in_channels, self->out_channels, self->k, self->s, self->p);
    else
        snprintf(buf, n, "Conv1d(%d->%d, k=%d, s=%d, p=%d, groups=%d)",
                 self->in_channels, self->out_channels, self->k, self->s, self->p, self->groups);
}

/* ========================= ConvTranspose2d ========================= */

static void convt2d_release(ConvTranspose2d *self)
{
    free(self->W);   free(self->b);
    free(self->gW);  free(self->gb);
    free(self->mW);  free(self->vW);
    free(self->mb);  free(self->vb);
    free(self->xbuf);
    free(self->rows);
    self->W = self->b = self->gW = self->gb = NULL;
    self->mW = self->vW = self->mb = self->vb = NULL;
    self->xbuf = NULL;
    self->rows = NULL;
}

static void convt2d_vt_forward(Module *m, const Tensor *x, Tensor *out)
{
    convt2d_forward(MODULE_SELF(ConvTranspose2d, m), x, out);
}

static void convt2d_vt_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    convt2d_backward(MODULE_SELF(ConvTranspose2d, m), dout, dx);
}

static void convt2d_vt_zero_grad(Module *m)
{
    convt2d_zero_grad(MODULE_SELF(ConvTranspose2d, m));
}

static void convt2d_vt_free(Module *m)
{
    convt2d_release(MODULE_SELF(ConvTranspose2d, m));
}

static int convt2d_vt_param_count(Module *m)
{
    return convt2d_param_count(MODULE_SELF(ConvTranspose2d, m));
}

static void convt2d_vt_to_string(Module *m, char *buf, size_t n)
{
    convt2d_to_string(MODULE_SELF(ConvTranspose2d, m), buf, n);
}

static const ModuleVT convt2d_vt = {
    "ConvTranspose2d",
    convt2d_vt_forward, convt2d_vt_backward, convt2d_vt_zero_grad,
    convt2d_vt_free, convt2d_vt_param_count, convt2d_vt_to_string
};

ConvTranspose2d *convt2d_create(int in_channels, int out_channels, int kernel_size,
                                int stride, int padding, int output_padding, int groups)
{
    if (in_channels < 1 || out_channels < 1)
        die("convt2d_create", "输入/输出通道数必须 >= 1");
    if (kernel_size < 1) die("convt2d_create", "kernel_size 必须 >= 1");
    if (stride < 1)      die("convt2d_create", "stride 必须 >= 1");
    if (padding < 0)     die("convt2d_create", "padding 必须 >= 0");
    if (output_padding < 0) die("convt2d_create", "output_padding 必须 >= 0");
    if (output_padding >= stride)
        die("convt2d_create", "output_padding 必须小于 stride（同 PyTorch）");
    if (groups < 1)      die("convt2d_create", "groups 必须 >= 1");
    if (in_channels % groups != 0)
        die("convt2d_create", "in_channels 必须能被 groups 整除");
    if (out_channels % groups != 0)
        die("convt2d_create", "out_channels 必须能被 groups 整除");

    ConvTranspose2d *self = xcalloc(1, sizeof *self);
    self->mod.vt = &convt2d_vt;
    self->mod.self = self;

    self->in_channels = in_channels;
    self->out_channels = out_channels;
    self->kh = kernel_size;
    self->kw = kernel_size;
    self->sh = stride;
    self->sw = stride;
    self->ph = padding;
    self->pw = padding;
    self->oph = output_padding;
    self->opw = output_padding;
    self->groups = groups;
    self->cg = in_channels / groups;
    self->og = out_channels / groups;

    const size_t w_n = convt2d_weight_count(self);
    self->W  = xcalloc(w_n, sizeof(double));
    self->gW = xcalloc(w_n, sizeof(double));
    self->mW = xcalloc(w_n, sizeof(double));
    self->vW = xcalloc(w_n, sizeof(double));

    const size_t b_n = (size_t)out_channels;
    self->b  = xcalloc(b_n, sizeof(double));
    self->gb = xcalloc(b_n, sizeof(double));
    self->mb = xcalloc(b_n, sizeof(double));
    self->vb = xcalloc(b_n, sizeof(double));
    return self;
}

void convt2d_free(ConvTranspose2d *self)
{
    if (!self) return;
    convt2d_release(self);
    free(self);
}

void convt2d_out_shape(const ConvTranspose2d *self, int H, int W, int *ho, int *wo)
{
    if (H < 1 || W < 1) die("convt2d_out_shape", "输入空间尺寸必须 >= 1");
    /* Python：ho = (H-1)*sh - 2*ph + kh + oph，且 ho <= 0 时报错 */
    const int h = (H - 1) * self->sh - 2 * self->ph + self->kh + self->oph;
    const int w = (W - 1) * self->sw - 2 * self->pw + self->kw + self->opw;
    if (h < 1 || w < 1)
        die("convt2d_out_shape", "输出空间尺寸 <= 0（padding 太大或 output_padding 太小）");
    *ho = h;
    *wo = w;
}

void convt2d_bind_shape(ConvTranspose2d *self, int C, int H, int W)
{
    if (C != self->in_channels)
        die("convt2d_bind_shape", "通道数与 in_channels 不一致");
    int ho = 0, wo = 0;
    convt2d_out_shape(self, H, W, &ho, &wo);
    if (!self->xbuf)
        self->xbuf = xcalloc((size_t)C * (size_t)H * (size_t)W, sizeof(double));
    if (!self->rows)
        self->rows = xcalloc((size_t)ho, sizeof(double));
    self->in_c = C;
    self->in_h = H;
    self->in_w = W;
    self->out_h = ho;
    self->out_w = wo;
    self->bound = 1;
}

void convt2d_zero_grad(ConvTranspose2d *self)
{
    memset(self->gW, 0, convt2d_weight_count(self) * sizeof(double));
    memset(self->gb, 0, (size_t)self->out_channels * sizeof(double));
}

void convt2d_forward(ConvTranspose2d *self, const Tensor *x, Tensor *out)
{
    if (x->ndim != 3) die("convt2d_forward", "输入必须是 (C,H,W)");
    const int want_x[3] = { self->in_channels, x->shape[1], x->shape[2] };
    tensor_check_shape(x, 3, want_x, "convt2d_forward(x)");

    const int C = x->shape[0];
    const int H = x->shape[1];
    const int W = x->shape[2];
    if (!self->bound) convt2d_bind_shape(self, C, H, W);
    else if (H != self->in_h || W != self->in_w)
        die("convt2d_forward", "输入空间尺寸与绑定时不一致（需要显式重新绑定）");

    const int ho = self->out_h;
    const int wo = self->out_w;
    const int want_out[3] = { self->out_channels, ho, wo };
    tensor_check_shape(out, 3, want_out, "convt2d_forward(out)");

    /* backward 要读输入值 */
    memcpy(self->xbuf, x->data, (size_t)C * (size_t)H * (size_t)W * sizeof(double));

    /* Python：先把输出全部填成 bias，再按 ci/co 把输入散写到输出上 */
    for (int co = 0; co < self->out_channels; co++) {
        double *oc_plane = out->data + (size_t)co * (size_t)ho * (size_t)wo;
        for (int i = 0; i < ho * wo; i++) oc_plane[i] = self->b[co];
    }

    const size_t plane_out = (size_t)ho * (size_t)wo;
    const size_t kk_n = (size_t)self->kh * (size_t)self->kw;
    for (int ci = 0; ci < self->in_channels; ci++) {
        const int co0 = (ci / self->cg) * self->og;
        const double *xc = self->xbuf + (size_t)ci * (size_t)H * (size_t)W;
        for (int co = 0; co < self->og; co++) {
            const double *w = self->W + ((size_t)ci * (size_t)self->og + (size_t)co) * kk_n;
            double *oc_plane = out->data + (size_t)(co0 + co) * plane_out;
            for (int oh = 0; oh < H; oh++) {
                const int base_h = oh * self->sh - self->ph;
                const double *xrow = xc + (size_t)oh * (size_t)W;
                for (int ow = 0; ow < W; ow++) {
                    const double v = xrow[ow];
                    const int base_w = ow * self->sw - self->pw;
                    for (int kh = 0; kh < self->kh; kh++) {
                        const int ih = base_h + kh;
                        if (ih < 0 || ih >= ho) continue;
                        double *dstrow = oc_plane + (size_t)ih * (size_t)wo;
                        const double *wrow = w + (size_t)kh * (size_t)self->kw;
                        for (int kw = 0; kw < self->kw; kw++) {
                            const int iw = base_w + kw;
                            if (iw < 0 || iw >= wo) continue;
                            dstrow[iw] += v * wrow[kw];   /* Python：dst[iw] += v * wrow[kw] */
                        }
                    }
                }
            }
        }
    }
}

void convt2d_backward(ConvTranspose2d *self, const Tensor *dout, Tensor *dx)
{
    if (!self->bound) die("convt2d_backward", "还没有 forward 过，没有输入缓存可用");

    const int C = self->in_c;
    const int H = self->in_h;
    const int W = self->in_w;
    const int ho = self->out_h;
    const int wo = self->out_w;

    const int want_dout[3] = { self->out_channels, ho, wo };
    tensor_check_shape(dout, 3, want_dout, "convt2d_backward(dout)");
    const int want_dx[3] = { C, H, W };
    tensor_check_shape(dx, 3, want_dx, "convt2d_backward(dx)");

    memset(dx->data, 0, (size_t)C * (size_t)H * (size_t)W * sizeof(double));

    const size_t plane_out = (size_t)ho * (size_t)wo;
    const size_t kk_n = (size_t)self->kh * (size_t)self->kw;

    /* 与 forward 完全对偶：遍历输入位置的 (oh,ow)，把输出侧的 dout 拉回来，
     * 同一圈里顺手累加 grad_w 并为 dx 求一个和（acc）。
     * 越界判断必须在 kh/kw 内部逐项做，不能提前 break（base 可能是负的）。 */
    for (int ci = 0; ci < self->in_channels; ci++) {
        const int co0 = (ci / self->cg) * self->og;
        for (int co = 0; co < self->og; co++) {
            const size_t wrow_off = ((size_t)ci * (size_t)self->og + (size_t)co) * kk_n;
            const double *w = self->W + wrow_off;
            double *gw = self->gW + wrow_off;
            const double *oc_plane = dout->data + (size_t)(co0 + co) * plane_out;
            for (int oh = 0; oh < H; oh++) {
                const int base_h = oh * self->sh - self->ph;
                const double *xrow = self->xbuf + ((size_t)ci * (size_t)H + (size_t)oh) * (size_t)W;
                double *dxrow = dx->data + ((size_t)ci * (size_t)H + (size_t)oh) * (size_t)W;
                for (int ow = 0; ow < W; ow++) {
                    const int base_w = ow * self->sw - self->pw;
                    const double xv = xrow[ow];
                    double acc = 0.0;
                    for (int kh = 0; kh < self->kh; kh++) {
                        const int ih = base_h + kh;
                        if (ih < 0 || ih >= ho) continue;
                        const double *drow = oc_plane + (size_t)ih * (size_t)wo;
                        const double *wrow = w + (size_t)kh * (size_t)self->kw;
                        double *grow = gw + (size_t)kh * (size_t)self->kw;
                        for (int kw = 0; kw < self->kw; kw++) {
                            const int iw = base_w + kw;
                            if (iw < 0 || iw >= wo) continue;
                            const double dv = drow[iw];
                            grow[kw] += dv * xv;      /* Python：g_w[kh][kw] += dv * xv */
                            acc += dv * wrow[kw];     /* Python：acc += dv * w[kh][kw] */
                        }
                    }
                    dxrow[ow] += acc;
                }
            }
        }
    }

    /* grad_b 必须在 ci 循环之外：放在里面会被放大 cin/groups 倍（Python 注释里的原 bug）。
     * Python 写的是 grad_b[co] += sum(sum(row) for row in dout[co])，
     * 也就是「每行先 sum_neumaier，再对 ho 个行和 sum_neumaier」，两级补偿求和。 */
    for (int co = 0; co < self->out_channels; co++) {
        const double *oc_plane = dout->data + (size_t)co * plane_out;
        for (int oh = 0; oh < ho; oh++)
            self->rows[oh] = sum_neumaier(oc_plane + (size_t)oh * (size_t)wo, wo);
        self->gb[co] += sum_neumaier(self->rows, ho);
    }
}

int convt2d_param_count(ConvTranspose2d *self)
{
    return (int)(convt2d_weight_count(self) + (size_t)self->out_channels);
}

void convt2d_set_weights(ConvTranspose2d *self, const double *W, const double *b)
{
    const size_t w_n = convt2d_weight_count(self);
    if (W) memcpy(self->W, W, w_n * sizeof(double));
    if (b) memcpy(self->b, b, (size_t)self->out_channels * sizeof(double));
}

void convt2d_get_weights(const ConvTranspose2d *self, double *W, double *b)
{
    const size_t w_n = convt2d_weight_count(self);
    if (W) memcpy(W, self->W, w_n * sizeof(double));
    if (b) memcpy(b, self->b, (size_t)self->out_channels * sizeof(double));
}

void convt2d_init_he(ConvTranspose2d *self, unsigned long long *rng_state)
{
    /* Python：fan_in = cg * kh * kw; std = sqrt(2.0 / fan_in)，顺序 ci -> co -> kh -> kw */
    const int fan_in = self->cg * self->kh * self->kw;
    const double std = sqrt(2.0 / (double)fan_in);
    const size_t w_n = convt2d_weight_count(self);
    for (size_t i = 0; i < w_n; i++) self->W[i] = rng_gauss(rng_state) * std;
    for (int co = 0; co < self->out_channels; co++) self->b[co] = 0.0;
}

Module *convt2d_as_module(ConvTranspose2d *self)
{
    return &self->mod;
}

void convt2d_to_string(const ConvTranspose2d *self, char *buf, size_t n)
{
    if (self->groups == 1 && self->oph == 0)
        snprintf(buf, n, "ConvTranspose2d(%d->%d, k=%d, s=%d, p=%d)",
                 self->in_channels, self->out_channels, self->kh, self->sh, self->ph);
    else
        snprintf(buf, n, "ConvTranspose2d(%d->%d, k=%d, s=%d, p=%d, output_padding=%d, groups=%d)",
                 self->in_channels, self->out_channels, self->kh, self->sh, self->ph,
                 self->oph, self->groups);
}
