/*
 * layers_conv.h —— CNN 三件套（Conv2d / MaxPool2d / Flatten）的 C17 移植接口
 *
 * 对应 AI.py 里的同名类：
 *   Conv2d(in_channels, out_channels, kernel_size, stride, padding)
 *       -> conv2d_create(in_channels, out_channels, kernel_size, stride, padding)
 *   MaxPool2d(kernel_size, stride=None) -> maxpool2d_create(kernel_size, stride)
 *   Flatten()                           -> flatten_create()
 *
 * 张量形状约定（与 Python 版一致，一律行主序、double、连续）：
 *   Conv2d   : 输入 (C, H, W)  ->  输出 (out_channels, Ho, Wo)
 *   MaxPool2d: 输入 (C, H, W)  ->  输出 (C, Ho, Wo)
 *   Flatten  : 输入 (C, H, W)  ->  输出 (C*H*W,)
 *
 * 数值约定（细节见 layers_conv.c 里每一处的注释）：
 *   - 凡是 Python 里写 sum(...) / sum(map(_mul, ...)) 的地方，一律走 numeric.h 的
 *     sum_neumaier / dot_neumaier；Python 里本来是朴素 += 的地方（dcol、_col2im、
 *     grad_w/grad_b 的累积）就用普通 +。多重循环下的加法顺序决定逐位结果，
 *     因此循环嵌套顺序也照抄 Python。
 *   - kernel_size / stride / padding 只支持 int。Python 版还接受 (int,int) 元组，
 *     这里按需只做标量版本；真要非方形核时再按 (kh,kw) 扩展即可。
 *   - 随机初始化（He）不追求与 Python 逐位一致（那边是 Mersenne Twister + 极坐标法）；
 *     需要跨语言逐位比对时用 conv2d_set_weights() 从外部灌入权重。
 *
 * 内存与形状约定（本项目的硬要求）：
 *   - forward / backward 热路径不调用 malloc：im2col / dcol / dxpad / argmax 掩码
 *     等临时缓冲在"绑定输入尺寸"时一次性分配，之后只 memset + 复用。
 *     绑定方式二选一：显式调用 conv2d_bind_shape() / maxpool2d_bind_shape()，
 *     或者让第一次 forward 按输入形状懒绑定（只发生一次）。绑定之后输入形状
 *     再变化会直接报错退出，而不是在热路径里悄悄重新分配。
 *   - 形状不匹配一律 tensor_check_shape() 报错退出，绝不静默截断。
 *   - forward 把结果"写入" out；backward 把梯度"写入" dx（不是累加，与 Python
 *     返回新列表的做法一致），同时把参数梯度"累加"到模块内部 gW / gb
 *     （调用方负责按 batch 求平均，并在需要时先调 xxx_zero_grad()）。
 *
 * 所有权约定（与 tensor.h 的 module_free 配套，重要）：
 *   - 每个层结构体的第一个成员必须是 Module mod，于是 &layer->mod == layer；
 *     module_free() 末尾的 free(m) 因此恰好释放整个层结构体。
 *   - xxx_as_module() 只是返回 &self->mod（create 时已填好 vtable 与 self），
 *     可以反复调用、返回同一地址，不额外分配任何东西。
 *   - 释放二选一，不要混用：
 *       a) 单独使用：xxx_free(layer)；
 *       b) 交给容器（Sequential / ResidualBlock）：容器对该 Module 调
 *          module_free()，它会释放层的内部缓冲并释放层结构体本身。
 *     两条路径都会释放层结构体，所以不要对同一层把两者都调一遍。
 */
#ifndef AI_LAYERS_CONV_H
#define AI_LAYERS_CONV_H

#include <stddef.h>

#include "tensor.h"

/* ============================== Conv2d ============================== */

typedef struct {
    Module mod;             /* 必须是第一个成员，理由见文件头的"所有权约定" */

    /* 超参：对应 Python 的 in_channels / out_channels / (kh,kw) / (sh,sw) / (ph,pw) */
    int in_channels;
    int out_channels;
    int kh, kw;
    int sh, sw;
    int ph, pw;

    /* 参数与梯度：W 展平成 [oc][ic][kh][kw]（oc 最外层，b 长度 out_channels），
     * 与 Python 的 self.weights / self.biases / self.grad_w / self.grad_b 同序 */
    double *W;
    double *b;
    double *gW;
    double *gb;

    /* Adam 状态：对应 Python 的 m_w / v_w / m_b / v_b。
     * 本文件不实现优化器，先把状态分配好，留给后续的 optimizer 移植统一使用 */
    double *mW, *vW;
    double *mb, *vb;

    /* 前向缓存：对应 Python 的 self.x_shape / h_out / w_out / x_pad / x_col
     * 注意形状字段命名为 in_c/in_h/in_w：不能叫 W，否则与权重成员 W 重名 */
    int in_c, in_h, in_w;   /* 输入形状 (C,H,W) */
    int out_h, out_w;       /* 输出空间尺寸 */
    int pad_h, pad_w;       /* 填充后的尺寸 H+2*ph, W+2*pw */
    double *xpad;           /* (C, Hp, Wp)，零填充后的输入 */
    double *col;            /* im2col 矩阵 (col_rows, col_cols)，行主序 = Python 的 self.x_col */
    int col_rows;           /* C*kh*kw */
    int col_cols;           /* Ho*Wo */

    /* 反向缓冲（绑定尺寸时一次性分配，热路径零分配） */
    double *dcol;           /* (col_rows, col_cols) = Python backward 里的 dcol */
    double *dxpad;          /* (C, Hp, Wp) = Python 的 dx_padded */
    double *scratch;        /* 长度 col_rows：前向点积的乘积暂存，见 conv2d_forward 注释 */

    int bound;              /* 1 = 已绑定输入尺寸 */
} Conv2d;

/* 创建一个卷积层：kernel/stride/padding 均为标量（对应 Python 的 int 形式）。
 * 要求 kernel_size >= 1、stride >= 1、padding >= 0，否则报错退出。
 * 与 ai.c 的 mlp_create 一致：参数只做零初始化，需要 He 随机初始化时由调用方
 * 自己调 conv2d_init_he(conv, &seed)（Python 版是在构造函数里直接 He 初始化的） */
Conv2d *conv2d_create(int in_channels, int out_channels,
                      int kernel_size, int stride, int padding);
void    conv2d_free(Conv2d *self);

/* 绑定输入的空间尺寸并一次性分配 im2col / dcol / dxpad / 暂存缓冲。
 * 必须在第一次 forward 之前调用（或依赖 forward 的懒绑定）；重复绑定同一形状是空操作，
 * 形状变化时会重新分配——只应在"换输入尺寸"这种冷路径上调用 */
void conv2d_bind_shape(Conv2d *self, int C, int H, int W);

/* 输出空间尺寸，对应 Python 的 _img_shape：ho = (H + 2*ph - kh)//sh + 1 */
void conv2d_out_shape(const Conv2d *self, int H, int W, int *ho, int *wo);

void conv2d_zero_grad(Conv2d *self);
void conv2d_forward(Conv2d *self, const Tensor *x, Tensor *out);
void conv2d_backward(Conv2d *self, const Tensor *dout, Tensor *dx);
int  conv2d_param_count(Conv2d *self);

/* 从外部灌入权重（跨语言等价性验证用）：
 * W 按 Python 的 self.weights[oc][ic][kh][kw] 顺序展平（oc 最外层），长度
 * out_channels*in_channels*kh*kw；b 长度 out_channels。b 可以传 NULL（保持原值） */
void conv2d_set_weights(Conv2d *self, const double *W, const double *b);
/* 取出当前权重（两个目标都可以传 NULL 表示不需要） */
void conv2d_get_weights(const Conv2d *self, double *W, double *b);

/* He 初始化：std = sqrt(2 / fan_in)，fan_in = in_channels*kh*kw；
 * 与 ai.c 的 he_init 一样用调用方传入的 xorshift 状态（不追求与 Python 逐位一致） */
void conv2d_init_he(Conv2d *self, unsigned long long *rng_state);

Module *conv2d_as_module(Conv2d *self);
void    conv2d_to_string(const Conv2d *self, char *buf, size_t n);

/* ============================ MaxPool2d ============================ */

typedef struct {
    Module mod;             /* 必须是第一个成员 */

    int kh, kw;
    int sh, sw;

    /* 前向缓存：in_c/in_h/in_w 对应 Python 的 self.x_shape，mask 对应 self.mask。
     * Python 存的是窗口内 (kh,kw) 元组，这里存等价的展平偏移 kh*kw_dim + kw */
    int in_c, in_h, in_w;
    int out_h, out_w;
    int *mask;              /* C*Ho*Wo 个窗口内偏移 */

    int bound;
} MaxPool2d;

/* stride <= 0 表示与 kernel_size 相同（对应 Python 的 stride=None）。
 * 要求 kernel_size >= 1 */
MaxPool2d *maxpool2d_create(int kernel_size, int stride);
void       maxpool2d_free(MaxPool2d *self);

/* 绑定输入尺寸并分配 argmax 掩码（同 Conv2d 的懒绑定约定） */
void maxpool2d_bind_shape(MaxPool2d *self, int C, int H, int W);
/* 输出空间尺寸，对应 Python 的 ho = (H - kh)//sh + 1 */
void maxpool2d_out_shape(const MaxPool2d *self, int H, int W, int *ho, int *wo);

void maxpool2d_zero_grad(MaxPool2d *self);   /* 无参数，空操作（保留统一接口） */
void maxpool2d_forward(MaxPool2d *self, const Tensor *x, Tensor *out);
void maxpool2d_backward(MaxPool2d *self, const Tensor *dout, Tensor *dx);
int  maxpool2d_param_count(MaxPool2d *self);

Module *maxpool2d_as_module(MaxPool2d *self);
void    maxpool2d_to_string(const MaxPool2d *self, char *buf, size_t n);

/* ============================= Flatten ============================= */

typedef struct {
    Module mod;             /* 必须是第一个成员 */

    int in_c, in_h, in_w;   /* 最近一次 forward 的输入形状，对应 Python 的 self.x_shape */
    int bound;
} Flatten;

Flatten *flatten_create(void);
void     flatten_free(Flatten *self);

void flatten_zero_grad(Flatten *self);       /* 无参数，空操作（保留统一接口） */
void flatten_forward(Flatten *self, const Tensor *x, Tensor *out);
void flatten_backward(Flatten *self, const Tensor *dout, Tensor *dx);
int  flatten_param_count(Flatten *self);

Module *flatten_as_module(Flatten *self);
void    flatten_to_string(const Flatten *self, char *buf, size_t n);

#endif /* AI_LAYERS_CONV_H */
