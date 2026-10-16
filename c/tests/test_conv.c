/*
 * tests/test_conv.c —— Conv2d / MaxPool2d / Flatten 的自测程序
 *
 * 编译（在 c/ 目录下执行）：
 *   gcc -std=c17 -O2 -Wall -Wextra -o tests/test_conv.exe tests/test_conv.c layers_conv.c tensor.c -lm
 *   tests/test_conv.exe
 *
 * 额外可选的"热路径零分配 + 释放路径无泄漏"验证（用链接器的 --wrap 统计
 * malloc/calloc/realloc/free，默认构建不参与编译，见 check_noalloc.c 的同款做法）：
 *   gcc -std=c17 -O2 -Wall -Wextra -DCONV_NOALLOC -Wl,--wrap=malloc -Wl,--wrap=calloc \
 *       -Wl,--wrap=realloc -Wl,--wrap=free \
 *       -o tests/test_conv_noalloc.exe tests/test_conv.c layers_conv.c tensor.c -lm
 *   tests/test_conv_noalloc.exe
 *
 * 覆盖内容：
 *   1) Conv2d 的 grad_w / grad_b / dx 中心差分校验（eps=1e-6，阈值 1e-5），
 *      做法与 AI.py 的 test_ai.py::TestConv2dGradients 一致：
 *      构造标量损失 L = Σ coef[c][i][j] * out[c][i][j]（于是 dL/dout = coef），
 *      解析梯度用 backward 求，数值梯度用中心差分 (L(w+eps)-L(w-eps))/(2*eps) 求。
 *   2) MaxPool2d：forward 形状、argmax 掩码的合法性（值等于窗口最大值、并列时取
 *      窗口内第一个）与独立性、backward 的守恒性（Σdx == Σdout）与差分校验。
 *   3) Flatten：forward/backward 的形状与逐位一致性（顺序必须是 c/h/w）。
 *   4) 三个类的 Module vtable 路径与直接调用逐位一致（供 Sequential 容器使用）。
 */
#include "../layers_conv.h"
#include "../numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(CONV_NOALLOC)
/* 统计热路径里的分配/释放次数（只在本文件里定义 __wrap_*；
 * 四个 --wrap 必须同时给，否则 __real_free 会链接不上） */
void *__real_malloc(size_t n);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *p, size_t n);
void  __real_free(void *p);
static long g_allocs = 0;
static long g_frees = 0;
void *__wrap_malloc(size_t n)            { g_allocs++; return __real_malloc(n); }
void *__wrap_calloc(size_t c, size_t s)  { g_allocs++; return __real_calloc(c, s); }
void *__wrap_realloc(void *p, size_t n)  { g_allocs++; return __real_realloc(p, n); }
void  __wrap_free(void *p)               { if (p) g_frees++; __real_free(p); }
#endif

#define FD_EPS  1e-6
#define FD_TOL  1e-5

/* ========================= 测试小工具 ========================= */

static void *t_xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "test: out of memory\n");
        exit(1);
    }
    return p;
}

/* 确定性伪随机（只用来造测试数据；与 Python 无关，那边是 Mersenne Twister） */
static unsigned long long g_rng = 0x243F6A8885A308D3ULL;

static void rng_seed(unsigned long long s)
{
    g_rng = s ? s : 0x9E3779B97F4A7C15ULL;
}

static unsigned long long rng_u64(void)
{
    unsigned long long x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(void)
{
    return (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_gauss(void)
{
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static void fill_gauss(double *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = rng_gauss();
}

static Tensor *tensor_gauss3(int a, int b, int c)
{
    const int shape[3] = { a, b, c };
    Tensor *t = tensor_new(3, shape);
    fill_gauss(t->data, (size_t)t->size);
    return t;
}

/* ================== 1) Conv2d 的有限差分校验 ================== */

/* L = Σ coef * out（与 Python 的 sum(coef[c][i][j]*o[c][i][j] ...) 同序，Neumaier） */
static double conv_loss(Conv2d *net, const Tensor *x, const Tensor *coef,
                        Tensor *out, double *prod)
{
    conv2d_forward(net, x, out);
    for (int i = 0; i < out->size; i++) prod[i] = coef->data[i] * out->data[i];
    return sum_neumaier(prod, out->size);
}

static int fd_check_conv(int C, int OC, int K, int S, int P, int H, int W,
                         unsigned long long seed)
{
    const int fan_in = C * K * K;
    const size_t w_n = (size_t)OC * fan_in;

    Conv2d *net = conv2d_create(C, OC, K, S, P);

    /* 参数：测试自己留一份 W/b，逐参数扰动后再灌回层里（等价于 Python 里直接改
     * conv.weights / conv.biases）。He 的 std 与 Python 版一致，b 取非零值以便
     * 顺带覆盖 bias 的加法路径（Python 的测试里 b 恒为 0） */
    double *Wt = t_xmalloc(w_n * sizeof(double));
    double *bt = t_xmalloc((size_t)OC * sizeof(double));
    rng_seed(seed);
    const double std = sqrt(2.0 / (double)fan_in);
    for (size_t i = 0; i < w_n; i++) Wt[i] = rng_gauss() * std;
    for (int oc = 0; oc < OC; oc++) bt[oc] = 0.1 * rng_gauss();
    conv2d_set_weights(net, Wt, bt);

    int ho = 0, wo = 0;
    conv2d_out_shape(net, H, W, &ho, &wo);
    const int out_shape[3] = { OC, ho, wo };
    const int x_shape[3] = { C, H, W };

    Tensor *x    = tensor_gauss3(C, H, W);
    Tensor *out  = tensor_new(3, out_shape);
    Tensor *coef = tensor_gauss3(OC, ho, wo);   /* dL/dout = coef */
    Tensor *dx   = tensor_new(3, x_shape);
    double *prod = t_xmalloc((size_t)out->size * sizeof(double));

    /* 解析梯度：先清零再 backward（grad_w/grad_b 在 Python 里是 += 累积的） */
    conv2d_zero_grad(net);
    conv2d_forward(net, x, out);
    conv2d_backward(net, coef, dx);
    /* 下面的差分只调用 forward，不会动 gW/gb/dx，所以可以直接拿它们比较 */

    double worst_w = 0.0, worst_b = 0.0, worst_x = 0.0;

    for (size_t i = 0; i < w_n; i++) {
        double old = Wt[i];
        Wt[i] = old + FD_EPS; conv2d_set_weights(net, Wt, NULL);
        double lp = conv_loss(net, x, coef, out, prod);
        Wt[i] = old - FD_EPS; conv2d_set_weights(net, Wt, NULL);
        double lm = conv_loss(net, x, coef, out, prod);
        Wt[i] = old;          conv2d_set_weights(net, Wt, NULL);

        double err = fabs((lp - lm) / (2.0 * FD_EPS) - net->gW[i]);
        if (err > worst_w) worst_w = err;
    }

    for (int oc = 0; oc < OC; oc++) {
        double old = bt[oc];
        bt[oc] = old + FD_EPS; conv2d_set_weights(net, NULL, bt);
        double lp = conv_loss(net, x, coef, out, prod);
        bt[oc] = old - FD_EPS; conv2d_set_weights(net, NULL, bt);
        double lm = conv_loss(net, x, coef, out, prod);
        bt[oc] = old;          conv2d_set_weights(net, NULL, bt);

        double err = fabs((lp - lm) / (2.0 * FD_EPS) - net->gb[oc]);
        if (err > worst_b) worst_b = err;
    }

    for (int i = 0; i < x->size; i++) {
        double old = x->data[i];
        x->data[i] = old + FD_EPS;
        double lp = conv_loss(net, x, coef, out, prod);
        x->data[i] = old - FD_EPS;
        double lm = conv_loss(net, x, coef, out, prod);
        x->data[i] = old;

        double err = fabs((lp - lm) / (2.0 * FD_EPS) - dx->data[i]);
        if (err > worst_x) worst_x = err;
    }

    int ok = (worst_w < FD_TOL) && (worst_b < FD_TOL) && (worst_x < FD_TOL);

    printf("  C=%d OC=%d K=%d stride=%d pad=%d  H=%d W=%d  ->  输出 (%d,%d)\n",
           C, OC, K, S, P, H, W, ho, wo);
    printf("    grad_w: 最大误差 %.3e   grad_b: %.3e   dx: %.3e   %s\n",
           worst_w, worst_b, worst_x, ok ? "[通过]" : "[失败]");

    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
    free(prod); free(Wt); free(bt);
    conv2d_free(net);
    return ok;
}

/* ================== 2) MaxPool2d 的校验 ================== */

/* 池化 backward 的差分校验：L = Σ coef*out，连续输入下 argmax 稳定（无并列） */
static int fd_check_pool(int C, int K, int S, int H, int W, unsigned long long seed)
{
    rng_seed(seed);

    MaxPool2d *pool = maxpool2d_create(K, S);
    int ho = 0, wo = 0;
    maxpool2d_out_shape(pool, H, W, &ho, &wo);

    const int x_shape[3] = { C, H, W };
    const int o_shape[3] = { C, ho, wo };

    Tensor *x    = tensor_gauss3(C, H, W);
    Tensor *out  = tensor_new(3, o_shape);
    Tensor *coef = tensor_gauss3(C, ho, wo);
    Tensor *dx   = tensor_new(3, x_shape);
    double *prod = t_xmalloc((size_t)out->size * sizeof(double));

    maxpool2d_forward(pool, x, out);
    maxpool2d_backward(pool, coef, dx);

    double worst = 0.0;
    for (int i = 0; i < x->size; i++) {
        double old = x->data[i];
        x->data[i] = old + FD_EPS;
        maxpool2d_forward(pool, x, out);
        for (int j = 0; j < out->size; j++) prod[j] = coef->data[j] * out->data[j];
        double lp = sum_neumaier(prod, out->size);
        x->data[i] = old - FD_EPS;
        maxpool2d_forward(pool, x, out);
        for (int j = 0; j < out->size; j++) prod[j] = coef->data[j] * out->data[j];
        double lm = sum_neumaier(prod, out->size);
        x->data[i] = old;

        double err = fabs((lp - lm) / (2.0 * FD_EPS) - dx->data[i]);
        if (err > worst) worst = err;
    }

    int ok = worst < FD_TOL;
    printf("  K=%d stride=%d  C=%d H=%d W=%d  ->  输出 (%d,%d)：dx 最大误差 %.3e   %s\n",
           K, S, C, H, W, ho, wo, worst, ok ? "[通过]" : "[失败]");

    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
    free(prod);
    maxpool2d_free(pool);
    return ok;
}

/* argmax 掩码的合法性：out 必须等于掩码位置上的输入值，且是窗口内的最大值；
 * 并列时必须取窗口内第一个位置（Python 从 -inf 起、用严格 > 比较） */
static int check_pool_mask(int C, int K, int S, int H, int W, unsigned long long seed)
{
    rng_seed(seed);

    MaxPool2d *pool = maxpool2d_create(K, S);
    int ho = 0, wo = 0;
    maxpool2d_out_shape(pool, H, W, &ho, &wo);

    const int o_shape[3] = { C, ho, wo };
    Tensor *x   = tensor_gauss3(C, H, W);
    Tensor *out = tensor_new(3, o_shape);
    maxpool2d_forward(pool, x, out);

    int bad = 0;
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < ho; oh++) {
            for (int ow = 0; ow < wo; ow++) {
                size_t idx = ((size_t)c * ho + oh) * wo + ow;
                int pos = pool->mask[idx];
                int mkh = pos / K, mkw = pos % K;
                int mih = oh * S + mkh, miw = ow * S + mkw;
                double mv = x->data[((size_t)c * H + mih) * W + miw];
                if (mv != out->data[idx]) bad++;                 /* 掩码位置就是最大值 */

                int first = -1;
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * S + kh, iw = ow * S + kw;
                        double v = x->data[((size_t)c * H + ih) * W + iw];
                        if (v > mv) bad++;                       /* 不该有更大的值 */
                        if (v == mv && first < 0) first = kh * K + kw;
                    }
                }
                if (first != pos) bad++;                         /* 并列时必须取第一个 */
            }
        }
    }

    printf("  K=%d stride=%d  C=%d H=%d W=%d  ->  argmax 掩码检查：%s\n",
           K, S, C, H, W, bad == 0 ? "[通过]" : "[失败]");

    tensor_free(x); tensor_free(out);
    maxpool2d_free(pool);
    return bad == 0;
}

/* backward 的基本性质：Σdx == Σdout；stride == kernel（窗口不重叠）时
 * dx 的非零个数恰好等于输出元素个数，且每个非零值就是对应窗口的 dout */
static int check_pool_backward_props(int C, int K, int S, int H, int W, unsigned long long seed)
{
    rng_seed(seed);

    MaxPool2d *pool = maxpool2d_create(K, S);
    int ho = 0, wo = 0;
    maxpool2d_out_shape(pool, H, W, &ho, &wo);

    const int x_shape[3] = { C, H, W };
    const int o_shape[3] = { C, ho, wo };
    Tensor *x    = tensor_gauss3(C, H, W);
    Tensor *out  = tensor_new(3, o_shape);
    Tensor *dout = tensor_gauss3(C, ho, wo);
    Tensor *dx   = tensor_new(3, x_shape);

    maxpool2d_forward(pool, x, out);
    maxpool2d_backward(pool, dout, dx);

    double s_out = sum_neumaier(dout->data, dout->size);
    double s_dx  = sum_neumaier(dx->data, dx->size);
    double scale = fabs(s_out) > 1.0 ? fabs(s_out) : 1.0;
    int ok_sum = fabs(s_out - s_dx) <= 1e-12 * scale;

    int ok_bijection = 1;
    if (S == K) {   /* 窗口不重叠：dx 与 dout 一一对应 */
        int nz = 0;
        for (int i = 0; i < dx->size; i++) if (dx->data[i] != 0.0) nz++;
        ok_bijection = (nz == out->size);
    }

    printf("  K=%d stride=%d  C=%d H=%d W=%d  ->  Σdx==Σdout: %s（%.15g vs %.15g）  %s\n",
           K, S, C, H, W, ok_sum ? "是" : "否", s_dx, s_out,
           ok_bijection ? (S == K ? "非零个数与输出一致" : "(窗口重叠，跳过一一对应检查)") : "非零个数不对");

    tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
    maxpool2d_free(pool);
    return ok_sum && ok_bijection;
}

/* 并列（输入为常数）时：掩码必须全是 0（窗口内第一个），
 * dout 全 1 时 dx 是 0/1 掩码，且 1 恰好落在每个窗口的左上角 */
static int check_pool_tie(void)
{
    const int C = 2, H = 6, W = 5, K = 2, S = 2;
    MaxPool2d *pool = maxpool2d_create(K, S);
    int ho = 0, wo = 0;
    maxpool2d_out_shape(pool, H, W, &ho, &wo);

    const int x_shape[3] = { C, H, W };
    const int o_shape[3] = { C, ho, wo };
    Tensor *x    = tensor_new(3, x_shape);
    Tensor *out  = tensor_new(3, o_shape);
    Tensor *dout = tensor_new(3, o_shape);
    Tensor *dx   = tensor_new(3, x_shape);
    tensor_fill(x, 3.5);            /* 常数输入：所有窗口内全部并列 */
    tensor_fill(dout, 1.0);

    maxpool2d_forward(pool, x, out);
    maxpool2d_backward(pool, dout, dx);

    int bad = 0;
    for (int i = 0; i < out->size; i++) {
        if (out->data[i] != 3.5) bad++;          /* 最大值就是常数 */
        if (pool->mask[i] != 0) bad++;           /* 并列必须取窗口内第一个 */
    }
    int ones = 0;
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < ho; oh++) {
            for (int ow = 0; ow < wo; ow++) {
                size_t idx = ((size_t)c * H + oh * S) * W + ow * S;
                if (dx->data[idx] != 1.0) bad++;  /* 梯度落在窗口左上角 */
            }
        }
    }
    for (int i = 0; i < dx->size; i++) if (dx->data[i] != 0.0) ones++;
    if (ones != C * ho * wo) bad++;               /* 且只落在这些位置 */

    printf("  常数输入并列时取窗口内第一个 + dout 全 1 的梯度掩码：%s\n",
           bad == 0 ? "[通过]" : "[失败]");

    tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
    maxpool2d_free(pool);
    return bad == 0;
}

/* ================== 3) Flatten 的校验 ================== */

static int check_flatten(void)
{
    const int C = 2, H = 3, W = 4;
    const int n = C * H * W;
    const int x_shape[3] = { C, H, W };
    const int f_shape[1] = { n };

    Flatten *flat = flatten_create();
    Tensor *x     = tensor_new(3, x_shape);
    Tensor *out   = tensor_new(1, f_shape);
    Tensor *delta = tensor_new(1, f_shape);
    Tensor *dx    = tensor_new(3, x_shape);

    /* 输入用互不相同的值，顺序错了立刻能看出来 */
    for (int i = 0; i < n; i++) x->data[i] = (double)i * 0.5 - 3.0;
    for (int i = 0; i < n; i++) delta->data[i] = 100.0 + (double)i;

    flatten_forward(flat, x, out);

    int bad = 0;
    if (out->size != n) bad++;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                /* Python 的 append 顺序就是 c -> h -> w，展平后下标 = (c*H+h)*W+w */
                size_t idx = ((size_t)c * H + h) * W + w;
                if (out->data[idx] != x->data[idx]) bad++;
            }
        }
    }

    flatten_backward(flat, delta, dx);
    if (dx->size != n) bad++;
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                size_t idx = ((size_t)c * H + h) * W + w;
                if (dx->data[idx] != delta->data[idx]) bad++;
            }
        }
    }
    /* backward 是"赋值"语义：把 dx 先弄脏也应该被完整覆盖 */
    tensor_fill(dx, -1.0);
    flatten_backward(flat, delta, dx);
    if (memcmp(dx->data, delta->data, (size_t)n * sizeof(double)) != 0) bad++;

    printf("  (C,H,W)=(%d,%d,%d) -> (%d,)  展平顺序 + 反向还原（含覆盖语义）：%s\n",
           C, H, W, n, bad == 0 ? "[通过]" : "[失败]");

    tensor_free(x); tensor_free(out); tensor_free(delta); tensor_free(dx);
    flatten_free(flat);
    return bad == 0;
}

/* ================== 4) Module vtable 路径 ================== */

static int check_module_vtable(void)
{
    int bad = 0;

    printf("  Module 接口（供 Sequential / ResidualBlock 这类容器使用）：\n");

    /* --- Conv2d：module 路径与直接调用必须逐位一致 --- */
    const int C = 1, OC = 2, K = 2, H = 5, W = 5;
    const int x_shape[3] = { C, H, W };

    int ho = 0, wo = 0;
    Conv2d *net = conv2d_create(C, OC, K, 1, 0);
    conv2d_out_shape(net, H, W, &ho, &wo);
    const int real_o[3] = { OC, ho, wo };

    rng_seed(20240918);
    size_t w_n = (size_t)OC * C * K * K;
    double *Wt = t_xmalloc(w_n * sizeof(double));
    double *bt = t_xmalloc((size_t)OC * sizeof(double));
    fill_gauss(Wt, w_n);
    fill_gauss(bt, (size_t)OC);
    conv2d_set_weights(net, Wt, bt);

    Tensor *x    = tensor_gauss3(C, H, W);
    Tensor *o1   = tensor_new(3, real_o);
    Tensor *o2   = tensor_new(3, real_o);
    Tensor *coef = tensor_gauss3(OC, ho, wo);
    Tensor *d1   = tensor_new(3, x_shape);
    Tensor *d2   = tensor_new(3, x_shape);

    Module *m = conv2d_as_module(net);
    if (m != conv2d_as_module(net)) { printf("    as_module 返回值不稳定\n"); bad++; }
    if (module_param_count(m) != conv2d_param_count(net)) { printf("    param_count 不一致\n"); bad++; }
    if (conv2d_param_count(net) != OC * C * K * K + OC) { printf("    param_count 数值不对\n"); bad++; }

    conv2d_forward(net, x, o1);
    module_forward(m, x, o2);
    if (memcmp(o1->data, o2->data, (size_t)o1->size * sizeof(double)) != 0) {
        printf("    module_forward 与 conv2d_forward 不一致\n"); bad++;
    }

    conv2d_zero_grad(net);
    conv2d_backward(net, coef, d1);
    double *gW_ref = t_xmalloc(w_n * sizeof(double));
    double *gb_ref = t_xmalloc((size_t)OC * sizeof(double));
    memcpy(gW_ref, net->gW, w_n * sizeof(double));
    memcpy(gb_ref, net->gb, (size_t)OC * sizeof(double));

    module_zero_grad(m);
    module_backward(m, coef, d2);
    if (memcmp(net->gW, gW_ref, w_n * sizeof(double)) != 0 ||
        memcmp(net->gb, gb_ref, (size_t)OC * sizeof(double)) != 0 ||
        memcmp(d1->data, d2->data, (size_t)d1->size * sizeof(double)) != 0) {
        printf("    module_backward 与 conv2d_backward 不一致\n"); bad++;
    }

    char buf[64];
    buf[0] = '\0';
    m->vt->to_string(m, buf, sizeof(buf));
    if (buf[0] == '\0') { printf("    to_string 没写出内容\n"); bad++; }

    /* 容器路径释放：module_free 会释放层内部缓冲并释放层结构体本身 */
    module_free(m);
    free(Wt); free(bt); free(gW_ref); free(gb_ref);
    tensor_free(x); tensor_free(o1); tensor_free(o2);
    tensor_free(coef); tensor_free(d1); tensor_free(d2);

    /* --- MaxPool2d / Flatten：走一遍 forward/backward/param_count/free --- */
    MaxPool2d *pool = maxpool2d_create(2, 2);
    Module *pm = maxpool2d_as_module(pool);
    const int px_shape[3] = { 1, 4, 4 };
    const int po_shape[3] = { 1, 2, 2 };
    Tensor *px = tensor_gauss3(1, 4, 4);
    Tensor *po = tensor_new(3, po_shape);
    Tensor *pd = tensor_new(3, po_shape);
    Tensor *pdx = tensor_new(3, px_shape);
    module_forward(pm, px, po);
    module_backward(pm, pd, pdx);
    if (module_param_count(pm) != 0) { printf("    MaxPool2d param_count 应为 0\n"); bad++; }
    tensor_free(px); tensor_free(po); tensor_free(pd); tensor_free(pdx);
    module_free(pm);

    Flatten *flat = flatten_create();
    Module *fm = flatten_as_module(flat);
    Tensor *fx = tensor_gauss3(1, 4, 4);
    const int ff_shape[1] = { 16 };
    Tensor *fo = tensor_new(1, ff_shape);
    Tensor *fd = tensor_new(1, ff_shape);
    Tensor *fdx = tensor_new(3, px_shape);
    module_forward(fm, fx, fo);
    module_backward(fm, fd, fdx);
    if (module_param_count(fm) != 0) { printf("    Flatten param_count 应为 0\n"); bad++; }
    buf[0] = '\0';
    fm->vt->to_string(fm, buf, sizeof(buf));
    if (buf[0] == '\0') { printf("    Flatten to_string 没写出内容\n"); bad++; }
    tensor_free(fx); tensor_free(fo); tensor_free(fd); tensor_free(fdx);
    module_free(fm);

    printf("    Conv2d/MaxPool2d/Flatten 的 vtable 与直接调用逐位一致：%s\n",
           bad == 0 ? "[通过]" : "[失败]");
    return bad == 0;
}

/* ================== 5) 热路径零分配（可选） ================== */

#if defined(CONV_NOALLOC)
static int check_noalloc(void)
{
    const int C = 2, OC = 4, K = 3, H = 8, W = 8;
    const int x_shape[3] = { C, H, W };
    const int px_shape[3] = { 2, 8, 8 };

    Conv2d *net = conv2d_create(C, OC, K, 1, 1);
    int ho = 0, wo = 0;
    conv2d_out_shape(net, H, W, &ho, &wo);
    const int o_shape[3] = { OC, ho, wo };

    MaxPool2d *pool = maxpool2d_create(2, 2);
    const int po_shape[3] = { 2, 4, 4 };

    Tensor *x    = tensor_gauss3(C, H, W);
    Tensor *out  = tensor_new(3, o_shape);
    Tensor *coef = tensor_gauss3(OC, ho, wo);
    Tensor *dx   = tensor_new(3, x_shape);
    Tensor *px   = tensor_gauss3(2, 8, 8);
    Tensor *po   = tensor_new(3, po_shape);
    Tensor *pd   = tensor_new(3, po_shape);
    Tensor *pdx  = tensor_new(3, px_shape);

    /* 热身：第一次 forward 会懒绑定并分配 im2col / dcol / dxpad / argmax 掩码 */
    conv2d_forward(net, x, out);
    conv2d_backward(net, coef, dx);
    maxpool2d_forward(pool, px, po);
    maxpool2d_backward(pool, pd, pdx);

    long before = g_allocs;
    for (int it = 0; it < 200; it++) {
        conv2d_zero_grad(net);
        conv2d_forward(net, x, out);
        conv2d_backward(net, coef, dx);
        maxpool2d_forward(pool, px, po);
        maxpool2d_backward(pool, pd, pdx);
    }
    long used = g_allocs - before;

    printf("  200 轮 Conv2d(2->4,k=3,pad=1,8x8) + MaxPool2d forward/backward：\n");
    printf("    分配次数 = %ld  %s\n", used,
           used == 0 ? "[通过：热路径零分配]" : "[失败：热路径里仍然 malloc]");

    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
    tensor_free(px); tensor_free(po); tensor_free(pd); tensor_free(pdx);
    conv2d_free(net);
    maxpool2d_free(pool);
    return used == 0;
}

/* 释放路径的泄漏/重复释放检查：统计"分配次数 - 释放次数"的净变化。
 * 覆盖 xxx_free() 与容器用的 module_free() 两条路径，以及换尺寸重绑定
 * （conv2d_bind_shape 会先 free 旧缓冲再重新分配） */
static int check_release(void)
{
    /* 形状：conv(2->3,k=3,s=1,p=1) 6x6 -> 6x6；重绑定到 5x5 -> 5x5
     *       pool(k=2,s=2) 6x6 -> 3x3；flatten 2x6x6 -> (72,)
     *       conv(1->2,k=2,s=1,p=0) 1x6x6 -> 2x5x5；pool(k=3,s=1) -> 1x4x4 */
    const int xs[3]    = { 2, 6, 6 };
    const int co[3]    = { 3, 6, 6 };
    const int po[3]    = { 2, 3, 3 };
    const int xs2[3]   = { 2, 5, 5 };
    const int co2[3]   = { 3, 5, 5 };
    const int fs[1]    = { 2 * 6 * 6 };
    const int qx_s[3]  = { 1, 6, 6 };
    const int qo_s[3]  = { 2, 5, 5 };
    const int qp_s[3]  = { 1, 4, 4 };
    const int qf_s[1]  = { 1 * 6 * 6 };

    long a0 = g_allocs, f0 = g_frees;

    /* --- 1) 单独使用：xxx_free() --- */
    Conv2d *net = conv2d_create(2, 3, 3, 1, 1);
    MaxPool2d *pool = maxpool2d_create(2, 0);      /* stride<=0 -> 等于 kernel */
    Flatten *flat = flatten_create();

    Tensor *x   = tensor_new(3, xs);
    Tensor *out = tensor_new(3, co);
    Tensor *dx  = tensor_new(3, xs);
    Tensor *x2  = tensor_new(3, xs2);
    Tensor *o2  = tensor_new(3, co2);
    Tensor *d2  = tensor_new(3, xs2);
    Tensor *pout = tensor_new(3, po);
    Tensor *pdx  = tensor_new(3, xs);
    Tensor *fx   = tensor_new(3, xs);
    Tensor *fo   = tensor_new(1, fs);
    Tensor *fdx  = tensor_new(3, xs);
    fill_gauss(x->data, (size_t)x->size);
    fill_gauss(x2->data, (size_t)x2->size);
    fill_gauss(fx->data, (size_t)fx->size);

    conv2d_zero_grad(net);
    conv2d_forward(net, x, out);            /* 懒绑定：在这里分配 im2col 等缓冲 */
    conv2d_backward(net, out, dx);
    conv2d_bind_shape(net, 2, 5, 5);        /* 换尺寸：释放旧缓冲 + 重新分配 */
    conv2d_forward(net, x2, o2);
    conv2d_backward(net, o2, d2);

    maxpool2d_forward(pool, x, pout);
    maxpool2d_backward(pool, pout, pdx);
    maxpool2d_zero_grad(pool);

    flatten_forward(flat, fx, fo);
    flatten_backward(flat, fo, fdx);
    flatten_zero_grad(flat);

    tensor_free(x); tensor_free(out); tensor_free(dx);
    tensor_free(x2); tensor_free(o2); tensor_free(d2);
    tensor_free(pout); tensor_free(pdx);
    tensor_free(fx); tensor_free(fo); tensor_free(fdx);
    conv2d_free(net);
    maxpool2d_free(pool);
    flatten_free(flat);

    /* --- 2) 交给容器：module_free()（释放层内部缓冲 + 层结构体本身） --- */
    Conv2d *net2 = conv2d_create(1, 2, 2, 1, 0);
    MaxPool2d *pool2 = maxpool2d_create(3, 1);
    Flatten *flat2 = flatten_create();
    Tensor *qx  = tensor_new(3, qx_s);
    Tensor *qo  = tensor_new(3, qo_s);
    Tensor *qd  = tensor_new(3, qx_s);
    Tensor *qpo = tensor_new(3, qp_s);
    Tensor *qfo = tensor_new(1, qf_s);
    fill_gauss(qx->data, (size_t)qx->size);

    conv2d_forward(net2, qx, qo);           /* 懒绑定 */
    conv2d_backward(net2, qo, qd);
    maxpool2d_forward(pool2, qx, qpo);
    maxpool2d_backward(pool2, qpo, qd);
    flatten_forward(flat2, qx, qfo);
    flatten_backward(flat2, qfo, qd);

    tensor_free(qx); tensor_free(qo); tensor_free(qd);
    tensor_free(qpo); tensor_free(qfo);
    module_free(conv2d_as_module(net2));
    module_free(maxpool2d_as_module(pool2));
    module_free(flatten_as_module(flat2));

    long dalloc = g_allocs - a0;
    long dfree  = g_frees  - f0;
    int ok = (dalloc == dfree);

    printf("  释放路径（xxx_free / module_free / 换尺寸重绑定）：\n");
    printf("    分配 %ld 次，释放 %ld 次  %s\n", dalloc, dfree,
           ok ? "[通过：无泄漏、无重复释放]" : "[失败：次数不相等]");
    return ok;
}
#endif

/* ============================== main ============================== */

int main(void)
{
    int fails = 0;

    printf("========== 1) Conv2d 有限差分校验（eps=1e-6, 阈值 1e-5） ==========\n");
    /* 与 AI.py 的 test_ai.py::TestConv2dGradients 同两套配置 */
    fails += !fd_check_conv(1, 2, 2, 1, 0, 5, 5, 3);          /* seed=3 对齐 Python */
    fails += !fd_check_conv(2, 2, 3, 2, 1, 6, 6, 7);          /* seed=7 对齐 Python */
    /* 附加配置：k=3 / stride=3 / pad=2 / C=2，覆盖更大的 padding 与跳步 */
    fails += !fd_check_conv(2, 3, 3, 3, 2, 7, 7, 11);

    printf("\n========== 2) MaxPool2d 校验 ==========\n");
    fails += !check_pool_mask(2, 2, 2, 6, 5, 21);
    fails += !check_pool_backward_props(2, 2, 2, 6, 5, 22);
    fails += !check_pool_tie();
    fails += !fd_check_pool(2, 2, 2, 6, 5, 23);               /* 窗口不重叠 */
    fails += !check_pool_backward_props(1, 3, 1, 5, 5, 24);   /* 窗口重叠 */
    fails += !fd_check_pool(1, 3, 1, 5, 5, 25);               /* 窗口重叠 + 差分 */

    printf("\n========== 3) Flatten 校验 ==========\n");
    fails += !check_flatten();

    printf("\n========== 4) Module vtable 校验 ==========\n");
    fails += !check_module_vtable();

#if defined(CONV_NOALLOC)
    printf("\n========== 5) 热路径零分配 / 释放路径校验 ==========\n");
    fails += !check_noalloc();
    fails += !check_release();
#else
    printf("\n（未定义 CONV_NOALLOC，跳过 --wrap 的零分配/释放统计）\n");
#endif

    printf("\n========== 汇总：%s（失败项 %d） ==========\n",
           fails == 0 ? "全部通过" : "存在失败", fails);
    return fails == 0 ? 0 : 1;
}
