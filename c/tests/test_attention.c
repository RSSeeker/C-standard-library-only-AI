/*
 * tests/test_attention.c —— Transformer 组件的自测程序
 *
 * 覆盖：
 *   1. MultiHeadAttention forward 形状 (seq_len, d_model) → (seq_len, d_model)
 *   2. MultiHeadAttention 的数值梯度（W_q/W_k/W_v/W_o/b_?/输入）中心差分校验
 *      —— 历史上的「整体转置」bug 就是靠这个查出来的
 *   3. FeedForward 数值梯度 + 形状
 *   4. Sequential：前向串联顺序、反向逆序、参数量/清零/递归释放
 *   5. ResidualBlock：dx = dout + 子层输入梯度（有限差分验证）
 *   6. TransformerEncoderLayer（dropout=0，退化恒等）：形状 + 各参数数值梯度抽查
 *   7. TransformerEncoder（2 层）：形状 + 数值梯度抽查 + sgd_step
 *   8. Sequential(PositionalEncoding + TransformerEncoderLayer) 集成数值梯度
 *   9. PositionalEncoding：与 Python 公式逐项对照（sin/cos 表）
 *  10. Embedding：查表、同索引重复累加、grad_weight 数值梯度
 *
 * 全部通过返回 0；任何一项超差返回 1。
 * 编译：gcc -std=c17 -O2 -Wall -Wextra -o tests/test_attention.exe \
 *           tests/test_attention.c attention.c containers.c tensor.c -lm
 */
#include "../attention.h"
#include "../containers.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------ 测试框架 ------------------------------ */

static int    g_fail = 0;
static double g_eps = 1e-6;    /* 中心差分步长（题目要求 1e-6） */
static double g_tol = 1e-5;    /* 误差容差（题目要求 1e-5） */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "测试: out of memory\n"); exit(2); }
    return p;
}

static void check(const char *what, double worst, double tol)
{
    int ok = (worst <= tol);
    if (!ok) g_fail++;
    printf("  %-30s 最大误差 %.3e  %s\n", what, worst, ok ? "[OK]" : "[超出容差]");
}

/* 确定性「伪随机」填充：不用 rand()，保证每次运行、每个平台结果一致 */
static double deter(int i, double scale)
{
    double v = sin((double)i * 12.9898 + 78.233) * 43758.5453;
    v = v - floor(v);                 /* [0,1) */
    return (v - 0.5) * 2.0 * scale;
}

static void fill_deter(double *p, int n, double scale)
{
    for (int i = 0; i < n; i++) p[i] = deter(i, scale);
}

/* --------------------- 损失与中心差分通用工具 ---------------------
 * L = Σ coef[t][j] * out[t][j]，则 dL/dout = coef，正好可以直接喂给 backward。
 */

typedef struct {
    int kind;                       /* 0 = 直接用 mha_forward，1 = 用 Module 接口 */
    MultiHeadAttention *mha;
    Module *mod;
    Tensor *x, *coef, *out;
} LossCtx;

static double loss_eval(void *ctx)
{
    LossCtx *c = (LossCtx *)ctx;
    if (c->kind == 0) mha_forward(c->mha, c->x, c->out);
    else              module_forward(c->mod, c->x, c->out);
    double s = 0.0;
    for (int i = 0; i < c->out->size; i++) s += c->coef->data[i] * c->out->data[i];
    return s;
}

/* 对参数数组 p（长度 n，stride 用于抽样）做中心差分，和解析梯度 analytic 逐项比较。
 * p 直接指向模块内部的参数数组，所以 loss_eval 会用到被扰动后的参数。 */
static double fd_check(const char *what, double *p, const double *analytic, int n, int stride,
                       double (*loss)(void *), void *ctx)
{
    double worst = 0.0;
    int at = -1;
    if (stride < 1) stride = 1;
    for (int i = 0; i < n; i += stride) {
        double saved = p[i];
        p[i] = saved + g_eps;
        double lp = loss(ctx);
        p[i] = saved - g_eps;
        double lm = loss(ctx);
        p[i] = saved;
        double num = (lp - lm) / (2.0 * g_eps);
        double err = fabs(num - analytic[i]);
        if (err > worst) { worst = err; at = i; }
    }
    (void)at;
    check(what, worst, g_tol);
    return worst;
}

/* --------------------- 测试用的假模块（供容器测试） --------------------- */

typedef struct {
    double w;               /* out = w * x */
    int log_id;             /* 调用日志用的编号 */
    double *grads;          /* 3 个假参数梯度，用来验证 zero_grad 递归 */
} FakeData;

#define LOG_MAX 64
static int g_log[LOG_MAX];
static int g_log_n = 0;
static void log_call(int id) { if (g_log_n < LOG_MAX) g_log[g_log_n++] = id; }

static void fake_forward(Module *m, const Tensor *x, Tensor *out)
{
    FakeData *d = m->self;
    log_call(d->log_id);
    tensor_check_shape(out, x->ndim, x->shape, "fake_forward(out)");
    for (int i = 0; i < x->size; i++) out->data[i] = d->w * x->data[i];
}

static void fake_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    FakeData *d = m->self;
    log_call(d->log_id);
    tensor_check_shape(dx, dout->ndim, dout->shape, "fake_backward(dx)");
    for (int i = 0; i < dout->size; i++) {
        dx->data[i] = d->w * dout->data[i];
        d->grads[i % 3] += dout->data[i];      /* 假的参数梯度累积 */
    }
}

static void fake_zero_grad(Module *m)
{
    FakeData *d = m->self;
    for (int i = 0; i < 3; i++) d->grads[i] = 0.0;
}

static void fake_free(Module *m)
{
    FakeData *d = m->self;
    if (!d) return;
    free(d->grads);
    free(d);
    m->self = NULL;
}

static int fake_param_count(Module *m) { (void)m; return 3; }

static void fake_to_string(Module *m, char *buf, size_t n)
{
    FakeData *d = m->self;
    snprintf(buf, n, "FakeScale(w=%g)", d->w);
}

static const ModuleVT FAKE_VT = {
    "FakeScale", fake_forward, fake_backward, fake_zero_grad,
    fake_free, fake_param_count, fake_to_string
};

static Module *fake_create(double w, int log_id)
{
    FakeData *d = xmalloc(sizeof(*d));
    memset(d, 0, sizeof(*d));
    d->w = w;
    d->log_id = log_id;
    d->grads = xmalloc(3 * sizeof(double));
    memset(d->grads, 0, 3 * sizeof(double));
    Module *m = xmalloc(sizeof(*m));
    m->vt = &FAKE_VT;
    m->self = d;
    return m;
}

static double *fake_grads(Module *m) { return ((FakeData *)m->self)->grads; }

/* ------------------------------ 1/2. MHA ------------------------------ */

static void test_mha(void)
{
    const int d_model = 8, heads = 2, T = 5, max_seq = 8;
    MultiHeadAttention *m = mha_create(d_model, heads, max_seq, 0.0);

    /* 灌入确定性权重 */
    const int dd = d_model * d_model;
    double *W[4], *B[4];
    for (int k = 0; k < 4; k++) {
        W[k] = xmalloc((size_t)dd * sizeof(double));
        B[k] = xmalloc((size_t)d_model * sizeof(double));
        fill_deter(W[k], dd, 0.6);
        fill_deter(B[k], d_model, 0.4);
    }
    mha_set_params(m, W[0], B[0], W[1], B[1], W[2], B[2], W[3], B[3]);
    for (int k = 0; k < 4; k++) { free(W[k]); free(B[k]); }

    int xshape[2] = { T, d_model };
    Tensor *x = tensor_new(2, xshape);
    Tensor *out = tensor_new(2, xshape);
    Tensor *coef = tensor_new(2, xshape);
    Tensor *dx = tensor_new(2, xshape);
    fill_deter(x->data, x->size, 1.0);
    fill_deter(coef->data, coef->size, 1.0);

    printf("\n=== 1. MultiHeadAttention forward 形状 ===\n");
    mha_forward(m, x, out);
    int ok = (out->ndim == 2 && out->shape[0] == T && out->shape[1] == d_model);
    printf("  (%d,%d) -> (%d,%d)  %s\n", T, d_model, out->shape[0], out->shape[1],
           ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;
    /* 参考值（跨语言比对时也用这几个数） */
    printf("  out[0][0..2] = %.15g %.15g %.15g\n", out->data[0], out->data[1], out->data[2]);

    printf("\n=== 2. MultiHeadAttention 数值梯度（中心差分 eps=%.0e，容差 %.0e）===\n",
           g_eps, g_tol);
    LossCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 0; ctx.mha = m; ctx.x = x; ctx.coef = coef; ctx.out = out;

    mha_zero_grad(m);
    mha_forward(m, x, out);
    mha_backward(m, coef, dx);

    fd_check("grad_W_q", m->W_q, m->grad_W_q, dd, 1, loss_eval, &ctx);
    fd_check("grad_W_k", m->W_k, m->grad_W_k, dd, 1, loss_eval, &ctx);
    fd_check("grad_W_v", m->W_v, m->grad_W_v, dd, 1, loss_eval, &ctx);
    fd_check("grad_W_o", m->W_o, m->grad_W_o, dd, 1, loss_eval, &ctx);
    fd_check("grad_b_q", m->b_q, m->grad_b_q, d_model, 1, loss_eval, &ctx);
    fd_check("grad_b_k", m->b_k, m->grad_b_k, d_model, 1, loss_eval, &ctx);
    fd_check("grad_b_v", m->b_v, m->grad_b_v, d_model, 1, loss_eval, &ctx);
    fd_check("grad_b_o", m->b_o, m->grad_b_o, d_model, 1, loss_eval, &ctx);
    fd_check("d_x（输入梯度）", x->data, dx->data, x->size, 1, loss_eval, &ctx);

    /* 反向的梯度是「累加」语义：重新 forward 后再跑一次反向，梯度应翻倍
     * （注意必须先 forward：backward 用的是最近一次 forward 的缓存，
     *   上面 FD 循环最后把 x 扰动过又改回来了，缓存里的 x 还是扰动值） */
    mha_forward(m, x, out);
    double before = m->grad_W_q[0];
    mha_backward(m, coef, dx);
    printf("  梯度累加语义: grad_W_q[0] %.15g -> %.15g %s\n", before, m->grad_W_q[0],
           (fabs(m->grad_W_q[0] - 2.0 * before) < 1e-12) ? "[OK]" : "[错误]");
    if (fabs(m->grad_W_q[0] - 2.0 * before) >= 1e-12) g_fail++;

    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
    mha_free(m);
}

/* ------------------------------ 3. FeedForward ------------------------------ */

static void test_ff(void)
{
    const int d_model = 8, d_ff = 16, T = 4, max_seq = 8;
    FeedForward *f = ff_create(d_model, d_ff, max_seq);
    double *W1 = xmalloc((size_t)d_ff * d_model * sizeof(double));
    double *b1 = xmalloc((size_t)d_ff * sizeof(double));
    double *W2 = xmalloc((size_t)d_model * d_ff * sizeof(double));
    double *b2 = xmalloc((size_t)d_model * sizeof(double));
    fill_deter(W1, d_ff * d_model, 0.6);
    fill_deter(b1, d_ff, 0.5);
    fill_deter(W2, d_model * d_ff, 0.6);
    fill_deter(b2, d_model, 0.5);
    ff_set_params(f, W1, b1, W2, b2);

    int xs[2] = { T, d_model };
    Tensor *x = tensor_new(2, xs), *out = tensor_new(2, xs);
    Tensor *coef = tensor_new(2, xs), *dx = tensor_new(2, xs);
    fill_deter(x->data, x->size, 1.0);
    fill_deter(coef->data, coef->size, 1.0);

    printf("\n=== 3. FeedForward 形状与数值梯度 ===\n");
    ff_forward(f, x, out);
    int ok = (out->shape[0] == T && out->shape[1] == d_model);
    printf("  (%d,%d) -> (%d,%d)  %s\n", T, d_model, out->shape[0], out->shape[1],
           ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;

    LossCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1;

    /* 模块包装会接管所有权，这里单独包一个用于 FD 的实例 */
    FeedForward *f2 = ff_create(d_model, d_ff, max_seq);
    ff_set_params(f2, W1, b1, W2, b2);
    ctx.mod = ff_module(f2);
    ctx.x = x; ctx.coef = coef; ctx.out = out;

    ff_zero_grad(f2);
    module_forward(ctx.mod, x, out);
    module_backward(ctx.mod, coef, dx);
    fd_check("grad_W1", f2->W1, f2->grad_W1, d_ff * d_model, 1, loss_eval, &ctx);
    fd_check("grad_b1", f2->b1, f2->grad_b1, d_ff, 1, loss_eval, &ctx);
    fd_check("grad_W2", f2->W2, f2->grad_W2, d_model * d_ff, 1, loss_eval, &ctx);
    fd_check("grad_b2", f2->b2, f2->grad_b2, d_model, 1, loss_eval, &ctx);
    fd_check("d_x（输入梯度）", x->data, dx->data, x->size, 1, loss_eval, &ctx);
    module_free(ctx.mod);                      /* 连带 f2 一起释放 */

    free(W1); free(b1); free(W2); free(b2);
    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
    ff_free(f);
}

/* --------------------- 4/5. Sequential / ResidualBlock --------------------- */

static void test_containers(void)
{
    const int n = 6;
    int xs[1] = { n };
    Tensor *x = tensor_new(1, xs), *out = tensor_new(1, xs);
    Tensor *coef = tensor_new(1, xs), *dx = tensor_new(1, xs);
    fill_deter(x->data, n, 1.0);
    fill_deter(coef->data, n, 1.0);

    printf("\n=== 4. Sequential：前向串联 / 反向逆序 ===\n");
    Module *mods[2];
    mods[0] = fake_create(2.0, 1);
    mods[1] = fake_create(3.0, 2);
    Module *seq = sequential_create(mods, 2);   /* 接管所有权 */

    g_log_n = 0;
    module_forward(seq, x, out);
    module_backward(seq, coef, dx);
    /* 期望：前向 1 → 2，反向 2 → 1 */
    int order_ok = (g_log_n == 4 && g_log[0] == 1 && g_log[1] == 2 &&
                    g_log[2] == 2 && g_log[3] == 1);
    printf("  调用顺序 = [%d %d %d %d]，期望 [1 2 2 1]  %s\n",
           g_log[0], g_log[1], g_log[2], g_log[3], order_ok ? "[OK]" : "[错误]");
    if (!order_ok) g_fail++;

    double fw_worst = 0.0;
    for (int i = 0; i < n; i++) fw_worst = fmax(fw_worst, fabs(out->data[i] - 6.0 * x->data[i]));
    check("forward = 3*(2*x) = 6x", fw_worst, 1e-12);

    double bw_worst = 0.0;
    for (int i = 0; i < n; i++) bw_worst = fmax(bw_worst, fabs(dx->data[i] - 6.0 * coef->data[i]));
    check("backward = 2*(3*dout) = 6*dout", bw_worst, 1e-12);

    LossCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = seq; ctx.x = x; ctx.coef = coef; ctx.out = out;
    fd_check("Sequential 链式输入梯度", x->data, dx->data, n, 1, loss_eval, &ctx);

    printf("\n=== 5. ResidualBlock：dx = dout + 子层梯度 ===\n");
    Module *sub = fake_create(2.0, 3);
    Module *res = residual_create(sub);
    g_log_n = 0;
    module_forward(res, x, out);
    module_backward(res, coef, dx);
    double res_worst = 0.0;
    for (int i = 0; i < n; i++)
        res_worst = fmax(res_worst, fabs(out->data[i] - 3.0 * x->data[i]));  /* (1+2)x */
    check("forward = x + 2x = 3x", res_worst, 1e-12);
    double resb_worst = 0.0;
    for (int i = 0; i < n; i++) {
        double want = coef->data[i] + 2.0 * coef->data[i];   /* dout + sublayer(dout) */
        resb_worst = fmax(resb_worst, fabs(dx->data[i] - want));
    }
    check("backward = dout + 2*dout", resb_worst, 1e-12);

    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = res; ctx.x = x; ctx.coef = coef; ctx.out = out;
    fd_check("ResidualBlock 输入梯度", x->data, dx->data, n, 1, loss_eval, &ctx);

    printf("\n=== 6. 容器：param_count / zero_grad / 递归 free ===\n");
    int pc_seq = module_param_count(seq);
    int pc_res = module_param_count(res);
    printf("  param_count(Sequential(2 个假模块)) = %d，期望 6  %s\n", pc_seq,
           pc_seq == 6 ? "[OK]" : "[错误]");
    if (pc_seq != 6) g_fail++;
    printf("  param_count(ResidualBlock(1 个假模块)) = %d，期望 3  %s\n", pc_res,
           pc_res == 3 ? "[OK]" : "[错误]");
    if (pc_res != 3) g_fail++;

    /* 累一些假的参数梯度，再递归清零 */
    module_backward(seq, coef, dx);
    double g0 = fake_grads(mods[0])[0], g1 = fake_grads(mods[1])[0];
    module_zero_grad(seq);
    int zg_ok = (g0 != 0.0 && g1 != 0.0 &&
                 fake_grads(mods[0])[0] == 0.0 && fake_grads(mods[1])[0] == 0.0);
    printf("  递归 zero_grad：累加值 %.3g/%.3g 清零后 %.3g/%.3g  %s\n",
           g0, g1, fake_grads(mods[0])[0], fake_grads(mods[1])[0], zg_ok ? "[OK]" : "[错误]");
    if (!zg_ok) g_fail++;

    module_free(seq);   /* 递归释放两个假模块 */
    module_free(res);   /* 递归释放子层 */
    printf("  递归 free：已释放（无泄漏检查工具时以不崩溃为准）  [OK]\n");

    printf("\n=== 6b. 空 Sequence（恒等）与改变形状的链 ===\n");
    Module *empty = sequential_create(NULL, 0);
    module_forward(empty, x, out);
    module_backward(empty, coef, dx);
    double idw = 0.0;
    for (int i = 0; i < n; i++) {
        idw = fmax(idw, fabs(out->data[i] - x->data[i]));
        idw = fmax(idw, fabs(dx->data[i] - coef->data[i]));
    }
    check("空 Sequential = 恒等", idw, 0.0);
    module_free(empty);

    /* Sequential(Embedding → FakeScale)：(T) → (T,E)：中间形状必须显式声明 */
    const int vocab2 = 5, E2 = 3, T2 = 4;
    Embedding *e3 = emb_create(vocab2, E2, 8);
    double *W2 = xmalloc((size_t)vocab2 * E2 * sizeof(double));
    fill_deter(W2, vocab2 * E2, 0.7);
    emb_set_weight(e3, W2);
    free(W2);
    Module *chain2[2];
    chain2[0] = emb_module(e3);
    chain2[1] = fake_create(1.5, 9);
    int mid[2] = { T2, E2 };
    Module *seq2 = sequential_create_shaped(chain2, 2, 2, mid);
    int is2[1] = { T2 };
    int os2[2] = { T2, E2 };
    Tensor *it = tensor_new(1, is2), *o2 = tensor_new(2, os2);
    Tensor *c2 = tensor_new(2, os2), *d2 = tensor_new(1, is2);
    double idx2[4] = { 2.0, 0.0, 2.0, 4.0 };
    for (int t = 0; t < T2; t++) it->data[t] = idx2[t];
    fill_deter(c2->data, c2->size, 1.0);
    module_forward(seq2, it, o2);
    double sh = 0.0;
    for (int t = 0; t < T2; t++)
        for (int j = 0; j < E2; j++)
            sh = fmax(sh, fabs(o2->data[(size_t)t * E2 + j] -
                               1.5 * e3->weight[(size_t)(int)idx2[t] * E2 + j]));
    check("Embedding→FakeScale 前向", sh, 1e-15);

    module_zero_grad(seq2);
    module_backward(seq2, c2, d2);
    double shg = 0.0;
    for (int v = 0; v < vocab2; v++)
        for (int j = 0; j < E2; j++) {
            double want = 0.0;
            for (int t = 0; t < T2; t++)
                if ((int)idx2[t] == v) want += 1.5 * c2->data[(size_t)t * E2 + j];
            shg = fmax(shg, fabs(e3->grad_weight[(size_t)v * E2 + j] - want));
        }
    check("链内 Embedding grad_weight", shg, 1e-15);
    double d2w = 0.0;
    for (int t = 0; t < T2; t++) d2w = fmax(d2w, fabs(d2->data[t]));
    check("链末端 dx（索引梯度=0）", d2w, 0.0);
    module_free(seq2);
    tensor_free(it); tensor_free(o2); tensor_free(c2); tensor_free(d2);

    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
}

/* --------------------- 7/8. TransformerEncoderLayer / Encoder --------------------- */

/* 给一层灌入确定性参数（避免用随机初值做数值梯度检查） */
static void fill_layer_params(TransformerEncoderLayer *cl, int d_model, int d_ff, int salt)
{
    const int dd = d_model * d_model;
    fill_deter(cl->self_attn->W_q, dd, 0.4 + 0.01 * salt);
    fill_deter(cl->self_attn->W_k, dd, 0.4 + 0.01 * salt);
    fill_deter(cl->self_attn->W_v, dd, 0.4 + 0.01 * salt);
    fill_deter(cl->self_attn->W_o, dd, 0.4 + 0.01 * salt);
    fill_deter(cl->self_attn->b_q, d_model, 0.2);
    fill_deter(cl->self_attn->b_k, d_model, 0.2);
    fill_deter(cl->self_attn->b_v, d_model, 0.2);
    fill_deter(cl->self_attn->b_o, d_model, 0.2);
    fill_deter(cl->ffn->W1, d_ff * d_model, 0.4);
    fill_deter(cl->ffn->b1, d_ff, 0.3);
    fill_deter(cl->ffn->W2, d_model * d_ff, 0.4);
    fill_deter(cl->ffn->b2, d_model, 0.3);
    for (int j = 0; j < d_model; j++) {
        cl->ln1.gamma[j] = 1.0 + deter(j + salt, 0.2);
        cl->ln2.gamma[j] = 1.0 + deter(j + 100 * salt, 0.2);
        cl->ln1.beta[j] = deter(j + 10 * salt, 0.2);
        cl->ln2.beta[j] = deter(j + 20 * salt, 0.2);
    }
}

/* 把一层的全部参数复制到另一层（用来比较不同 dropout 设置下的行为） */
static void copy_layer_params(TransformerEncoderLayer *dst, const TransformerEncoderLayer *src)
{
    const int d = src->d_model, dff = src->ffn->d_ff;
    const size_t dd = (size_t)d * d;
    memcpy(dst->self_attn->W_q, src->self_attn->W_q, dd * sizeof(double));
    memcpy(dst->self_attn->W_k, src->self_attn->W_k, dd * sizeof(double));
    memcpy(dst->self_attn->W_v, src->self_attn->W_v, dd * sizeof(double));
    memcpy(dst->self_attn->W_o, src->self_attn->W_o, dd * sizeof(double));
    memcpy(dst->self_attn->b_q, src->self_attn->b_q, (size_t)d * sizeof(double));
    memcpy(dst->self_attn->b_k, src->self_attn->b_k, (size_t)d * sizeof(double));
    memcpy(dst->self_attn->b_v, src->self_attn->b_v, (size_t)d * sizeof(double));
    memcpy(dst->self_attn->b_o, src->self_attn->b_o, (size_t)d * sizeof(double));
    memcpy(dst->ffn->W1, src->ffn->W1, (size_t)dff * d * sizeof(double));
    memcpy(dst->ffn->b1, src->ffn->b1, (size_t)dff * sizeof(double));
    memcpy(dst->ffn->W2, src->ffn->W2, (size_t)d * dff * sizeof(double));
    memcpy(dst->ffn->b2, src->ffn->b2, (size_t)d * sizeof(double));
    memcpy(dst->ln1.gamma, src->ln1.gamma, (size_t)d * sizeof(double));
    memcpy(dst->ln1.beta, src->ln1.beta, (size_t)d * sizeof(double));
    memcpy(dst->ln2.gamma, src->ln2.gamma, (size_t)d * sizeof(double));
    memcpy(dst->ln2.beta, src->ln2.beta, (size_t)d * sizeof(double));
}

static void test_transformer(void)
{
    const int d_model = 8, heads = 2, d_ff = 16, T = 4, max_seq = 6;
    const int dd = d_model * d_model;
    TransformerEncoderLayer *l = tel_create(d_model, heads, d_ff, max_seq, 0.0, 12345);

    /* 灌确定性参数 */
    double *buf;
    buf = xmalloc((size_t)dd * sizeof(double));
    fill_deter(buf, dd, 0.5);
    memcpy(l->self_attn->W_q, buf, (size_t)dd * sizeof(double));
    fill_deter(buf, dd, 0.5);
    memcpy(l->self_attn->W_k, buf, (size_t)dd * sizeof(double));
    fill_deter(buf, dd, 0.5);
    memcpy(l->self_attn->W_v, buf, (size_t)dd * sizeof(double));
    fill_deter(buf, dd, 0.5);
    memcpy(l->self_attn->W_o, buf, (size_t)dd * sizeof(double));
    fill_deter(l->self_attn->b_q, d_model, 0.3);
    fill_deter(l->self_attn->b_k, d_model, 0.3);
    fill_deter(l->self_attn->b_v, d_model, 0.3);
    fill_deter(l->self_attn->b_o, d_model, 0.3);
    fill_deter(l->ffn->W1, d_ff * d_model, 0.5);
    fill_deter(l->ffn->b1, d_ff, 0.4);
    fill_deter(l->ffn->W2, d_model * d_ff, 0.5);
    fill_deter(l->ffn->b2, d_model, 0.4);
    fill_deter(l->ln1.gamma, d_model, 0.3);
    fill_deter(l->ln1.beta, d_model, 0.3);
    fill_deter(l->ln2.gamma, d_model, 0.3);
    fill_deter(l->ln2.beta, d_model, 0.3);
    for (int j = 0; j < d_model; j++) {   /* gamma 别离 1 太远，避免数值病态 */
        l->ln1.gamma[j] += 1.0;
        l->ln2.gamma[j] += 1.0;
    }
    free(buf);

    int xs[2] = { T, d_model };
    Tensor *x = tensor_new(2, xs), *out = tensor_new(2, xs);
    Tensor *coef = tensor_new(2, xs), *dx = tensor_new(2, xs);
    fill_deter(x->data, x->size, 1.0);
    fill_deter(coef->data, coef->size, 1.0);

    printf("\n=== 7. TransformerEncoderLayer（dropout=0）形状与数值梯度抽查 ===\n");
    tel_forward(l, x, out);
    int ok = (out->shape[0] == T && out->shape[1] == d_model);
    printf("  (%d,%d) -> (%d,%d)  %s\n", T, d_model, out->shape[0], out->shape[1],
           ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;
    printf("  out[0][0..2] = %.15g %.15g %.15g\n", out->data[0], out->data[1], out->data[2]);

    Module *lmod = tel_module(l);   /* 接管 l 的所有权 */
    tel_zero_grad(l);
    module_forward(lmod, x, out);
    module_backward(lmod, coef, dx);

    LossCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = lmod; ctx.x = x; ctx.coef = coef; ctx.out = out;

    /* 参数多，抽样步长放大；每项都覆盖到若干元素 */
    fd_check("attn.grad_W_q（抽样）", l->self_attn->W_q, l->self_attn->grad_W_q, dd, 7, loss_eval, &ctx);
    fd_check("attn.grad_W_k（抽样）", l->self_attn->W_k, l->self_attn->grad_W_k, dd, 7, loss_eval, &ctx);
    fd_check("attn.grad_W_v（抽样）", l->self_attn->W_v, l->self_attn->grad_W_v, dd, 7, loss_eval, &ctx);
    fd_check("attn.grad_W_o（抽样）", l->self_attn->W_o, l->self_attn->grad_W_o, dd, 7, loss_eval, &ctx);
    fd_check("attn.grad_b_q", l->self_attn->b_q, l->self_attn->grad_b_q, d_model, 1, loss_eval, &ctx);
    fd_check("attn.grad_b_o", l->self_attn->b_o, l->self_attn->grad_b_o, d_model, 1, loss_eval, &ctx);
    fd_check("ffn.grad_W1（抽样）", l->ffn->W1, l->ffn->grad_W1, d_ff * d_model, 13, loss_eval, &ctx);
    fd_check("ffn.grad_b1（抽样）", l->ffn->b1, l->ffn->grad_b1, d_ff, 3, loss_eval, &ctx);
    fd_check("ffn.grad_W2（抽样）", l->ffn->W2, l->ffn->grad_W2, d_model * d_ff, 13, loss_eval, &ctx);
    fd_check("ffn.grad_b2（抽样）", l->ffn->b2, l->ffn->grad_b2, d_model, 3, loss_eval, &ctx);
    fd_check("ln1.grad_gamma", l->ln1.gamma, l->ln1.ggamma, d_model, 1, loss_eval, &ctx);
    fd_check("ln1.grad_beta", l->ln1.beta, l->ln1.gbeta, d_model, 1, loss_eval, &ctx);
    fd_check("ln2.grad_gamma", l->ln2.gamma, l->ln2.ggamma, d_model, 1, loss_eval, &ctx);
    fd_check("ln2.grad_beta", l->ln2.beta, l->ln2.gbeta, d_model, 1, loss_eval, &ctx);
    fd_check("d_x（输入梯度）", x->data, dx->data, x->size, 1, loss_eval, &ctx);

    printf("\n=== 7b. Dropout 路径（train=0 恒等 / train=1 mask 语义）===\n");
    TransformerEncoderLayer *ld = tel_create(d_model, heads, d_ff, max_seq, 0.5, 2025);
    copy_layer_params(ld, l);
    Tensor *out2 = tensor_new(2, xs), *dx2 = tensor_new(2, xs);
    tel_set_train(ld, 0);
    tel_forward(ld, x, out2);
    tel_forward(l, x, out);        /* 重新算一次干净的 p=0 前向（FD 循环最后扰动过参数） */
    double dw = 0.0;
    for (int i = 0; i < out->size; i++) dw = fmax(dw, fabs(out2->data[i] - out->data[i]));
    check("train=0 前向与 dropout=0 逐位一致", dw, 0.0);
    tel_zero_grad(ld);
    tel_backward(ld, coef, dx2);
    double dbw = 0.0;
    for (int i = 0; i < dx->size; i++) dbw = fmax(dbw, fabs(dx2->data[i] - dx->data[i]));
    check("train=0 反向与 dropout=0 逐位一致", dbw, 0.0);

    tel_set_train(ld, 1);
    tel_forward(ld, x, out2);
    int mask_ok = 1, n_zero = 0, n_tot = T * d_model;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d_model; j++) {
            double mk = ld->dropout1.mask[(size_t)t * d_model + j];
            if (!(mk == 0.0 || fabs(mk - 2.0) < 1e-15)) mask_ok = 0;   /* p=0.5 → 1/(1-p)=2 */
            if (mk == 0.0) n_zero++;
        }
    printf("  mask 取值合法（0 或 1/(1-p)=2）：%s，置零比例 %d/%d\n",
           mask_ok ? "[OK]" : "[错误]", n_zero, n_tot);
    if (!mask_ok) g_fail++;
    if (n_zero == 0 || n_zero == n_tot) g_fail++;      /* 全 0 或全 1 都说明 RNG 有问题 */
    Tensor *out3 = tensor_new(2, xs);
    tel_forward(ld, x, out3);
    double diff = 0.0;
    for (int i = 0; i < out2->size; i++) diff = fmax(diff, fabs(out3->data[i] - out2->data[i]));
    printf("  两次训练态前向的差异 = %.3e  %s\n", diff,
           diff > 0.0 ? "[OK]（mask 是随机的）" : "[错误]（mask 没变）");
    if (!(diff > 0.0)) g_fail++;
    tel_free(ld);
    tensor_free(out2); tensor_free(out3); tensor_free(dx2);

    printf("\n=== 8. TransformerEncoder（2 层）形状 / 数值梯度 / sgd_step ===\n");
    TransformerEncoder *e = te_create(d_model, heads, d_ff, 2, max_seq, 0.0, 777);
    for (int li = 0; li < 2; li++) fill_layer_params(e->layers[li], d_model, d_ff, li);
    Module *emod = te_module(e);
    te_forward(e, x, out);
    ok = (out->shape[0] == T && out->shape[1] == d_model);
    printf("  (%d,%d) -> (%d,%d)  %s\n", T, d_model, out->shape[0], out->shape[1],
           ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;
    printf("  out[0][0..2] = %.15g %.15g %.15g\n", out->data[0], out->data[1], out->data[2]);

    te_zero_grad(e);
    te_backward(e, coef, dx);
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = emod; ctx.x = x; ctx.coef = coef; ctx.out = out;
    fd_check("layer0.attn.grad_W_q", e->layers[0]->self_attn->W_q,
             e->layers[0]->self_attn->grad_W_q, dd, 11, loss_eval, &ctx);
    fd_check("layer1.attn.grad_W_v", e->layers[1]->self_attn->W_v,
             e->layers[1]->self_attn->grad_W_v, dd, 11, loss_eval, &ctx);
    fd_check("layer1.ffn.grad_W2", e->layers[1]->ffn->W2,
             e->layers[1]->ffn->grad_W2, d_model * d_ff, 17, loss_eval, &ctx);
    fd_check("layer0.ln1.grad_gamma", e->layers[0]->ln1.gamma,
             e->layers[0]->ln1.ggamma, d_model, 1, loss_eval, &ctx);
    fd_check("输入梯度 d_x", x->data, dx->data, x->size, 1, loss_eval, &ctx);

    /* sgd_step：param -= lr*grad（抽查第一个元素） */
    const double lr = 1e-3;
    double w_before = e->layers[0]->self_attn->W_q[0];
    double g_used = e->layers[0]->self_attn->grad_W_q[0];
    te_sgd_step(e, lr);
    double w_after = e->layers[0]->self_attn->W_q[0];
    double want = w_before - lr * g_used;
    printf("  sgd_step: W_q[0] %.17g -> %.17g（期望 %.17g）  %s\n", w_before, w_after, want,
           fabs(w_after - want) < 1e-18 ? "[OK]" : "[错误]");
    if (fabs(w_after - want) >= 1e-18) g_fail++;

    /* 3 层 + T < max_seq：专门覆盖乒乓缓冲里「按 T 取视图」的那条路径
     * （曾经写成把整块 (max_seq_len,d) 缓冲当下一层的输入，2 层时不会暴露） */
    printf("\n=== 8b. TransformerEncoder（3 层，T < max_seq_len）===\n");
    TransformerEncoder *e3 = te_create(d_model, heads, d_ff, 3, max_seq + 2, 0.0, 31);
    for (int li = 0; li < 3; li++) fill_layer_params(e3->layers[li], d_model, d_ff, li + 3);
    Module *e3mod = te_module(e3);
    te_forward(e3, x, out);
    ok = (out->shape[0] == T && out->shape[1] == d_model);
    printf("  (%d,%d) -> (%d,%d)（max_seq=%d）  %s\n", T, d_model, out->shape[0], out->shape[1],
           max_seq + 2, ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;
    te_zero_grad(e3);
    te_backward(e3, coef, dx);
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = e3mod; ctx.x = x; ctx.coef = coef; ctx.out = out;
    fd_check("layer1（中间层）grad_W_q", e3->layers[1]->self_attn->W_q,
             e3->layers[1]->self_attn->grad_W_q, dd, 5, loss_eval, &ctx);
    fd_check("layer2（顶层）ffn.grad_W1", e3->layers[2]->ffn->W1,
             e3->layers[2]->ffn->grad_W1, d_ff * d_model, 19, loss_eval, &ctx);
    fd_check("layer2.ln2.grad_beta", e3->layers[2]->ln2.beta,
             e3->layers[2]->ln2.gbeta, d_model, 1, loss_eval, &ctx);
    fd_check("3 层输入梯度 d_x", x->data, dx->data, x->size, 1, loss_eval, &ctx);
    module_free(e3mod);

    printf("\n=== 9. Sequential(PositionalEncoding → TransformerEncoderLayer) 集成 ===\n");
    PositionalEncoding *pe = pe_create(d_model, max_seq);
    TransformerEncoderLayer *l2 = tel_create(d_model, heads, d_ff, max_seq, 0.0, 4242);
    for (int j = 0; j < d_model; j++) {
        l2->ln1.gamma[j] = 1.0; l2->ln2.gamma[j] = 1.0;
        l2->ln1.beta[j] = 0.0;  l2->ln2.beta[j] = 0.0;
    }
    fill_deter(l2->self_attn->W_q, dd, 0.4);
    fill_deter(l2->self_attn->W_k, dd, 0.4);
    fill_deter(l2->self_attn->W_v, dd, 0.4);
    fill_deter(l2->self_attn->W_o, dd, 0.4);
    fill_deter(l2->self_attn->b_q, d_model, 0.2);
    fill_deter(l2->self_attn->b_k, d_model, 0.2);
    fill_deter(l2->self_attn->b_v, d_model, 0.2);
    fill_deter(l2->self_attn->b_o, d_model, 0.2);
    fill_deter(l2->ffn->W1, d_ff * d_model, 0.4);
    fill_deter(l2->ffn->b1, d_ff, 0.3);
    fill_deter(l2->ffn->W2, d_model * d_ff, 0.4);
    fill_deter(l2->ffn->b2, d_model, 0.3);

    Module *chain[2];
    chain[0] = pe_module(pe);
    chain[1] = tel_module(l2);
    Module *seq = sequential_create(chain, 2);   /* (T,d) → (T,d)，形状不变，无需声明 */
    module_forward(seq, x, out);
    ok = (out->shape[0] == T && out->shape[1] == d_model);
    printf("  Sequential(PE, TEL) (%d,%d) -> (%d,%d)  %s\n", T, d_model, out->shape[0],
           out->shape[1], ok ? "[OK]" : "[形状错误]");
    if (!ok) g_fail++;
    module_zero_grad(seq);
    module_backward(seq, coef, dx);
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = seq; ctx.x = x; ctx.coef = coef; ctx.out = out;
    fd_check("链内 TEL.attn.grad_W_o", l2->self_attn->W_o, l2->self_attn->grad_W_o, dd, 9,
             loss_eval, &ctx);
    fd_check("链内 TEL.ffn.grad_b1", l2->ffn->b1, l2->ffn->grad_b1, d_ff, 5, loss_eval, &ctx);
    fd_check("链的输入梯度（含 PE 直通）", x->data, dx->data, x->size, 1, loss_eval, &ctx);

    module_free(seq);          /* 递归释放 PE 与 TEL */
    module_free(lmod);         /* 释放第 7 节的层 */
    module_free(emod);         /* 递归释放 2 层 Encoder */
    tensor_free(x); tensor_free(out); tensor_free(coef); tensor_free(dx);
}

/* ------------------------------ 10. PositionalEncoding ------------------------------ */

static void test_pe(void)
{
    const int d_model = 6, max_len = 5;
    PositionalEncoding *p = pe_create(d_model, max_len);

    printf("\n=== 10. PositionalEncoding（与 Python 公式对照）===\n");
    /* 直接按 Python 公式重算，逐项比较 */
    double worst = 0.0;
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            double div = exp((double)i * -log(10000.0) / (double)d_model);
            double e_sin = sin((double)pos * div);
            double e_cos = (i + 1 < d_model) ? cos((double)pos * div) : 0.0;
            worst = fmax(worst, fabs(p->pe[(size_t)pos * d_model + i] - e_sin));
            if (i + 1 < d_model)
                worst = fmax(worst, fabs(p->pe[(size_t)pos * d_model + i + 1] - e_cos));
        }
    }
    check("sin/cos 表", worst, 1e-15);

    int xs[2] = { 3, d_model };
    Tensor *x = tensor_new(2, xs), *out = tensor_new(2, xs), *dx = tensor_new(2, xs);
    fill_deter(x->data, x->size, 1.0);
    pe_forward(p, x, out);
    double fw = 0.0;
    for (int t = 0; t < 3; t++)
        for (int j = 0; j < d_model; j++)
            fw = fmax(fw, fabs(out->data[(size_t)t * d_model + j] -
                               (x->data[(size_t)t * d_model + j] + p->pe[(size_t)t * d_model + j])));
    check("forward = x + PE", fw, 1e-15);

    fill_deter(out->data, out->size, 1.0);
    pe_backward(out, dx);
    double bw = 0.0;
    for (int i = 0; i < dx->size; i++) bw = fmax(bw, fabs(dx->data[i] - out->data[i]));
    check("backward 直通", bw, 0.0);

    pe_free(p);
    tensor_free(x); tensor_free(out); tensor_free(dx);
}

/* ------------------------------ 11. Embedding ------------------------------ */

static void test_embedding(void)
{
    const int vocab = 7, E = 4, T = 5;
    Embedding *e = emb_create(vocab, E, 8);
    double *W = xmalloc((size_t)vocab * E * sizeof(double));
    fill_deter(W, vocab * E, 0.8);
    emb_set_weight(e, W);

    int idx[5] = { 3, 1, 3, 0, 6 };     /* 故意让 3 出现两次 */
    int os[2] = { T, E };
    Tensor *out = tensor_new(2, os);
    Tensor *coef = tensor_new(2, os);
    Tensor *dout = tensor_new(2, os);
    fill_deter(coef->data, coef->size, 1.0);

    printf("\n=== 11. Embedding：查表 / 重复索引累加 / 数值梯度 ===\n");
    emb_forward(e, idx, T, out);
    double fw = 0.0;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < E; j++)
            fw = fmax(fw, fabs(out->data[(size_t)t * E + j] - W[(size_t)idx[t] * E + j]));
    check("forward 查表", fw, 0.0);

    emb_zero_grad(e);
    emb_backward(e, coef, NULL);
    /* 手工期望：grad_weight[idx][j] = Σ_{t: idx[t]==idx} coef[t][j] */
    double gw = 0.0;
    for (int v = 0; v < vocab; v++) {
        for (int j = 0; j < E; j++) {
            double want = 0.0;
            for (int t = 0; t < T; t++) if (idx[t] == v) want += coef->data[(size_t)t * E + j];
            gw = fmax(gw, fabs(e->grad_weight[(size_t)v * E + j] - want));
        }
    }
    check("grad_weight（含重复索引）", gw, 1e-15);

    /* 通过 Module 接口做数值梯度（输入是索引张量） */
    int is[1] = { T };
    Tensor *it = tensor_new(1, is);
    Tensor *dix = tensor_new(1, is);      /* 索引的「梯度」只能是 0，这里验证契约不崩 */
    for (int t = 0; t < T; t++) it->data[t] = (double)idx[t];
    Embedding *e2 = emb_create(vocab, E, 8);
    emb_set_weight(e2, W);
    Module *em = emb_module(e2);
    LossCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.kind = 1; ctx.mod = em; ctx.x = it; ctx.coef = coef; ctx.out = out;
    module_zero_grad(em);
    module_forward(em, it, out);
    module_backward(em, coef, dix);    /* dx 必须是 (T) 的一维张量 */
    double dix_worst = 0.0;
    for (int t = 0; t < T; t++) dix_worst = fmax(dix_worst, fabs(dix->data[t]));
    check("backward 写回的索引梯度为 0", dix_worst, 0.0);
    fd_check("grad_weight（数值梯度）", e2->weight, e2->grad_weight, vocab * E, 1, loss_eval, &ctx);
    module_free(em);                   /* 连带 e2 */

    emb_free(e);
    free(W);
    tensor_free(out); tensor_free(coef); tensor_free(dout); tensor_free(it); tensor_free(dix);
}

/* ------------------------------ main ------------------------------ */

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* 关掉缓冲：万一崩了也能看到崩在哪一步 */
    printf("Transformer 组件自测（C17 移植自 AI.py）\n");
    printf("中心差分步长 eps=%.0e，误差容差=%.0e\n", g_eps, g_tol);

    test_mha();
    test_ff();
    test_containers();
    test_transformer();
    test_pe();
    test_embedding();

    printf("\n========================================\n");
    if (g_fail == 0) {
        printf("全部通过（%d 项超差）\n", g_fail);
        return 0;
    }
    printf("存在 %d 项超差\n", g_fail);
    return 1;
}
