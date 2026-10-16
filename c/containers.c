/*
 * containers.c —— Sequential / ResidualBlock 的实现
 *
 * 契约见 containers.h 顶部注释。这里只强调两件事：
 *   1. forward / backward 稳态零 malloc：中间缓冲在 create 或首次 forward 时分配好，
 *      之后按当前形状取「视图」（Tensor 的 owns_data=0），形状变大才重新分配。
 *   2. Sequential 在 backward 里复用 forward 的中间缓冲，所以子模块必须自己缓存
 *      反向需要的量（本项目里所有模块都是把输入拷一份，从不只存外部指针）。
 */
#include "containers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "containers: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "containers: out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

/* 把 (*slot) 这块缓冲按 (ndim, shape) 取视图；容量不足时先扩容。
 * 行主序下「取前 size 个元素」就是想要的那块连续内存，所以变小不用重新分配。 */
static Tensor buf_view(Tensor **slot, int ndim, const int *shape)
{
    const int size = tensor_size_of(ndim, shape);
    if (*slot == NULL) {
        *slot = tensor_new(ndim, shape);
    } else if ((*slot)->size < size) {
        tensor_free(*slot);
        *slot = tensor_new(ndim, shape);
    }
    Tensor t;
    t.ndim = ndim;
    for (int i = 0; i < TENSOR_MAX_DIM; i++) t.shape[i] = (i < ndim) ? shape[i] : 1;
    t.size = size;
    t.data = (*slot)->data;
    t.owns_data = 0;
    return t;
}

/* ==========================================================================
 *  Sequential
 * ========================================================================== */

/* 先声明函数，好在创建函数之前就把 vtable 定下来 */
static void seq_forward(Module *m, const Tensor *x, Tensor *out);
static void seq_backward(Module *m, const Tensor *dout, Tensor *dx);
static void seq_zero_grad(Module *m);
static void seq_free(Module *m);
static int  seq_param_count(Module *m);
static void seq_to_string(Module *m, char *buf, size_t n);

static const ModuleVT SEQUENTIAL_MODULE_VT = {
    "Sequential", seq_forward, seq_backward, seq_zero_grad,
    seq_free, seq_param_count, seq_to_string
};

static void seq_record_stage(Sequential *s, int i, const Tensor *t)
{
    s->stage_ndim[i] = t->ndim;
    for (int k = 0; k < TENSOR_MAX_DIM; k++)
        s->stage_shape[i * TENSOR_MAX_DIM + k] = (k < t->ndim) ? t->shape[k] : 1;
}

static Module *sequential_create_impl(Module *const *modules, int n_modules,
                                      int ndim, const int *shape)
{
    if (n_modules < 0) {
        fprintf(stderr, "sequential_create: n_modules=%d 非法\n", n_modules);
        exit(1);
    }
    Sequential *s = xmalloc(sizeof(*s));
    memset(s, 0, sizeof(*s));
    s->n_modules = n_modules;
    s->modules = (Module **)xcalloc((size_t)(n_modules > 0 ? n_modules : 1), sizeof(Module *));
    for (int i = 0; i < n_modules; i++) {
        if (!modules[i]) {
            fprintf(stderr, "sequential_create: 第 %d 个模块为 NULL\n", i);
            exit(1);
        }
        s->modules[i] = modules[i];
    }
    /* stage 需要 n+1 项；空容器（n=0）的 forward/backward 也会记两次，所以再宽一格 */
    s->stage_ndim = (int *)xmalloc((size_t)(n_modules + 2) * sizeof(int));
    s->stage_shape = (int *)xmalloc((size_t)(n_modules + 2) * TENSOR_MAX_DIM * sizeof(int));
    for (int i = 0; i <= n_modules + 1; i++) {
        s->stage_ndim[i] = -1;      /* -1 = 还没 forward 过 */
        for (int k = 0; k < TENSOR_MAX_DIM; k++) s->stage_shape[i * TENSOR_MAX_DIM + k] = 1;
    }
    if (shape) {
        s->shape_declared = 1;
        s->decl_ndim = ndim;
        for (int k = 0; k < TENSOR_MAX_DIM; k++)
            s->decl_shape[k] = (k < ndim) ? shape[k] : 1;
        /* 预先分配好，稳态完全不分配 */
        s->scratch[0] = tensor_new(ndim, shape);
        s->scratch[1] = tensor_new(ndim, shape);
    }
    Module *m = xmalloc(sizeof(*m));
    m->vt = &SEQUENTIAL_MODULE_VT;
    m->self = s;
    return m;
}

Module *sequential_create(Module *const *modules, int n_modules)
{
    return sequential_create_impl(modules, n_modules, 0, NULL);
}

Module *sequential_create_shaped(Module *const *modules, int n_modules,
                                 int ndim, const int *shape)
{
    if (!shape || ndim < 1 || ndim > TENSOR_MAX_DIM) {
        fprintf(stderr, "sequential_create_shaped: 形状参数非法 (ndim=%d)\n", ndim);
        exit(1);
    }
    return sequential_create_impl(modules, n_modules, ndim, shape);
}

/* 中间张量的形状：声明过就用声明的，否则沿用 parent 的形状 */
static void seq_mid_shape(const Sequential *s, const Tensor *parent, int *ndim, int *shape)
{
    if (s->shape_declared) {
        *ndim = s->decl_ndim;
        for (int k = 0; k < TENSOR_MAX_DIM; k++) shape[k] = s->decl_shape[k];
    } else {
        *ndim = parent->ndim;
        for (int k = 0; k < TENSOR_MAX_DIM; k++) shape[k] = parent->shape[k];
    }
}

static void seq_forward(Module *m, const Tensor *x, Tensor *out)
{
    Sequential *s = m->self;
    if (s->n_modules == 0) {            /* 空容器：恒等（Python 里 forward 直接返回 x） */
        tensor_copy(out, x);
        seq_record_stage(s, 0, x);
        seq_record_stage(s, 1, x);
        return;
    }
    if (s->n_modules == 1) {
        module_forward(s->modules[0], x, out);
        seq_record_stage(s, 0, x);
        seq_record_stage(s, 1, out);
        return;
    }

    seq_record_stage(s, 0, x);
    const Tensor *cur = x;
    Tensor views[2];
    for (int i = 0; i < s->n_modules - 1; i++) {
        int nd, shape[TENSOR_MAX_DIM];
        seq_mid_shape(s, cur, &nd, shape);
        Tensor *v = &views[i % 2];
        *v = buf_view(&s->scratch[i % 2], nd, shape);
        module_forward(s->modules[i], cur, v);
        /* 记录「第 i+1 个模块的输入形状」：模块只能写进我们给的视图，
         * 所以视图的形状就是它实际的输出形状（模块内部会做形状检查） */
        seq_record_stage(s, i + 1, v);
        cur = v;
    }
    module_forward(s->modules[s->n_modules - 1], cur, out);
    seq_record_stage(s, s->n_modules, out);
}

static void seq_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    Sequential *s = m->self;
    if (s->n_modules == 0) {
        tensor_copy(dx, dout);
        return;
    }
    if (s->n_modules == 1) {
        module_backward(s->modules[0], dout, dx);
        return;
    }
    if (s->stage_ndim[0] < 0) {
        fprintf(stderr, "sequential_backward: 还没调用过 forward（无反传所需的形状信息）\n");
        exit(1);
    }
    /* 形状校验：dout 必须是最后一层的输出形状，dx 必须是整条链的输入形状 */
    tensor_check_shape(dout, s->stage_ndim[s->n_modules],
                       s->stage_shape + (size_t)s->n_modules * TENSOR_MAX_DIM,
                       "sequential_backward(dout)");
    tensor_check_shape(dx, s->stage_ndim[0], s->stage_shape, "sequential_backward(dx)");

    /* 第 i 层（i>=1）的输入梯度形状 = stage[i]，写进 scratch[(i-1)%2]；
     * 与前向 buf[i%2] 的对应关系一致，且写入的缓冲与正在读的缓冲奇偶不同，不会别名。 */
    const Tensor *dcur = dout;
    Tensor views[2];
    for (int i = s->n_modules - 1; i >= 1; i--) {
        Tensor *v = &views[(i - 1) % 2];
        *v = buf_view(&s->scratch[(i - 1) % 2], s->stage_ndim[i],
                      s->stage_shape + (size_t)i * TENSOR_MAX_DIM);
        module_backward(s->modules[i], dcur, v);
        dcur = v;
    }
    module_backward(s->modules[0], dcur, dx);
}

static void seq_zero_grad(Module *m)
{
    Sequential *s = m->self;
    for (int i = 0; i < s->n_modules; i++) module_zero_grad(s->modules[i]);
}

static void seq_free(Module *m)
{
    Sequential *s = m->self;
    if (!s) return;
    for (int i = 0; i < s->n_modules; i++) module_free(s->modules[i]);   /* 递归，拥有所有权 */
    free(s->modules);
    tensor_free(s->scratch[0]);
    tensor_free(s->scratch[1]);
    free(s->stage_ndim);
    free(s->stage_shape);
    free(s);
    m->self = NULL;
}

static int seq_param_count(Module *m)
{
    Sequential *s = m->self;
    int n = 0;
    for (int i = 0; i < s->n_modules; i++) n += module_param_count(s->modules[i]);
    return n;
}

static void seq_to_string(Module *m, char *buf, size_t n)
{
    Sequential *s = m->self;
    size_t used = 0;
    int w = snprintf(buf, n, "Sequential(%d): ", s->n_modules);
    if (w < 0) return;
    used = (size_t)w;
    for (int i = 0; i < s->n_modules && used + 1 < n; i++) {
        const char *nm = s->modules[i]->vt->name ? s->modules[i]->vt->name : "?";
        w = snprintf(buf + used, n - used, "%s%s", i ? " -> " : "", nm);
        if (w < 0) return;
        used += (size_t)w;
    }
}

/* ==========================================================================
 *  ResidualBlock
 * ========================================================================== */

static Tensor res_view(ResidualBlock *r, const Tensor *like)
{
    r->buf_ndim = like->ndim;
    for (int k = 0; k < TENSOR_MAX_DIM; k++)
        r->buf_shape[k] = (k < like->ndim) ? like->shape[k] : 1;
    return buf_view(&r->buf, like->ndim, like->shape);
}

static void res_forward(Module *m, const Tensor *x, Tensor *out)
{
    ResidualBlock *r = m->self;
    /* out = x + sublayer(x)；残差要求子层保持形状（Python 版也是靠 `for j in range(len(x[0]))` 隐式要求） */
    tensor_check_shape(out, x->ndim, x->shape, "residual_forward(out)");
    Tensor v = res_view(r, x);
    module_forward(r->sublayer, x, &v);
    for (int i = 0; i < x->size; i++) out->data[i] = x->data[i] + v.data[i];
}

static void res_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    ResidualBlock *r = m->self;
    /* 历史 bug 备忘：Python 版这里曾经直接 `dout + sublayer.backward(dout)`，
     * 子层返回元组（Layer / LayerNorm）时 TypeError；C 版用 dx 出参天然没有这个问题。 */
    tensor_check_shape(dx, dout->ndim, dout->shape, "residual_backward(dx)");
    Tensor v = res_view(r, dout);
    module_backward(r->sublayer, dout, &v);
    for (int i = 0; i < dout->size; i++) dx->data[i] = dout->data[i] + v.data[i];
}

static void res_zero_grad(Module *m)
{
    ResidualBlock *r = m->self;
    module_zero_grad(r->sublayer);
}

static void res_free(Module *m)
{
    ResidualBlock *r = m->self;
    if (!r) return;
    module_free(r->sublayer);       /* 递归，拥有所有权 */
    tensor_free(r->buf);
    free(r);
    m->self = NULL;
}

static int res_param_count(Module *m)
{
    ResidualBlock *r = m->self;
    return module_param_count(r->sublayer);
}

static void res_to_string(Module *m, char *buf, size_t n)
{
    ResidualBlock *r = m->self;
    const char *nm = (r->sublayer && r->sublayer->vt->name) ? r->sublayer->vt->name : "?";
    snprintf(buf, n, "ResidualBlock(%s)", nm);
}

static const ModuleVT RESIDUAL_MODULE_VT = {
    "ResidualBlock", res_forward, res_backward, res_zero_grad,
    res_free, res_param_count, res_to_string
};

Module *residual_create(Module *sublayer)
{
    if (!sublayer) {
        fprintf(stderr, "residual_create: sublayer 为 NULL\n");
        exit(1);
    }
    ResidualBlock *r = xmalloc(sizeof(*r));
    memset(r, 0, sizeof(*r));
    r->sublayer = sublayer;
    r->buf = NULL;
    Module *m = xmalloc(sizeof(*m));
    m->vt = &RESIDUAL_MODULE_VT;
    m->self = r;
    return m;
}
