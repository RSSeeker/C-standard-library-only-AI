/*
 * tensor.c —— 张量与通用模块接口的实现
 */
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 与 ai.c 里一致的分配行为：失败即退出，避免到处判空 */
static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "tensor: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

int tensor_size_of(int ndim, const int *shape)
{
    int s = 1;
    for (int i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

Tensor *tensor_new(int ndim, const int *shape)
{
    if (ndim < 1 || ndim > TENSOR_MAX_DIM) {
        fprintf(stderr, "tensor_new: ndim %d 超出 1..%d\n", ndim, TENSOR_MAX_DIM);
        exit(1);
    }
    Tensor *t = xmalloc(sizeof(Tensor));
    memset(t, 0, sizeof(*t));
    t->ndim = ndim;
    for (int i = 0; i < TENSOR_MAX_DIM; i++) t->shape[i] = (i < ndim) ? shape[i] : 1;
    t->size = tensor_size_of(ndim, shape);
    t->data = xmalloc((size_t)t->size * sizeof(double));
    memset(t->data, 0, (size_t)t->size * sizeof(double));
    t->owns_data = 1;
    return t;
}

Tensor *tensor_new_like(const Tensor *t)
{
    return tensor_new(t->ndim, t->shape);
}

void tensor_free(Tensor *t)
{
    if (!t) return;
    if (t->owns_data) free(t->data);
    free(t);
}

void tensor_zero(Tensor *t)
{
    memset(t->data, 0, (size_t)t->size * sizeof(double));
}

void tensor_copy(Tensor *dst, const Tensor *src)
{
    tensor_check_shape(dst, src->ndim, src->shape, "tensor_copy");
    memcpy(dst->data, src->data, (size_t)src->size * sizeof(double));
}

void tensor_fill(Tensor *t, double v)
{
    for (int i = 0; i < t->size; i++) t->data[i] = v;
}

int tensor_same_shape(const Tensor *a, const Tensor *b)
{
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

/* 形状不符时立刻报错：这个项目里"静默按较短的那个截断"曾经导致过真实 bug */
int tensor_check_shape(const Tensor *t, int ndim, const int *shape, const char *who)
{
    int size = tensor_size_of(ndim, shape);
    int ok = (t->ndim == ndim) && (t->size == size);
    if (ok) {
        for (int i = 0; i < ndim; i++) if (t->shape[i] != shape[i]) { ok = 0; break; }
    }
    if (!ok) {
        fprintf(stderr, "%s: shape mismatch, expected ndim=%d [", who, ndim);
        for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%d", i ? "," : "", shape[i]);
        fprintf(stderr, "] got ndim=%d [", t->ndim);
        for (int i = 0; i < t->ndim; i++) fprintf(stderr, "%s%d", i ? "," : "", t->shape[i]);
        fprintf(stderr, "]\n");
        exit(1);
    }
    return 1;
}

void tensor_print_shape(const Tensor *t, const char *tag)
{
    printf("%s: [", tag);
    for (int i = 0; i < t->ndim; i++) printf("%s%d", i ? "," : "", t->shape[i]);
    printf("] size=%d\n", t->size);
}

/* ---------- 模块接口转发 ---------- */

void module_forward(Module *m, const Tensor *x, Tensor *out)
{
    if (!m || !m->vt->forward) {
        fprintf(stderr, "module_forward: 该模块未实现 forward\n");
        exit(1);
    }
    m->vt->forward(m, x, out);
}

void module_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    if (!m || !m->vt->backward) {
        fprintf(stderr, "module_backward: 该模块未实现 backward\n");
        exit(1);
    }
    m->vt->backward(m, dout, dx);
}

void module_zero_grad(Module *m)
{
    if (m && m->vt->zero_grad) m->vt->zero_grad(m);
}

void module_free(Module *m)
{
    if (!m) return;
    if (m->vt->free) m->vt->free(m);
    free(m);
}

int module_param_count(Module *m)
{
    return (m && m->vt->param_count) ? m->vt->param_count(m) : 0;
}
