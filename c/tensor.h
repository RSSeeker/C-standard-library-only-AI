/*
 * tensor.h —— 张量与通用模块接口
 *
 * 为什么需要它：Python 版的 CNN / RNN / 注意力涉及多维数据（C×H×W、(T, D) 等），
 * 而 ai.h 里的 MLP 用的是扁平数组。这里提供一个最小张量类型，
 * 让后续模块（Conv2d / MaxPool2d / BatchNorm / RNN / Attention…）有统一的数据载体。
 *
 * 约定（与 Python 版一致的地方会特别注明）：
 *   - 一律行主序、double、连续内存
 *   - ndim 只支持 1..4（够覆盖本项目的全部形状）
 *   - 所有函数不做隐式分配；forward/backward 的形状由调用方保证，
 *     不匹配时通过 tensor_check_shape() 报错而不是静默截断
 *   - 反向传播的写入目标由调用方提供（dx），参数梯度累加到模块内部缓冲
 */
#ifndef AI_TENSOR_H
#define AI_TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIM 4

typedef struct {
    int ndim;                    /* 1..4 */
    int shape[TENSOR_MAX_DIM];   /* 末尾维度为 1 的视为无效维 */
    int size;                    /* 元素总数 = prod(shape) */
    double *data;                /* 长度 size，行主序 */
    int owns_data;               /* 1 = 需要 free，0 = 只是视图 */
} Tensor;

/* ---------- 生命周期 ---------- */
Tensor *tensor_new(int ndim, const int *shape);          /* 分配并清零 */
Tensor *tensor_new_like(const Tensor *t);
void    tensor_free(Tensor *t);
void    tensor_zero(Tensor *t);
void    tensor_copy(Tensor *dst, const Tensor *src);      /* 形状必须一致 */
void    tensor_fill(Tensor *t, double v);

/* ---------- 形状 ---------- */
int  tensor_size_of(int ndim, const int *shape);
int  tensor_same_shape(const Tensor *a, const Tensor *b);
int  tensor_check_shape(const Tensor *t, int ndim, const int *shape, const char *who);
void tensor_print_shape(const Tensor *t, const char *tag);

/* ---------- 通用模块接口（供 Sequential / ResidualBlock 这类容器使用） ----------
 * Python 版靠鸭子类型，C 里用这个 vtable 表达同一件事。
 * 注意 backward 的契约与 Python 一致：把梯度写进 dx，同时把参数梯度
 * 累加到模块内部（调用方负责按 batch 求平均与清零）。
 */
typedef struct Module Module;

typedef struct {
    const char *name;
    void (*forward)(Module *m, const Tensor *x, Tensor *out);
    void (*backward)(Module *m, const Tensor *dout, Tensor *dx);
    void (*zero_grad)(Module *m);
    void (*free)(Module *m);
    int  (*param_count)(Module *m);
    void (*to_string)(Module *m, char *buf, size_t n);
} ModuleVT;

struct Module {
    const ModuleVT *vt;
    void *self;
};

void module_forward(Module *m, const Tensor *x, Tensor *out);
void module_backward(Module *m, const Tensor *dout, Tensor *dx);
void module_zero_grad(Module *m);
void module_free(Module *m);
int  module_param_count(Module *m);

#endif /* AI_TENSOR_H */
