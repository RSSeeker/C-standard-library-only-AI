/*
 * containers.h —— 通用容器：Sequential / ResidualBlock（移植自 AI.py）
 *
 * 与 Python 版的关键差异（有意为之的简化）：
 *
 *   Python 版的 backward 返回值有两种约定，全靠 isinstance 判断：
 *       - 非元组（list）：整体就是传给上一层的 delta（Dropout / Embedding / 注意力……）
 *       - 元组 (..., delta)：取末位（Layer / LayerNorm）
 *       Sequential.backward 里还额外写了 `if result is not None: delta = ...`，
 *       因为 Embedding 之类没有输入梯度时会返回 None。
 *   C 版统一成 tensor.h 的 Module vtable 契约：
 *       backward(m, dout, dx) —— 把「对输入的梯度」写进 dx（没有就写 0），
 *   于是「元组取末位 / 非元组整体 / None」三种歧义都不存在，容器里不需要任何类型判断。
 *   这是对 Python 约定的简化，语义等价。
 *
 *   ResidualBlock 在 Python 里曾经有个真实 bug：backward 直接拿 sublayer.backward()
 *   的返回值去加 dout，没处理「返回元组」的情况，于是 sublayer 是 Layer / LayerNorm 时
 *   会 TypeError（元组不能和 list 相加）。C 版用 Module vtable 天然避开这个问题；
 *   另外 Python 版的 ResidualBlock 至今仍没有 Sequential 那样的 `None` 判断，
 *   所以 ResidualBlock(Embedding) 在 Python 里会因 `dout + None` 直接崩，
 *   C 版因为 Embedding 的 backward 会往 dx 里写 0 而不会崩（但语义上把 Embedding
 *   放在残差里本身就没有意义，见答复里的说明）。
 *
 * 所有权：Sequential / ResidualBlock 都 **拥有** 自己的子模块，
 * module_free() 会递归释放子模块（以及子模块的子模块）。
 * 传给 create 的 Module* 不要再单独 free；modules 数组本身会被复制一份，调用方可以释放原数组。
 */
#ifndef AI_CONTAINERS_H
#define AI_CONTAINERS_H

#include "tensor.h"

/* ==========================================================================
 *  Sequential：依次调用 forward，逆序调用 backward
 * ========================================================================== */
typedef struct {
    Module **modules;       /* 拥有；free 时递归释放每个子模块 */
    int n_modules;

    /* 中间张量缓冲（乒乓两块）。
     * 默认：形状取第一次 forward 时输入张量的形状（即假定整条链保持形状不变）。
     * 链里有改变形状的模块时（例如 Embedding: (T) → (T, E)），
     * 必须用 sequential_create_shaped() 显式声明中间形状。
     * 缓冲只在「需要的形状比当前容量大」时才重新分配，稳态热路径零 malloc
     * （与 ai.h 里 Trainer 缓冲的懒分配约定一致）。 */
    Tensor *scratch[2];
    int decl_ndim;
    int decl_shape[TENSOR_MAX_DIM];
    int shape_declared;

    /* 每层输入的形状：stage[i] 是第 i 个模块输入的形状（stage[0] = 输入，stage[n] = 输出）。
     * 前向时记录，反向时用它给缓冲取视图（这样各层形状不同也能对）。 */
    int *stage_ndim;
    int *stage_shape;       /* (n_modules + 1) * TENSOR_MAX_DIM */
} Sequential;

/* 创建 Sequential：接管 modules 中每个 Module 的所有权（数组内容会被复制） */
Module *sequential_create(Module *const *modules, int n_modules);
/* 显式声明「模块之间的中间张量形状」（链中有改变形状的模块时使用） */
Module *sequential_create_shaped(Module *const *modules, int n_modules,
                                 int ndim, const int *shape);

/* ==========================================================================
 *  ResidualBlock：out = x + sublayer(x)
 * ========================================================================== */
typedef struct {
    Module *sublayer;       /* 拥有 */
    int buf_ndim;
    int buf_shape[TENSOR_MAX_DIM];
    Tensor *buf;            /* 子层输出 / 子层输入梯度的缓冲 */
} ResidualBlock;

Module *residual_create(Module *sublayer);

#endif /* AI_CONTAINERS_H */
