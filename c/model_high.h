/*
 * model_high.h —— 高层封装（对应 Python 的 make_classifier / make_regressor / quick_train）
 *
 * Python 版这三个函数只是把 MLP + Model 组合起来，C 版同样保持"薄封装"：
 * 返回已经配好损失/优化器的 Trainer，调用方再用 trainer_fit 训练。
 *
 * 与 Python 的对应关系：
 *   make_classifier(input_dim, num_classes, hidden_layers, act, lr, optimizer)
 *     -> model_make_classifier(...)   softmax + cross_entropy
 *   make_regressor(input_dim, output_dim, hidden_layers, act, lr, optimizer)
 *     -> model_make_regressor(...)    linear + mse
 *   quick_train(x, y, task, ...)
 *     -> model_quick_train(...)       建模型 + 训练一步到位
 */
#ifndef AI_MODEL_HIGH_H
#define AI_MODEL_HIGH_H

#include "ai.h"

/* hidden 为长度 n_hidden 的隐藏层宽度数组（可为 NULL 表示没有隐藏层） */
Trainer *model_make_classifier(int input_dim, int num_classes,
                               const int *hidden, int n_hidden,
                               Activation act, OptKind opt, double lr);
Trainer *model_make_regressor(int input_dim, int output_dim,
                              const int *hidden, int n_hidden,
                              Activation act, OptKind opt, double lr);

/* 释放 Trainer 与其网络（对应 Python 里模型对象整体释放） */
void model_free(Trainer *t);

/*
 * 一步到位训练（对应 quick_train）：
 *   task_is_classification = 1 时，y 走 y_idx（类别索引），否则走 y（回归目标）
 *   hidden / n_hidden 同上面两个函数
 * 返回训练好的 Trainer，失败返回 NULL。
 */
Trainer *model_quick_train(const Dataset *train, int task_is_classification,
                           const int *hidden, int n_hidden,
                           const FitConfig *cfg, OptKind opt, double lr);

#endif /* AI_MODEL_HIGH_H */
