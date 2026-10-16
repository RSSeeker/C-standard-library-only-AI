/*
 * model_high.c —— 高层封装实现
 */
#include "model_high.h"

#include <stdio.h>
#include <stdlib.h>

/* sizes = [input_dim] + hidden + [输出维度]；隐藏层用 act，输出层用 out_act */
static Trainer *make_trainer(int input_dim, int output_dim,
                             const int *hidden, int n_hidden,
                             Activation act, Activation out_act,
                             LossKind loss, OptKind opt, double lr)
{
    int n_sizes = n_hidden + 2;
    int *sizes = malloc((size_t)n_sizes * sizeof(int));
    Activation *acts = malloc((size_t)(n_sizes - 1) * sizeof(Activation));
    if (!sizes || !acts) {
        fprintf(stderr, "model_high: out of memory\n");
        exit(1);
    }
    sizes[0] = input_dim;
    for (int i = 0; i < n_hidden; i++) sizes[i + 1] = hidden[i];
    sizes[n_sizes - 1] = output_dim;
    /* 与 Python 一致：隐藏层全用 act，最后一层用 out_act */
    for (int i = 0; i < n_sizes - 2; i++) acts[i] = act;
    acts[n_sizes - 2] = out_act;

    MLP *mlp = mlp_create(sizes, acts, n_sizes);

    /* 与 Python 的 MLP 一致：He 初始化（这里用自带 RNG，不追求与 MT 逐位一致） */
    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < mlp->n_layers; i++) he_init(&mlp->layers[i], &seed);

    Trainer *t = malloc(sizeof(Trainer));
    if (!t) {
        fprintf(stderr, "model_high: out of memory\n");
        exit(1);
    }
    trainer_init(t, mlp, loss, opt, lr);

    free(sizes);
    free(acts);
    return t;
}

Trainer *model_make_classifier(int input_dim, int num_classes,
                               const int *hidden, int n_hidden,
                               Activation act, OptKind opt, double lr)
{
    return make_trainer(input_dim, num_classes, hidden, n_hidden,
                        act, ACT_SOFTMAX, LOSS_CROSS_ENTROPY, opt, lr);
}

Trainer *model_make_regressor(int input_dim, int output_dim,
                              const int *hidden, int n_hidden,
                              Activation act, OptKind opt, double lr)
{
    return make_trainer(input_dim, output_dim, hidden, n_hidden,
                        act, ACT_LINEAR, LOSS_MSE, opt, lr);
}

void model_free(Trainer *t)
{
    if (!t) return;
    trainer_release(t);
    mlp_free(t->net);
    free(t);
}

Trainer *model_quick_train(const Dataset *train, int task_is_classification,
                           const int *hidden, int n_hidden,
                           const FitConfig *cfg, OptKind opt, double lr)
{
    Trainer *t;
    if (task_is_classification) {
        /* 类别数 = 最大索引 + 1（Python 用 len(set(y))；调用方给的 y_idx 需连续） */
        int num_classes = 1;
        for (int i = 0; i < train->n; i++)
            if (train->y_idx[i] + 1 > num_classes) num_classes = train->y_idx[i] + 1;
        t = model_make_classifier(train->in_dim, num_classes, hidden, n_hidden,
                                  ACT_LEAKY_RELU, opt, lr);
    } else {
        t = model_make_regressor(train->in_dim, train->out_dim, hidden, n_hidden,
                                 ACT_LEAKY_RELU, opt, lr);
    }
    if (!t) return NULL;

    FitConfig c = *cfg;
    trainer_fit(t, train, NULL, &c, NULL, NULL);
    return t;
}
