/*
 * demo.c —— 对应 README「快速开始」的两段示例，纯 C 实现
 *   (1) 分类：3 维特征 3 类，Softmax + CrossEntropy + Adam
 *   (2) 回归：目标 y = [x0², x0·x1]，20 个样本
 * 输出刻意保持 ASCII，避免控制台代码页问题。
 *
 * 用法: demo
 */
#include "ai.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

static double now_sec(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(void)
{
    /* ================= 1) 分类 ================= */
    printf("=== classification: quick_train equivalent ===\n");
    {
        const double x[6][3] = {
            { 0.10, 0.20, 0.30 }, { 0.40, 0.50, 0.60 }, { 0.70, 0.80, 0.90 },
            { 0.15, 0.25, 0.35 }, { 0.45, 0.55, 0.65 }, { 0.75, 0.85, 0.95 },
        };
        const int y[6] = { 0, 1, 2, 0, 1, 2 };

        const int sizes[] = { 3, 32, 3 };
        const Activation acts[] = { ACT_LEAKY_RELU, ACT_SOFTMAX };
        MLP *net = mlp_create(sizes, acts, 3);
        unsigned long long seed = 42ULL;
        he_init(&net->layers[0], &seed);
        he_init(&net->layers[1], &seed);

        Trainer t;
        trainer_init(&t, net, LOSS_CROSS_ENTROPY, OPT_ADAM, 0.01);

        Dataset train = { &x[0][0], NULL, y, 6, 3, 3 };
        FitConfig cfg = { 0 };
        cfg.epochs = 300;
        cfg.batch_size = 6;
        cfg.shuffle = 1;
        cfg.patience = 0;
        cfg.seed = 1;

        double t0 = now_sec();
        int done = trainer_fit(&t, &train, NULL, &cfg, NULL, NULL);
        double dt = now_sec() - t0;
        printf("  trained %d epochs in %.3f s, accuracy = %.4f\n",
               done, dt, trainer_accuracy(&t, &train));

        double pred[3];
        for (int i = 0; i < 6; i++) {
            mlp_forward(net, x[i], pred);
            int best = 0;
            for (int j = 1; j < 3; j++) if (pred[j] > pred[best]) best = j;
            printf("    x=[%.2f %.2f %.2f] -> pred %d (true %d) %s\n",
                   x[i][0], x[i][1], x[i][2], best, y[i], best == y[i] ? "OK" : "MISS");
        }
        trainer_release(&t);
        mlp_free(net);
    }

    /* ================= 2) 回归 ================= */
    printf("\n=== regression: y = [x0^2, x0*x1] ===\n");
    {
        enum { N = 20, IN = 2, OUT = 2 };
        double x[N][IN], y[N][OUT];
        for (int i = 0; i < N; i++) {
            double v = 0.2 + 1.1 * (double)i / (double)(N - 1);   /* 0.2 ~ 1.3 */
            x[i][0] = v;
            x[i][1] = v * 1.5;
            y[i][0] = v * v;
            y[i][1] = v * v * 1.5;
        }

        const int sizes[] = { IN, 32, 32, OUT };
        const Activation acts[] = { ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_LINEAR };
        MLP *net = mlp_create(sizes, acts, 4);
        unsigned long long seed = 2024ULL;
        he_init(&net->layers[0], &seed);
        he_init(&net->layers[1], &seed);
        he_init(&net->layers[2], &seed);

        Trainer t;
        trainer_init(&t, net, LOSS_MSE, OPT_ADAM, 0.03);

        Dataset train = { &x[0][0], &y[0][0], NULL, N, IN, OUT };
        FitConfig cfg = { 0 };
        cfg.epochs = 1500;
        cfg.batch_size = 20;
        cfg.shuffle = 1;
        cfg.val_split = 0.1;
        cfg.patience = 100;
        cfg.seed = 7;

        double t0 = now_sec();
        int done = trainer_fit(&t, &train, NULL, &cfg, NULL, NULL);
        double dt = now_sec() - t0;
        printf("  trained %d epochs in %.3f s, R2 = %.4f\n",
               done, dt, trainer_r2(&t, &train));

        double q[IN] = { 0.6, 0.9 };
        double pred[OUT];
        mlp_forward(net, q, pred);
        printf("  predict([0.60, 0.90]) = [%.4f, %.4f]   (expected [0.3600, 0.5400])\n",
               pred[0], pred[1]);
        trainer_release(&t);
        mlp_free(net);
    }
    return 0;
}
