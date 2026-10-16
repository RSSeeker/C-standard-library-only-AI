/*
 * bench.c —— 与 Python 版做对等性能对比
 *
 * 工作负载与 tools/c_bench.py 完全一致：
 *   MLP [64, 256, 128, 8]，1000 个样本，5 轮，batch=32，Adam(lr=0.001)，shuffle
 * 只计时训练循环本身（数据生成不计入）。
 *
 * 用法: bench [epochs] [samples]
 */
#include "ai.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void)
{
    /* 用 C89 就有的 clock()：测的是 CPU 时间，且不依赖 POSIX/Win32 扩展 */
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    int epochs = (argc > 1) ? atoi(argv[1]) : 5;
    int n = (argc > 2) ? atoi(argv[2]) : 1000;

    const int sizes[] = { 64, 256, 128, 8 };
    const Activation acts[] = { ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_LINEAR };
    const int in_dim = sizes[0], out_dim = sizes[3];

    /* ---------- 数据（不计时） ---------- */
    unsigned long long rs = 12345ULL;
    double *X = malloc((size_t)n * in_dim * sizeof(double));
    double *Y = malloc((size_t)n * out_dim * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < in_dim; k++) X[(size_t)i * in_dim + k] = (double)((i * 7 + k * 13) % 100) / 50.0 - 1.0;
        for (int k = 0; k < out_dim; k++) {
            double s = 0.0;
            for (int t = 0; t < 8; t++) s += X[(size_t)i * in_dim + (k * 8 + t) % in_dim];
            Y[(size_t)i * out_dim + k] = s / 8.0;
        }
    }
    (void)rs;

    Dataset train = { X, Y, NULL, n, in_dim, out_dim };
    Trainer t;
    trainer_init(&t, mlp_create(sizes, acts, 4), LOSS_MSE, OPT_ADAM, 0.001);

    FitConfig cfg = { 0 };
    cfg.epochs = epochs;
    cfg.batch_size = 32;
    cfg.patience = 0;
    cfg.val_split = 0.0;
    cfg.shuffle = 1;
    cfg.verbose = 0;
    cfg.seed = 42;
    cfg.grad_clip = 0.0;

    unsigned long long seed = 999ULL;
    he_init(&t.net->layers[0], &seed);
    he_init(&t.net->layers[1], &seed);
    he_init(&t.net->layers[2], &seed);

    int params = 0;
    for (int i = 0; i < t.net->n_layers; i++)
        params += t.net->layers[i].fan_in * t.net->layers[i].fan_out + t.net->layers[i].fan_out;

    printf("C: MLP[64,256,128,8] params=%d, samples=%d, epochs=%d, batch=32, Adam(0.001)\n",
           params, n, epochs);
    fflush(stdout);
    double t0 = now_sec();
    int done = trainer_fit(&t, &train, NULL, &cfg, NULL, NULL);
    double dt = now_sec() - t0;

    Dataset d = train;
    double r2 = trainer_r2(&t, &d);
    printf("  trained %d epochs in %.3f s, train R2 = %.6f\n", done, dt, r2);
    printf("  %.1f ms/epoch, %.1f us per sample-epoch\n",
           dt * 1000.0 / done, dt * 1e6 / ((double)done * n));

    trainer_release(&t);
    mlp_free(t.net);
    free(X); free(Y);
    return 0;
}
