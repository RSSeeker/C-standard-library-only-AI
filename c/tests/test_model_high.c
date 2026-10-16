/*
 * test_model_high.c —— 高层封装自测（make_classifier / make_regressor / quick_train）
 *
 *   ./test_model_high self
 */
#include "../ai.h"
#include "../model_high.h"
#include "../model_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int mode_self(void)
{
    int ok = 1;

    /* ---------- 分类：3 维特征 3 类 ---------- */
    {
        static const double x[6][3] = {
            { 0.10, 0.20, 0.30 }, { 0.40, 0.50, 0.60 }, { 0.70, 0.80, 0.90 },
            { 0.15, 0.25, 0.35 }, { 0.45, 0.55, 0.65 }, { 0.75, 0.85, 0.95 },
        };
        static const int y[6] = { 0, 1, 2, 0, 1, 2 };
        const int hidden[1] = { 32 };

        Trainer *t = model_make_classifier(3, 3, hidden, 1, ACT_LEAKY_RELU, OPT_ADAM, 0.01);
        Dataset train = { &x[0][0], NULL, y, 6, 3, 3 };
        FitConfig cfg = { 0 };
        cfg.epochs = 300;
        cfg.batch_size = 6;
        cfg.shuffle = 1;
        cfg.seed = 1;

        trainer_fit(t, &train, NULL, &cfg, NULL, NULL);
        double acc = trainer_accuracy(t, &train);
        printf("[self] make_classifier: 准确率 = %.4f（期望 1.0）\n", acc);
        if (acc < 0.999) ok = 0;
        /* 输出层必须是 softmax + CE */
        printf("        输出层激活 = %s，损失 = %s\n",
               act_name_of(t->net->layers[t->net->n_layers - 1].act),
               t->loss == LOSS_CROSS_ENTROPY ? "cross_entropy" : "mse");
        if (t->net->layers[t->net->n_layers - 1].act != ACT_SOFTMAX) ok = 0;
        model_free(t);
    }

    /* ---------- 回归：y = [x0^2, x0*x1] ---------- */
    {
        enum { N = 20, IN = 2, OUT = 2 };
        static double x[N][IN], y[N][OUT];
        for (int i = 0; i < N; i++) {
            double v = 0.2 + 1.1 * (double)i / (double)(N - 1);
            x[i][0] = v;
            x[i][1] = v * 1.5;
            y[i][0] = v * v;
            y[i][1] = v * v * 1.5;
        }
        const int hidden[2] = { 32, 32 };

        Trainer *t = model_make_regressor(IN, OUT, hidden, 2, ACT_LEAKY_RELU, OPT_ADAM, 0.03);
        Dataset train = { &x[0][0], &y[0][0], NULL, N, IN, OUT };
        FitConfig cfg = { 0 };
        cfg.epochs = 1500;
        cfg.batch_size = N;
        cfg.shuffle = 1;
        cfg.val_split = 0.1;
        cfg.patience = 100;
        cfg.seed = 7;

        int done = trainer_fit(t, &train, NULL, &cfg, NULL, NULL);
        double r2 = trainer_r2(t, &train);
        printf("[self] make_regressor: %d 轮后 R2 = %.4f（期望 > 0.99）\n", done, r2);
        if (r2 < 0.99) ok = 0;
        model_free(t);
    }

    /* ---------- quick_train 两条路径 ---------- */
    {
        static const double x[4][2] = { { 0.1, 0.2 }, { 0.8, 0.9 }, { 0.15, 0.25 }, { 0.85, 0.95 } };
        static const int y[4] = { 0, 1, 0, 1 };
        static const double yr[4][1] = { { 0.1 }, { 0.8 }, { 0.15 }, { 0.85 } };
        const int hidden[1] = { 16 };

        Dataset dc = { &x[0][0], NULL, y, 4, 2, 2 };
        FitConfig cfg = { 0 };
        cfg.epochs = 400;
        cfg.batch_size = 4;
        cfg.shuffle = 1;
        cfg.seed = 3;
        Trainer *tc = model_quick_train(&dc, 1, hidden, 1, &cfg, OPT_ADAM, 0.02);
        double acc = trainer_accuracy(tc, &dc);
        printf("[self] quick_train(分类): 准确率 = %.4f\n", acc);
        if (acc < 0.999) ok = 0;
        model_free(tc);

        Dataset dr = { &x[0][0], &yr[0][0], NULL, 4, 2, 1 };
        Trainer *tr = model_quick_train(&dr, 0, hidden, 1, &cfg, OPT_ADAM, 0.02);
        double r2 = trainer_r2(tr, &dr);
        printf("[self] quick_train(回归): R2 = %.4f\n", r2);
        if (r2 < 0.9) ok = 0;
        model_free(tr);
    }

    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "self";
    if (strcmp(mode, "self") == 0) return mode_self();
    fprintf(stderr, "未知模式: %s\n", mode);
    return 2;
}
