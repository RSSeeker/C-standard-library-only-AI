/*
 * test_sched.c —— 学习率调度与梯度裁剪的自测 + 供 Python 侧比对的 dump
 *
 *   ./test_sched self   基本断言
 *   ./test_sched dump   打印各调度器在每个 epoch 的 lr（供 Python 逐位比对）
 */
#include "../ai.h"
#include "../sched.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int mode_dump(void)
{
    Optimizer opt;
    LRScheduler s;

    /* StepLR: initial_lr=0.1, step_size=3, gamma=0.5 */
    optimizer_init(&opt, OPT_SGD, 0.1);
    lrsched_init_step(&s, &opt, 3, 0.5);
    printf("STEP");
    for (int e = 1; e <= 10; e++) {
        lrsched_step(&s, e);
        printf(" %.17g", opt.lr);
    }
    printf("\n");

    /* ExponentialLR: gamma=0.9 */
    optimizer_init(&opt, OPT_SGD, 0.1);
    lrsched_init_exponential(&s, &opt, 0.9);
    printf("EXP");
    for (int e = 1; e <= 8; e++) {
        lrsched_step(&s, e);
        printf(" %.17g", opt.lr);
    }
    printf("\n");

    /* CosineAnnealingLR: T_max=5, eta_min=0.01（验证周期重启） */
    optimizer_init(&opt, OPT_SGD, 0.1);
    lrsched_init_cosine(&s, &opt, 5, 0.01);
    printf("COSINE");
    for (int e = 1; e <= 12; e++) {
        lrsched_step(&s, e);
        printf(" %.17g", opt.lr);
    }
    printf("\n");

    /* 裁剪：构造已知梯度，打印裁剪前范数与裁剪后的值 */
    const int sizes[] = { 3, 2 };
    const Activation acts[] = { ACT_LINEAR };
    MLP *m = mlp_create(sizes, acts, 2);
    double vals[8] = { 3.0, -4.0, 0.5, 0.0, -0.25, 2.0, -1.0, 0.75 };
    for (int k = 0; k < 6; k++) m->layers[0].gW[k] = vals[k];
    m->layers[0].gb[0] = vals[6];
    m->layers[0].gb[1] = vals[7];

    double norm = clip_grad_norm(m, 1.0);
    printf("CLIP_NORM norm=%.17g", norm);
    for (int k = 0; k < 6; k++) printf(" %.17g", m->layers[0].gW[k]);
    for (int j = 0; j < 2; j++) printf(" %.17g", m->layers[0].gb[j]);
    printf("\n");

    /* 按值裁剪（重新灌入原值） */
    for (int k = 0; k < 6; k++) m->layers[0].gW[k] = vals[k];
    for (int j = 0; j < 2; j++) m->layers[0].gb[j] = vals[6 + j];
    clip_grad_value(m, 1.5);
    printf("CLIP_VALUE");
    for (int k = 0; k < 6; k++) printf(" %.17g", m->layers[0].gW[k]);
    for (int j = 0; j < 2; j++) printf(" %.17g", m->layers[0].gb[j]);
    printf("\n");

    mlp_free(m);
    return 0;
}

static int mode_self(void)
{
    Optimizer opt;
    LRScheduler s;
    int ok = 1;

    /* StepLR 只在 epoch % step_size == 0 时变，且是 initial_lr * gamma^(epoch/step) */
    optimizer_init(&opt, OPT_SGD, 0.1);
    lrsched_init_step(&s, &opt, 3, 0.5);
    lrsched_step(&s, 1);
    double lr1 = opt.lr;
    lrsched_step(&s, 2);
    double lr2 = opt.lr;
    lrsched_step(&s, 3);
    double lr3 = opt.lr;
    printf("[self] StepLR: e1=%.6f e2=%.6f e3=%.6f\n", lr1, lr2, lr3);
    if (lr1 != 0.1 || lr2 != 0.1) ok = 0;
    if (fabs(lr3 - 0.05) > 1e-15) ok = 0;

    /* CosineAnnealingLR 的周期重启：epoch=1 与 epoch=T_max+1 应相同 */
    optimizer_init(&opt, OPT_SGD, 0.1);
    lrsched_init_cosine(&s, &opt, 5, 0.0);
    lrsched_step(&s, 1);
    double a = opt.lr;
    lrsched_step(&s, 6);
    double b = opt.lr;
    printf("[self] Cosine 周期重启: e1=%.17g e6=%.17g 一致=%s\n", a, b, (a == b) ? "是" : "否");
    if (a != b) ok = 0;

    /* 裁剪：范数应等于手算值 */
    const int sizes[] = { 2, 1 };
    const Activation acts[] = { ACT_LINEAR };
    MLP *m = mlp_create(sizes, acts, 2);
    m->layers[0].gW[0] = 3.0;
    m->layers[0].gW[1] = 4.0;
    m->layers[0].gb[0] = 0.0;
    double norm = clip_grad_norm(m, 100.0);
    printf("[self] 未超阈值时范数=%.17g（应为 5）\n", norm);
    if (norm != 5.0) ok = 0;
    double n2 = clip_grad_norm(m, 1.0);
    printf("[self] 裁剪后 gW=[%.6f %.6f] 范数记录=%.17g\n",
           m->layers[0].gW[0], m->layers[0].gW[1], n2);
    if (fabs(m->layers[0].gW[0] - 0.6) > 1e-15 || fabs(m->layers[0].gW[1] - 0.8) > 1e-15) ok = 0;

    mlp_free(m);
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "self";
    if (strcmp(mode, "self") == 0) return mode_self();
    if (strcmp(mode, "dump") == 0) return mode_dump();
    fprintf(stderr, "未知模式: %s\n", mode);
    return 2;
}
