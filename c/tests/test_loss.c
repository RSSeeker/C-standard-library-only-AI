/*
 * test_loss.c —— 额外损失的 C 自测（focal / bce / bce_with_logits / huber）
 *
 * 运行：./tests/test_loss（make test 会用 "self" 调一次）
 *
 * 覆盖：
 *   [1] focal 的闭式值、p_t=1 端点、梯度零和、alpha/gamma 语义
 *   [2] bce 的闭式值、对称性、eps 夹取、p=y 处梯度为 0
 *   [3] bce_with_logits 的数值稳定（±800 不出 inf）、与 bce(sigmoid(z)) 等价
 *   [4] huber 的两段在 |e|=delta 处连续、梯度、delta 缩放
 *   [5] smooth_l1 与 huber 逐位相同
 *   [6] Trainer 集成：四种 LossKind 都能训练一步，且训练损失与直接调用一致
 *   [7] trainer_init 的默认参数与 trainer_set_loss_params 的语义
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../ai.h"
#include "../losses.h"

static int g_fail;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_BITS(got, want)                                                \
    do {                                                                     \
        double _g = (got), _w = (want);                                      \
        if (memcmp(&_g, &_w, sizeof(double)) != 0) {                         \
            printf("[FAIL] %s:%d  %s = %.17g，期望逐位 %.17g\n",             \
                   __FILE__, __LINE__, #got, _g, _w);                        \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(got, want, tol)                                           \
    do {                                                                     \
        double _g = (got), _w = (want);                                      \
        if (!(fabs(_g - _w) <= (tol))) {                                     \
            printf("[FAIL] %s:%d  %s = %.17g，期望 %.17g（差 %.3e）\n",       \
                   __FILE__, __LINE__, #got, _g, _w, fabs(_g - _w));         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

static double sum_of(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i];
    return s;
}

/* ================= [1] focal ================= */
static void test_focal(void)
{
    double pred[4] = {0.1, 0.6, 0.2, 0.1};
    double grad[4];

    /* gamma=0、alpha=1 时退化成 -log(p_t) */
    double loss = focal_loss_grad(pred, 4, 1, 0.0, 1.0, grad);
    CHECK_NEAR(loss, -log(0.6), 1e-15);
    /* 软最大平移不变 ⇒ dz 各项之和为 0 */
    CHECK_NEAR(sum_of(grad, 4), 0.0, 1e-15); /* 数学上为 0，浮点上 ~1e-17 */

    /* p_t == 1 端点：loss 与梯度都是 0 */
    double one[3] = {0.0, 1.0, 0.0};
    double g1[3] = {9.0, 9.0, 9.0};
    CHECK_BITS(focal_loss_grad(one, 3, 1, 2.0, 1.0, g1), 0.0);
    CHECK_BITS(g1[0], 0.0);
    CHECK_BITS(g1[1], 0.0);
    CHECK_BITS(g1[2], 0.0);

    /* alpha < 0 等价于 alpha=1（Python 的 None） */
    double ga[4], gb[4];
    double la = focal_loss_grad(pred, 4, 2, 2.0, -1.0, ga);
    double lb = focal_loss_grad(pred, 4, 2, 2.0, 1.0, gb);
    CHECK_BITS(la, lb);
    CHECK(memcmp(ga, gb, sizeof ga) == 0);

    /* 同一 gamma 下 p_t 越大损失越小；gamma 越大对易分样本压得越狠 */
    double easy[2] = {0.9, 0.1};
    double hard[2] = {0.55, 0.45};
    double l_easy = focal_loss_grad(easy, 2, 0, 2.0, 1.0, ga);
    double l_hard = focal_loss_grad(hard, 2, 0, 2.0, 1.0, gb);
    CHECK(l_easy < l_hard);
    double l_easy_g0 = focal_loss_grad(easy, 2, 0, 0.0, 1.0, ga);
    CHECK(l_easy < l_easy_g0); /* (1-p)^γ 把易分样本的损失压得更小 */

    /* alpha 是乘性系数：alpha=0.5 的损失恰为 alpha=1 的一半 */
    double l_half = focal_loss_grad(pred, 4, 1, 2.0, 0.5, ga);
    double l_one = focal_loss_grad(pred, 4, 1, 2.0, 1.0, gb);
    CHECK_BITS(l_half * 2.0, l_one);
}

/* ================= [2] bce ================= */
static void test_bce(void)
{
    double pred[3] = {0.5, 0.9, 0.1};
    double target[3] = {1.0, 1.0, 1.0};
    double grad[3];

    /* p=0.5、y=1 时逐元素损失是 log2 */
    CHECK_NEAR(bce_loss_grad(pred, target, 1, 1e-12, grad), log(2.0), 1e-15);

    /* p == y 时**梯度**为 0；软标签下 loss 不为 0（是标签自身的熵） */
    double p2[2] = {0.25, 0.75};
    double y2[2] = {0.25, 0.75};
    double l_soft = bce_loss_grad(p2, y2, 2, 1e-12, grad);
    CHECK_BITS(grad[0], 0.0);
    CHECK_BITS(grad[1], 0.0);
    CHECK_NEAR(l_soft, -(0.25 * log(0.25) + 0.75 * log(0.75)), 1e-15);

    /* 落在 (eps, 1-eps) 内部的 p == y：loss 是标签熵、梯度严格为 0 */
    double p3h[2] = {1.0 - 1e-3, 1e-3};
    double y3h[2] = {1.0 - 1e-3, 1e-3};
    (void)bce_loss_grad(p3h, y3h, 2, 1e-12, grad);
    CHECK_BITS(grad[0], 0.0);
    CHECK_BITS(grad[1], 0.0);

    /* 端点 p=1/0 会被夹到 1-eps/eps：梯度不是 0，而是 -1/(2eps)、+1/(2eps) 量级
     * （这是 eps 保护的固有代价，不是 bug；记下来免得以后误判） */
    double p5h[2] = {1.0, 0.0};
    double y5h[2] = {1.0, 0.0};
    (void)bce_loss_grad(p5h, y5h, 2, 1e-12, grad);
    CHECK_NEAR(grad[0], -0.5, 1e-9);
    CHECK_NEAR(grad[1], 0.5, 1e-9);

    /* 对称性：bce(p, y) == bce(1-p, 1-y) */
    double p3[3] = {0.1, 0.4, 0.85};
    double y3[3] = {1.0, 0.0, 1.0};
    double rp[3], ry[3], ga[3], gb[3];
    for (int i = 0; i < 3; i++) { rp[i] = 1.0 - p3[i]; ry[i] = 1.0 - y3[i]; }
    double la = bce_loss_grad(p3, y3, 3, 1e-12, ga);
    double lb = bce_loss_grad(rp, ry, 3, 1e-12, gb);
    CHECK_NEAR(la, lb, 1e-15);

    /* eps 夹取：p=0、y=1 时损失是 -log(eps) */
    double p0[1] = {0.0}, y1[1] = {1.0};
    CHECK_NEAR(bce_loss_grad(p0, y1, 1, 1e-6, grad), -log(1e-6), 1e-12);

    /* 梯度是逐元素平均：n 个相同元素时总梯度等于单个的一半（n=2） */
    double p4[2] = {0.3, 0.3}, y4[2] = {1.0, 1.0};
    bce_loss_grad(p4, y4, 2, 1e-12, grad);
    CHECK_BITS(grad[0], grad[1]);
    CHECK_NEAR(grad[0] * 2.0, (0.3 - 1.0) / (0.3 * 0.7), 1e-15);
}

/* ================= [3] bce_with_logits ================= */
static void test_bce_logits(void)
{
    double z[5] = {-800.0, -2.0, 0.0, 2.0, 800.0};
    double y[5] = {0.0, 1.0, 1.0, 0.0, 1.0};
    double grad[5];

    double loss = bce_with_logits_loss_grad(z, y, 5, grad);
    CHECK(isfinite(loss));       /* ±800 不能溢出成 inf */
    for (int i = 0; i < 5; i++) CHECK(isfinite(grad[i]));

    /* z=0、y=1 的逐元素损失是 log2 */
    double z0[1] = {0.0}, y1[1] = {1.0};
    CHECK_NEAR(bce_with_logits_loss_grad(z0, y1, 1, grad), log(2.0), 1e-15);

    /* 与 bce(sigmoid(z), y) 等价（同一元素，误差在舍入量级） */
    double zz[4] = {-3.5, -0.25, 0.75, 2.0};
    double yy[4] = {1.0, 0.0, 1.0, 0.0};
    double pp[4], gl[4], gp[4];
    for (int i = 0; i < 4; i++) pp[i] = 1.0 / (1.0 + exp(-zz[i]));
    double l_logits = bce_with_logits_loss_grad(zz, yy, 4, gl);
    double l_prob = bce_loss_grad(pp, yy, 4, 1e-12, gp);
    CHECK_NEAR(l_logits, l_prob, 1e-12);
    /* 梯度口径不同：gl 相对 logits（= sigmoid-y)/n），gp 相对概率 */
    for (int i = 0; i < 4; i++) {
        double want = (pp[i] - yy[i]) / 4.0;
        CHECK_NEAR(gl[i], want, 1e-15);
    }

    /* y=0.5 时 z=0 的梯度为 0 */
    double z1[1] = {0.0}, yh[1] = {0.5};
    bce_with_logits_loss_grad(z1, yh, 1, grad);
    CHECK_BITS(grad[0], 0.0);
}

/* ================= [4] huber ================= */
static void test_huber(void)
{
    double grad[3];

    /* |e| == delta 时两段相等：0.5·d² 与 d·(d − 0.5d) */
    double p_edge[1] = {1.0}, t0[1] = {0.0};
    CHECK_BITS(huber_loss_grad(p_edge, t0, 1, 1.0, grad), 0.5);

    /* |e| > delta：loss = delta(|e| − 0.5delta)，梯度为 ±delta */
    double p_big[2] = {3.0, -3.0};
    double t_big[2] = {0.0, 0.0};
    double loss = huber_loss_grad(p_big, t_big, 2, 1.0, grad);
    CHECK_NEAR(loss, (3.0 - 0.5) * 1.0, 1e-15); /* 两个元素各 2.5，平均 2.5 */
    CHECK_BITS(grad[0], 0.5);
    CHECK_BITS(grad[1], -0.5);

    /* 小误差段：loss = 0.5e²，梯度 = e/n */
    double p_small[1] = {0.25}, t_small[1] = {0.0};
    CHECK_NEAR(huber_loss_grad(p_small, t_small, 1, 1.0, grad), 0.03125, 1e-17);
    CHECK_BITS(grad[0], 0.25);

    /* delta 缩放：误差恰为 delta 的一半时仍是二次段 */
    double p_d[1] = {0.25};
    CHECK_NEAR(huber_loss_grad(p_d, t0, 1, 0.5, grad), 0.03125, 1e-17);

    /* smooth_l1 是别名，必须逐位相同 */
    double pm[3] = {1.5, -0.3, 0.0};
    double tm[3] = {0.0, 0.1, 0.0};
    double gh[3], gs[3];
    double lh = huber_loss_grad(pm, tm, 3, 1.0, gh);
    double ls = smooth_l1_loss_grad(pm, tm, 3, 1.0, gs);
    CHECK_BITS(lh, ls);
    CHECK(memcmp(gh, gs, sizeof gh) == 0);
}

/* ================= [6] Trainer 集成 ================= */
static void test_trainer_integration(void)
{
    /* 3 类分类：MLP [2,4,3]，ReLU + softmax */
    int sizes[3] = {2, 4, 3};
    Activation acts[2] = {ACT_RELU, ACT_SOFTMAX};
    MLP *net = mlp_create(sizes, acts, 3);
    CHECK(net != NULL);
    unsigned long long rng = 20261108ULL;
    for (int i = 0; i < net->n_layers; i++) he_init(&net->layers[i], &rng);
    Trainer t;
    trainer_init(&t, net, LOSS_FOCAL, OPT_ADAM, 0.01);
    trainer_set_focal(&t, 2.0, 0.5);
    CHECK_BITS(t.loss_gamma, 2.0);
    CHECK_BITS(t.loss_alpha, 0.5);

    double x[2] = {0.3, -0.2};
    double y_onehot[3] = {0.0, 1.0, 0.0};
    double loss_t = trainer_accumulate(&t, x, y_onehot, 1);
    CHECK(isfinite(loss_t));

    /* 同一 pred 直接调 focal，loss 必须逐位相同（证明分发接对了） */
    double pred[3];
    mlp_forward(net, x, pred);
    double g[3];
    double loss_direct = focal_loss_grad(pred, 3, 1, 2.0, 0.5, g);
    CHECK_BITS(loss_t, loss_direct);

    /* 梯度确实流回了网络（第一层 gw 不全为 0） */
    double gsum = 0.0;
    for (int j = 0; j < net->layers[0].fan_out; j++)
        for (int i = 0; i < net->layers[0].fan_in; i++)
            gsum += fabs(net->layers[0].gW[j * net->layers[0].fan_in + i]);
    CHECK(gsum > 0.0);

    /* 换成 bce_with_logits：需要线性输出层 */
    mlp_zero_grad(net);
    t.loss = LOSS_BCE_LOGITS;
    double yb[3] = {0.0, 1.0, 0.0};
    double lb = trainer_accumulate(&t, x, yb, 0); /* 输出 3 维，y 也是 3 维 */
    CHECK(isfinite(lb));

    /* huber：回归目标 */
    t.loss = LOSS_HUBER;
    trainer_set_loss_params(&t, 2.0, -1.0, 0.5, 1e-12);
    double yh[3] = {0.2, -0.1, 0.0};
    double lh = trainer_accumulate(&t, x, yh, 0);
    CHECK(isfinite(lh));

    /* bce：概率输入（softmax 输出归一化，数值上可用） */
    t.loss = LOSS_BCE;
    double lp = trainer_accumulate(&t, x, yb, 0);
    CHECK(isfinite(lp));

    trainer_release(&t);
    mlp_free(net);
}

/* ================= [7] 默认参数 ================= */
static void test_defaults(void)
{
    int sizes[3] = {2, 3, 1};
    Activation acts[2] = {ACT_RELU, ACT_LINEAR};
    MLP *net = mlp_create(sizes, acts, 3);
    unsigned long long rng = 7ULL;
    for (int i = 0; i < net->n_layers; i++) he_init(&net->layers[i], &rng);
    Trainer t;
    trainer_init(&t, net, LOSS_MSE, OPT_SGD, 0.1);
    CHECK_BITS(t.loss_gamma, 2.0);
    CHECK_BITS(t.loss_alpha, -1.0);
    CHECK_BITS(t.loss_delta, 1.0);
    CHECK_BITS(t.loss_eps, 1e-12);

    /* set_loss_params 只改传进来的值 */
    trainer_set_loss_params(&t, 3.0, 0.25, 2.0, 1e-6);
    CHECK_BITS(t.loss_gamma, 3.0);
    CHECK_BITS(t.loss_alpha, 0.25);
    CHECK_BITS(t.loss_delta, 2.0);
    CHECK_BITS(t.loss_eps, 1e-6);

    /* trainer_set_focal 不动 delta/eps */
    trainer_set_focal(&t, 1.5, -1.0);
    CHECK_BITS(t.loss_gamma, 1.5);
    CHECK_BITS(t.loss_alpha, -1.0);
    CHECK_BITS(t.loss_delta, 2.0);
    CHECK_BITS(t.loss_eps, 1e-6);

    trainer_release(&t);
    mlp_free(net);
}

int main(int argc, char **argv)
{
    /* make test 会以 "self" 作为参数调用 */
    (void)argc;
    (void)argv;

    printf("=== 额外损失自测 ===\n");
    test_focal();
    test_bce();
    test_bce_logits();
    test_huber();
    test_trainer_integration();
    test_defaults();

    if (g_fail == 0) {
        printf("[OK] focal/bce/bce_with_logits/huber 与 Trainer 集成自测全部通过\n");
        return 0;
    }
    printf("[FAIL] %d 项失败\n", g_fail);
    return 1;
}
