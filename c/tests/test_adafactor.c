/*
 * test_adafactor.c —— Adafactor 的单元测试（数值 + 非法参数）
 *
 * 用法（在 c/ 目录）：./tests/test_adafactor.exe
 * 逐位对拍在 compare_adafactor.c；这里测手算值、行列布局、eps2 分支与错误路径。
 */
#include "../optim.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        const double _a = (a), _b = (b), _t = (tol);                         \
        if (!(fabs(_a - _b) <= _t)) {                                        \
            printf("  FAIL %s:%d  %s = %.17g 与 %.17g 差 %.3e > %.0e\n",     \
                   __FILE__, __LINE__, #a, _a, _b, fabs(_a - _b), _t);       \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_ERR(call, sub)                                                 \
    do {                                                                     \
        const Adafactor *_p = (call);                                        \
        if (_p) {                                                            \
            printf("  FAIL %s:%d  %s 本应失败却成功了\n", __FILE__,          \
                   __LINE__, #call);                                         \
            g_fail++;                                                        \
            adafactor_free((Adafactor *)_p);                                 \
        } else if (!err) {                                                   \
            printf("  FAIL %s:%d  %s 失败但 err 为空\n", __FILE__,           \
                   __LINE__, #call);                                         \
            g_fail++;                                                        \
        } else if (!strstr(err, (sub))) {                                    \
            printf("  FAIL %s:%d  错误串 '%s' 不含 '%s'\n", __FILE__,        \
                   __LINE__, err, (sub));                                    \
            g_fail++;                                                        \
        }                                                                    \
        err = NULL;                                                          \
    } while (0)

/* ---------- 1. 1x1 手算：lr_t=1, v=1 -> 一步走到 0 ---------- */
static void test_hand_computed(void)
{
    printf("[1] 手算 1x1（g=1, lr=1, eps2=1e-3）\n");
    double w = 1.0, b = 0.0, gw = 1.0, gb = 1.0;
    OptLayer l = {&w, &b, &gw, &gb, 1, 1};
    const char *err = NULL;
    Adafactor *o = adafactor_create(&l, 1, 1.0, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;

    /* t=1: decay=0, om=1 -> R=C=1, mean_R=1, lr_t=1*max(1e-3,1)=1, v=1
     *      w = 1 - 1*1/sqrt(1+1e-30) = 0；V_b=1 -> b = 0 - 1 = -1 */
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, 0.0, 0.0);
    CHECK_NEAR(b, -1.0, 0.0);
    CHECK_NEAR(adafactor_row(o, 0)[0], 1.0, 0.0);
    CHECK_NEAR(adafactor_col(o, 0)[0], 1.0, 0.0);
    CHECK_NEAR(adafactor_v_b(o, 0)[0], 1.0, 0.0);
    CHECK(adafactor_step_count(o) == 1);
    adafactor_free(o);

    /* 线性性质：g=2, lr=1 -> lr_t = max(1e-3, 2) = 2, v = 4, 更新 = 2*2/2 = 2 -> w = -1 */
    w = 1.0; b = 0.0; gw = 2.0; gb = 0.0;
    o = adafactor_create(&l, 1, 1.0, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, -1.0, 0.0);
    CHECK_NEAR(b, 0.0, 0.0);   /* 偏置梯度 0 -> 不动 */
    adafactor_free(o);
}

/* ---------- 2. 行列布局（行主序映射）---------- */
static void test_row_col_layout(void)
{
    printf("[2] R/C 的行列均值与行主序映射\n");
    /* 2x2 权重 gw = [[1,2],[3,4]]，t=1 时 decay=0 -> R/C 就是纯均值 */
    double w[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    double gw[4] = {1, 2, 3, 4}, gb[2] = {0, 0};
    OptLayer l = {w, b, gw, gb, 2, 2};
    const char *err = NULL;
    Adafactor *o = adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adafactor_step(o, &err) == 0);
    /* R[0] = (1^2+2^2)/2 = 2.5, R[1] = (3^2+4^2)/2 = 12.5 */
    CHECK_NEAR(adafactor_row(o, 0)[0], 2.5, 0.0);
    CHECK_NEAR(adafactor_row(o, 0)[1], 12.5, 0.0);
    /* C[0] = (1^2+3^2)/2 = 5, C[1] = (2^2+4^2)/2 = 10 */
    CHECK_NEAR(adafactor_col(o, 0)[0], 5.0, 0.0);
    CHECK_NEAR(adafactor_col(o, 0)[1], (4.0 + 16.0) / 2.0, 0.0);
    /* 元素 (0,0) 的更新用 R[0]*C[0]/mean_R = 2.5*5/7.5 -> 参数应当减小 */
    CHECK(w[0] < 0.0);
    CHECK(w[3] < 0.0);
    adafactor_free(o);
}

/* ---------- 3. 权重衰减作用在更新前的参数上（dyadic 精确断言）---------- */
static void test_decay_order(void)
{
    printf("[3] 解耦权重衰减的顺序（dyadic：0.875 -> 0.765625）\n");
    double w = 1.0, b = 0.0, gw = 0.0, gb = 0.0;
    OptLayer l = {&w, &b, &gw, &gb, 1, 1};
    const char *err = NULL;
    /* 梯度为 0 -> 更新量只有 -lr*wd*w；lr=0.25, wd=0.5 全是二进制精确值 */
    Adafactor *o = adafactor_create(&l, 1, 0.25, 1e-30, 1e-3, 0.999, 0.5, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, 0.875, 0.0);       /* 1 - 0.25*0.5*1 */
    CHECK_NEAR(w, 1.0 - 0.25 * 0.5, 0.0);
    CHECK_NEAR(b, 0.0, 0.0);         /* b 从 0 起 -> 衰减项也是 0 */
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, 0.765625, 0.0);    /* 0.875 - 0.25*0.5*0.875 */
    /* 若 wd 误作用在更新后的参数上，第二步会得到 0.78125 */
    CHECK(fabs(w - 0.78125) > 1e-6);
    adafactor_free(o);
}

/* ---------- 4. eps2 / relative step size 分支 ---------- */
static void test_eps2_branch(void)
{
    printf("[4] lr_t = lr*max(eps2, rms(g)) 的两侧\n");
    double w = 1.0, b = 0.0, gw = 4e-4, gb = 0.0;
    OptLayer l = {&w, &b, &gw, &gb, 1, 1};
    const char *err = NULL;
    /* rms(g)=4e-4 < eps2=1e-3 -> lr_t = 1*1e-3；v=1.6e-7, sqrt(v)=4e-4
     * 更新 = 1e-3 * 4e-4/4e-4 = 1e-3 -> w ≈ 0.999（而不是 4e-4 或 1） */
    Adafactor *o = adafactor_create(&l, 1, 1.0, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, 0.999, 1e-12);
    CHECK(fabs(w - 0.9999) > 1e-6);  /* 若没有 max() 的 eps2 下限，会走 1e-4 这一档 */
    adafactor_free(o);

    /* 大梯度：rms=1 > eps2 -> 走 rms 一侧 */
    w = 1.0; b = 0.0; gw = 1.0;
    o = adafactor_create(&l, 1, 1.0, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    CHECK(adafactor_step(o, &err) == 0);
    CHECK_NEAR(w, 0.0, 0.0);
    adafactor_free(o);
}

/* ---------- 5. 全零梯度：mean_R<=0 取 1.0 的分支 ---------- */
static void test_zero_grads(void)
{
    printf("[5] 全零梯度（mean_R 保护 + 只剩 wd）\n");
    double w = 1.0, b = 2.0, gw = 0.0, gb = 0.0;
    OptLayer l = {&w, &b, &gw, &gb, 1, 1};
    const char *err = NULL;
    Adafactor *o = adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 0.999, 0.1, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adafactor_step(o, &err) == 0);
    /* R=C=0 -> mean_R=0 -> 取 1.0；v=0；sqrt(0+1e-30)=1e-15；
     * 梯度为 0 所以更新项是 0；参数只乘 (1-lr*wd)=0.99 */
    CHECK_NEAR(w, 0.99, 0.0);
    CHECK_NEAR(b, 1.98, 0.0);
    CHECK_NEAR(adafactor_row(o, 0)[0], 0.0, 0.0);
    CHECK_NEAR(adafactor_v_b(o, 0)[0], 0.0, 0.0);
    CHECK(!isnan(w) && !isnan(b));   /* 关键：没有 0/0 -> NaN */
    adafactor_free(o);
}

/* ---------- 6. 非法参数 ---------- */
static void test_invalid(void)
{
    printf("[6] 非法参数\n");
    double w = 1.0, b = 0.0, gw = 1.0, gb = 1.0;
    OptLayer l = {&w, &b, &gw, &gb, 1, 1};
    const char *err = NULL;

    CHECK_ERR(adafactor_create(&l, 1, 0.1, 0.0, 1e-3, 0.999, 0.0, &err), "eps");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, -1e-9, 1e-3, 0.999, 0.0, &err), "eps");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, 0.0, 0.999, 0.0, &err), "eps");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, -0.5, 0.999, 0.0, &err), "eps");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, -0.1, 0.0, &err), "beta2");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 1.0, 0.0, &err), "beta2");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 1.5, 0.0, &err), "beta2");
    CHECK_ERR(adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 0.999, -0.1, &err), "weight_decay");
    CHECK_ERR(adafactor_create(NULL, 1, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err), "layers");
    CHECK_ERR(adafactor_create(&l, 0, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err), "layers");

    OptLayer bad = l;
    bad.fan_in = 0;
    CHECK_ERR(adafactor_create(&bad, 1, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err), "fan_in");
    bad = l;
    bad.gw = NULL;
    CHECK_ERR(adafactor_create(&bad, 1, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err), "NULL");

    /* beta2=0 是合法下界 */
    Adafactor *edge = adafactor_create(&l, 1, 0.1, 1e-30, 1e-3, 0.0, 0.0, &err);
    CHECK(edge != NULL);
    adafactor_free(edge);

    CHECK(adafactor_create(&l, 1, 0.1, 0.0, 1e-3, 0.999, 0.0, NULL) == NULL);
    CHECK(adafactor_step(NULL, &err) == -1);
    CHECK(err != NULL);
}

/* ---------- 7. 状态隔离 + 访问器 ---------- */
static void test_state(void)
{
    printf("[7] 状态隔离与访问器边界\n");
    double wa = 1.0, ba = 0.0, gwa = 1.0, gba = 1.0;
    double wb = 1.0, bb = 0.0, gwb = 1.0, gbb = 1.0;
    OptLayer la = {&wa, &ba, &gwa, &gba, 1, 1};
    OptLayer lb = {&wb, &bb, &gwb, &gbb, 1, 1};
    OptLayer ls[2] = {la, lb};
    const char *err = NULL;
    Adafactor *o = adafactor_create(ls, 2, 0.1, 1e-30, 1e-3, 0.999, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adafactor_num_layers(o) == 2);
    CHECK(adafactor_lr(o) == 0.1);
    CHECK(adafactor_eps(o) == 1e-30);
    CHECK(adafactor_eps2(o) == 1e-3);
    CHECK(adafactor_beta2(o) == 0.999);
    CHECK(adafactor_weight_decay(o) == 0.0);
    CHECK(adafactor_row(o, 0) != NULL);
    CHECK(adafactor_col(o, 1) != NULL);
    CHECK(adafactor_v_b(o, 1) != NULL);
    CHECK(adafactor_row(o, -1) == NULL);
    CHECK(adafactor_row(o, 2) == NULL);
    CHECK(adafactor_col(o, 2) == NULL);
    CHECK(adafactor_v_b(o, 9) == NULL);

    CHECK(adafactor_step(o, &err) == 0);
    CHECK(adafactor_step(o, &err) == 0);
    /* 两层参数相同、梯度相同 -> 状态必须相同（没有共享缓冲） */
    CHECK(wa == wb);
    CHECK(adafactor_row(o, 0)[0] == adafactor_row(o, 1)[0]);

    adafactor_set_lr(o, 0.5);
    CHECK(adafactor_lr(o) == 0.5);
    adafactor_free(o);

    adafactor_free(NULL);
    CHECK(adafactor_step_count(NULL) == 0);
    CHECK(adafactor_num_layers(NULL) == 0);
    CHECK(adafactor_lr(NULL) == 0.0);
    CHECK(adafactor_row(NULL, 0) == NULL);
}

int main(void)
{
    printf("== test_adafactor ==\n");
    test_hand_computed();
    test_row_col_layout();
    test_decay_order();
    test_eps2_branch();
    test_zero_grads();
    test_invalid();
    test_state();

    if (g_fail == 0) {
        printf("[OK] test_adafactor 全部通过\n");
        return 0;
    }
    printf("[失败] test_adafactor 有 %d 处断言失败\n", g_fail);
    return 1;
}
