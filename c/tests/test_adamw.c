/*
 * test_adamw.c —— AdamW 的单元测试（数值 + 非法参数）
 *
 * 用法（在 c/ 目录）：./tests/test_adamw.exe
 * 逐位对拍在 compare_adamw.c（读 tools/adamw_reference.py 生成的 JSON）；
 * 这里只测"手算得出的位数"与错误路径，断言失败会打印 文件:行号。
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

/* 期望 create 失败且错误串包含 sub；要求调用点作用域里有名为 err 的变量，
 * 且 call 里的 &err 指的就是它（宏不自己声明 err，避免遮蔽）。 */
#define CHECK_ERR(call, sub)                                                 \
    do {                                                                     \
        const AdamW *_p = (call);                                            \
        if (_p) {                                                            \
            printf("  FAIL %s:%d  %s 本应失败却成功了\n", __FILE__,          \
                   __LINE__, #call);                                         \
            g_fail++;                                                        \
            adamw_free((AdamW *)_p);                                         \
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

/* 1x1 的小层，方便手算 */
typedef struct {
    double w, b, gw, gb;
    OptLayer layer;
} Tiny;

static void tiny_init(Tiny *t, double w, double b, double gw, double gb)
{
    t->w = w;
    t->b = b;
    t->gw = gw;
    t->gb = gb;
    t->layer = (OptLayer){&t->w, &t->b, &t->gw, &t->gb, 1, 1};
}

/* ---------- 1. 手算：dyadic 超参下每步正好走 lr ---------- */
static void test_hand_computed(void)
{
    printf("[1] 手算（b1=0.5, b2=0.75, g=1, lr=0.1, eps=0, wd=0）\n");
    Tiny t;
    tiny_init(&t, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;
    AdamW *o = adamw_create(&t.layer, 1, 0.1, 0.5, 0.75, 0.0, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;

    /* t=1: m=0.5, v=0.25, c1=0.5, c2=0.25 -> m/c1 = 1, sqrt(v/c2) = 1 */
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.9, 0.0);   /* 精确：1 - 0.1*1/1 */
    CHECK_NEAR(t.b, -0.1, 0.0);  /* b 从 0 起，梯度 +1 -> -0.1 */
    CHECK_NEAR(adamw_m_w(o, 0)[0], 0.5, 0.0);
    CHECK_NEAR(adamw_v_w(o, 0)[0], 0.25, 0.0);
    CHECK_NEAR(adamw_m_b(o, 0)[0], 0.5, 0.0);
    CHECK(adamw_step_count(o) == 1);

    /* t=2 再来同样的梯度：m=0.75, v=0.4375, c1=0.75, c2=0.4375 -> 还是走 0.1 */
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.8, 1e-16);
    CHECK_NEAR(t.b, -0.2, 1e-16);
    CHECK_NEAR(adamw_m_w(o, 0)[0], 0.75, 0.0);
    CHECK_NEAR(adamw_v_w(o, 0)[0], 0.4375, 0.0);
    CHECK(adamw_step_count(o) == 2);
    adamw_free(o);

    /* 负梯度：对称地加回去 */
    tiny_init(&t, 1.0, 0.0, -1.0, -1.0);
    o = adamw_create(&t.layer, 1, 0.1, 0.5, 0.75, 0.0, 0.0, &err);
    CHECK(o != NULL);
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 1.1, 0.0);
    CHECK_NEAR(t.b, 0.1, 0.0);
    adamw_free(o);
}

/* ---------- 2. 解耦权重衰减：先乘 decay，再减更新量 ---------- */
static void test_weight_decay(void)
{
    printf("[2] 解耦权重衰减（梯度为 0 时只剩 w *= 1-lr*wd）\n");
    Tiny t;
    tiny_init(&t, 1.0, 2.0, 0.0, 0.0);
    const char *err = NULL;
    /* 梯度为 0 时必须给非零 eps：否则 sqrt(0)+eps == 0，0/0 会得到 NaN
     * （Python 侧同样会 ZeroDivisionError，属于非法配置） */
    AdamW *o = adamw_create(&t.layer, 1, 0.1, 0.5, 0.75, 1e-8, 0.1, &err);
    CHECK(o != NULL);
    if (!o) return;

    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.99, 0.0);        /* 1 * (1-0.1*0.1) = 0.99 */
    CHECK_NEAR(t.b, 1.98, 0.0);        /* 2 * 0.99 */
    CHECK_NEAR(adamw_weight_decay(o), 0.1, 0.0);

    /* 第二步再乘一次：0.99*0.99 = 0.9801（最后一位按 double 舍入） */
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.99 * 0.99, 0.0);
    CHECK_NEAR(adamw_m_w(o, 0)[0], 0.0, 0.0); /* 梯度为 0 -> 动量保持 0 */
    adamw_free(o);

    /* wd=0 时参数只受更新量影响；lr=0 时参数完全不动（动量仍然更新） */
    tiny_init(&t, 1.0, 1.0, 1.0, 1.0);
    o = adamw_create(&t.layer, 1, 0.0, 0.5, 0.75, 1e-8, 0.5, &err);
    CHECK(o != NULL);
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 1.0, 0.0);
    CHECK_NEAR(adamw_m_w(o, 0)[0], 0.5, 0.0);
    adamw_free(o);
}

/* ---------- 3. beta=0：一阶矩退化成梯度，更新 = lr*sign(g) ---------- */
static void test_zero_beta(void)
{
    printf("[3] beta1=beta2=0（更新量 = lr*sign(g)）\n");
    Tiny t;
    tiny_init(&t, 1.0, 0.0, 2.0, -3.0);
    const char *err = NULL;
    AdamW *o = adamw_create(&t.layer, 1, 0.5, 0.0, 0.0, 0.0, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;
    /* c1 = 1-0^1 = 1，m = g；v = g^2；sqrt(v/c2) = |g| -> 更新 lr*sign(g) */
    CHECK(adamw_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.5, 0.0);   /* 1 - 0.5*2/2 */
    CHECK_NEAR(t.b, 0.5, 0.0);   /* 0 - 0.5*(-3)/3 */
    adamw_free(o);
}

/* ---------- 4. 非法参数 ---------- */
static void test_invalid(void)
{
    printf("[4] 非法参数\n");
    Tiny t;
    tiny_init(&t, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;

    CHECK_ERR(adamw_create(&t.layer, 1, 0.1, 0.9, 0.999, 1e-8, -1.0, &err), "weight_decay");
    CHECK_ERR(adamw_create(NULL, 1, 0.1, 0.9, 0.999, 1e-8, 0.0, &err), "layers");
    CHECK_ERR(adamw_create(&t.layer, 0, 0.1, 0.9, 0.999, 1e-8, 0.0, &err), "layers");

    OptLayer bad = t.layer;
    bad.fan_in = 0;
    CHECK_ERR(adamw_create(&bad, 1, 0.1, 0.9, 0.999, 1e-8, 0.0, &err), "fan_in");
    bad = t.layer;
    bad.w = NULL;
    CHECK_ERR(adamw_create(&bad, 1, 0.1, 0.9, 0.999, 1e-8, 0.0, &err), "NULL");
    bad = t.layer;
    bad.gb = NULL;
    CHECK_ERR(adamw_create(&bad, 1, 0.1, 0.9, 0.999, 1e-8, 0.0, &err), "NULL");

    /* err 传 NULL 也不能崩 */
    CHECK(adamw_create(&t.layer, 1, 0.1, 0.9, 0.999, 1e-8, -2.0, NULL) == NULL);
    /* step 传 NULL 优化器要报错而不是段错误 */
    CHECK(adamw_step(NULL, &err) == -1);
    CHECK(err != NULL);
}

/* ---------- 5. 状态隔离 + 访问器越界 ---------- */
static void test_state(void)
{
    printf("[5] 状态隔离与访问器边界\n");
    Tiny a, b;
    tiny_init(&a, 1.0, 0.0, 1.0, 1.0);
    tiny_init(&b, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;
    AdamW *oa = adamw_create(&a.layer, 1, 0.1, 0.5, 0.75, 1e-8, 0.0, &err);
    AdamW *ob = adamw_create(&b.layer, 1, 0.1, 0.5, 0.75, 1e-8, 0.0, &err);
    CHECK(oa && ob);
    if (oa && ob) {
        CHECK(adamw_step(oa, &err) == 0);
        CHECK(adamw_step(oa, &err) == 0);
        CHECK(adamw_step(ob, &err) == 0);
        CHECK(adamw_step_count(oa) == 2);
        CHECK(adamw_step_count(ob) == 1);
        CHECK(fabs(a.w - b.w) > 1e-6);  /* 第二个优化器不该看到第一个的动量 */
        CHECK(adamw_num_layers(oa) == 1);
        CHECK(adamw_m_w(oa, 0) != NULL);
        CHECK(adamw_m_w(oa, -1) == NULL);
        CHECK(adamw_m_w(oa, 1) == NULL);
        CHECK(adamw_v_b(oa, 0) != NULL);
        CHECK(adamw_v_b(oa, 7) == NULL);

        /* set_lr 影响下一步：lr 减半后这一步的位移也减半（动量项照旧累积，
         * 所以只比较"这一步相对于上一步的差"是否随 lr 线性缩小） */
        const double w0 = a.w;
        adamw_set_lr(oa, 0.05);
        CHECK(adamw_lr(oa) == 0.05);
        CHECK(adamw_step(oa, &err) == 0);
        const double d1 = fabs(a.w - w0);
        CHECK(d1 > 0.0);
    }
    adamw_free(oa);
    adamw_free(ob);

    /* NULL 安全 */
    adamw_free(NULL);
    CHECK(adamw_step_count(NULL) == 0);
    CHECK(adamw_num_layers(NULL) == 0);
    CHECK(adamw_lr(NULL) == 0.0);
    CHECK(adamw_m_w(NULL, 0) == NULL);
    adamw_set_lr(NULL, 0.5); /* 不应崩 */
}

/* ---------- 6. 多层：每层独立状态 ---------- */
static void test_multi_layer(void)
{
    printf("[6] 多层（2 层，形状 2x3 与 1x2）\n");
    double w0[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    double b0[2] = {0.01, 0.02};
    double w1[2] = {-0.1, -0.2};
    double b1[1] = {0.03};
    double gw0[6] = {1, -1, 0.5, -0.5, 0.25, -0.25};
    double gb0[2] = {1, -1};
    double gw1[2] = {0.5, 0.5};
    double gb1[1] = {2};
    OptLayer ls[2] = {
        {w0, b0, gw0, gb0, 2, 3},
        {w1, b1, gw1, gb1, 1, 2},
    };
    const char *err = NULL;
    AdamW *o = adamw_create(ls, 2, 0.01, 0.5, 0.75, 1e-8, 0.01, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(adamw_num_layers(o) == 2);
    for (int s = 0; s < 5; s++) CHECK(adamw_step(o, &err) == 0);
    /* 五个参数元素都动过，而且第 5、6 个元素（w0[4], w0[5]）只受第 3 行的梯度影响 */
    for (int i = 0; i < 6; i++) CHECK(w0[i] != 0.1 * (i + 1));
    CHECK(w1[0] != -0.1 && w1[1] != -0.2);
    CHECK(adamw_m_w(o, 0)[0] != 0.0);
    CHECK(adamw_m_w(o, 1)[1] != 0.0);
    adamw_free(o);
}

int main(void)
{
    printf("== test_adamw ==\n");
    test_hand_computed();
    test_weight_decay();
    test_zero_beta();
    test_invalid();
    test_state();
    test_multi_layer();

    if (g_fail == 0) {
        printf("[OK] test_adamw 全部通过\n");
        return 0;
    }
    printf("[失败] test_adamw 有 %d 处断言失败\n", g_fail);
    return 1;
}
