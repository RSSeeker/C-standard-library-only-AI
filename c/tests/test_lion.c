/*
 * test_lion.c —— Lion 的单元测试（数值 + 非法参数）
 *
 * 用法（在 c/ 目录）：./tests/test_lion.exe
 * 逐位对拍在 compare_lion.c（读 tools/lion_reference.py 生成的 JSON）；
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

/* 要求 create 失败且错误串包含 sub；依赖调用点作用域里的 err 变量 */
#define CHECK_ERR(call, sub)                                                 \
    do {                                                                     \
        const Lion *_p = (call);                                             \
        if (_p) {                                                            \
            printf("  FAIL %s:%d  %s 本应失败却成功了\n", __FILE__,          \
                   __LINE__, #call);                                         \
            g_fail++;                                                        \
            lion_free((Lion *)_p);                                           \
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

/* ---------- 1. 手算：符号动量让每步正好走 lr ---------- */
static void test_hand_computed(void)
{
    printf("[1] 手算（b1=0.5, b2=0.5, g=1, lr=0.1, wd=0）\n");
    Tiny t;
    tiny_init(&t, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;
    Lion *o = lion_create(&t.layer, 1, 0.1, 0.5, 0.5, 0.0, &err);
    CHECK(o != NULL);
    if (!o) return;

    /* t=1: c = 0.5*0 + 0.5*1 = 0.5 -> sign = 1 -> w -= 0.1 */
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.9, 0.0);
    CHECK_NEAR(t.b, -0.1, 0.0);        /* b 从 0 起 */
    CHECK_NEAR(lion_m_w(o, 0)[0], 0.5, 0.0);
    CHECK_NEAR(lion_m_b(o, 0)[0], 0.5, 0.0);
    CHECK(lion_step_count(o) == 1);

    /* t=2: c = 0.5*0.5 + 0.5 = 0.75 -> 还是 sign = 1 */
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.8, 1e-16);
    CHECK_NEAR(t.b, -0.2, 1e-16);
    CHECK_NEAR(lion_m_w(o, 0)[0], 0.75, 0.0);
    CHECK(lion_step_count(o) == 2);
    lion_free(o);

    /* 负梯度：符号翻转，参数加回去 */
    tiny_init(&t, 1.0, 0.0, -1.0, -1.0);
    o = lion_create(&t.layer, 1, 0.1, 0.5, 0.5, 0.0, &err);
    CHECK(o != NULL);
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 1.1, 0.0);
    CHECK_NEAR(t.b, 0.1, 0.0);
    lion_free(o);
}

/* ---------- 2. sign(0) == 0：c 恰好为 0 时更新量只剩 wd*p ---------- */
static void test_zero_sign(void)
{
    printf("[2] sign(0)==0 与 wd 作用在更新前的参数上\n");
    Tiny t;
    tiny_init(&t, 1.0, 2.0, 0.0, 0.0);   /* 梯度为 0，动量初始为 0 -> c == 0 */
    const char *err = NULL;
    Lion *o = lion_create(&t.layer, 1, 0.5, 0.9, 0.99, 0.1, &err);
    CHECK(o != NULL);
    if (!o) return;
    /* upd = sign(0) + 0.1*1.0 = 0.1 -> w = 1 - 0.5*0.1 = 0.95（精确） */
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.95, 0.0);
    CHECK_NEAR(t.b, 1.9, 0.0);
    CHECK_NEAR(lion_m_w(o, 0)[0], 0.0, 0.0);
    lion_free(o);

    /* wd=0 且 c==0：参数一动不动（只有 sign(c)=0 且没有衰减项） */
    tiny_init(&t, 1.0, 1.0, 0.0, 0.0);
    o = lion_create(&t.layer, 1, 0.5, 0.9, 0.99, 0.0, &err);
    CHECK(o != NULL);
    for (int s = 0; s < 5; s++) CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 1.0, 0.0);
    CHECK_NEAR(t.b, 1.0, 0.0);
    lion_free(o);

    /* 衰减用的是"更新前"的 w：dyadic 数值下可以精确断言
     * step1: upd = 0.5*1.0 = 0.5 -> w = 1 - 0.25*0.5 = 0.875
     * step2: upd = 0.5*0.875 = 0.4375 -> w = 0.875 - 0.25*0.4375 = 0.765625
     * 若误用更新后的 w（0.75），step2 会得到 0.78125，差得很明显。 */
    tiny_init(&t, 1.0, 0.0, 0.0, 0.0);
    o = lion_create(&t.layer, 1, 0.25, 0.9, 0.99, 0.5, &err);
    CHECK(o != NULL);
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.875, 0.0);
    CHECK(lion_step(o, &err) == 0);
    CHECK_NEAR(t.w, 0.765625, 0.0);
    lion_free(o);
}

/* ---------- 3. 非法参数 ---------- */
static void test_invalid(void)
{
    printf("[3] 非法参数\n");
    Tiny t;
    tiny_init(&t, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;

    CHECK_ERR(lion_create(&t.layer, 1, 0.1, -0.1, 0.99, 0.0, &err), "beta1");
    CHECK_ERR(lion_create(&t.layer, 1, 0.1, 1.1, 0.99, 0.0, &err), "beta1");
    CHECK_ERR(lion_create(&t.layer, 1, 0.1, 0.9, -0.1, 0.0, &err), "beta2");
    CHECK_ERR(lion_create(&t.layer, 1, 0.1, 0.9, 2.0, 0.0, &err), "beta2");
    CHECK_ERR(lion_create(&t.layer, 1, 0.1, 0.9, 0.99, -0.1, &err), "weight_decay");
    CHECK_ERR(lion_create(NULL, 1, 0.1, 0.9, 0.99, 0.0, &err), "layers");
    CHECK_ERR(lion_create(&t.layer, 0, 0.1, 0.9, 0.99, 0.0, &err), "layers");

    OptLayer bad = t.layer;
    bad.fan_out = 0;
    CHECK_ERR(lion_create(&bad, 1, 0.1, 0.9, 0.99, 0.0, &err), "fan_out");
    bad = t.layer;
    bad.b = NULL;
    CHECK_ERR(lion_create(&bad, 1, 0.1, 0.9, 0.99, 0.0, &err), "NULL");
    bad = t.layer;
    bad.gw = NULL;
    CHECK_ERR(lion_create(&bad, 1, 0.1, 0.9, 0.99, 0.0, &err), "NULL");

    /* 边界值必须合法：beta 取 0 和 1 都允许 */
    Lion *edge = lion_create(&t.layer, 1, 0.1, 0.0, 1.0, 0.0, &err);
    CHECK(edge != NULL);
    lion_free(edge);

    CHECK(lion_create(&t.layer, 1, 0.1, -1e-9, 0.99, 0.0, NULL) == NULL);
    CHECK(lion_step(NULL, &err) == -1);
    CHECK(err != NULL);
}

/* ---------- 4. 状态隔离 + 访问器 ---------- */
static void test_state(void)
{
    printf("[4] 状态隔离与访问器边界\n");
    Tiny a, b;
    tiny_init(&a, 1.0, 0.0, 1.0, 1.0);
    tiny_init(&b, 1.0, 0.0, 1.0, 1.0);
    const char *err = NULL;
    Lion *oa = lion_create(&a.layer, 1, 0.1, 0.5, 0.5, 0.0, &err);
    Lion *ob = lion_create(&b.layer, 1, 0.1, 0.5, 0.5, 0.0, &err);
    CHECK(oa && ob);
    if (oa && ob) {
        CHECK(lion_step(oa, &err) == 0);
        CHECK(lion_step(oa, &err) == 0);
        CHECK(lion_step(ob, &err) == 0);
        CHECK(lion_step_count(oa) == 2);
        CHECK(lion_step_count(ob) == 1);
        CHECK(fabs(a.w - b.w) > 1e-6);
        CHECK(lion_num_layers(oa) == 1);
        CHECK(lion_weight_decay(oa) == 0.0);
        CHECK(lion_beta1(oa) == 0.5);
        CHECK(lion_beta2(oa) == 0.5);
        CHECK(lion_m_w(oa, 0) != NULL);
        CHECK(lion_m_b(oa, 0) != NULL);
        CHECK(lion_m_w(oa, -1) == NULL);
        CHECK(lion_m_w(oa, 1) == NULL);
        CHECK(lion_m_b(oa, 9) == NULL);

        lion_set_lr(oa, 0.5);
        CHECK(lion_lr(oa) == 0.5);
        const double w0 = a.w;
        CHECK(lion_step(oa, &err) == 0);
        CHECK_NEAR(a.w, w0 - 0.5 * 1.0, 0.0); /* c 仍是正的 -> 走满 0.5 */
    }
    lion_free(oa);
    lion_free(ob);

    lion_free(NULL);
    CHECK(lion_step_count(NULL) == 0);
    CHECK(lion_num_layers(NULL) == 0);
    CHECK(lion_lr(NULL) == 0.0);
    CHECK(lion_m_w(NULL, 0) == NULL);
    lion_set_lr(NULL, 0.5);
}

/* ---------- 5. 多层 ---------- */
static void test_multi_layer(void)
{
    printf("[5] 多层（2 层，形状 2x3 与 1x2）\n");
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
    Lion *o = lion_create(ls, 2, 0.01, 0.5, 0.5, 0.01, &err);
    CHECK(o != NULL);
    if (!o) return;
    CHECK(lion_num_layers(o) == 2);
    for (int s = 0; s < 7; s++) CHECK(lion_step(o, &err) == 0);
    /* 每一行都动过：sign 更新量是 ±lr（再叠加衰减项） */
    for (int i = 0; i < 6; i++) {
        const double before = 0.1 * (i + 1);
        CHECK(fabs(w0[i] - before) > 0.0);
    }
    CHECK(w1[0] != -0.1 && w1[1] != -0.2);
    CHECK(lion_m_w(o, 0)[0] != 0.0);
    CHECK(lion_m_w(o, 1)[1] != 0.0);
    CHECK(lion_step_count(o) == 7);
    lion_free(o);
}

int main(void)
{
    printf("== test_lion ==\n");
    test_hand_computed();
    test_zero_sign();
    test_invalid();
    test_state();
    test_multi_layer();

    if (g_fail == 0) {
        printf("[OK] test_lion 全部通过\n");
        return 0;
    }
    printf("[失败] test_lion 有 %d 处断言失败\n", g_fail);
    return 1;
}
