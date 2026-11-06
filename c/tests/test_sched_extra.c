/*
 * test_sched_extra.c —— WarmupLR / ReduceLROnPlateau 的单元测试
 *
 * 用法（在 c/ 目录）：./tests/test_sched_extra.exe
 * 逐位对拍在 compare_sched_extra.c；这里测手算值、回调时序与错误路径。
 */
#include "../sched.h"

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

/* ---------- 1. 线性 warmup 的手算 ---------- */
static void test_warmup_linear(void)
{
    printf("[1] 线性 warmup（initial=1, warmup=4, start=0）\n");
    double lr = 1.0;
    const char *err = NULL;
    WarmupLR *w = warmuplr_create(&lr, 4, 0.0, &err);
    CHECK(w != NULL);
    if (!w) return;
    CHECK(warmuplr_initial_lr(w) == 1.0);
    CHECK(warmuplr_warmup_epochs(w) == 4);
    CHECK(warmuplr_start_factor(w) == 0.0);

    CHECK_NEAR(warmuplr_step(w, 1), 0.25, 0.0);
    CHECK_NEAR(warmuplr_step(w, 2), 0.5, 0.0);
    CHECK_NEAR(warmuplr_step(w, 3), 0.75, 0.0);
    CHECK_NEAR(warmuplr_step(w, 4), 1.0, 0.0);
    CHECK_NEAR(warmuplr_step(w, 5), 1.0, 0.0);   /* warmup 结束后保持 initial */
    CHECK_NEAR(lr, 1.0, 0.0);
    warmuplr_free(w);

    /* start_factor=1：全程停在 initial */
    lr = 0.5;
    w = warmuplr_create(&lr, 3, 1.0, &err);
    CHECK(w != NULL);
    for (int ep = 1; ep <= 5; ep++) CHECK_NEAR(warmuplr_step(w, ep), 0.5, 0.0);
    warmuplr_free(w);

    /* epoch <= 0 按 1 处理 */
    lr = 0.8;
    w = warmuplr_create(&lr, 2, 0.0, &err);
    CHECK(w != NULL);
    CHECK_NEAR(warmuplr_step(w, 0), 0.4, 0.0);    /* 0 -> 1 -> 0.8*(0+0.5) */
    CHECK_NEAR(warmuplr_step(w, -7), 0.4, 0.0);
    CHECK_NEAR(warmuplr_step(w, 2), 0.8, 0.0);
    warmuplr_free(w);
}

/* ---------- 2. base_scheduler 的调用时序 ---------- */
typedef struct {
    double *lr;
    double base_lr;
    int epochs[8];
    int n;
} BaseCtx;

static void base_halve(void *ctx, int epoch)
{
    BaseCtx *b = (BaseCtx *)ctx;
    b->epochs[b->n++] = epoch;
    double v = b->base_lr;
    for (int i = 0; i < epoch; i++) v *= 0.5;
    *b->lr = v;
}

static void test_base_scheduler(void)
{
    printf("[2] base_scheduler 只在 warmup 之后被调用，且收到绝对 epoch\n");
    double lr = 2.0;
    BaseCtx base;
    memset(&base, 0, sizeof base);
    base.lr = &lr;
    base.base_lr = 2.0;
    const char *err = NULL;
    WarmupLR *w = warmuplr_create(&lr, 2, 0.0, &err);
    CHECK(w != NULL);
    if (!w) return;
    warmuplr_set_base(w, base_halve, &base);

    CHECK_NEAR(warmuplr_step(w, 1), 1.0, 0.0);   /* warmup 中：2*0.5 */
    CHECK(base.n == 0);
    CHECK_NEAR(warmuplr_step(w, 2), 2.0, 0.0);   /* warmup 末：2*1.0 */
    CHECK(base.n == 0);
    CHECK_NEAR(warmuplr_step(w, 3), 0.25, 0.0);  /* base: 2*0.5^3 */
    CHECK(base.n == 1 && base.epochs[0] == 3);
    CHECK_NEAR(warmuplr_step(w, 4), 0.125, 0.0);
    CHECK(base.n == 2 && base.epochs[1] == 4);

    /* 取消 base：回到 initial_lr 就停 */
    warmuplr_set_base(w, NULL, NULL);
    CHECK_NEAR(warmuplr_step(w, 5), 2.0, 0.0);
    CHECK(base.n == 2);
    warmuplr_free(w);
}

/* ---------- 3. ReduceLROnPlateau: min 模式 ---------- */
static void test_plateau_min(void)
{
    printf("[3] plateau min 模式（patience=1）\n");
    double lr = 1.0;
    const char *err = NULL;
    ReduceLROnPlateau *p = plateau_create(&lr, 0.5, 1, 0.0, PLATEAU_MIN, 1e-4, -1, &err);
    CHECK(p != NULL);
    if (!p) return;
    CHECK(plateau_initial_lr(p) == 1.0);
    CHECK(!plateau_has_best(p));
    CHECK(isnan(plateau_best(p)));

    /* 第一个指标只记录 best */
    CHECK_NEAR(plateau_step(p, 1.0), 1.0, 0.0);
    CHECK(plateau_has_best(p) && plateau_best(p) == 1.0);
    CHECK(plateau_num_bad(p) == 0);

    /* 改善（相对 threshold 真的变小）：best 更新，num_bad 清零 */
    CHECK_NEAR(plateau_step(p, 0.5), 1.0, 0.0);
    CHECK(plateau_best(p) == 0.5);
    CHECK(plateau_num_bad(p) == 0);

    /* 第 1 次没改善：num_bad=1，还没到 patience+1 */
    CHECK_NEAR(plateau_step(p, 0.5), 1.0, 0.0);
    CHECK(plateau_num_bad(p) == 1);
    CHECK(plateau_reductions(p) == 0);

    /* 第 2 次没改善 -> num_bad=2 > patience=1 -> lr *= 0.5 */
    CHECK_NEAR(plateau_step(p, 0.5), 0.5, 0.0);
    CHECK(plateau_num_bad(p) == 0);
    CHECK(plateau_reductions(p) == 1);

    /* 任意大的改善都会重置 */
    CHECK_NEAR(plateau_step(p, 0.1), 0.5, 0.0);
    CHECK(plateau_best(p) == 0.1);
    CHECK(plateau_num_bad(p) == 0);
    plateau_free(p);
}

/* ---------- 4. max 模式 + min_lr 下限 + max_reductions ---------- */
static void test_plateau_limits(void)
{
    printf("[4] plateau max 模式 / min_lr 下限 / max_reductions\n");
    double lr = 1.0;
    const char *err = NULL;
    ReduceLROnPlateau *p = plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MAX, 1e-4, -1, &err);
    CHECK(p != NULL);
    if (!p) return;
    CHECK(plateau_mode(p) == PLATEAU_MAX);
    CHECK_NEAR(plateau_step(p, 0.5), 1.0, 0.0);  /* best=0.5 */
    CHECK_NEAR(plateau_step(p, 0.6), 1.0, 0.0);  /* 变大 = 改善 */
    CHECK(plateau_best(p) == 0.6);
    CHECK_NEAR(plateau_step(p, 0.6), 0.5, 0.0);  /* patience=0 -> 立刻降 */
    CHECK(plateau_reductions(p) == 1);
    plateau_free(p);

    /* min_lr 下限：lr 不会低于 0.3 */
    lr = 1.0;
    p = plateau_create(&lr, 0.5, 0, 0.3, PLATEAU_MIN, 1e-4, -1, &err);
    CHECK(p != NULL);
    if (p) {
        for (int i = 0; i < 6; i++) plateau_step(p, 1.0);
        CHECK_NEAR(lr, 0.3, 0.0);
        CHECK_NEAR(plateau_lr(p), 0.3, 0.0);
        plateau_free(p);
    }

    /* max_reductions=1：只降一次 */
    lr = 1.0;
    p = plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, 1, &err);
    CHECK(p != NULL);
    if (p) {
        CHECK(plateau_max_reductions(p) == 1);
        for (int i = 0; i < 8; i++) plateau_step(p, 1.0);
        CHECK_NEAR(lr, 0.5, 0.0);
        CHECK(plateau_reductions(p) == 1);
        plateau_free(p);
    }
}

/* ---------- 5. NaN 指标 ---------- */
static void test_plateau_nan(void)
{
    printf("[5] NaN 指标不参与比较\n");
    double lr = 1.0;
    const char *err = NULL;
    ReduceLROnPlateau *p = plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, -1, &err);
    CHECK(p != NULL);
    if (!p) return;
    plateau_step(p, 1.0);
    CHECK(plateau_num_bad(p) == 0);
    CHECK_NEAR(plateau_step(p, (double)NAN), 1.0, 0.0);
    CHECK(plateau_num_bad(p) == 0);      /* NaN 既不 +1 也不降 lr */
    CHECK(plateau_reductions(p) == 0);
    CHECK(plateau_best(p) == 1.0);
    CHECK_NEAR(plateau_step(p, 1.0), 0.5, 0.0);   /* patience=0 -> 第一次停滞就降 */
    plateau_free(p);
}

/* ---------- 6. 非法参数 ---------- */
static void test_invalid(void)
{
    printf("[6] 非法参数\n");
    double lr = 1.0;
    const char *err = NULL;

    CHECK(warmuplr_create(NULL, 5, 0.0, &err) == NULL);
    CHECK(err && strstr(err, "lr"));
    err = NULL;
    CHECK(warmuplr_create(&lr, 0, 0.0, &err) == NULL);
    CHECK(err && strstr(err, "warmup_epochs"));
    err = NULL;
    CHECK(warmuplr_create(&lr, 3, -0.1, &err) == NULL);
    CHECK(err && strstr(err, "start_factor"));
    err = NULL;
    CHECK(warmuplr_create(&lr, 3, 1.5, &err) == NULL);
    CHECK(err && strstr(err, "start_factor"));
    err = NULL;
    /* 边界值合法 */
    WarmupLR *w = warmuplr_create(&lr, 1, 1.0, &err);
    CHECK(w != NULL);
    warmuplr_free(w);
    /* 非法时不应改写 lr */
    lr = 0.7;
    CHECK(warmuplr_create(&lr, 0, 0.0, &err) == NULL);
    CHECK(lr == 0.7);

    err = NULL;
    CHECK(plateau_create(NULL, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "lr"));
    err = NULL;
    CHECK(plateau_create(&lr, 0.0, 0, 0.0, PLATEAU_MIN, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "factor"));
    err = NULL;
    CHECK(plateau_create(&lr, 1.0, 0, 0.0, PLATEAU_MIN, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "factor"));
    err = NULL;
    CHECK(plateau_create(&lr, 0.5, 0, 0.0, (PlateauMode)7, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "mode"));
    err = NULL;
    CHECK(plateau_create(&lr, 0.5, -1, 0.0, PLATEAU_MIN, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "patience"));
    err = NULL;
    CHECK(plateau_create(&lr, 0.5, 0, -0.1, PLATEAU_MIN, 1e-4, -1, &err) == NULL);
    CHECK(err && strstr(err, "min_lr"));
    err = NULL;
    CHECK(plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, -2, &err) == NULL);
    CHECK(err && strstr(err, "max_reductions"));
    err = NULL;

    /* -1 与 0 都是合法的 max_reductions */
    ReduceLROnPlateau *p = plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, 0, &err);
    CHECK(p != NULL);
    plateau_free(p);
    CHECK(plateau_create(&lr, 0.5, 0, 0.0, PLATEAU_MIN, 1e-4, -1, NULL) != NULL);

    /* NULL 接收者的行为 */
    CHECK(warmuplr_step(NULL, 3) == 0.0);
    warmuplr_set_base(NULL, NULL, NULL);
    warmuplr_free(NULL);
    CHECK(plateau_step(NULL, 1.0) == 0.0);
    plateau_free(NULL);
    CHECK(plateau_has_best(NULL) == 0);
}

int main(void)
{
    printf("== test_sched_extra ==\n");
    test_warmup_linear();
    test_base_scheduler();
    test_plateau_min();
    test_plateau_limits();
    test_plateau_nan();
    test_invalid();

    if (g_fail == 0) {
        printf("[OK] test_sched_extra 全部通过\n");
        return 0;
    }
    printf("[失败] test_sched_extra 有 %d 处断言失败\n", g_fail);
    return 1;
}
