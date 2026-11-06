/*
 * test_activations.c —— SiLU / Mish / ELU / SELU 单元测试
 *
 * 运行：./tests/test_activations（不需要参数；make test 会用 "self" 调一次也行）
 *
 * 期望值全部是在 CPython 3.14 上调 AI.py 的 *_scalar 得到的字面量。
 * 比较用相对误差 1e-15（除了明确要求逐位的边界值，如 0、±1、-0.0、inf 附近），
 * 因为 C 侧 libm 与 Python 的 UCRT 在个别输入上会差 1 ULP。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../activations.h"
#include "../ai.h"

static int g_fail;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("[FAIL] %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(got, want, tol)                                           \
    do {                                                                     \
        double _g = (got), _w = (want);                                      \
        double _d = fabs(_g - _w);                                           \
        double _s = fabs(_w) > 1.0 ? fabs(_w) : 1.0;                         \
        if (!(_d <= (tol) * _s)) {                                           \
            printf("[FAIL] %s:%d  %s = %.17g，期望 %.17g（差 %.3e）\n",       \
                   __FILE__, __LINE__, #got, _g, _w, _d);                    \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_BITS(got, want)                                                \
    do {                                                                     \
        double _g = (got), _w = (want);                                      \
        if (memcmp(&_g, &_w, sizeof(double)) != 0) {                         \
            printf("[FAIL] %s:%d  %s = %.17g，期望逐位 %.17g\n",              \
                   __FILE__, __LINE__, #got, _g, _w);                        \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

static void sec_boundaries(void)
{
    printf("[1] 分段边界（z=0 走 silu 的 >= 分支、elu/selu 的 exp 分支）\n");
    CHECK_BITS(silu_scalar(0.0), 0.0);
    CHECK_BITS(dsilu_scalar(0.0), 0.5);          /* sigmoid(0) * (1 + 0) */
    CHECK_BITS(mish_scalar(0.0), 0.0);
    CHECK_NEAR(dmish_scalar(0.0), 0.6, 1e-15);   /* tanh(log 2) */
    CHECK_BITS(elu_scalar(0.0), 0.0);            /* exp(0)-1 == 0 */
    CHECK_BITS(delu_scalar(0.0), 1.0);
    CHECK_BITS(selu_scalar(0.0), 0.0);
    CHECK_NEAR(dselu_scalar(0.0), 1.7580993408473766, 1e-15); /* lambda*alpha */
    CHECK_NEAR(softplus_scalar(0.0), 0.6931471805599453, 1e-15);

    /* -0.0 也算 >= 0：silu(-0.0) 走除法分支，结果保留符号位 */
    double mz = silu_scalar(-0.0);
    CHECK_BITS(mz, -0.0);
    CHECK(signbit(mz) != 0);
}

static void sec_reference_values(void)
{
    printf("[2] 与 AI.py 的标量值对照（z=1 / z=-1）\n");
    CHECK_NEAR(silu_scalar(1.0), 0.7310585786300049, 1e-15);
    CHECK_NEAR(silu_scalar(-1.0), -0.2689414213699951, 1e-15);
    CHECK_NEAR(dsilu_scalar(1.0), 0.9276705118714869, 1e-15);
    CHECK_NEAR(dsilu_scalar(-1.0), 0.07232948812851325, 1e-15);
    CHECK_NEAR(mish_scalar(1.0), 0.8650983882673103, 1e-15);
    CHECK_NEAR(mish_scalar(-1.0), -0.30340146137410895, 1e-15);
    CHECK_NEAR(dmish_scalar(1.0), 1.0490362200997922, 1e-15);
    CHECK_NEAR(dmish_scalar(-1.0), 0.059216755877395, 1e-15);
    CHECK_NEAR(elu_scalar(1.0), 1.0, 0.0);
    CHECK_NEAR(elu_scalar(-1.0), -0.6321205588285577, 1e-15);
    CHECK_NEAR(delu_scalar(-1.0), 0.36787944117144233, 1e-15);
    CHECK_NEAR(selu_scalar(1.0), 1.0507009873554805, 1e-15);
    CHECK_NEAR(selu_scalar(-1.0), -1.1113307378125625, 1e-15);
    CHECK_NEAR(dselu_scalar(-1.0), 0.646768603034814, 1e-15);
    CHECK_NEAR(softplus_scalar(1.0), 1.3132616875182228, 1e-15);
    CHECK_NEAR(softplus_scalar(-1.0), 0.31326168751822286, 1e-15);
}

static void sec_saturation(void)
{
    printf("[3] 大负数饱和（exp 下溢到 0）与 softplus 的 30 阈值\n");
    /* exp(-800) 下溢为 0：elu -> -1，selu -> -lambda*alpha，导数为 0 */
    CHECK_BITS(elu_scalar(-800.0), -1.0);
    CHECK_NEAR(selu_scalar(-800.0), -1.7580993408473766, 0.0);
    CHECK_BITS(delu_scalar(-800.0), 0.0);
    CHECK_BITS(dselu_scalar(-800.0), 0.0);
    CHECK_BITS(softplus_scalar(-800.0), 0.0);
    /* silu(-800) = -800*0/(1+0) = -0.0 */
    CHECK_BITS(silu_scalar(-800.0), -0.0);
    CHECK_BITS(mish_scalar(-800.0), -0.0);
    CHECK_BITS(dmish_scalar(-800.0), 0.0);
    CHECK_BITS(dsilu_scalar(-800.0), -0.0);

    /* z > 30 直接返回 z；z == 30 仍走 log1p(exp(30)) ≈ 30 */
    CHECK_BITS(softplus_scalar(30.1), 30.1);
    CHECK_BITS(softplus_scalar(1000.0), 1000.0);
    CHECK(softplus_scalar(30.0) >= 30.0 && softplus_scalar(30.0) < 30.0000001);
    /* 正半轴：elu/selu 恒等映射，dselu 为常数 */
    CHECK_BITS(elu_scalar(3.5), 3.5);
    CHECK_BITS(delu_scalar(3.5), 1.0);
    CHECK_NEAR(selu_scalar(3.5), 1.0507009873554805 * 3.5, 0.0);
    CHECK_BITS(dselu_scalar(3.5), 1.0507009873554805);
}

static void sec_vectors(void)
{
    printf("[4] 向量版本逐元素等于标量版本\n");
    const double z[] = {-800.0, -3.0, -1.0, -1e-8, 0.0, 1e-8, 0.5, 1.0, 2.0, 30.0, 31.0};
    const int n = (int)(sizeof(z) / sizeof(z[0]));
    double out[16];

#define CHECK_VEC(fn_vec, fn_scalar)                                          \
    do {                                                                      \
        memset(out, 0, sizeof out);                                           \
        fn_vec(z, out, n);                                                    \
        for (int i = 0; i < n; i++) {                                         \
            double want = fn_scalar(z[i]);                                    \
            if (memcmp(&out[i], &want, sizeof(double)) != 0) {                 \
                printf("[FAIL] %s:%d  %s[%d] = %.17g，标量版 %.17g\n",         \
                       __FILE__, __LINE__, #fn_vec, i, out[i], want);          \
                g_fail++;                                                     \
            }                                                                 \
        }                                                                     \
    } while (0)

    CHECK_VEC(silu_vec, silu_scalar);
    CHECK_VEC(dsilu_vec, dsilu_scalar);
    CHECK_VEC(mish_vec, mish_scalar);
    CHECK_VEC(dmish_vec, dmish_scalar);
    CHECK_VEC(elu_vec, elu_scalar);
    CHECK_VEC(delu_vec, delu_scalar);
    CHECK_VEC(selu_vec, selu_scalar);
    CHECK_VEC(dselu_vec, dselu_scalar);
    CHECK_VEC(softplus_vec, softplus_scalar);

    /* n <= 0 必须是无操作（且不崩） */
    out[0] = 123.0;
    silu_vec(z, out, 0);
    silu_vec(z, out, -5);
    CHECK_BITS(out[0], 123.0);
}

static void sec_monotone(void)
{
    printf("[5] 数值性质：silu/mish 在 z>=0 上严格递增、selu 在 0 处连续\n");
    /* 注意 silu/mish 在整个实数轴上并不单调：两者在 z≈-1.28 处取最小值，
     * 所以只在 z>=0 上断言严格递增。 */
    double prev = silu_scalar(0.0);
    for (int i = 1; i <= 50; i++) {
        double z = i * 0.1;
        double v = silu_scalar(z);
        CHECK(v > prev);
        prev = v;
    }
    prev = mish_scalar(0.0);
    for (int i = 1; i <= 50; i++) {
        double z = i * 0.1;
        double v = mish_scalar(z);
        CHECK(v > prev);
        prev = v;
    }
    /* selu 在 0 两侧极限都是 0（严格 > 判据下 0 走 exp 分支得 0） */
    CHECK_NEAR(selu_scalar(1e-12), 1.0507009873554805e-12, 1e-12);
    CHECK_NEAR(selu_scalar(-1e-12), -1.7580993408473766e-12, 1e-12);
}

static void sec_mlp_integration(void)
{
    printf("[6] MLP 的 act_apply/act_deriv 已接入新激活\n");
    const double z[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
    const int n = (int)(sizeof(z) / sizeof(z[0]));
    double out[5];
    double want[5];

    act_apply(ACT_SILU, z, out, n);
    silu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);
    act_deriv(ACT_SILU, z, out, out, n);   /* act_deriv 只用 z，a 传什么都不影响 */
    dsilu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);

    act_apply(ACT_MISH, z, out, n);
    mish_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);
    act_deriv(ACT_MISH, z, out, out, n);
    dmish_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);

    act_apply(ACT_ELU, z, out, n);
    elu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);
    act_deriv(ACT_ELU, z, out, out, n);
    delu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);

    act_apply(ACT_SELU, z, out, n);
    selu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);
    act_deriv(ACT_SELU, z, out, out, n);
    dselu_vec(z, want, n);
    CHECK(memcmp(out, want, sizeof want) == 0);
}

int main(int argc, char **argv)
{
    /* make test 会以 "self" 作为参数调用，这里不需要分派模式 */
    (void)argc;
    (void)argv;

    printf("== test_activations ==\n");
    sec_boundaries();
    sec_reference_values();
    sec_saturation();
    sec_vectors();
    sec_monotone();
    sec_mlp_integration();

    if (g_fail == 0) {
        printf("[OK] test_activations 全部通过\n");
        return 0;
    }
    printf("[FAIL] test_activations 有 %d 处失败\n", g_fail);
    return 1;
}
