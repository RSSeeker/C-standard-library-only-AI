/*
 * test_rmsnorm.c —— RMSNorm 的单元测试
 *
 * 用法（在 c/ 目录）：./tests/test_rmsnorm.exe
 * 跨语言逐位对拍在 compare_rmsnorm.c；这里测手算值、语义细节与错误路径。
 *
 * RMSNorm 只有 sqrt 与四则运算，Python 与 C 的运算顺序一致时结果必须逐位相同，
 * 所以下面用 memcmp 判等（不用容差）。手算期望值是用 AI.py 同一个表达式算出来的
 * 十进制字面量（repr 17 位，可无损往返）。
 */
#include "../layers_norm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

#define CHECK_BITS(got, want)                                                \
    do {                                                                     \
        const double _g = (got), _w = (want);                                \
        if (memcmp(&_g, &_w, sizeof(double)) != 0) {                         \
            printf("  FAIL %s:%d  %s = %.17g 与 %.17g 不是逐位相同（差 %.3e）\n", \
                   __FILE__, __LINE__, #got, _g, _w, fabs(_g - _w));         \
            g_fail++;                                                        \
        }                                                                    \
    } while (0)

/* ---------- 1. 手算：d=2, eps=0 ---------- */
static void test_hand_d2(void)
{
    printf("[1] 手算 d=2（gamma=[2,1], x=[3,4], eps=0）\n");
    RMSNorm *rn = rmsnorm_create(2, 0.0);
    const double gamma[2] = { 2.0, 1.0 };
    rmsnorm_set_params(rn, gamma);
    CHECK(rmsnorm_gamma(rn)[0] == 2.0 && rmsnorm_gamma(rn)[1] == 1.0);
    CHECK(rmsnorm_param_count(rn) == 2);

    const int shape[1] = { 2 };
    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);
    Tensor *dout = tensor_new(1, shape);
    Tensor *dx = tensor_new(1, shape);

    x->data[0] = 3.0;
    x->data[1] = 4.0;
    rmsnorm_forward(rn, x, y);
    /* mean_sq = (9+16)/2 = 12.5，r = 1/sqrt(12.5) */
    CHECK_BITS(rmsnorm_inv_rms(rn), 0.282842712474619);
    CHECK_BITS(y->data[0], 1.697056274847714);
    CHECK_BITS(y->data[1], 1.131370849898476);

    dout->data[0] = 1.0;
    dout->data[1] = 0.0;
    rmsnorm_backward(rn, dout, dx);
    const double *gg = rmsnorm_grad_gamma(rn);
    CHECK_BITS(gg[0], 0.848528137423857);   /* dout*x*r */
    CHECK_BITS(gg[1], 0.0);
    /* g = 1*2*3 + 0 = 6；dx_j = gamma_j*dout_j*r - g*x_j*r^3/2 */
    CHECK_BITS(dx->data[0], 0.3620386719675123);
    CHECK_BITS(dx->data[1], -0.27152900397563423);

    tensor_free(x);
    tensor_free(y);
    tensor_free(dout);
    tensor_free(dx);
    rmsnorm_free(rn);
}

/* ---------- 2. 均匀输入：r = 1/|c| ---------- */
static void test_uniform(void)
{
    printf("[2] 均匀输入（x=[2,2], gamma=[1,1] -> r=0.5，g=0）\n");
    RMSNorm *rn = rmsnorm_create(2, 0.0);
    const int shape[1] = { 2 };
    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);
    Tensor *dout = tensor_new(1, shape);
    Tensor *dx = tensor_new(1, shape);

    x->data[0] = 2.0;
    x->data[1] = 2.0;
    rmsnorm_forward(rn, x, y);
    CHECK_BITS(rmsnorm_inv_rms(rn), 0.5);
    CHECK_BITS(y->data[0], 1.0);
    CHECK_BITS(y->data[1], 1.0);

    /* dout=[1,-1] -> g = 1*1*2 + (-1)*1*2 = 0 -> dx 只剩第一项 */
    dout->data[0] = 1.0;
    dout->data[1] = -1.0;
    rmsnorm_backward(rn, dout, dx);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], 1.0);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[1], -1.0);
    CHECK_BITS(dx->data[0], 0.5);
    CHECK_BITS(dx->data[1], -0.5);

    tensor_free(x);
    tensor_free(y);
    tensor_free(dout);
    tensor_free(dx);
    rmsnorm_free(rn);
}

/* ---------- 3. grad_gamma 是覆盖赋值（不是累加） ---------- */
static void test_grad_overwrite(void)
{
    printf("[3] grad_gamma 覆盖语义 + zero_grad\n");
    RMSNorm *rn = rmsnorm_create(2, 0.0);
    const int shape[1] = { 2 };
    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);
    Tensor *dout = tensor_new(1, shape);
    Tensor *dx = tensor_new(1, shape);
    x->data[0] = 2.0;
    x->data[1] = 2.0;

    rmsnorm_forward(rn, x, y);
    dout->data[0] = 1.0;
    dout->data[1] = 1.0;
    rmsnorm_backward(rn, dout, dx);
    const double first0 = rmsnorm_grad_gamma(rn)[0];
    const double first1 = rmsnorm_grad_gamma(rn)[1];
    CHECK_BITS(first0, 1.0);

    /* 同一层不 zero_grad 再反向一次：Python 是覆盖，值应与刚才完全相同，
     * 而不是变成两倍（这正是与 LayerNorm 的语义差异） */
    dout->data[0] = 1.0;
    dout->data[1] = 1.0;
    rmsnorm_backward(rn, dout, dx);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], first0);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[1], first1);

    /* 换一个 dout：只反映这一次 */
    dout->data[0] = 3.0;
    dout->data[1] = 0.0;
    rmsnorm_backward(rn, dout, dx);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], 3.0);   /* 3*2*0.5 */
    CHECK_BITS(rmsnorm_grad_gamma(rn)[1], 0.0);

    rmsnorm_zero_grad(rn);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], 0.0);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[1], 0.0);

    tensor_free(x);
    tensor_free(y);
    tensor_free(dout);
    tensor_free(dx);
    rmsnorm_free(rn);
}

/* ---------- 4. Module 包装 ---------- */
static void test_module(void)
{
    printf("[4] Module 包装（forward/backward/param_count/to_string）\n");
    RMSNorm *rn = rmsnorm_create(3, 1e-6);
    const double gamma[3] = { 1.0, 2.0, 0.5 };
    rmsnorm_set_params(rn, gamma);
    Module *m = rmsnorm_as_module(rn);
    CHECK(module_param_count(m) == 3);

    char buf[64];
    m->vt->to_string(m, buf, sizeof buf);
    CHECK(strstr(buf, "RMSNorm(3") != NULL);

    const int shape[1] = { 3 };
    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);
    Tensor *dout = tensor_new(1, shape);
    Tensor *dx = tensor_new(1, shape);
    const double xv[3] = { 0.5, -1.5, 2.5 };
    const double dv[3] = { 0.25, 0.5, -0.75 };
    memcpy(x->data, xv, sizeof xv);
    memcpy(dout->data, dv, sizeof dv);

    module_forward(m, x, y);
    CHECK_BITS(rmsnorm_inv_rms(rn), 0.5855399433908525);
    CHECK_BITS(y->data[0], 0.29276997169542623);
    CHECK_BITS(y->data[1], -1.7566198301725575);
    CHECK_BITS(y->data[2], 0.7319249292385656);

    module_backward(m, dout, dx);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], 0.07319249292385656);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[1], -0.4391549575431394);
    CHECK_BITS(rmsnorm_grad_gamma(rn)[2], -1.0978873938578484);
    CHECK_BITS(dx->data[0], 0.2237598804101119);
    CHECK_BITS(dx->data[1], 0.35341525970365606);
    CHECK_BITS(dx->data[2], 0.16729699404042425);

    tensor_free(x);
    tensor_free(y);
    tensor_free(dout);
    tensor_free(dx);
    module_free(m);   /* 同时释放 rn */
}

/* ---------- 5. 错误路径（在子进程里跑，因为既有风格是 exit(1)） ---------- */
static int run_child(const char *self, const char *mode)
{
    char cmd[512];
    snprintf(cmd, sizeof cmd, "\"%s\" %s", self, mode);
    const int rc = system(cmd);
    return rc;
}

static void test_invalid(const char *self)
{
    printf("[5] 错误路径（子进程，期望非 0 退出码）\n");
    const char *modes[] = { "bad-d", "bad-setparams", "backward-before-forward",
                            "bad-shape-x", "bad-shape-dout" };
    for (int i = 0; i < 5; i++) {
        const int rc = run_child(self, modes[i]);
        printf("  %-24s 退出码 %d\n", modes[i], rc);
        CHECK(rc != 0);
    }
    /* 正常路径的退出码应当是 0 */
    CHECK(run_child(self, "self-ok") == 0);
}

/* ---------- 6. d=1 与 eps 的作用 ---------- */
static void test_d1_eps(void)
{
    printf("[6] d=1 与 eps\n");
    RMSNorm *rn = rmsnorm_create(1, 0.0);
    const int shape[1] = { 1 };
    Tensor *x = tensor_new(1, shape);
    Tensor *y = tensor_new(1, shape);
    Tensor *dout = tensor_new(1, shape);
    Tensor *dx = tensor_new(1, shape);

    x->data[0] = -2.0;
    rmsnorm_forward(rn, x, y);
    CHECK_BITS(rmsnorm_inv_rms(rn), 0.5);   /* 1/sqrt(4) */
    CHECK_BITS(y->data[0], -1.0);           /* gamma=1, x=-2, r=0.5 */

    dout->data[0] = 1.0;
    rmsnorm_backward(rn, dout, dx);
    /* g = 1*1*(-2) = -2；dx = 1*1*0.5 - (-2)*(-2)*0.125/1 = 0.5 - 0.5 = 0 */
    CHECK_BITS(rmsnorm_grad_gamma(rn)[0], -1.0);
    CHECK_BITS(dx->data[0], 0.0);

    /* eps 变大时 r 变小 */
    rmsnorm_free(rn);
    rn = rmsnorm_create(1, 3.0);
    x->data[0] = 1.0;
    rmsnorm_forward(rn, x, y);
    CHECK_BITS(rmsnorm_inv_rms(rn), 0.5);   /* 1/sqrt(1+3) */

    tensor_free(x);
    tensor_free(y);
    tensor_free(dout);
    tensor_free(dx);
    rmsnorm_free(rn);
}

int main(int argc, char **argv)
{
    /* 子进程模式：故意触发 exit(1)，供 test_invalid 检查退出码 */
    const char *mode = argc > 1 ? argv[1] : NULL;
    if (mode && strcmp(mode, "self") == 0) mode = NULL;   /* make test 的约定参数 */
    if (mode) {
        if (strcmp(mode, "self-ok") == 0) {
            RMSNorm *rn = rmsnorm_create(2, 1e-6);
            const int shape[1] = { 2 };
            Tensor *x = tensor_new(1, shape);
            Tensor *y = tensor_new(1, shape);
            x->data[0] = 1.0;
            x->data[1] = 2.0;
            rmsnorm_forward(rn, x, y);
            tensor_free(x);
            tensor_free(y);
            rmsnorm_free(rn);
            return 0;
        }
        if (strcmp(mode, "bad-d") == 0) {
            rmsnorm_create(0, 1e-6);
            return 0;   /* 不该走到这里 */
        }
        if (strcmp(mode, "bad-setparams") == 0) {
            RMSNorm *rn = rmsnorm_create(2, 1e-6);
            rmsnorm_set_params(rn, NULL);
            rmsnorm_free(rn);
            return 0;
        }
        if (strcmp(mode, "backward-before-forward") == 0) {
            RMSNorm *rn = rmsnorm_create(2, 1e-6);
            const int shape[1] = { 2 };
            Tensor *dout = tensor_new(1, shape);
            Tensor *dx = tensor_new(1, shape);
            rmsnorm_backward(rn, dout, dx);
            return 0;
        }
        if (strcmp(mode, "bad-shape-x") == 0) {
            RMSNorm *rn = rmsnorm_create(2, 1e-6);
            const int bad[1] = { 3 };
            Tensor *x = tensor_new(1, bad);
            Tensor *y = tensor_new(1, bad);
            rmsnorm_forward(rn, x, y);
            return 0;
        }
        if (strcmp(mode, "bad-shape-dout") == 0) {
            RMSNorm *rn = rmsnorm_create(2, 1e-6);
            const int shape[1] = { 2 };
            const int bad[1] = { 3 };
            Tensor *x = tensor_new(1, shape);
            Tensor *y = tensor_new(1, shape);
            Tensor *dout = tensor_new(1, bad);
            Tensor *dx = tensor_new(1, bad);
            rmsnorm_forward(rn, x, y);
            rmsnorm_backward(rn, dout, dx);
            return 0;
        }
        printf("未知模式 %s\n", mode);
        return 2;
    }

    printf("== test_rmsnorm ==\n");
    test_hand_d2();
    test_uniform();
    test_grad_overwrite();
    test_module();
    test_d1_eps();
    test_invalid(argv[0]);

    if (g_fail == 0) {
        printf("[OK] test_rmsnorm 全部通过\n");
        return 0;
    }
    printf("[失败] test_rmsnorm 有 %d 处断言失败\n", g_fail);
    return 1;
}
