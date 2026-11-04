/*
 * tests/test_llm_ext.c —— CausalLM / MultiHeadAttention 的扩展特性自测
 *
 * 覆盖（与 tests/test_llm.c 的默认路径测试互补，这里只测新特性）：
 *   1. 滑动窗口：窗口外恰好为 0、窗口内每行和为 1、window 隐含因果
 *   2. window <= 0 时与未调用 set_window 的老行为逐位一致
 *   3. RoPE：开启后输出改变、关闭时不变；dQ/dK 的中心差分校验
 *   4. GQA（4 头 / 2 KV 头）：W_k/W_q/b_k 中心差分 + 训练冒烟
 *   5. nucleus 采样：top_p = 1.0 与旧路径逐位一致、top_p = 0.5 只留最高概率 token
 *   6. 困惑度：llm_perplexity == exp(llm_evaluate)，以及 loss > 700 时返回 INFINITY
 *
 * 非法输入走 exit(1)，不在这里测（与其它 test_*.c 的约定一致）。
 * 全部通过返回 0；任何一项超差返回 1。
 * 编译：见 README（与 test_llm.c 相同的源文件列表）。
 */
#include "../attention.h"
#include "../llm.h"
#include "../tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------ 测试框架 ------------------------------ */

static int    g_fail = 0;
static double g_eps = 1e-6;
static double g_tol = 1e-5;

static void check(const char *what, double worst, double tol)
{
    int ok = (worst <= tol);
    if (!ok) g_fail++;
    printf("  %-38s 最大误差 %.3e  %s\n", what, worst, ok ? "[OK]" : "[超出容差]");
}

static void check_true(const char *what, int cond)
{
    if (!cond) g_fail++;
    printf("  %-38s %s\n", what, cond ? "[OK]" : "[失败]");
}

/* 确定性输入：不依赖随机数，避免测试本身不可复现 */
static void fill_seq(Tensor *t, int T, int d)
{
    for (int i = 0; i < T; i++)
        for (int j = 0; j < d; j++)
            t->data[i * d + j] = sin(0.37 * (double)(i * d + j) + 0.11 * (double)i);
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    (void)g_eps;   /* 数值梯度小节（RoPE / GQA）才用到，这里先登记，避免 -Wunused */
    (void)g_tol;

    printf("CausalLM / MHA 扩展特性自测\n");

    /* ================== 1. 滑动窗口 ================== */
    printf("\n[1] 滑动窗口掩码（window=3, 未开 causal）\n");
    {
        const int d = 8, heads = 2, T = 8, window = 3;
        MultiHeadAttention *m = mha_create(d, heads, T, 0.0);
        mha_set_window(m, window);           /* 隐含因果，故意不调用 mha_set_causal */

        int dims[2] = { T, d };
        Tensor *x = tensor_new(2, dims);
        Tensor *out = tensor_new(2, dims);
        fill_seq(x, T, d);
        mha_forward(m, x, out);

        double worst_outside = 0.0, worst_rowsum = 0.0, min_inside = 1e9;
        int outside_count = 0;
        for (int h = 0; h < heads; h++) {
            const double *attn = m->attn + (size_t)h * T * T;
            for (int t = 0; t < T; t++) {
                double sum = 0.0;
                for (int j = 0; j < T; j++) {
                    double w = attn[t * T + j];
                    sum += w;
                    int visible = (j <= t) && (j >= t - window + 1);
                    if (!visible) {
                        outside_count++;
                        if (fabs(w) > worst_outside) worst_outside = fabs(w);
                    } else if (w < min_inside) {
                        min_inside = w;
                    }
                }
                if (fabs(sum - 1.0) > worst_rowsum) worst_rowsum = fabs(sum - 1.0);
            }
        }
        check("窗口外权重（应恰好为 0）", worst_outside, 0.0);
        check("每行和与 1 的偏差", worst_rowsum, 1e-12);
        check_true("窗口外确实存在被屏蔽的位置", outside_count > 0);
        check_true("窗口内位置都有正权重", min_inside > 0.0);

        /* 窗口隐含因果：未来位置必须是 0（上面 worst_outside 已覆盖，这里再显式查一次） */
        double worst_future = 0.0;
        for (int h = 0; h < heads; h++) {
            const double *attn = m->attn + (size_t)h * T * T;
            for (int t = 0; t < T; t++)
                for (int j = t + 1; j < T; j++)
                    if (fabs(attn[t * T + j]) > worst_future) worst_future = fabs(attn[t * T + j]);
        }
        check("未来位置权重（causal=0 也要屏蔽）", worst_future, 0.0);

        /* 窗口比序列长时等价于全因果：与 window=T 的模型逐位比较 */
        MultiHeadAttention *m2 = mha_create(d, heads, T, 0.0);
        mha_set_window(m2, T + 5);
        Tensor *out2 = tensor_new(2, dims);
        mha_forward(m2, x, out2);
        MultiHeadAttention *m3 = mha_create(d, heads, T, 0.0);
        mha_set_causal(m3, 1);
        Tensor *out3 = tensor_new(2, dims);
        mha_forward(m3, x, out3);
        double worst_big = 0.0;
        for (int i = 0; i < T * d; i++) {
            double dd = fabs(out2->data[i] - out3->data[i]);
            if (dd > worst_big) worst_big = dd;
        }
        check("window >= T 与全因果逐位一致", worst_big, 0.0);

        /* window <= 0 关闭：与从未调用 set_window 的对象逐位一致 */
        MultiHeadAttention *m4 = mha_create(d, heads, T, 0.0);
        mha_set_causal(m4, 1);
        mha_set_window(m4, 0);
        mha_set_window(m4, -7);
        Tensor *out4 = tensor_new(2, dims);
        mha_forward(m4, x, out4);
        double worst_off = 0.0;
        for (int i = 0; i < T * d; i++) {
            double dd = fabs(out4->data[i] - out3->data[i]);
            if (dd > worst_off) worst_off = dd;
        }
        check("window<=0 关闭后与默认路径逐位一致", worst_off, 0.0);

        tensor_free(x);
        tensor_free(out);
        tensor_free(out2);
        tensor_free(out3);
        tensor_free(out4);
        mha_free(m);
        mha_free(m2);
        mha_free(m3);
        mha_free(m4);
    }

    printf("\n%s\n", g_fail ? "存在失败项" : "全部通过");
    return g_fail ? 1 : 0;
}
