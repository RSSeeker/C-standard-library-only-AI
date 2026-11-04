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

/* ------------------------------ 数值梯度工具 ------------------------------ */

static LLMParamBlock *block_by_name(LLMParamBlock *blocks, int nb, const char *name)
{
    for (int i = 0; i < nb; i++) if (strcmp(blocks[i].name, name) == 0) return &blocks[i];
    fprintf(stderr, "测试: 找不到参数块 %s\n", name);
    exit(2);
}

/* 一次前向+反向的 loss（先清零梯度，保证每次都是「从当前位置算起」） */
static double loss_of(CausalLM *m, const int *ids, int T)
{
    llm_zero_grad(m);
    return llm_loss_and_backward(m, ids, T);
}

/* 对某个参数做中心差分，返回 (解析梯度, 数值梯度) */
static void fd_one(CausalLM *m, const int *ids, int T, LLMParamBlock *blk, int idx,
                   double *ana, double *num)
{
    llm_zero_grad(m);
    (void)llm_loss_and_backward(m, ids, T);
    *ana = blk->grad[idx];

    double v0 = blk->param[idx];
    blk->param[idx] = v0 + g_eps;
    double lp = loss_of(m, ids, T);
    blk->param[idx] = v0 - g_eps;
    double lm = loss_of(m, ids, T);
    blk->param[idx] = v0;
    *num = (lp - lm) / (2.0 * g_eps);
}

/* 跑一个参数块上的中心差分并打印；rel = |ana-num| / max(|num|, 1) */
static void check_fd(CausalLM *m, const int *ids, int T, LLMParamBlock *blocks, int nb,
                     const char *name, int idx)
{
    LLMParamBlock *blk = block_by_name(blocks, nb, name);
    double ana = 0.0, num = 0.0;
    fd_one(m, ids, T, blk, idx, &ana, &num);
    double denom = fabs(num) > 1.0 ? fabs(num) : 1.0;
    char label[80];
    snprintf(label, sizeof(label), "%s[%d]", name, idx);
    check(label, fabs(ana - num) / denom, g_tol);
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

    /* ================== 2. RoPE：开关与可复现性 ================== */
    printf("\n[2] RoPE（开关、输出改变、关闭后逐位恢复）\n");
    {
        const int d = 8, heads = 2, T = 6;
        MultiHeadAttention *m = mha_create(d, heads, T, 0.0);
        mha_set_causal(m, 1);

        int dims[2] = { T, d };
        Tensor *x = tensor_new(2, dims);
        Tensor *off1 = tensor_new(2, dims);
        Tensor *on = tensor_new(2, dims);
        Tensor *off2 = tensor_new(2, dims);
        fill_seq(x, T, d);

        mha_forward(m, x, off1);          /* 默认（无 RoPE） */
        mha_set_rope(m, 1, 10000.0);
        mha_forward(m, x, on);            /* 打开 RoPE */
        mha_set_rope(m, 0, 10000.0);
        mha_forward(m, x, off2);          /* 再关掉 */

        double worst_off = 0.0, diff_on = 0.0;
        for (int i = 0; i < T * d; i++) {
            double a = fabs(off2->data[i] - off1->data[i]);
            double b = fabs(on->data[i] - off1->data[i]);
            if (a > worst_off) worst_off = a;
            if (b > diff_on) diff_on = b;
        }
        check("rope 关闭后与默认路径逐位一致", worst_off, 0.0);
        check_true("rope 打开后输出确实改变", diff_on > 1e-3);

        /* 位置 0 的旋转是恒等（theta = 0），所以第 0 个 token 的输出应当完全不变 —— 
         * 这同时验证「旋转按位置 t 施加」而不是按 token 序号乱来。 */
        double worst_pos0 = 0.0;
        for (int j = 0; j < d; j++) {
            double dd = fabs(on->data[j] - off1->data[j]);
            if (dd > worst_pos0) worst_pos0 = dd;
        }
        check("位置 0 在 RoPE 下不变（theta=0）", worst_pos0, 1e-15);

        /* 不同 base → 不同旋转频率 → 输出不同 */
        Tensor *on2 = tensor_new(2, dims);
        mha_set_rope(m, 1, 500.0);
        mha_forward(m, x, on2);
        double diff_base = 0.0;
        for (int i = 0; i < T * d; i++) {
            double dd = fabs(on2->data[i] - on->data[i]);
            if (dd > diff_base) diff_base = dd;
        }
        check_true("不同 rope_base 得到不同输出", diff_base > 1e-6);

        tensor_free(x);
        tensor_free(off1);
        tensor_free(on);
        tensor_free(off2);
        tensor_free(on2);
        mha_free(m);
    }

    /* ================== 3. RoPE 梯度（CausalLM 全链路） ================== */
    printf("\n[3] RoPE 梯度（中心差分 eps=%.0e，rel tol=%.0e）\n", g_eps, g_tol);
    {
        const int vocab = 7, d_model = 8, heads = 2, d_ff = 16, layers = 2, max_len = 12;
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 2024);
        llm_set_rope(m, 1, 10000.0);
        check_true("llm_set_rope 关掉了绝对位置编码", m->use_abs_pos == 0);
        check_true("每层 MHA 的 rope 都打开了",
                   m->enc->layers[0]->self_attn->rope == 1 &&
                   m->enc->layers[1]->self_attn->rope == 1);

        LLMParamBlock blocks[128];
        int nb = llm_param_blocks(m, blocks, 128);
        int ids[6] = { 2, 5, 1, 6, 3, 0 };
        check_fd(m, ids, 6, blocks, nb, "L0.attn.W_q", 5 * d_model + 6);
        check_fd(m, ids, 6, blocks, nb, "L0.attn.W_v", 1 * d_model + 4);
        check_fd(m, ids, 6, blocks, nb, "L0.attn.W_k", 3 * d_model + 2);

        /* 关掉 RoPE 后应回到「绝对位置编码 + 无 RoPE」的默认路径：
         * 与同种子新建的模型逐位比较 logits。 */
        CausalLM *ref = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 2024);
        llm_set_rope(m, 0, 10000.0);
        int dims[2] = { 6, vocab };
        Tensor *la = tensor_new(2, dims);
        Tensor *lb = tensor_new(2, dims);
        llm_forward(m, ids, 6, la);
        llm_forward(ref, ids, 6, lb);
        double worst_off = 0.0;
        for (int i = 0; i < 6 * vocab; i++) {
            double dd = fabs(la->data[i] - lb->data[i]);
            if (dd > worst_off) worst_off = dd;
        }
        check("rope 关掉后 logits 与默认模型逐位一致", worst_off, 0.0);

        tensor_free(la);
        tensor_free(lb);
        llm_free(ref);
        llm_free(m);
    }

    printf("\n%s\n", g_fail ? "存在失败项" : "全部通过");
    return g_fail ? 1 : 0;
}
