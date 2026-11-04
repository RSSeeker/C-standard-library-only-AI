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

/* --------------------- MHA 级数值梯度工具（GQA 小节用） ---------------------
 * CausalLM 没有暴露 num_kv_heads，因此 GQA 的梯度校验直接在 MultiHeadAttention 上做：
 * 取标量损失 L = 0.5 * sum(out^2)，则 dL/dout = out（无需额外目标张量）。
 */
static double mha_sq_loss(MultiHeadAttention *m, const Tensor *x, Tensor *out)
{
    mha_forward(m, x, out);
    double s = 0.0;
    for (int i = 0; i < out->size; i++) s += 0.5 * out->data[i] * out->data[i];
    return s;
}

static void mha_fd(MultiHeadAttention *m, const Tensor *x, Tensor *out, Tensor *dout, Tensor *dx,
                   double *param, const double *grad, int idx, const char *name)
{
    mha_zero_grad(m);
    (void)mha_sq_loss(m, x, out);
    memcpy(dout->data, out->data, (size_t)out->size * sizeof(double));
    mha_backward(m, dout, dx);
    const double ana = grad[idx];

    const double v0 = param[idx];
    param[idx] = v0 + g_eps;
    const double lp = mha_sq_loss(m, x, out);
    param[idx] = v0 - g_eps;
    const double lm = mha_sq_loss(m, x, out);
    param[idx] = v0;
    const double num = (lp - lm) / (2.0 * g_eps);

    const double denom = fabs(num) > 1.0 ? fabs(num) : 1.0;
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

    /* ================== 4. GQA（4 头 / 2 KV 头） ================== */
    printf("\n[4] GQA / MQA（4 头 / 2 KV 头，d_model=8）\n");
    {
        const int d = 8, heads = 4, kv_heads = 2, T = 6, dk = d / heads;
        MultiHeadAttention *m = mha_create_gqa(d, heads, kv_heads, T, 0.0);
        mha_set_causal(m, 1);

        check_true("kv_dim == num_kv_heads * d_k", m->kv_dim == kv_heads * dk);
        check_true("kv_group == num_heads / num_kv_heads", m->kv_group == heads / kv_heads);
        /* 4 头全量是 4*d²+4*d；GQA 省掉 (heads-kv_heads) 组 KV 投影及其 bias：
         *   2 * ((heads-kv_heads)*dk*d + (heads-kv_heads)*dk) = 2*(2*2*8 + 2*2) = 72 */
        check_true("参数数正好省下 2 组 KV 投影与 bias",
                   mha_param_count(m) == 4 * d * d + 4 * d - 72);

        int dims[2] = { T, d };
        Tensor *x = tensor_new(2, dims);
        Tensor *out = tensor_new(2, dims);
        Tensor *dout = tensor_new(2, dims);
        Tensor *dx = tensor_new(2, dims);
        fill_seq(x, T, d);

        /* W_k / W_v 的梯度：共享 KV 头的累加是否正确 —— 数值梯度说了算 */
        mha_fd(m, x, out, dout, dx, m->W_q, m->grad_W_q, 1 * d + 5, "W_q");
        mha_fd(m, x, out, dout, dx, m->W_q, m->grad_W_q, 3 * d + 2, "W_q");
        mha_fd(m, x, out, dout, dx, m->W_k, m->grad_W_k, 3 * d + 2, "W_k");
        mha_fd(m, x, out, dout, dx, m->W_k, m->grad_W_k, 0 * d + 4, "W_k");
        mha_fd(m, x, out, dout, dx, m->W_v, m->grad_W_v, 2 * d + 7, "W_v");
        mha_fd(m, x, out, dout, dx, m->b_k, m->grad_b_k, 5, "b_k");
        mha_fd(m, x, out, dout, dx, m->b_q, m->grad_b_q, 6, "b_q");
        mha_fd(m, x, out, dout, dx, m->b_v, m->grad_b_v, 1, "b_v");

        /* 头映射校验：把 2 组 KV 头「复制」成 4 组后，GQA 与普通多头应逐位一致
         * （第 h 个 Q 头用第 h / kv_group 个 KV 头，复制后两者看到完全相同的 k/v）。 */
        MultiHeadAttention *mfull = mha_create(d, heads, T, 0.0);
        mha_set_causal(mfull, 1);
        double *Wk_full = (double *)malloc((size_t)d * d * sizeof(double));
        double *Wv_full = (double *)malloc((size_t)d * d * sizeof(double));
        for (int g = 0; g < heads / kv_heads; g++) {   /* 每个 kv 头复制 kv_group 份 */
            for (int hh = 0; hh < kv_heads; hh++) {
                int dst_head = hh * (heads / kv_heads) + g;
                memcpy(Wk_full + (size_t)dst_head * dk * d,
                       m->W_k + (size_t)hh * dk * d, (size_t)dk * d * sizeof(double));
                memcpy(Wv_full + (size_t)dst_head * dk * d,
                       m->W_v + (size_t)hh * dk * d, (size_t)dk * d * sizeof(double));
            }
        }
        double *bk_full = (double *)malloc((size_t)d * sizeof(double));
        double *bv_full = (double *)malloc((size_t)d * sizeof(double));
        for (int g = 0; g < heads / kv_heads; g++)
            for (int hh = 0; hh < kv_heads; hh++) {
                int dst_head = hh * (heads / kv_heads) + g;
                memcpy(bk_full + (size_t)dst_head * dk, m->b_k + (size_t)hh * dk,
                       (size_t)dk * sizeof(double));
                memcpy(bv_full + (size_t)dst_head * dk, m->b_v + (size_t)hh * dk,
                       (size_t)dk * sizeof(double));
            }
        mha_set_params(mfull, m->W_q, m->b_q, Wk_full, bk_full, Wv_full, bv_full, m->W_o, m->b_o);
        Tensor *out_full = tensor_new(2, dims);
        mha_forward(m, x, out);
        mha_forward(mfull, x, out_full);
        double worst_map = 0.0;
        for (int i = 0; i < T * d; i++) {
            double dd = fabs(out->data[i] - out_full->data[i]);
            if (dd > worst_map) worst_map = dd;
        }
        check("KV 头复制后与普通多头逐位一致", worst_map, 0.0);
        free(Wk_full); free(Wv_full); free(bk_full); free(bv_full);

        /* 训练冒烟：用解析梯度对 GQA 模型做 20 步 SGD，loss 必须明显下降。
         * 损失 0.5*||out||² 是二次型，步长过大（> 2/λ_max）会发散，这里取 0.002。 */
        double l_first = 0.0, l_last = 0.0;
        const double lr = 0.002;
        for (int step = 0; step < 20; step++) {
            mha_zero_grad(m);
            double L = mha_sq_loss(m, x, out);
            if (step == 0) l_first = L;
            memcpy(dout->data, out->data, (size_t)out->size * sizeof(double));
            mha_backward(m, dout, dx);
            for (int i = 0; i < d * d; i++) {
                m->W_q[i] -= lr * m->grad_W_q[i];
                m->W_o[i] -= lr * m->grad_W_o[i];
            }
            for (int i = 0; i < m->kv_dim * d; i++) {
                m->W_k[i] -= lr * m->grad_W_k[i];
                m->W_v[i] -= lr * m->grad_W_v[i];
            }
            for (int i = 0; i < d; i++) { m->b_q[i] -= lr * m->grad_b_q[i]; m->b_o[i] -= lr * m->grad_b_o[i]; }
            for (int i = 0; i < m->kv_dim; i++) { m->b_k[i] -= lr * m->grad_b_k[i]; m->b_v[i] -= lr * m->grad_b_v[i]; }
        }
        l_last = mha_sq_loss(m, x, out);
        check_true("GQA 训练冒烟：loss 明显下降", l_last < 0.5 * l_first);
        printf("    GQA loss: %.6f → %.6f\n", l_first, l_last);

        tensor_free(x);
        tensor_free(out);
        tensor_free(dout);
        tensor_free(dx);
        tensor_free(out_full);
        mha_free(m);
        mha_free(mfull);
    }

    /* ================== 5. nucleus 采样（top_p） ================== */
    printf("\n[5] nucleus 采样（top_p）\n");
    {
        const int vocab = 8, d_model = 8, heads = 2, d_ff = 16, layers = 1, max_len = 12;
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 7);

        /* 先过拟合一条短序列，让下一步分布变尖 —— 这样 top_p = 0.5 只可能留下 argmax */
        int seq[8] = { 1, 4, 2, 7, 3, 5, 0, 6 };
        const int *seqs[1] = { seq };
        int lens[1] = { 8 };
        (void)llm_fit(m, seqs, lens, 1, 80, 0.02, 0, 1, NULL);

        int prompt[2] = { 1, 4 };
        const int max_new = 4;
        int a[6], b[6], g[6];
        llm_generate(m, prompt, 2, max_new, 1.0, 0, a, 42);
        llm_generate_p(m, prompt, 2, max_new, 1.0, 0, 1.0, b, 42);
        check_true("top_p = 1.0 与 llm_generate 逐位一致",
                   memcmp(a, b, sizeof(a)) == 0);

        /* top_p = 1.0 时不应多抽随机数：同一 seed 连续两次结果相同 */
        int c[6];
        llm_generate_p(m, prompt, 2, max_new, 1.0, 0, 1.0, c, 42);
        check_true("top_p = 1.0 同 seed 可复现", memcmp(b, c, sizeof(b)) == 0);

        /* 前提：分布足够尖，argmax 概率 > 0.5（否则下面的等价性检查没有意义） */
        int dims_l[2] = { 2, vocab };
        Tensor *lg = tensor_new(2, dims_l);
        llm_forward(m, prompt, 2, lg);
        double p[8];
        for (int v = 0; v < vocab; v++) p[v] = lg->data[(size_t)(2 - 1) * vocab + v];
        double mx = p[0];
        for (int v = 1; v < vocab; v++) if (p[v] > mx) mx = p[v];
        double sum = 0.0;
        for (int v = 0; v < vocab; v++) { p[v] = exp(p[v] - mx); sum += p[v]; }
        double p_max = 1.0 / sum;   /* argmax 的概率（softmax 后 argmax 项的指数为 1） */
        check_true("模型已足够尖（p_max > 0.5）", p_max > 0.5);
        printf("    argmax 概率 = %.6f\n", p_max);

        /* top_p = 0.5：只保留累积概率刚达到 0.5 的最小集合 → 当 p_max > 0.5 时就是 argmax，
         * 因此与贪心（top_k = 1）逐 token 相同，任意 seed 都一样。 */
        llm_generate(m, prompt, 2, max_new, 1.0, 1, g, 42);
        int ok = 1;
        for (unsigned long long sd = 0; sd < 5; sd++) {
            int t[6];
            llm_generate_p(m, prompt, 2, max_new, 1.0, 0, 0.5, t, sd);
            if (memcmp(t, g, sizeof(g)) != 0) ok = 0;
        }
        check_true("top_p = 0.5 在尖分布上等价于贪心（5 个 seed）", ok);

        /* top_p = 1.0 仍然随机：同一模型不同 seed 通常给出不同序列（这里只验证语义合法） */
        int d[6];
        int n = llm_generate_p(m, prompt, 2, max_new, 1.0, 0, 1.0, d, 12345);
        check_true("top_p = 1.0 返回完整长度", n == 2 + max_new);
        int in_range = 1;
        for (int i = 0; i < n; i++) if (d[i] < 0 || d[i] >= vocab) in_range = 0;
        check_true("生成的 token 都在词表范围内", in_range);

        tensor_free(lg);
        llm_free(m);
    }

    /* ================== 6. 困惑度 ================== */
    printf("\n[6] 困惑度（llm_perplexity）\n");
    {
        const int vocab = 8, d_model = 8, heads = 2, d_ff = 16, layers = 1, max_len = 12;
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 11);
        int s1[6] = { 1, 4, 2, 7, 3, 5 };
        int s2[4] = { 6, 0, 3, 1 };
        const int *seqs[2] = { s1, s2 };
        int lens[2] = { 6, 4 };
        (void)llm_fit(m, seqs, lens, 2, 10, 0.01, 0, 1, NULL);

        double ev = llm_evaluate(m, seqs, lens, 2);
        double pp = llm_perplexity(m, seqs, lens, 2);
        check("perplexity == exp(evaluate)", fabs(pp - exp(ev)), 1e-12);
        printf("    evaluate = %.6f  perplexity = %.6f\n", ev, pp);

        /* 未训练的模型再验一次（另一条权重、另一个 loss 量级） */
        CausalLM *m2 = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 13);
        double ev2 = llm_evaluate(m2, seqs, lens, 2);
        double pp2 = llm_perplexity(m2, seqs, lens, 2);
        check("未训练模型也满足 perplexity == exp(evaluate)", fabs(pp2 - exp(ev2)), 1e-12);

        /* 关于 loss > 700 的保护分支：llm_evaluate 与 Python 的 evaluate 一样把单位置概率
         * 夹在 1e-12，因此平均 loss 的上界是 -log(1e-12) ≈ 27.631，永远到不了 700 ——
         * 也就是说这个分支是纯防御性的（与 Python 的 perplexity() 完全对应），无法通过
         * 公开 API 触发，这里不做（也无法做）行为测试。 */

        llm_free(m);
        llm_free(m2);
    }

    printf("\n%s\n", g_fail ? "存在失败项" : "全部通过");
    return g_fail ? 1 : 0;
}
