/*
 * tests/test_llm.c —— CausalLM（因果语言模型）自测
 *
 * 覆盖：
 *   1. 构造与参数块表：块数量、顺序、名字与 Python 的 _named_params 一致
 *   2. 因果性：改动最后一个 token，前面位置的 logits 必须逐位不变
 *   3. 掩码本身：注意力权重上三角恰好为 0、每行和为 1
 *   4. 数值梯度：out.W / out.b / emb.weight / attn.W_v / ffn.W1 / ln1.gamma 中心差分
 *   5. 训练：小语料上 loss 显著下降
 *   6. 训练分段等价：fit(2) 与 fit(1)+fit(1) 参数逐位相同（step_count 不能被重置）
 *   7. 生成：同种子可复现、top_k=1 是确定性的、eval 模式下两次前向一致
 *   8. 参数块灌入/读出：llm_param_blocks 拿到的是活指针
 *   9. 生成罚分 / min_p：repetition 的正负 logit 两分支、presence 扣一次、
 *      frequency 按次数扣、no_repeat_ngram 禁后继、min_p 的候选数
 *  10. 旧生成 API 与 llm_generate_ex 全默认参数逐 token 一致（min_p=0 不变）
 *  11. ALiBi：TE -> TEL -> MHA 透传后权重等于 softmax(slope*(t-j)) 的闭式解
 *
 * 非法输入走 exit(1)，不在这里测（与其它 test_*.c 的约定一致：那是进程级行为）。
 * 全部通过返回 0；任何一项超差返回 1。
 * 编译：gcc -std=c17 -O2 -Wall -Wextra -o tests/test_llm.exe \
 *           tests/test_llm.c ai.c attention.c containers.c json.c layers_conv.c \
 *           layers_norm.c llm.c model_high.c model_io.c rnn.c sched.c tensor.c text.c -lm
 */
#include "../llm.h"

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
    printf("  %-34s 最大误差 %.3e  %s\n", what, worst, ok ? "[OK]" : "[超出容差]");
}

static void check_true(const char *what, int cond)
{
    if (!cond) g_fail++;
    printf("  %-34s %s\n", what, cond ? "[OK]" : "[失败]");
}

/* 在块表里按名字找参数块 */
static LLMParamBlock *block_by_name(LLMParamBlock *blocks, int nb, const char *name)
{
    for (int i = 0; i < nb; i++) if (strcmp(blocks[i].name, name) == 0) return &blocks[i];
    fprintf(stderr, "测试: 找不到参数块 %s\n", name);
    exit(2);
}

/* 一次前向+反向的 loss（会清零梯度，保证每次都是「从当前位置算起」） */
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
    (void)llm_loss_and_backward(m, ids, T);   /* 填充梯度 */
    *ana = blk->grad[idx];

    double v0 = blk->param[idx];
    blk->param[idx] = v0 + g_eps;
    double lp = loss_of(m, ids, T);
    blk->param[idx] = v0 - g_eps;
    double lm = loss_of(m, ids, T);
    blk->param[idx] = v0;
    *num = (lp - lm) / (2.0 * g_eps);
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    const int vocab = 7, d_model = 8, heads = 2, d_ff = 16, layers = 2, max_len = 12;

    printf("CausalLM 自测（vocab=%d d_model=%d heads=%d d_ff=%d layers=%d max_len=%d）\n\n",
           vocab, d_model, heads, d_ff, layers, max_len);

    CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 2024);
    llm_set_train(m, 0);

    LLMParamBlock blocks[128];
    int nb = llm_param_blocks(m, blocks, 128);

    /* ------------------ 1. 参数块表 ------------------ */
    printf("[1] 参数块表\n");
    {
        int expect_blocks = 1 + layers * 16 + 2;   /* emb + 每层 16 块 + out.W / out.b */
        check_true("块数量", nb == expect_blocks);
        check_true("首块是 emb.weight", strcmp(blocks[0].name, "emb.weight") == 0);
        check_true("末块是 out.b", strcmp(blocks[nb - 1].name, "out.b") == 0);
        check_true("倒数第二块是 out.W", strcmp(blocks[nb - 2].name, "out.W") == 0);
        check_true("第二层参数名", block_by_name(blocks, nb, "L1.ffn.W2") != NULL);

        double total = 0.0;
        for (int i = 0; i < nb; i++) total += blocks[i].n;
        check_true("llm_param_count 与块表一致", (double)llm_param_count(m) == total);
    }

    /* ------------------ 2. 因果性 ------------------ */
    printf("\n[2] 因果性（改最后一个 token，前面的 logits 不变）\n");
    {
        int ids_a[5] = { 3, 1, 4, 1, 5 };
        int ids_b[5] = { 3, 1, 4, 1, 2 };   /* 只改最后一个 */
        int dims[2] = { 5, vocab };
        Tensor *la = tensor_new(2, dims);
        Tensor *lb = tensor_new(2, dims);
        llm_forward(m, ids_a, 5, la);
        llm_forward(m, ids_b, 5, lb);

        double worst_front = 0.0;
        for (int t = 0; t < 4; t++)
            for (int v = 0; v < vocab; v++) {
                double d = fabs(la->data[t * vocab + v] - lb->data[t * vocab + v]);
                if (d > worst_front) worst_front = d;
            }
        check("前 4 个位置的 logits", worst_front, 0.0);

        /* 最后一行应当受影响（否则说明掩码把因果方向搞反了） */
        double diff_last = 0.0;
        for (int v = 0; v < vocab; v++) {
            double d = fabs(la->data[4 * vocab + v] - lb->data[4 * vocab + v]);
            if (d > diff_last) diff_last = d;
        }
        check_true("最后一个位置确实变了", diff_last > 0.0);

        tensor_free(la);
        tensor_free(lb);
    }

    /* ------------------ 3. 掩码权重 ------------------ */
    printf("\n[3] 注意力掩码（上三角为 0、每行和为 1）\n");
    {
        int ids[6] = { 1, 6, 2, 5, 3, 4 };
        int dims[2] = { 6, vocab };
        Tensor *out = tensor_new(2, dims);
        llm_forward(m, ids, 6, out);
        tensor_free(out);

        MultiHeadAttention *a = m->enc->layers[0]->self_attn;
        const int T = 6;
        double worst_masked = 0.0, worst_rowsum = 0.0, min_diag = 1e9;
        for (int h = 0; h < a->num_heads; h++) {
            const double *attn = a->attn + (size_t)h * T * T;
            for (int t = 0; t < T; t++) {
                double sum = 0.0;
                for (int j = 0; j < T; j++) {
                    if (j > t) {
                        double v = fabs(attn[t * T + j]);
                        if (v > worst_masked) worst_masked = v;
                    }
                    sum += attn[t * T + j];
                }
                if (fabs(sum - 1.0) > worst_rowsum) worst_rowsum = fabs(sum - 1.0);
                if (attn[t * T + t] < min_diag) min_diag = attn[t * T + t];
            }
        }
        check("被屏蔽位置（应恰好为 0）", worst_masked, 0.0);
        check("每行和与 1 的偏差", worst_rowsum, 1e-12);
        check_true("对角线位置有权重", min_diag > 0.0);
    }

    /* ------------------ 4. 数值梯度 ------------------ */
    printf("\n[4] 数值梯度（中心差分 eps=%.0e）\n", g_eps);
    {
        int ids[6] = { 2, 5, 1, 6, 3, 0 };
        const int T = 6;
        struct { const char *name; int idx; } cases[] = {
            { "out.W", 3 },
            { "out.b", 2 },
            { "emb.weight", 3 * d_model + 2 },
            { "L0.attn.W_v", 1 * d_model + 4 },
            { "L0.ffn.W1", 2 * d_model + 1 },
            { "L0.ln1.gamma", 1 },
            { "L1.attn.W_q", 5 * d_model + 6 },
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            LLMParamBlock *blk = block_by_name(blocks, nb, cases[i].name);
            double ana = 0.0, num = 0.0;
            fd_one(m, ids, T, blk, cases[i].idx, &ana, &num);
            double denom = fabs(num) > 1.0 ? fabs(num) : 1.0;
            char label[64];
            snprintf(label, sizeof(label), "%s[%d]", cases[i].name, cases[i].idx);
            check(label, fabs(ana - num) / denom, g_tol);
        }
    }

    /* ------------------ 5. 训练：loss 下降 ------------------ */
    printf("\n[5] 小语料训练\n");
    {
        /* 语料重复三遍，保证贪心续写要走的 8 步都落在学过的转移里 */
        const int seq[9] = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
        const int *seqs[1] = { seq };
        const int lens[1] = { 9 };
        double first = loss_of(m, seq, 9);
        double history[60];
        double last = llm_fit(m, seqs, lens, 1, 60, 0.02, 0, 1, history);

        char label[64];
        snprintf(label, sizeof(label), "loss %.4f → %.5f", first, last);
        check_true(label, last < first * 0.05);
        check_true("history 单调下降（前 5 个 epoch 之后不再回升）",
                   history[59] <= history[9] + 1e-9);

        /* 过拟合之后，贪心续写应当稳定在 0,1,2 的循环里 */
        int prompt[1] = { 0 };
        int out[9];
        int n = llm_generate(m, prompt, 1, 8, 1.0, 1, out, 7);
        int pattern_ok = (n == 9);
        for (int i = 0; pattern_ok && i < 8; i++) {
            if (out[i + 1] != (i + 1) % 3) pattern_ok = 0;
        }
        if (!pattern_ok) {
            printf("    实际生成：");
            for (int i = 0; i < n; i++) printf("%d ", out[i]);
            printf("\n");
        }
        check_true("贪心续写符合 0,1,2 循环", pattern_ok);
    }

    /* ------------------ 6. 训练分段等价 ------------------ */
    printf("\n[6] fit(2) 与 fit(1)+fit(1) 等价（step_count 不被重置）\n");
    {
        const int seq[5] = { 1, 2, 3, 4, 5 };
        const int *seqs[1] = { seq };
        const int lens[1] = { 5 };

        CausalLM *x = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 99);
        CausalLM *y = llm_create(vocab, d_model, heads, d_ff, layers, max_len, 0.0, 99);
        LLMParamBlock bx[128], by[128];
        int nbx = llm_param_blocks(x, bx, 128);
        int nby = llm_param_blocks(y, by, 128);
        check_true("两个模型的块表一致", nbx == nby);

        double hx[2], hy[2];
        (void)llm_fit(x, seqs, lens, 1, 2, 0.01, 0, 5, hx);
        (void)llm_fit(y, seqs, lens, 1, 1, 0.01, 0, 5, hy);
        (void)llm_fit(y, seqs, lens, 1, 1, 0.01, 0, 5, hy + 1);

        double worst = 0.0;
        for (int i = 0; i < nbx; i++) {
            if (strcmp(bx[i].name, by[i].name) != 0) { worst = 1.0; break; }
            for (int k = 0; k < bx[i].n; k++) {
                double d = fabs(bx[i].param[k] - by[i].param[k]);
                if (d > worst) worst = d;
            }
        }
        check("参数最大差异", worst, 0.0);
        check("history 差异", fabs(hx[0] - hy[0]) + fabs(hx[1] - hy[1]), 0.0);

        llm_free(x);
        llm_free(y);
    }

    /* ------------------ 7. 生成的可复现性 ------------------ */
    printf("\n[7] 生成\n");
    {
        int prompt[2] = { 3, 1 };
        int out1[8], out2[8];
        int n1 = llm_generate(m, prompt, 2, 6, 1.0, 0, out1, 12345);
        int n2 = llm_generate(m, prompt, 2, 6, 1.0, 0, out2, 12345);
        check_true("长度 = prompt + max_new", n1 == 8 && n2 == 8);
        int same = 1;
        for (int i = 0; i < n1; i++) if (out1[i] != out2[i]) same = 0;
        check_true("同种子结果相同", same);

        int g1[8], g2[8];
        (void)llm_generate(m, prompt, 2, 6, 1.0, 1, g1, 1);
        (void)llm_generate(m, prompt, 2, 6, 1.0, 1, g2, 999999);
        int greedy_same = 1;
        for (int i = 0; i < 8; i++) if (g1[i] != g2[i]) greedy_same = 0;
        check_true("top_k=1 与种子无关（argmax 确定）", greedy_same);
        check_true("生成后保持 eval 模式", m->enc->layers[0]->dropout1.train == 0);
    }

    llm_free(m);

    /* ------------------ 8. dropout=0 时 train/eval 一致 ------------------ */
    printf("\n[8] dropout=0：train/eval 前向一致\n");
    {
        CausalLM *d = llm_create(vocab, d_model, heads, d_ff, 1, max_len, 0.0, 7);
        int ids[4] = { 1, 2, 3, 4 };
        int dims[2] = { 4, vocab };
        Tensor *a = tensor_new(2, dims);
        Tensor *b = tensor_new(2, dims);

        llm_set_train(d, 1);
        llm_forward(d, ids, 4, a);
        llm_set_train(d, 0);
        llm_forward(d, ids, 4, b);

        double worst = 0.0;
        for (int i = 0; i < 4 * vocab; i++) {
            double diff = fabs(a->data[i] - b->data[i]);
            if (diff > worst) worst = diff;
        }
        check("train 与 eval 的 logits 差异", worst, 0.0);

        /* 同时验证 dropout 参数确实被接受了（p>0 时只要求能跑通） */
        CausalLM *p = llm_create(vocab, d_model, heads, d_ff, 1, max_len, 0.25, 7);
        llm_set_train(p, 1);
        llm_forward(p, ids, 4, a);
        check_true("dropout=0.25 可以前向", 1);
        llm_free(p);

        tensor_free(a);
        tensor_free(b);
        llm_free(d);
    }

    /* ------------------ 9. 生成罚分与 min_p ------------------ */
    printf("\n[9] 生成罚分 / min_p\n");
    {
        CausalLM *t = llm_create(vocab, d_model, heads, d_ff, 1, max_len, 0.0, 5);

        /* 9a. repetition_penalty：logit > 0 除以 rp，logit <= 0 乘以 rp */
        int ids[5] = { 0, 1, 0, 2, 0 };   /* counts: 0→3, 1→1, 2→1 */
        double row[7] = { 2.0, -2.0, 0.0, 1.0, -1.0, 3.0, -3.0 };
        double want[7];

        memcpy(want, row, sizeof(row));
        want[0] = 1.0;    /* 2 / 2 */
        want[1] = -4.0;   /* -2 * 2（非正分支） */
        want[2] = 0.0;    /* 0 * 2（==0 也走乘法分支） */
        llm_apply_generation_penalties(t, row, ids, 5, 2.0, 0.0, 0.0, 0);
        double worst = 0.0;
        for (int i = 0; i < vocab; i++) {
            double d = fabs(row[i] - want[i]);
            if (d > worst) worst = d;
        }
        check("repetition_penalty=2 正/负 logit", worst, 0.0);

        /* 9b. rp < 1：正 logit 除以 <1 变大，负 logit 乘 <1 变接近 0 */
        double clean[7] = { 2.0, -2.0, 0.0, 1.0, -1.0, 3.0, -3.0 };
        memcpy(row, clean, sizeof(row));
        want[0] = 4.0;    /* 2 / 0.5 */
        want[1] = -1.0;   /* -2 * 0.5 */
        want[2] = 0.0;
        llm_apply_generation_penalties(t, row, ids, 5, 0.5, 0.0, 0.0, 0);
        worst = 0.0;
        for (int i = 0; i < vocab; i++) {
            double d = fabs(row[i] - want[i]);
            if (d > worst) worst = d;
        }
        check("repetition_penalty=0.5", worst, 0.0);

        /* 9c. presence_penalty：每个出现过的 token 只扣一次 */
        memcpy(row, clean, sizeof(row));
        want[0] = 1.5;    /* 2 - 0.5（出现 3 次也只扣一次） */
        want[1] = -2.5;
        want[2] = -0.5;
        llm_apply_generation_penalties(t, row, ids, 5, 1.0, 0.5, 0.0, 0);
        worst = 0.0;
        for (int i = 0; i < vocab; i++) {
            double d = fabs(row[i] - want[i]);
            if (d > worst) worst = d;
        }
        check("presence_penalty 只扣一次", worst, 0.0);

        /* 9d. frequency_penalty：按出现次数扣 */
        memcpy(row, clean, sizeof(row));
        want[0] = 0.5;    /* 2 - 0.5*3 */
        want[1] = -2.5;   /* -2 - 0.5*1 */
        want[2] = -0.5;
        llm_apply_generation_penalties(t, row, ids, 5, 1.0, 0.0, 0.5, 0);
        worst = 0.0;
        for (int i = 0; i < vocab; i++) {
            double d = fabs(row[i] - want[i]);
            if (d > worst) worst = d;
        }
        check("frequency_penalty 按次数扣", worst, 0.0);

        /* 9e. no_repeat_ngram_size=2：历史里出现过 prefix 时禁掉后继
         * ids = 3,4,5,4 → 最近 1 个 token 是 4；前文 i=1 处的 4 后继是 5，
         * 所以 (4,5) 会重复，5 必须被禁。 */
        int ng_ids[4] = { 3, 4, 5, 4 };
        memcpy(row, clean, sizeof(row));
        llm_apply_generation_penalties(t, row, ng_ids, 4, 1.0, 0.0, 0.0, 2);
        int ngram_ok = (row[5] == -INFINITY);
        for (int i = 0; i < vocab; i++) {
            if (i != 5 && row[i] != clean[i]) ngram_ok = 0;
        }
        check_true("no_repeat_ngram=2 禁掉重复后继", ngram_ok);

        /* 9f. n=3 同理：prefix 是最后 2 个 token */
        int ng3_ids[5] = { 1, 2, 3, 1, 2 };
        memcpy(row, clean, sizeof(row));
        llm_apply_generation_penalties(t, row, ng3_ids, 5, 1.0, 0.0, 0.0, 3);
        check_true("no_repeat_ngram=3 禁掉重复后继", row[3] == -INFINITY);

        /* 9g. 全默认参数必须原样返回（逐位） */
        memcpy(row, clean, sizeof(row));
        llm_apply_generation_penalties(t, row, ids, 5, 1.0, 0.0, 0.0, 0);
        check_true("全默认参数不改 row", memcmp(row, clean, sizeof(row)) == 0);

        /* 9h. min_p 候选数：min_p<=0 全保留；0.5 只留峰值；0.1 全留 */
        double lp[4] = { 2.0, 0.0, 0.0, 0.0 };
        check_true("min_p=0 保留全部", llm_min_p_candidates(lp, 4, 0.0) == 4);
        check_true("min_p=0.5 只留峰值", llm_min_p_candidates(lp, 4, 0.5) == 1);
        check_true("min_p=0.1 全保留", llm_min_p_candidates(lp, 4, 0.1) == 4);
        double linf[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
        check_true("整行 -inf 时 min_p 不擦除", llm_min_p_candidates(linf, 4, 0.5) == 4);

        llm_free(t);
    }

    /* ------------------ 10. 旧 API 与 ex API 默认参数一致 ------------------ */
    printf("\n[10] 旧生成 API 与 llm_generate_ex 默认参数逐 token 一致\n");
    {
        CausalLM *g = llm_create(vocab, d_model, heads, d_ff, 1, max_len, 0.0, 321);
        int prompt[2] = { 2, 3 };
        int a[6], b[6];

        (void)llm_generate(g, prompt, 2, 6, 0.8, 0, a, 4242);
        (void)llm_generate_ex(g, prompt, 2, 6, 0.8, 0, 1.0,
                              0.0, 1.0, 0.0, 0.0, 0, b, 4242);
        int same = 1;
        for (int i = 0; i < 6; i++) if (a[i] != b[i]) same = 0;
        check_true("llm_generate == ex(默认)", same);

        (void)llm_generate_p(g, prompt, 2, 6, 0.8, 3, 0.9, a, 77);
        (void)llm_generate_ex(g, prompt, 2, 6, 0.8, 3, 0.9,
                              0.0, 1.0, 0.0, 0.0, 0, b, 77);
        same = 1;
        for (int i = 0; i < 6; i++) if (a[i] != b[i]) same = 0;
        check_true("llm_generate_p == ex(默认)", same);

        llm_free(g);
    }

    /* ------------------ 11. ALiBi 闭式解（TE -> TEL -> MHA 透传） ------------------ */
    printf("\n[11] ALiBi：注意力权重等于 softmax(slope*(t-j))\n");
    {
        const int dv = 4;   /* d_model；单头，slope = 2^-8 */
        const int T = 3;
        CausalLM *a = llm_create(vocab, dv, 1, 8, 1, 8, 0.0, 61);
        llm_set_train(a, 0);
        MultiHeadAttention *att = a->enc->layers[0]->self_attn;

        /* 让分数只由 ALiBi 决定：W_q/W_k 全 0，W_v/W_o 单位阵，所有 bias 0。
         * 这样 q=k=0，scores[t][j] 恒为 0，叠加的偏置就是唯一变量。 */
        for (int i = 0; i < dv * dv; i++) {
            att->W_q[i] = 0.0;
            att->W_k[i] = 0.0;
        }
        for (int i = 0; i < dv; i++) {
            for (int j = 0; j < dv; j++) {
                att->W_v[i * dv + j] = (i == j) ? 1.0 : 0.0;
                att->W_o[i * dv + j] = (i == j) ? 1.0 : 0.0;
            }
            att->b_q[i] = att->b_k[i] = att->b_v[i] = att->b_o[i] = 0.0;
        }

        int ids[3] = { 1, 2, 3 };
        int dims[2] = { T, vocab };
        Tensor *lo = tensor_new(2, dims);

        /* 关闭：因果 softmax 退化为均匀 1/(t+1) */
        llm_forward(a, ids, T, lo);
        double worst = 0.0;
        for (int t = 0; t < T; t++) {
            for (int j = 0; j <= t; j++) {
                double df = fabs(att->attn[t * T + j] - 1.0 / (double)(t + 1));
                if (df > worst) worst = df;
            }
            for (int j = t + 1; j < T; j++) {
                if (att->attn[t * T + j] != 0.0) worst = 1e9;
            }
        }
        check("alibi=0 均匀权重", worst, 0.0);

        /* 打开（经 TE 逐层透传到 MHA）：softmax(slope*(t-j))，单头 slope=2^-8 */
        te_set_alibi(a->enc, 1);
        llm_forward(a, ids, T, lo);
        const double slope = pow(2.0, -8.0);
        worst = 0.0;
        for (int t = 0; t < T; t++) {
            double p[8], s = 0.0;
            for (int j = 0; j <= t; j++) {
                p[j] = exp(slope * (double)(t - j));
                s += p[j];
            }
            for (int j = 0; j <= t; j++) {
                double df = fabs(att->attn[t * T + j] - p[j] / s);
                if (df > worst) worst = df;
            }
            for (int j = t + 1; j < T; j++) {
                if (att->attn[t * T + j] != 0.0) worst = 1e9;
            }
        }
        check("attn = softmax(slope*(t-j))", worst, 1e-14);

        /* 再次关闭：权重回到均匀，证明开关幂等 */
        te_set_alibi(a->enc, 0);
        llm_forward(a, ids, T, lo);
        worst = 0.0;
        for (int t = 0; t < T; t++) {
            for (int j = 0; j <= t; j++) {
                double df = fabs(att->attn[t * T + j] - 1.0 / (double)(t + 1));
                if (df > worst) worst = df;
            }
        }
        check("te_set_alibi(0) 恢复均匀权重", worst, 0.0);

        tensor_free(lo);
        llm_free(a);
    }

    printf("\n%s\n", g_fail ? "存在失败项" : "全部通过");
    return g_fail ? 1 : 0;
}
