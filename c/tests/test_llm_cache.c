/* test_llm_cache.c —— CausalLM 增量 KV cache 的逐位一致性 + 加速比
 *
 * 验证目标（任务 1）：
 *   [1] 单 token 增量前向 llm_forward_step 的 logits 与整段重算 llm_forward 的
 *       最后一行逐位相同（delta == 0），覆盖绝对位置编码与 RoPE 两种模式。
 *   [2] 生成循环（含窗口滑动重建）的每一步 logits 与「把当前窗口整段重算」逐位相同；
 *       窗口长度取 > / == / < 生成长度三种情形。
 *   [3] llm_generate_cached 与 llm_generate_p 的 token 序列完全相同（贪心、top_k、
 *       top_p、温度四类采样），且同 seed 重复调用可复现。
 *   [3b] 带 min_p / repetition_penalty / presence_penalty / frequency_penalty /
 *       no_repeat_ngram_size 时，llm_generate_ex 与 llm_generate_cached_ex 的
 *       token 序列逐位相同（含窗口滑动重建）；全默认参数时旧包装与 ex 路径一致。
 *   [4] clock() 计时：prompt 8 + 新增 40、d_model 32、2/4 层，打印加速比。
 *
 * 全部比较容差 0（bit-exact）；不涉及训练轨迹，因此不需要 1e-12 的容差。
 * 构建（在仓库根目录）：
 *   gcc -std=c17 -O2 -Wall -Wextra -o c/tests/test_llm_cache.exe \
 *       c/tests/test_llm_cache.c $SRC -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../llm.h"

static int g_fail = 0;
static int g_checks = 0;

static void check_true(const char *what, int cond)
{
    g_checks++;
    if (!cond) {
        printf("  FAIL  %s\n", what);
        g_fail++;
    }
}

static void check_row(const char *what, const double *got, const double *want, int n)
{
    g_checks++;
    double worst = 0.0;
    int at = -1;
    for (int i = 0; i < n; i++) {
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    if (!(worst == 0.0)) {
        printf("  FAIL  %s  (最大 |delta| = %.17g @ %d: got %.17g want %.17g)\n",
               what, worst, at, at >= 0 ? got[at] : 0.0, at >= 0 ? want[at] : 0.0);
        g_fail++;
    }
}

static void check_int(const char *what, int got, int want)
{
    g_checks++;
    if (got != want) {
        printf("  FAIL  %s  (got %d want %d)\n", what, got, want);
        g_fail++;
    }
}

static CausalLM *make_model(int V, int d, int h, int ff, int layers, int max_len,
                            int rope, unsigned long long seed)
{
    CausalLM *m = llm_create(V, d, h, ff, layers, max_len, 0.0, seed);
    llm_set_train(m, 0);
    llm_set_rope(m, rope, 10000.0);
    return m;
}

static int argmax_row(const double *row, int V)
{
    int best = 0;
    for (int v = 1; v < V; v++)
        if (row[v] > row[best]) best = v;
    return best;
}

/* ---------------- [1] 单步 logits vs 整段重算 ---------------- */

static void test_step_logits(void)
{
    const int V = 13, d = 32, h = 4, ff = 64, max_len = 16;
    const int ids[10] = { 2, 5, 1, 7, 3, 0, 4, 6, 2, 1 };
    printf("[1] 单 token 前向 vs 整段前向（last row），容差 0\n");

    for (int rope = 0; rope <= 1; rope++) {
        for (int layers = 1; layers <= 4; layers += 3) {   /* 1 层与 4 层 */
            CausalLM *m = make_model(V, d, h, ff, layers, max_len, rope,
                                     1000u + (unsigned)rope * 17u + (unsigned)layers);
            double step_logits[64];
            for (int T = 1; T <= 10; T++) {
                int dims[2] = { T, V };
                Tensor *full = tensor_new(2, dims);   /* (T, V) */
                llm_forward(m, ids, T, full);

                llm_cache_reset(m);
                for (int t = 0; t < T; t++)
                    llm_forward_step(m, ids[t], t, step_logits);
                /* 只比较最后一行（增量前向只产生这一行） */
                char what[128];
                snprintf(what, sizeof(what), "rope=%d L=%d T=%d", rope, layers, T);
                check_row(what, step_logits, full->data + (size_t)(T - 1) * V, V);
                tensor_free(full);
            }
            llm_free(m);
        }
    }
}

/* ---------------- [2] 生成循环逐步 logits（含滑动重建） ---------------- */

static void test_loop_logits(int max_len, int rope)
{
    const int V = 13, d = 32, h = 4, ff = 64, layers = 2;
    const int prompt[4] = { 3, 1, 4, 1 };
    const int max_new = 12;
    int ids[4 + 12];
    memcpy(ids, prompt, sizeof(prompt));

    CausalLM *m = make_model(V, d, h, ff, layers, max_len, rope,
                             7000u + (unsigned)max_len * 31u + (unsigned)rope);
    double logits[64];

    int len = 4;
    int start = 0;
    llm_cache_reset(m);
    for (int i = 0; i < len; i++) llm_forward_step(m, ids[i], i, logits);
    int pos = len;

    for (int step = 0; step < max_new; step++) {
        /* 当前窗口整段重算，和 cache 的最后一行逐位对照 */
        int T = len - start;
        int dims[2] = { T, V };
        Tensor *full = tensor_new(2, dims);
        llm_forward(m, ids + start, T, full);
        char what[128];
        snprintf(what, sizeof(what), "max_len=%d rope=%d step=%d", max_len, rope, step);
        check_row(what, logits, full->data + (size_t)(T - 1) * V, V);
        tensor_free(full);

        int pick = argmax_row(logits, V);   /* 贪心，只为推进序列 */
        ids[len++] = pick;
        int new_start = len > max_len ? len - max_len : 0;
        if (new_start == start) {
            llm_forward_step(m, pick, pos, logits);
            pos++;
        } else {
            start = new_start;
            llm_cache_reset(m);
            for (int i = 0; i < max_len; i++)
                llm_forward_step(m, ids[start + i], i, logits);
            pos = max_len;
        }
    }
    llm_free(m);
}

/* ---------------- [3] 生成 token 序列：cached vs 重算 ---------------- */

static void test_tokens(int max_len, int rope, double temperature, int top_k, double top_p)
{
    const int V = 13, d = 32, h = 4, ff = 64, layers = 2;
    const int prompt[4] = { 2, 7, 1, 5 };
    const int max_new = 12;
    int a[64], b[64];

    CausalLM *m = make_model(V, d, h, ff, layers, max_len, rope, 424242u);
    int na = llm_generate_p(m, prompt, 4, max_new, temperature, top_k, top_p, a, 1234567u);
    int nb = llm_generate_cached(m, prompt, 4, max_new, temperature, top_k, top_p, b, 1234567u);

    char what[160];
    snprintf(what, sizeof(what), "len  max_len=%d rope=%d T=%.1f k=%d p=%.2f",
             max_len, rope, temperature, top_k, top_p);
    check_int(what, nb, na);
    check_true("token 序列", na == nb && memcmp(a, b, (size_t)na * sizeof(int)) == 0);
    if (na != nb || memcmp(a, b, (size_t)na * sizeof(int)) != 0) {
        printf("    p   :");
        for (int i = 0; i < na; i++) printf(" %d", a[i]);
        printf("\n    cache:");
        for (int i = 0; i < nb; i++) printf(" %d", b[i]);
        printf("\n");
    }

    /* 同 seed 再跑一次 cached：cache 重置与 RNG 消耗都必须可复现 */
    int c[64];
    int nc = llm_generate_cached(m, prompt, 4, max_new, temperature, top_k, top_p, c, 1234567u);
    check_int("重复调用长度", nc, nb);
    check_true("同 seed 可复现", nc == nb && memcmp(b, c, (size_t)nb * sizeof(int)) == 0);

    llm_free(m);
}

/* ---------------- [3b] 带罚分 / min_p 的 cached vs 重算 ---------------- */

static void test_tokens_penalties(int max_len, int rope, double temperature,
                                  double min_p, double rp, double pp, double fp,
                                  int ngram)
{
    const int V = 13, d = 32, h = 4, ff = 64, layers = 2;
    const int prompt[4] = { 2, 7, 1, 5 };
    const int max_new = 18;            /* 超过 max_len=4/8 时必然触发滑动重建 */
    int a[64], b[64];

    CausalLM *m = make_model(V, d, h, ff, layers, max_len, rope, 555u);
    int na = llm_generate_ex(m, prompt, 4, max_new, temperature, 0, 1.0,
                             min_p, rp, pp, fp, ngram, a, 24680u);
    int nb = llm_generate_cached_ex(m, prompt, 4, max_new, temperature, 0, 1.0,
                                    min_p, rp, pp, fp, ngram, b, 24680u);

    char what[200];
    snprintf(what, sizeof(what),
             "ex len max_len=%d rope=%d T=%.1f min=%.2f rp=%.1f pp=%.1f fp=%.1f n=%d",
             max_len, rope, temperature, min_p, rp, pp, fp, ngram);
    check_int(what, nb, na);
    check_true("ex token 序列", na == nb && memcmp(a, b, (size_t)na * sizeof(int)) == 0);
    if (na != nb || memcmp(a, b, (size_t)na * sizeof(int)) != 0) {
        printf("    full :");
        for (int i = 0; i < na; i++) printf(" %d", a[i]);
        printf("\n    cache:");
        for (int i = 0; i < nb; i++) printf(" %d", b[i]);
        printf("\n");
    }

    /* 全默认参数时，旧包装必须和新 ex 路径给出同一串 token */
    if (min_p == 0.0 && rp == 1.0 && pp == 0.0 && fp == 0.0 && ngram == 0) {
        int c[64];
        int nc = llm_generate_cached(m, prompt, 4, max_new, temperature, 0, 1.0,
                                     c, 24680u);
        check_int("旧包装长度 == ex 默认", nc, nb);
        check_true("旧包装 == ex 默认",
                   nc == nb && memcmp(b, c, (size_t)nb * sizeof(int)) == 0);
    }

    llm_free(m);
}

/* ---------------- [4] 计时 ---------------- */

static double now_sec(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void bench(int layers, int max_len, int reps)
{
    const int V = 13, d = 32, h = 4, ff = 64;
    const int prompt[8] = { 2, 5, 1, 7, 3, 0, 4, 6 };
    const int max_new = 40;
    int a[128], b[128];

    CausalLM *m = make_model(V, d, h, ff, layers, max_len, 0, 90210u);

    /* 正确性：先确认这一组配置下两条路径 token 相同 */
    int na = llm_generate_p(m, prompt, 8, max_new, 1.0, 1, 1.0, a, 777u);
    int nb = llm_generate_cached(m, prompt, 8, max_new, 1.0, 1, 1.0, b, 777u);
    check_int("bench 长度", nb, na);
    check_true("bench token 序列", na == nb && memcmp(a, b, (size_t)na * sizeof(int)) == 0);

    double t0 = now_sec();
    for (int r = 0; r < reps; r++)
        llm_generate_p(m, prompt, 8, max_new, 1.0, 1, 1.0, a, 777u);
    double t_re = (now_sec() - t0) / (double)reps;

    t0 = now_sec();
    for (int r = 0; r < reps; r++)
        llm_generate_cached(m, prompt, 8, max_new, 1.0, 1, 1.0, b, 777u);
    double t_ca = (now_sec() - t0) / (double)reps;

    printf("    L=%d max_len=%-3d 重算 %8.3f ms   cached %8.3f ms   加速 %6.2fx\n",
           layers, max_len, t_re * 1e3, t_ca * 1e3, t_re / t_ca);

    llm_free(m);
}

int main(void)
{
    printf("=== CausalLM 增量 KV cache：逐位一致性与加速比 ===\n");

    test_step_logits();

    printf("[2] 生成循环逐步 logits vs 整段重算（max_len 大于/等于/小于生成长度），容差 0\n");
    /* prompt 4 + 新增 12 = 16；max_len 20/16/8/5/4 覆盖三种情形与更激进的滑动 */
    const int lens[5] = { 20, 16, 8, 5, 4 };
    for (int i = 0; i < 5; i++)
        for (int rope = 0; rope <= 1; rope++)
            test_loop_logits(lens[i], rope);

    printf("[3] token 序列 cached vs 重算（贪心 / top_k / top_p / 温度）\n");
    const int ml[5] = { 4, 5, 8, 16, 64 };
    const double temps[5] = { 1.0, 1.0, 0.8, 1.2, 1.0 };
    const int ks[5] = { 1, 3, 0, 5, 0 };
    const double ps[5] = { 1.0, 0.9, 0.9, 1.0, 0.5 };
    for (int i = 0; i < 5; i++)
        for (int rope = 0; rope <= 1; rope++)
            for (int c = 0; c < 5; c++)
                test_tokens(ml[i], rope, temps[c], ks[c], ps[c]);

    printf("[3b] token 序列 cached_ex vs generate_ex（罚分 / min_p / 窗口滑动）\n");
    {
        struct { int ml, rope, ng; double T, min_p, rp, pp, fp; } pc[] = {
            { 64, 0, 0, 1.0, 0.05, 1.0, 0.0, 0.0 },   /* 只开 min_p */
            { 16, 1, 0, 1.0, 0.00, 2.0, 0.0, 0.0 },   /* rp > 1（正分支） */
            {  8, 0, 0, 0.9, 0.00, 0.5, 0.0, 0.0 },   /* rp < 1 + 窗口滑动 */
            {  8, 1, 0, 1.0, 0.02, 1.5, 1.0, 0.5 },   /* presence + frequency */
            {  4, 0, 2, 0.8, 0.10, 1.0, 0.5, 0.5 },   /* 立即滑动 + ngram=2 */
            {  5, 1, 3, 1.0, 0.00, 1.0, 0.0, 0.0 },   /* ngram=3 */
            {  8, 0, 2, 0.9, 0.05, 2.0, 1.0, 1.0 },   /* 全开 */
        };
        int n = (int)(sizeof(pc) / sizeof(pc[0]));
        for (int i = 0; i < n; i++) {
            test_tokens_penalties(pc[i].ml, pc[i].rope, pc[i].T, pc[i].min_p,
                                  pc[i].rp, pc[i].pp, pc[i].fp, pc[i].ng);
        }
    }

    printf("[4] 计时：prompt 8 + 新增 40，d_model=32\n");
    bench(2, 64, 20);    /* 不滑动：理论收益最大 */
    bench(4, 64, 20);
    bench(2, 16, 20);    /* 每步都滑动：cached 每步重建 16 个 token，收益变小 */
    bench(4, 16, 20);

    printf("\n%d 项检查，%d 项失败\n", g_checks, g_fail);
    if (g_fail == 0) printf("PASS\n");
    else printf("FAIL\n");
    return g_fail ? 1 : 0;
}
