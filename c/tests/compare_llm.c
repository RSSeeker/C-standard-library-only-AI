/*
 * compare_llm.c —— 读取 tools/llm_reference.py 导出的参考数据，
 * 用 C 的 CausalLM 重算前向 / loss / 梯度 / Adam 一步 / 训练 / 生成，逐项比对。
 *
 * 用法: compare_llm [llm_reference.json] [容差]
 * 退出码: 0 = 全部在容差内（默认要求逐位相同），1 = 存在超差项
 */
#include "../json.h"
#include "../llm.h"
#include "../tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static double g_tol = 0.0;          /* 默认要求逐位相同 */

#define MAX_BLOCKS 128

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/llm_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static int need_int(const JsonValue *obj, const char *key, const char *who)
{
    return (int)json_num(need(obj, key, who), 0.0);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    double worst = 0.0;
    int at = -1, exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    int ok = (worst <= g_tol);
    if (!ok) g_fail++;
    printf("  %-34s 最大差异 %.3e  %s%s\n", what, worst,
           ok ? "[OK]" : "[超出容差]", exact ? "  (逐位相同)" : "");
    if (!ok) printf("      index %d: C=%.17g py=%.17g\n", at, got[at], want[at]);
}

static void report_scalar(const char *what, double got, double want)
{
    report(what, &got, &want, 1);
}

/* 把 JSON 里的 {name: [values]} 灌进模型（两边从同一组权重出发） */
static void inject(LLMParamBlock *blocks, int nb, const JsonValue *obj, const char *who)
{
    int count = 0;
    for (int i = 0; i < nb; i++) {
        const JsonValue *v = json_get(obj, blocks[i].name);
        if (!v) {
            fprintf(stderr, "%s 缺少参数块 \"%s\" 的数值\n", who, blocks[i].name);
            exit(2);
        }
        if (json_array_len(v) != blocks[i].n) {
            fprintf(stderr, "%s 参数块 \"%s\" 长度不符：JSON %d, C %d\n",
                    who, blocks[i].name, json_array_len(v), blocks[i].n);
            exit(2);
        }
        for (int k = 0; k < blocks[i].n; k++)
            blocks[i].param[k] = json_num(json_at(v, k), 0.0);
        count++;
    }
    if (count != nb) {
        fprintf(stderr, "%s 只灌了 %d/%d 个块\n", who, count, nb);
        exit(2);
    }
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "llm_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python c/tools/llm_reference.py\n");
        return 1;
    }
    printf("参考数据: %s（容差 %.1e）\n\n", path, g_tol);

    const JsonValue *cfg = need(root, "config", "root");
    int vocab = need_int(cfg, "vocab_size", "config");
    int d_model = need_int(cfg, "d_model", "config");
    int heads = need_int(cfg, "num_heads", "config");
    int d_ff = need_int(cfg, "d_ff", "config");
    int layers = need_int(cfg, "num_layers", "config");
    int max_len = need_int(cfg, "max_len", "config");
    double dropout = json_num(need(cfg, "dropout", "config"), 0.0);
    const JsonValue *init = need(root, "init", "root");

    /* ------------------ 1. 前向 logits ------------------ */
    {
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");
        llm_set_train(m, 0);

        const JsonValue *ids_v = need(root, "forward_ids", "root");
        int T = json_array_len(ids_v);
        int *ids = malloc((size_t)T * sizeof(int));
        for (int i = 0; i < T; i++) ids[i] = (int)json_num(json_at(ids_v, i), 0.0);

        int dims[2] = { T, vocab };
        Tensor *logits = tensor_new(2, dims);
        llm_forward(m, ids, T, logits);

        const JsonValue *want = need(root, "forward_logits", "root");
        double *w = malloc((size_t)T * vocab * sizeof(double));
        for (int t = 0; t < T; t++) {
            const JsonValue *row = json_at(want, t);
            for (int v = 0; v < vocab; v++)
                w[t * vocab + v] = json_num(json_at(row, v), 0.0);
        }
        printf("[1] 前向 logits（T=%d）\n", T);
        report("logits", logits->data, w, T * vocab);

        free(w);
        free(ids);
        tensor_free(logits);
        llm_free(m);
    }

    /* ------------------ 2. loss 与梯度 ------------------ */
    {
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");
        llm_set_train(m, 0);

        const JsonValue *ids_v = need(root, "loss_ids", "root");
        int T = json_array_len(ids_v);
        int *ids = malloc((size_t)T * sizeof(int));
        for (int i = 0; i < T; i++) ids[i] = (int)json_num(json_at(ids_v, i), 0.0);

        llm_zero_grad(m);
        double loss = llm_loss_and_backward(m, ids, T);

        printf("\n[2] loss 与梯度（T=%d）\n", T);
        report_scalar("loss", loss, json_num(need(root, "loss", "root"), 0.0));

        const JsonValue *grads = need(root, "grads", "root");
        double *want = malloc((size_t)llm_param_count(m) * sizeof(double));
        (void)want;
        for (int i = 0; i < nb; i++) {
            const JsonValue *v = json_get(grads, blocks[i].name);
            if (!v) {
                fprintf(stderr, "grads 缺少 \"%s\"\n", blocks[i].name);
                exit(2);
            }
            int n = blocks[i].n;
            double *w = malloc((size_t)n * sizeof(double));
            for (int k = 0; k < n; k++) w[k] = json_num(json_at(v, k), 0.0);
            report(blocks[i].name, blocks[i].grad, w, n);
            free(w);
        }
        free(want);
        free(ids);
        llm_free(m);
    }

    /* ------------------ 3. 一步 Adam ------------------ */
    {
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");
        llm_set_train(m, 0);

        const JsonValue *ids_v = need(root, "loss_ids", "root");
        int T = json_array_len(ids_v);
        int *ids = malloc((size_t)T * sizeof(int));
        for (int i = 0; i < T; i++) ids[i] = (int)json_num(json_at(ids_v, i), 0.0);

        llm_zero_grad(m);
        (void)llm_loss_and_backward(m, ids, T);
        llm_adam_step(m, json_num(need(root, "adam_lr", "root"), 0.0));

        printf("\n[3] 一步 Adam 之后的参数\n");
        const JsonValue *after = need(root, "after_adam", "root");
        for (int i = 0; i < nb; i++) {
            const JsonValue *v = json_get(after, blocks[i].name);
            int n = blocks[i].n;
            double *w = malloc((size_t)n * sizeof(double));
            for (int k = 0; k < n; k++) w[k] = json_num(json_at(v, k), 0.0);
            report(blocks[i].name, blocks[i].param, w, n);
            free(w);
        }
        free(ids);
        llm_free(m);
    }

    /* ------------------ 4. 训练 3 个 epoch ------------------ */
    {
        const JsonValue *seqs_v = need(root, "fit_seqs", "root");
        int nseq = json_array_len(seqs_v);
        const int **seqs = malloc((size_t)nseq * sizeof(*seqs));
        int *lens = malloc((size_t)nseq * sizeof(int));
        for (int s = 0; s < nseq; s++) {
            const JsonValue *sv = json_at(seqs_v, s);
            int T = json_array_len(sv);
            int *ids = malloc((size_t)T * sizeof(int));
            for (int i = 0; i < T; i++) ids[i] = (int)json_num(json_at(sv, i), 0.0);
            seqs[s] = ids;
            lens[s] = T;
        }

        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");

        int epochs = need_int(root, "fit_epochs", "root");
        double lr = json_num(need(root, "fit_lr", "root"), 0.0);
        double *history = malloc((size_t)epochs * sizeof(double));

        printf("\n[4] 训练 %d 个 epoch（lr=%.4g）\n", epochs, lr);
        printf("    参数轨迹容差 1e-12（msvcrt/ucrtbase 的 exp/log 1-ULP 差异会随步数累积）\n");

        /* 每个 epoch 单独调用一次 llm_fit（epochs=1）：step_count 与 Adam 状态都延续，
         * 与一次调用 fit(epochs=N) 等价，但能定位到「第几个 epoch 开始出现差异」。 */
        const JsonValue *per = json_get(root, "per_epoch_fit");
        for (int ep = 0; ep < epochs; ep++) {
            double h = 0.0;
            (void)llm_fit(m, seqs, lens, nseq, 1, lr, 0, 777, &h);
            history[ep] = h;
            if (!per) continue;
            const JsonValue *snap = json_at(per, ep);
            int diff_blocks = 0;
            double worst = 0.0;
            const char *worst_name = "";
            for (int i = 0; i < nb; i++) {
                const JsonValue *v = json_get(snap, blocks[i].name);
                int n = blocks[i].n, bad = 0;
                for (int k = 0; k < n; k++) {
                    double wv = json_num(json_at(v, k), 0.0);
                    double d = fabs(blocks[i].param[k] - wv);
                    if (d > 0.0) bad = 1;
                    if (d > worst) { worst = d; worst_name = blocks[i].name; }
                }
                if (bad) diff_blocks++;
            }
            printf("  epoch %d: 不一致块 %d/%d%s", ep + 1, diff_blocks, nb,
                   diff_blocks ? "" : "  (全部逐位相同)");
            if (diff_blocks) printf("  最大差异 %.3e @ %s", worst, worst_name);
            printf("\n");
        }
        const JsonValue *want_h = need(root, "fit_history", "root");
        double *w = malloc((size_t)epochs * sizeof(double));
        for (int i = 0; i < epochs; i++) w[i] = json_num(json_at(want_h, i), 0.0);
        report("history", history, w, epochs);
        free(w);

        /* 参数轨迹这一段放宽到 1e-12：本机工具链是 x86_64-w64-mingw32-msvcrt，
         * 而 CPython 用 ucrtbase，两者的 exp/log 有 1-ULP 差异（实测 exp 9/2000、
         * log 6/2000 不同），多步训练后会沿轨迹累积（第 3 个 epoch 起 b_k 这类
         * 抵消敏感的块出现 1e-13 量级差异）。结构错误在 1e-2 量级，仍会被抓住。 */
        const double saved_tol = g_tol;
        g_tol = 1e-12;

        const JsonValue *after = need(root, "after_fit", "root");
        for (int i = 0; i < nb; i++) {
            const JsonValue *v = json_get(after, blocks[i].name);
            int n = blocks[i].n;
            double *ww = malloc((size_t)n * sizeof(double));
            for (int k = 0; k < n; k++) ww[k] = json_num(json_at(v, k), 0.0);
            report(blocks[i].name, blocks[i].param, ww, n);
            free(ww);
        }

        double ev = llm_evaluate(m, seqs, lens, nseq);
        report_scalar("eval_loss", ev, json_num(need(root, "eval_loss", "root"), 0.0));

        /* perplexity = exp(loss)：loss 逐位相同，但 exp 在 msvcrt/ucrtbase 上可能差
         * 1 ULP，所以这一段单独放宽到 1e-12（与训练轨迹同一原因）。 */
        g_tol = 1e-12;
        double ppl = llm_perplexity(m, seqs, lens, nseq);
        report_scalar("perplexity", ppl, json_num(need(root, "perplexity", "root"), 0.0));
        g_tol = saved_tol;

        free(history);
        for (int s = 0; s < nseq; s++) free((void *)seqs[s]);
        free((void *)seqs);
        free(lens);
        llm_free(m);
    }

    /* ------------------ 5. 生成（top_k=1，与随机数无关） ------------------ */
    {
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");

        const JsonValue *pv = need(root, "gen_prompt", "root");
        int pn = json_array_len(pv);
        int *prompt = malloc((size_t)pn * sizeof(int));
        for (int i = 0; i < pn; i++) prompt[i] = (int)json_num(json_at(pv, i), 0.0);
        int max_new = need_int(root, "gen_new", "root");

        int *out = malloc((size_t)(pn + max_new) * sizeof(int));
        int total = llm_generate(m, prompt, pn, max_new, 1.0, 1, out, 999);

        const JsonValue *want = need(root, "gen_tokens", "root");
        printf("\n[5] 生成（prompt %d 个 token，新增 %d）\n", pn, max_new);
        int bad = 0;
        if (json_array_len(want) != total) {
            bad = 1;
        } else {
            for (int i = 0; i < total; i++) {
                int wv = (int)json_num(json_at(want, i), 0.0);
                if (wv != out[i]) {
                    printf("  位置 %d: C=%d py=%d\n", i, out[i], wv);
                    bad = 1;
                }
            }
        }
        if (bad) {
            g_fail++;
            printf("  %-34s [不一致]\n", "tokens");
        } else {
            printf("  %-34s [OK]  (逐位相同)\n", "tokens");
        }

        free(prompt);
        free(out);
        llm_free(m);
    }

    /* ------------------ 6. top_p 的确定性对拍 ------------------
     * top_p 很小（1e-9）时 nucleus 只含 argmax（p_max >= 1/V >> 1e-9），
     * 所以采样结果与 RNG 无关，必须与 top_k=1 的贪心序列完全相同。
     * 任意 top_p 下的随机 token 不能跨语言逐位比较：C 用 splitmix64，
     * Python 用 random.MT19937，消耗的随机数序列不同。 */
    {
        CausalLM *m = llm_create(vocab, d_model, heads, d_ff, layers, max_len,
                                 dropout, 12345);
        LLMParamBlock blocks[MAX_BLOCKS];
        int nb = llm_param_blocks(m, blocks, MAX_BLOCKS);
        inject(blocks, nb, init, "init");

        const JsonValue *pv = need(root, "gen_prompt", "root");
        int pn = json_array_len(pv);
        int *prompt = malloc((size_t)pn * sizeof(int));
        for (int i = 0; i < pn; i++) prompt[i] = (int)json_num(json_at(pv, i), 0.0);
        int max_new = need_int(root, "gen_new", "root");
        double top_p = json_num(need(root, "gen_top_p", "root"), 0.0);

        int *out = malloc((size_t)(pn + max_new) * sizeof(int));
        int total = llm_generate_p(m, prompt, pn, max_new, 1.0, 0, top_p, out, 999);
        const JsonValue *want = need(root, "gen_tokens_top_p", "root");

        printf("\n[6] top_p=%.1e 生成（nucleus 只含 argmax，与 RNG 无关）\n", top_p);
        int bad = 0;
        if (json_array_len(want) != total) {
            bad = 1;
        } else {
            for (int i = 0; i < total; i++) {
                int wv = (int)json_num(json_at(want, i), 0.0);
                if (wv != out[i]) {
                    printf("  位置 %d: C=%d py=%d\n", i, out[i], wv);
                    bad = 1;
                }
            }
        }
        if (bad) {
            g_fail++;
            printf("  %-34s [不一致]\n", "tokens");
        } else {
            printf("  %-34s [OK]  (逐位相同)\n", "tokens");
        }

        free(prompt);
        free(out);
        llm_free(m);
    }

    printf("\n%s\n", g_fail ? "存在超差项" : "全部通过");
    return g_fail ? 1 : 0;
}
