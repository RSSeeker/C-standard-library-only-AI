/*
 * compare_llama.c —— 读取 tools/llama_reference.py 生成的参考数据，
 * 用 C 的 LlamaModel 重算前向 logits / 参数量并逐项比对。
 *
 * 用法: compare_llama [llama_reference.json] [容差]
 */
#include "../json.h"
#include "../llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_fail = 0;
static int g_exact = 0;
static double g_tol = 1e-12;

static const JsonValue *need(const JsonValue *obj, const char *key)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/llama_reference.py）\n", key);
        exit(2);
    }
    return v;
}

static const JsonValue *find_weight(const JsonValue *weights, const char *name)
{
    int n = json_array_len(weights);
    for (int i = 0; i < n; i++) {
        const JsonValue *b = json_at(weights, i);
        const char *nm = json_str(json_get(b, "name"), "");
        if (nm && strcmp(nm, name) == 0) return b;
    }
    fprintf(stderr, "参考权重缺少 %s\n", name);
    exit(2);
}

static double *alloc_doubles_exact(const JsonValue *block, int expect, const char *who)
{
    const JsonValue *data = need(block, "data");
    int n = json_array_len(data);
    if (n != expect) {
        fprintf(stderr, "权重 %s 长度 %d，期望 %d\n", who, n, expect);
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) {
        fprintf(stderr, "compare_llama: 内存不足\n");
        exit(2);
    }
    for (int i = 0; i < n; i++) p[i] = json_num(json_at(data, i), 0.0);
    return p;
}

static void load_weight(const JsonValue *weights, const char *name, double *dst, int expect)
{
    const JsonValue *b = find_weight(weights, name);
    double *p = alloc_doubles_exact(b, expect, name);
    memcpy(dst, p, (size_t)expect * sizeof(double));
    free(p);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    g_checks++;
    double worst = 0.0;
    int at = -1;
    int exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    if (exact) g_exact++;
    if (worst > g_tol) {
        g_fail++;
        printf("    %-22s 最大差异 %.3e [超出容差] index %d C=%.17g py=%.17g\n",
               what, worst, at, at >= 0 ? got[at] : 0.0, at >= 0 ? want[at] : 0.0);
    } else {
        printf("    %-22s 最大差异 %.3e [OK]%s\n", what, worst, exact ? " (逐位相同)" : "");
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *path = (argc > 1) ? argv[1] : "llama_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/llama_reference.py %s\n", path);
        return 2;
    }

    const JsonValue *cfg = need(root, "config");
    int vocab = json_int(need(cfg, "vocab_size"), 0);
    int d_model = json_int(need(cfg, "d_model"), 0);
    int num_heads = json_int(need(cfg, "num_heads"), 0);
    int num_kv = json_int(need(cfg, "num_kv_heads"), 0);
    int d_ff = json_int(need(cfg, "d_ff"), 0);
    int num_layers = json_int(need(cfg, "num_layers"), 0);
    int max_len = json_int(need(cfg, "max_len"), 0);
    int want_params = json_int(need(cfg, "param_count"), 0);
    if (num_kv <= 0) num_kv = num_heads;
    int d_k = d_model / num_heads;
    int kv_dim = num_kv * d_k;

    LlamaModel *m = llama_create(vocab, d_model, num_heads, d_ff, num_layers,
                                 num_kv, max_len, 10000.0);
    const JsonValue *weights = need(root, "weights");

    /* Embedding */
    load_weight(weights, "emb.weight", m->emb->weight, vocab * d_model);

    /* 每层 */
    for (int li = 0; li < num_layers; li++) {
        LlamaBlock *blk = llama_block(m, li);
        char name[128];

#define LW(field, dst, count) \
        do { snprintf(name, sizeof(name), "L%d.%s", li, field); \
             load_weight(weights, name, (dst), (count)); } while (0)

        double *Wq = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wk = (double *)malloc((size_t)kv_dim * d_model * sizeof(double));
        double *Wv = (double *)malloc((size_t)kv_dim * d_model * sizeof(double));
        double *Wo = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *bq = (double *)malloc((size_t)d_model * sizeof(double));
        double *bk = (double *)malloc((size_t)kv_dim * sizeof(double));
        double *bv = (double *)malloc((size_t)kv_dim * sizeof(double));
        double *bo = (double *)malloc((size_t)d_model * sizeof(double));
        LW("attn.W_q", Wq, d_model * d_model);
        LW("attn.W_k", Wk, kv_dim * d_model);
        LW("attn.W_v", Wv, kv_dim * d_model);
        LW("attn.W_o", Wo, d_model * d_model);
        LW("attn.b_q", bq, d_model);
        LW("attn.b_k", bk, kv_dim);
        LW("attn.b_v", bv, kv_dim);
        LW("attn.b_o", bo, d_model);
        mha_set_params(llama_block_attn(blk), Wq, bq, Wk, bk, Wv, bv, Wo, bo);
        free(Wq); free(Wk); free(Wv); free(Wo);
        free(bq); free(bk); free(bv); free(bo);

        double *W1 = (double *)malloc((size_t)d_ff * d_model * sizeof(double));
        double *Wg = (double *)malloc((size_t)d_ff * d_model * sizeof(double));
        double *W2 = (double *)malloc((size_t)d_model * d_ff * sizeof(double));
        double *b1 = (double *)malloc((size_t)d_ff * sizeof(double));
        double *bg = (double *)malloc((size_t)d_ff * sizeof(double));
        double *b2 = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ffn.W1", W1, d_ff * d_model);
        LW("ffn.Wg", Wg, d_ff * d_model);
        LW("ffn.W2", W2, d_model * d_ff);
        LW("ffn.b1", b1, d_ff);
        LW("ffn.bg", bg, d_ff);
        LW("ffn.b2", b2, d_model);
        swiglu_set_params(llama_block_ffn(blk), W1, b1, Wg, bg, W2, b2);
        free(W1); free(Wg); free(W2); free(b1); free(bg); free(b2);

        double *g1 = (double *)malloc((size_t)d_model * sizeof(double));
        double *g2 = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ln1.gamma", g1, d_model);
        LW("ln2.gamma", g2, d_model);
        rmsnorm_set_params(llama_block_ln1(blk), g1);
        rmsnorm_set_params(llama_block_ln2(blk), g2);
        free(g1); free(g2);
#undef LW
    }

    double *gnorm = (double *)malloc((size_t)d_model * sizeof(double));
    load_weight(weights, "norm.gamma", gnorm, d_model);
    rmsnorm_set_params(llama_norm(m), gnorm);
    free(gnorm);

    double *Wout = (double *)malloc((size_t)vocab * d_model * sizeof(double));
    double *bout = (double *)malloc((size_t)vocab * sizeof(double));
    load_weight(weights, "out.W", Wout, vocab * d_model);
    load_weight(weights, "out.b", bout, vocab);
    llama_set_head(m, Wout, bout);
    free(Wout);
    free(bout);

    printf("config: vocab=%d d_model=%d heads=%d kv_heads=%d d_ff=%d layers=%d\n",
           vocab, d_model, num_heads, num_kv, d_ff, num_layers);
    int got_params = llama_param_count(m);
    g_checks++;
    if (got_params == want_params) {
        g_exact++;
        printf("    %-22s %d [OK] (逐位相同)\n", "param_count", got_params);
    } else {
        g_fail++;
        printf("    %-22s C=%d py=%d [不一致]\n", "param_count", got_params, want_params);
    }

    const JsonValue *cases = need(root, "cases");
    int n_cases = json_array_len(cases);
    for (int ci = 0; ci < n_cases; ci++) {
        const JsonValue *c = json_at(cases, ci);
        const char *name = json_str(need(c, "name"), "case");
        const JsonValue *ids_v = need(c, "ids");
        int T = json_array_len(ids_v);
        int *ids = (int *)malloc((size_t)T * sizeof(int));
        for (int i = 0; i < T; i++) ids[i] = json_int(json_at(ids_v, i), 0);

        const JsonValue *want_v = need(c, "logits");
        int rows = json_array_len(want_v);
        if (rows != T) {
            fprintf(stderr, "case %s logits 行数 %d != T %d\n", name, rows, T);
            return 2;
        }
        int cols = json_array_len(json_at(want_v, 0));
        double *want = (double *)malloc((size_t)T * cols * sizeof(double));
        for (int r = 0; r < T; r++) {
            const JsonValue *row = json_at(want_v, r);
            if (json_array_len(row) != cols) {
                fprintf(stderr, "case %s logits 列数不一致\n", name);
                return 2;
            }
            for (int j = 0; j < cols; j++) {
                want[(size_t)r * cols + j] = json_num(json_at(row, j), 0.0);
            }
        }
        if (cols != vocab) {
            fprintf(stderr, "case %s 列数 %d != vocab %d\n", name, cols, vocab);
            return 2;
        }

        double *got = (double *)malloc((size_t)T * vocab * sizeof(double));
        llama_forward(m, ids, T, got);
        report(name, got, want, T * vocab);
        free(ids);
        free(want);
        free(got);
    }

    llama_free(m);
    json_free(root);

    printf("=== llama: %d 项检查, 失败 %d, 逐位相同 %d ===\n", g_checks, g_fail, g_exact);
    return g_fail == 0 ? 0 : 1;
}
