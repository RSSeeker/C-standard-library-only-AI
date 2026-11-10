/*
 * compare_bert.c —— 读取 tools/bert_reference.py 生成的参考数据，
 * 用 C 的 BertModel 重算 hidden / MLM logits / NSP logits 并逐项比对。
 *
 * 用法: compare_bert [bert_reference.json] [容差]
 */
#include "../bert.h"
#include "../json.h"

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
        fprintf(stderr, "参考数据缺少字段 %s（请重新运行 tools/bert_reference.py）\n", key);
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
        fprintf(stderr, "compare_bert: 内存不足\n");
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
    const char *path = (argc > 1) ? argv[1] : "bert_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/bert_reference.py %s\n", path);
        return 2;
    }

    const JsonValue *cfg = need(root, "config");
    int vocab = json_int(need(cfg, "vocab_size"), 0);
    int d_model = json_int(need(cfg, "d_model"), 0);
    int num_heads = json_int(need(cfg, "num_heads"), 0);
    int d_ff = json_int(need(cfg, "d_ff"), 0);
    int num_layers = json_int(need(cfg, "num_layers"), 0);
    int max_len = json_int(need(cfg, "max_len"), 0);
    int want_params = json_int(need(cfg, "param_count"), 0);

    BertModel *m = bert_create(vocab, d_model, num_heads, d_ff, num_layers, max_len, 7);
    const JsonValue *weights = need(root, "weights");

    load_weight(weights, "tok_emb.weight", m->tok_emb->weight, vocab * d_model);
    load_weight(weights, "seg_emb.weight", m->seg_emb->weight, 2 * d_model);

    for (int li = 0; li < num_layers; li++) {
        TransformerEncoderLayer *layer = m->encoder->layers[li];
        char name[128];
        double *Wq = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wk = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wv = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *Wo = (double *)malloc((size_t)d_model * d_model * sizeof(double));
        double *bq = (double *)malloc((size_t)d_model * sizeof(double));
        double *bk = (double *)malloc((size_t)d_model * sizeof(double));
        double *bv = (double *)malloc((size_t)d_model * sizeof(double));
        double *bo = (double *)malloc((size_t)d_model * sizeof(double));
        if (!Wq || !Wk || !Wv || !Wo || !bq || !bk || !bv || !bo) exit(2);

#define LW(field, dst, count) \
        do { snprintf(name, sizeof(name), "enc.L%d.%s", li, field); \
             load_weight(weights, name, (dst), (count)); } while (0)

        LW("attn.W_q", Wq, d_model * d_model);
        LW("attn.W_k", Wk, d_model * d_model);
        LW("attn.W_v", Wv, d_model * d_model);
        LW("attn.W_o", Wo, d_model * d_model);
        LW("attn.b_q", bq, d_model);
        LW("attn.b_k", bk, d_model);
        LW("attn.b_v", bv, d_model);
        LW("attn.b_o", bo, d_model);
        mha_set_params(layer->self_attn, Wq, bq, Wk, bk, Wv, bv, Wo, bo);
        free(Wq); free(Wk); free(Wv); free(Wo);
        free(bq); free(bk); free(bv); free(bo);

        double *W1 = (double *)malloc((size_t)d_ff * d_model * sizeof(double));
        double *W2 = (double *)malloc((size_t)d_model * d_ff * sizeof(double));
        double *b1 = (double *)malloc((size_t)d_ff * sizeof(double));
        double *b2 = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ffn.W1", W1, d_ff * d_model);
        LW("ffn.W2", W2, d_model * d_ff);
        LW("ffn.b1", b1, d_ff);
        LW("ffn.b2", b2, d_model);
        ff_set_params(layer->ffn, W1, b1, W2, b2);
        free(W1); free(W2); free(b1); free(b2);

        double *g1 = (double *)malloc((size_t)d_model * sizeof(double));
        double *b1n = (double *)malloc((size_t)d_model * sizeof(double));
        double *g2 = (double *)malloc((size_t)d_model * sizeof(double));
        double *b2n = (double *)malloc((size_t)d_model * sizeof(double));
        LW("ln1.gamma", g1, d_model);
        LW("ln1.beta", b1n, d_model);
        LW("ln2.gamma", g2, d_model);
        LW("ln2.beta", b2n, d_model);
        memcpy(layer->ln1.gamma, g1, (size_t)d_model * sizeof(double));
        memcpy(layer->ln1.beta, b1n, (size_t)d_model * sizeof(double));
        memcpy(layer->ln2.gamma, g2, (size_t)d_model * sizeof(double));
        memcpy(layer->ln2.beta, b2n, (size_t)d_model * sizeof(double));
        free(g1); free(b1n); free(g2); free(b2n);
#undef LW
    }

    double *W_mlm = (double *)malloc((size_t)vocab * d_model * sizeof(double));
    double *b_mlm = (double *)malloc((size_t)vocab * sizeof(double));
    double *W_nsp = (double *)malloc((size_t)2 * d_model * sizeof(double));
    double *b_nsp = (double *)malloc((size_t)2 * sizeof(double));
    load_weight(weights, "mlm.W", W_mlm, vocab * d_model);
    load_weight(weights, "mlm.b", b_mlm, vocab);
    load_weight(weights, "nsp.W", W_nsp, 2 * d_model);
    load_weight(weights, "nsp.b", b_nsp, 2);
    bert_set_mlm(m, W_mlm, b_mlm);
    bert_set_nsp(m, W_nsp, b_nsp);
    free(W_mlm); free(b_mlm); free(W_nsp); free(b_nsp);

    printf("config: vocab=%d d_model=%d heads=%d d_ff=%d layers=%d\n",
           vocab, d_model, num_heads, d_ff, num_layers);

    int got_params = bert_param_count(m);
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
        const JsonValue *seg_v = need(c, "seg");
        int T = json_array_len(ids_v);
        int *ids = (int *)malloc((size_t)T * sizeof(int));
        int *seg = (int *)malloc((size_t)T * sizeof(int));
        if (!ids || !seg) exit(2);
        for (int i = 0; i < T; i++) {
            ids[i] = json_int(json_at(ids_v, i), 0);
            seg[i] = json_int(json_at(seg_v, i), 0);
        }

        int shape[2] = { T, d_model };
        Tensor *hidden = tensor_new(2, shape);
        bert_forward(m, ids, seg, T, hidden);

        const JsonValue *want_h = need(c, "hidden");
        double *wh = (double *)malloc((size_t)T * d_model * sizeof(double));
        for (int t = 0; t < T; t++) {
            const JsonValue *row = json_at(want_h, t);
            for (int j = 0; j < d_model; j++) wh[t * d_model + j] = json_num(json_at(row, j), 0.0);
        }
        char label[96];
        snprintf(label, sizeof(label), "%s.hidden", name);
        report(label, hidden->data, wh, T * d_model);
        free(wh);

        double *got_mlm = (double *)malloc((size_t)T * vocab * sizeof(double));
        double *want_mlm = (double *)malloc((size_t)T * vocab * sizeof(double));
        bert_mlm_logits(m, hidden, got_mlm);
        const JsonValue *want_m = need(c, "mlm");
        for (int t = 0; t < T; t++) {
            const JsonValue *row = json_at(want_m, t);
            for (int v = 0; v < vocab; v++) {
                want_mlm[t * vocab + v] = json_num(json_at(row, v), 0.0);
            }
        }
        snprintf(label, sizeof(label), "%s.mlm", name);
        report(label, got_mlm, want_mlm, T * vocab);
        free(got_mlm);
        free(want_mlm);

        double got_nsp[2];
        double want_nsp[2];
        bert_nsp_logits(m, hidden, got_nsp);
        const JsonValue *want_n = need(c, "nsp");
        for (int v = 0; v < 2; v++) want_nsp[v] = json_num(json_at(want_n, v), 0.0);
        snprintf(label, sizeof(label), "%s.nsp", name);
        report(label, got_nsp, want_nsp, 2);

        tensor_free(hidden);
        free(ids);
        free(seg);
    }

    printf("=== bert: %d 项检查, 失败 %d, 逐位相同 %d ===\n",
           g_checks, g_fail, g_exact);
    json_free(root);
    bert_free(m);
    return g_fail == 0 ? 0 : 1;
}
