/* compare_deberta.c -- 读取 tools/deberta_reference.py 生成的参考数据，
 * 用 C 的 DebertaModel 重算前向 logits / 参数量并逐项比对。
 *
 * 用法: compare_deberta [deberta_reference.json] [容差]
 */
#include "../json.h"
#include "../deberta.h"

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
        fprintf(stderr, "missing field %s\n", key);
        exit(2);
    }
    return v;
}

static double *read_doubles(const JsonValue *arr, int expect, const char *who)
{
    int n = json_array_len(arr);
    if (n != expect) {
        fprintf(stderr, "%s count %d expected %d\n", who, n, expect);
        exit(2);
    }
    double *p = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!p) exit(2);
    for (int i = 0; i < n; i++) p[i] = json_num(json_at(arr, i), 0.0);
    return p;
}

static void check_close(double a, double b, const char *what)
{
    g_checks++;
    double diff = fabs(a - b);
    if (diff == 0.0) g_exact++;
    if (diff > g_tol) {
        g_fail++;
        if (g_fail <= 10)
            fprintf(stderr, "FAIL %s: C=%.17g python=%.17g diff=%.3e\n", what, a, b, diff);
    }
}

static void check_int(int a, int b, const char *what)
{
    g_checks++;
    if (a == b) g_exact++;
    else {
        g_fail++;
        fprintf(stderr, "FAIL %s: C=%d python=%d\n", what, a, b);
    }
}

static void load_matrix(const JsonValue *w, double *dst, int rows, int cols, const char *who)
{
    const JsonValue *arr = need(w, "weights");
    if (json_array_len(arr) != rows) {
        fprintf(stderr, "%s rows mismatch\n", who);
        exit(2);
    }
    for (int i = 0; i < rows; i++) {
        const JsonValue *row = json_at(arr, i);
        if (json_array_len(row) != cols) {
            fprintf(stderr, "%s row %d cols mismatch\n", who, i);
            exit(2);
        }
        for (int j = 0; j < cols; j++) dst[(size_t)i * cols + j] = json_num(json_at(row, j), 0.0);
    }
}

static void load_bias(const JsonValue *w, double *dst, int n, const char *who)
{
    double *p = read_doubles(need(w, "biases"), n, who);
    memcpy(dst, p, (size_t)n * sizeof(double));
    free(p);
}

static void load_ln(const JsonValue *ln, LayerNorm *dst, int d, const char *who)
{
    double *g = read_doubles(need(ln, "gamma"), d, who);
    double *b = read_doubles(need(ln, "beta"), d, who);
    layernorm_set_params(dst, g, b);
    free(g);
    free(b);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "deberta_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "read %s failed: %s\n", path, err);
        return 2;
    }

    const JsonValue *cfg = need(root, "config");
    int vocab = json_int(json_get(cfg, "vocab_size"), 0);
    int d = json_int(json_get(cfg, "d_model"), 0);
    int ff = json_int(json_get(cfg, "d_ff"), 0);
    int layers = json_int(json_get(cfg, "num_layers"), 0);
    int max_len = json_int(json_get(cfg, "max_len"), 0);
    int max_rel = json_int(json_get(cfg, "max_rel"), 0);
    int want_params = json_int(json_get(cfg, "param_count"), 0);

    DebertaModel *m = deberta_create(vocab, d, ff, layers, max_len, max_rel, 1);
    check_int(deberta_param_count(m), want_params, "param_count");

    const JsonValue *weights = need(root, "weights");
    const JsonValue *emb = need(weights, "emb");
    for (int i = 0; i < vocab; i++) {
        const JsonValue *row = json_at(emb, i);
        for (int j = 0; j < d; j++) m->emb[(size_t)i * d + j] = json_num(json_at(row, j), 0.0);
    }

    const JsonValue *blocks = need(weights, "blocks");
    if (json_array_len(blocks) != layers) {
        fprintf(stderr, "blocks mismatch\n");
        return 2;
    }
    for (int li = 0; li < layers; li++) {
        const JsonValue *b = json_at(blocks, li);
        DebertaBlock *cb = &m->blocks[li];
        load_ln(need(b, "ln1"), cb->ln1, d, "ln1");
        load_ln(need(b, "ln2"), cb->ln2, d, "ln2");
        load_matrix(need(b, "Wq"), cb->Wq, d, d, "Wq");
        load_bias(need(b, "Wq"), cb->bq, d, "Wq.bias");
        load_matrix(need(b, "Wk"), cb->Wk, d, d, "Wk");
        load_bias(need(b, "Wk"), cb->bk, d, "Wk.bias");
        load_matrix(need(b, "Wv"), cb->Wv, d, d, "Wv");
        load_bias(need(b, "Wv"), cb->bv, d, "Wv.bias");
        load_matrix(need(b, "Wo"), cb->Wo, d, d, "Wo");
        load_bias(need(b, "Wo"), cb->bo, d, "Wo.bias");
        load_matrix(need(b, "W1"), cb->W1, ff, d, "W1");
        load_bias(need(b, "W1"), cb->b1, ff, "W1.bias");
        load_matrix(need(b, "W2"), cb->W2, d, ff, "W2");
        load_bias(need(b, "W2"), cb->b2, d, "W2.bias");
        {
            const JsonValue *rk = need(b, "Rk");
            const JsonValue *rq = need(b, "Rq");
            int n_rel = cb->n_rel;
            for (int r = 0; r < n_rel; r++) {
                const JsonValue *row_rk = json_at(rk, r);
                const JsonValue *row_rq = json_at(rq, r);
                for (int i = 0; i < d; i++) {
                    cb->Rk[(size_t)r * d + i] = json_num(json_at(row_rk, i), 0.0);
                    cb->Rq[(size_t)r * d + i] = json_num(json_at(row_rq, i), 0.0);
                }
            }
        }
    }

    load_ln(need(weights, "ln_out"), m->ln_out, d, "ln_out");
    load_matrix(need(weights, "W_out"), m->W_out, vocab, d, "W_out");
    load_bias(need(weights, "W_out"), m->b_out, vocab, "W_out.bias");

    const JsonValue *cases = need(root, "cases");
    for (int ci = 0; ci < json_array_len(cases); ci++) {
        const JsonValue *c = json_at(cases, ci);
        const JsonValue *ids_json = need(c, "ids");
        int T = json_array_len(ids_json);
        if (T > max_len) {
            fprintf(stderr, "case %d T=%d > max_len\n", ci, T);
            return 2;
        }
        int *ids = (int *)malloc((size_t)T * sizeof(int));
        for (int t = 0; t < T; t++) ids[t] = json_int(json_at(ids_json, t), 0);
        double *logits = (double *)calloc((size_t)T * vocab, sizeof(double));
        deberta_forward(m, ids, T, logits);
        const JsonValue *want = need(c, "logits");
        for (int t = 0; t < T; t++) {
            const JsonValue *row = json_at(want, t);
            for (int v = 0; v < vocab; v++) {
                char what[64];
                snprintf(what, sizeof(what), "logits[%d][%d]", t, v);
                check_close(logits[(size_t)t * vocab + v], json_num(json_at(row, v), 0.0), what);
            }
        }
        free(ids);
        free(logits);
    }

    printf("DeBERTa: %d checks, exact %d, failed %d, tol %.1e\n",
           g_checks, g_exact, g_fail, g_tol);
    deberta_free(m);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
