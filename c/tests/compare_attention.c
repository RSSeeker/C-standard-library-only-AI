/*
 * compare_attention.c —— 读取 tools/attention_reference.py 导出的参考数据，
 * 用 C 的 MultiHeadAttention / TransformerEncoderLayer 重算前向、反向与增量
 * 前向，逐项比对。
 *
 * 覆盖：普通因果 MHA、滑动窗口、RoPE、GQA、MQA、窗口+RoPE+GQA 组合、
 *       TEL（LN + FFN + 注意力）的 forward + backward。
 *
 * 用法: compare_attention [attention_reference.json] [容差]
 * 退出码: 0 = 全部在容差内（默认要求逐位相同），1 = 存在超差项
 */
#include "../attention.h"
#include "../json.h"
#include "../tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static double g_tol = 0.0;

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = obj ? json_get(obj, key) : NULL;
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/attention_reference.py）\n",
                who, key);
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

/* 把 JSON 数组读进新分配的 double[]，长度必须等于 expect */
static double *load_flat(const JsonValue *obj, const char *key, int expect, const char *who)
{
    const JsonValue *v = need(obj, key, who);
    if (json_array_len(v) != expect) {
        fprintf(stderr, "%s.%s 长度不符：JSON %d, 期望 %d\n",
                who, key, json_array_len(v), expect);
        exit(2);
    }
    double *a = malloc((size_t)expect * sizeof(double));
    if (!a) { fprintf(stderr, "内存不足\n"); exit(2); }
    for (int i = 0; i < expect; i++) a[i] = json_num(json_at(v, i), 0.0);
    return a;
}

static Tensor *tensor_flat(const JsonValue *v, int rows, int cols, const char *who)
{
    if (json_array_len(v) != rows * cols) {
        fprintf(stderr, "%s 长度不符：JSON %d, 期望 %d\n",
                who, json_array_len(v), rows * cols);
        exit(2);
    }
    int dims[2] = { rows, cols };
    Tensor *t = tensor_new(2, dims);
    for (int i = 0; i < rows * cols; i++)
        t->data[i] = json_num(json_at(v, i), 0.0);
    return t;
}

/* ---------------- MHA：forward / backward / forward_step ---------------- */

static void run_mha(const JsonValue *c)
{
    const JsonValue *cfg = need(c, "config", "case");
    const int d = need_int(cfg, "d_model", "config");
    const int h = need_int(cfg, "num_heads", "config");
    const int kv = need_int(cfg, "num_kv_heads", "config");
    const int max_seq = need_int(cfg, "max_seq", "config");
    const int causal = need_int(cfg, "causal", "config");
    const int window = need_int(cfg, "window", "config");
    const int rope = need_int(cfg, "rope", "config");
    const double rope_base = json_num(need(cfg, "rope_base", "config"), 10000.0);
    const int dk = d / h;
    const int kv_dim = kv * dk;

    const JsonValue *xv = need(c, "x", "case");
    const int T = json_array_len(xv) / d;
    if (T * d != json_array_len(xv)) {
        fprintf(stderr, "x 长度 %d 不是 d_model=%d 的整数倍\n", json_array_len(xv), d);
        exit(2);
    }

    MultiHeadAttention *m = (kv == h) ? mha_create(d, h, max_seq, 0.0)
                                      : mha_create_gqa(d, h, kv, max_seq, 0.0);
    mha_set_causal(m, causal);
    mha_set_window(m, window);
    if (rope) mha_set_rope(m, 1, rope_base);

    const JsonValue *p = need(c, "params", "case");
    double *Wq = load_flat(p, "W_q", d * d, "params");
    double *bq = load_flat(p, "b_q", d, "params");
    double *Wk = load_flat(p, "W_k", kv_dim * d, "params");
    double *bk = load_flat(p, "b_k", kv_dim, "params");
    double *Wv = load_flat(p, "W_v", kv_dim * d, "params");
    double *bv = load_flat(p, "b_v", kv_dim, "params");
    double *Wo = load_flat(p, "W_o", d * d, "params");
    double *bo = load_flat(p, "b_o", d, "params");
    mha_set_params(m, Wq, bq, Wk, bk, Wv, bv, Wo, bo);
    mha_zero_grad(m);

    Tensor *x = tensor_flat(xv, T, d, "x");
    Tensor *dout = tensor_flat(need(c, "dout", "case"), T, d, "dout");
    int dims[2] = { T, d };
    Tensor *out = tensor_new(2, dims);
    Tensor *dx = tensor_new(2, dims);

    mha_forward(m, x, out);
    mha_backward(m, dout, dx);

    const JsonValue *g = need(c, "grads", "case");
    printf("  out:\n");
    report("out", out->data, load_flat(c, "out", T * d, "case"), T * d);
    report("dx", dx->data, load_flat(c, "dx", T * d, "case"), T * d);

    printf("  参数梯度:\n");
    report("grad_W_q", m->grad_W_q, load_flat(g, "W_q", d * d, "grads"), d * d);
    report("grad_b_q", m->grad_b_q, load_flat(g, "b_q", d, "grads"), d);
    report("grad_W_k", m->grad_W_k, load_flat(g, "W_k", kv_dim * d, "grads"), kv_dim * d);
    report("grad_b_k", m->grad_b_k, load_flat(g, "b_k", kv_dim, "grads"), kv_dim);
    report("grad_W_v", m->grad_W_v, load_flat(g, "W_v", kv_dim * d, "grads"), kv_dim * d);
    report("grad_b_v", m->grad_b_v, load_flat(g, "b_v", kv_dim, "grads"), kv_dim);
    report("grad_W_o", m->grad_W_o, load_flat(g, "W_o", d * d, "grads"), d * d);
    report("grad_b_o", m->grad_b_o, load_flat(g, "b_o", d, "grads"), d);

    printf("  增量前向（forward_step，逐 token）:\n");
    {
        const JsonValue *want = need(c, "steps", "case");
        double *all = load_flat(c, "steps", T * d, "case");
        int bad = 0;
        mha_cache_reset(m);
        for (int t = 0; t < T; t++) {
            double *so = malloc((size_t)d * sizeof(double));
            mha_forward_step(m, x->data + (size_t)t * d, so);
            double worst = 0.0;
            for (int j = 0; j < d; j++) {
                double df = fabs(so[j] - all[t * d + j]);
                if (df > worst) worst = df;
            }
            if (worst > g_tol) {
                bad = 1;
                printf("      step %d 最大差异 %.3e\n", t, worst);
            }
            free(so);
        }
        if (bad) { g_fail++; printf("  %-34s [超出容差]\n", "steps"); }
        else printf("  %-34s [OK]  (逐位相同)\n", "steps");
        (void)want;
        free(all);
    }

    free(Wq); free(bq); free(Wk); free(bk); free(Wv); free(bv); free(Wo); free(bo);
    tensor_free(x); tensor_free(dout); tensor_free(out); tensor_free(dx);
    mha_free(m);
}

/* ---------------- TEL：forward / backward ---------------- */

static void run_tel(const JsonValue *c)
{
    const JsonValue *cfg = need(c, "config", "case");
    const int d = need_int(cfg, "d_model", "config");
    const int h = need_int(cfg, "num_heads", "config");
    const int ff = need_int(cfg, "d_ff", "config");
    const int max_seq = need_int(cfg, "max_seq", "config");
    const int causal = need_int(cfg, "causal", "config");
    const double dropout = json_num(need(cfg, "dropout", "config"), 0.0);
    const int dk = d / h;
    const int kv_dim = h * dk;   /* TEL 的 MHA 目前固定 num_kv_heads == num_heads */

    const JsonValue *xv = need(c, "x", "case");
    const int T = json_array_len(xv) / d;

    TransformerEncoderLayer *l = tel_create(d, h, ff, max_seq, dropout, 12345);
    tel_set_causal(l, causal);
    tel_set_train(l, 0);

    const JsonValue *p = need(c, "params", "case");
    double *Wq = load_flat(p, "W_q", d * d, "params");
    double *bq = load_flat(p, "b_q", d, "params");
    double *Wk = load_flat(p, "W_k", kv_dim * d, "params");
    double *bk = load_flat(p, "b_k", kv_dim, "params");
    double *Wv = load_flat(p, "W_v", kv_dim * d, "params");
    double *bv = load_flat(p, "b_v", kv_dim, "params");
    double *Wo = load_flat(p, "W_o", d * d, "params");
    double *bo = load_flat(p, "b_o", d, "params");
    mha_set_params(l->self_attn, Wq, bq, Wk, bk, Wv, bv, Wo, bo);

    double *W1 = load_flat(p, "W1", ff * d, "params");
    double *b1 = load_flat(p, "b1", ff, "params");
    double *W2 = load_flat(p, "W2", d * ff, "params");
    double *b2 = load_flat(p, "b2", d, "params");
    ff_set_params(l->ffn, W1, b1, W2, b2);

    double *ln1g = load_flat(p, "ln1_gamma", d, "params");
    double *ln1b = load_flat(p, "ln1_beta", d, "params");
    double *ln2g = load_flat(p, "ln2_gamma", d, "params");
    double *ln2b = load_flat(p, "ln2_beta", d, "params");
    memcpy(l->ln1.gamma, ln1g, (size_t)d * sizeof(double));
    memcpy(l->ln1.beta, ln1b, (size_t)d * sizeof(double));
    memcpy(l->ln2.gamma, ln2g, (size_t)d * sizeof(double));
    memcpy(l->ln2.beta, ln2b, (size_t)d * sizeof(double));

    tel_zero_grad(l);
    Tensor *x = tensor_flat(xv, T, d, "x");
    Tensor *dout = tensor_flat(need(c, "dout", "case"), T, d, "dout");
    int dims[2] = { T, d };
    Tensor *out = tensor_new(2, dims);
    Tensor *dx = tensor_new(2, dims);
    tel_forward(l, x, out);
    tel_backward(l, dout, dx);

    const JsonValue *g = need(c, "grads", "case");
    printf("  out / dx:\n");
    report("out", out->data, load_flat(c, "out", T * d, "case"), T * d);
    report("dx", dx->data, load_flat(c, "dx", T * d, "case"), T * d);

    printf("  注意力梯度:\n");
    report("grad_W_q", l->self_attn->grad_W_q, load_flat(g, "W_q", d * d, "grads"), d * d);
    report("grad_b_q", l->self_attn->grad_b_q, load_flat(g, "b_q", d, "grads"), d);
    report("grad_W_k", l->self_attn->grad_W_k, load_flat(g, "W_k", d * d, "grads"), d * d);
    report("grad_b_k", l->self_attn->grad_b_k, load_flat(g, "b_k", d, "grads"), d);
    report("grad_W_v", l->self_attn->grad_W_v, load_flat(g, "W_v", d * d, "grads"), d * d);
    report("grad_b_v", l->self_attn->grad_b_v, load_flat(g, "b_v", d, "grads"), d);
    report("grad_W_o", l->self_attn->grad_W_o, load_flat(g, "W_o", d * d, "grads"), d * d);
    report("grad_b_o", l->self_attn->grad_b_o, load_flat(g, "b_o", d, "grads"), d);

    printf("  FFN / LN 梯度:\n");
    report("grad_W1", l->ffn->grad_W1, load_flat(g, "W1", ff * d, "grads"), ff * d);
    report("grad_b1", l->ffn->grad_b1, load_flat(g, "b1", ff, "grads"), ff);
    report("grad_W2", l->ffn->grad_W2, load_flat(g, "W2", d * ff, "grads"), d * ff);
    report("grad_b2", l->ffn->grad_b2, load_flat(g, "b2", d, "grads"), d);
    report("ln1.gamma", l->ln1.ggamma, load_flat(g, "ln1_gamma", d, "grads"), d);
    report("ln1.beta", l->ln1.gbeta, load_flat(g, "ln1_beta", d, "grads"), d);
    report("ln2.gamma", l->ln2.ggamma, load_flat(g, "ln2_gamma", d, "grads"), d);
    report("ln2.beta", l->ln2.gbeta, load_flat(g, "ln2_beta", d, "grads"), d);

    free(Wq); free(bq); free(Wk); free(bk); free(Wv); free(bv); free(Wo); free(bo);
    free(W1); free(b1); free(W2); free(b2);
    free(ln1g); free(ln1b); free(ln2g); free(ln2b);
    tensor_free(x); tensor_free(dout); tensor_free(out); tensor_free(dx);
    tel_free(l);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "attention_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python c/tools/attention_reference.py\n");
        return 1;
    }

    const JsonValue *cases = need(root, "cases", "root");
    const int n = json_array_len(cases);
    printf("参考数据: %s（%d 个用例，容差 %.1e）\n", path, n, g_tol);

    for (int i = 0; i < n; i++) {
        const JsonValue *c = json_at(cases, i);
        const char *name = json_str(need(c, "name", "case"), "?");
        const char *kind = json_str(need(c, "kind", "case"), "?");
        printf("\n[%d] %s（%s）\n", i + 1, name, kind);
        if (strcmp(kind, "mha") == 0) run_mha(c);
        else if (strcmp(kind, "tel") == 0) run_tel(c);
        else {
            fprintf(stderr, "未知用例类型 %s\n", kind);
            g_fail++;
        }
    }

    printf("\n%s\n", g_fail ? "存在超差项" : "全部通过");
    json_free(root);
    return g_fail ? 1 : 0;
}
