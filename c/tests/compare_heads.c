/* ==========================================================================
 *  compare_heads.c：heads（序列标注头 / 多标签头 / MoE）与 AI.py 的逐位比对
 *
 *  用法：./tests/compare_heads heads_reference.json
 * ========================================================================== */
#include "../heads.h"
#include "../json.h"
#include "../attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_arrays = 0;
static int g_exact = 0;
static double g_worst = 0.0;
static int g_failed = 0;

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "[compare_heads] %s: 缺少键 %s\n", who, key);
        exit(2);
    }
    return v;
}

/* 把 JSON 数组读进 dst（长度 n） */
static void arr_vals(const JsonValue *arr, double *dst, int n, const char *who)
{
    int i;
    if (json_array_len(arr) != n) {
        fprintf(stderr, "[compare_heads] %s: 数组长度 %d != %d\n",
                who, json_array_len(arr), n);
        exit(2);
    }
    for (i = 0; i < n; i++) dst[i] = json_num(json_at(arr, i), 0.0);
}

static void report(const char *what, const double *got, const double *want,
                   int n, double tol)
{
    int i, bad = 0;
    double worst = 0.0;
    g_arrays++;
    for (i = 0; i < n; i++) {
        double d = fabs(got[i] - want[i]);
        if (d > worst) worst = d;
        if (!(d <= tol)) bad++;
    }
    if (worst > g_worst) g_worst = worst;
    if (bad == 0) g_exact++;
    else g_failed++;
    printf("    %-22s %s  n=%d  worst=%.3e%s\n", what,
           bad == 0 ? "[OK]" : "[超差]", n, worst,
           tol == 0.0 && bad == 0 ? "  (逐位)" : "");
}

static void report_scalar(const char *what, double got, double want, double tol)
{
    double a[1], b[1];
    a[0] = got;
    b[0] = want;
    report(what, a, b, 1, tol);
}

/* ---------- TokenClassificationHead ---------- */
static int run_token(const JsonValue *c)
{
    int d = json_int(need(c, "d_model", "token"), 0);
    int L = json_int(need(c, "num_labels", "token"), 0);
    int T = json_int(need(c, "T", "token"), 0);
    double tol = 1e-12;
    TokenHead *h = token_head_create(d, L);
    double *W = (double *)malloc(sizeof(double) * (size_t)L * d);
    double *b = (double *)malloc(sizeof(double) * (size_t)L);
    double *x = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *logits = (double *)malloc(sizeof(double) * (size_t)T * L);
    double *dlogits = (double *)malloc(sizeof(double) * (size_t)T * L);
    double *dx = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *want = (double *)malloc(sizeof(double) * (size_t)T * L * 2 +
                                    sizeof(double) * (size_t)L * d * 2);
    int *targets = (int *)malloc(sizeof(int) * (size_t)T);
    int *mask = (int *)malloc(sizeof(int) * (size_t)T);
    int *pred = (int *)malloc(sizeof(int) * (size_t)T);
    const JsonValue *jm = json_get(c, "mask");
    int i, all_masked = 1;
    double loss;

    arr_vals(need(c, "w", "token"), W, L * d, "token.w");
    arr_vals(need(c, "b", "token"), b, L, "token.b");
    arr_vals(need(c, "x", "token"), x, T * d, "token.x");
    for (i = 0; i < T; i++) targets[i] = json_int(json_at(need(c, "targets", "token"), i), 0);
    if (jm && json_array_len(jm) == T) {
        for (i = 0; i < T; i++) mask[i] = json_int(json_at(jm, i), 0);
    } else {
        for (i = 0; i < T; i++) mask[i] = 1;
    }
    for (i = 0; i < T; i++) if (mask[i]) all_masked = 0;
    (void)all_masked;

    token_head_set_params(h, W, b);
    token_head_zero_grad(h);
    token_head_forward(h, x, T, logits);
    loss = token_head_loss_grad(h, logits, targets, jm ? mask : NULL, T, dlogits);
    token_head_backward(h, dlogits, T, dx);
    token_head_predict(h, logits, T, jm ? mask : NULL, pred);

    arr_vals(need(c, "logits", "token"), want, T * L, "token.logits");
    report("logits", logits, want, T * L, tol);
    report_scalar("loss", loss, json_num(need(c, "loss", "token"), 0.0), tol);
    arr_vals(need(c, "dlogits", "token"), want, T * L, "token.dlogits");
    report("dlogits", dlogits, want, T * L, tol);
    arr_vals(need(c, "dx", "token"), want, T * d, "token.dx");
    report("dx", dx, want, T * d, tol);
    arr_vals(need(c, "grad_w", "token"), want, L * d, "token.grad_w");
    report("grad_W", h->gW, want, L * d, tol);
    arr_vals(need(c, "grad_b", "token"), want, L, "token.grad_b");
    report("grad_b", h->gb, want, L, tol);
    {
        const JsonValue *jp = need(c, "pred", "token");
        double *gp = (double *)malloc(sizeof(double) * (size_t)T);
        double *wp = (double *)malloc(sizeof(double) * (size_t)T);
        for (i = 0; i < T; i++) {
            gp[i] = (double)pred[i];
            wp[i] = (double)json_int(json_at(jp, i), 0);
        }
        report("predict", gp, wp, T, 0.0);
        free(gp);
        free(wp);
    }
    report_scalar("param_count", (double)token_head_param_count(h),
                  json_num(need(c, "param_count", "token"), 0.0), 0.0);

    free(W); free(b); free(x); free(logits); free(dlogits); free(dx); free(want);
    free(targets); free(mask); free(pred);
    token_head_free(h);
    return 0;
}

/* ---------- MultiLabelHead ---------- */
static int run_multilabel(const JsonValue *c)
{
    int fi = json_int(need(c, "fan_in", "multilabel"), 0);
    int L = json_int(need(c, "num_labels", "multilabel"), 0);
    double tol = 1e-12;
    MultiLabelHead *h = multilabel_head_create(fi, L);
    double *W = (double *)malloc(sizeof(double) * (size_t)L * fi);
    double *b = (double *)malloc(sizeof(double) * (size_t)L);
    double *x = (double *)malloc(sizeof(double) * (size_t)fi);
    double *probs = (double *)malloc(sizeof(double) * (size_t)L);
    double *dlogits = (double *)malloc(sizeof(double) * (size_t)L);
    double *dx = (double *)malloc(sizeof(double) * (size_t)fi);
    double *targets = (double *)malloc(sizeof(double) * (size_t)L);
    double *want = (double *)malloc(sizeof(double) * (size_t)L * 4);
    int *pred = (int *)malloc(sizeof(int) * (size_t)L);
    int i;
    double loss;

    arr_vals(need(c, "w", "multilabel"), W, L * fi, "multilabel.w");
    arr_vals(need(c, "b", "multilabel"), b, L, "multilabel.b");
    arr_vals(need(c, "x", "multilabel"), x, fi, "multilabel.x");
    arr_vals(need(c, "targets", "multilabel"), targets, L, "multilabel.targets");

    multilabel_head_set_params(h, W, b);
    multilabel_head_zero_grad(h);
    multilabel_head_forward(h, x, probs);
    loss = multilabel_head_loss_grad(h, probs, targets, dlogits);
    multilabel_head_backward(h, dlogits, dx);
    multilabel_head_predict(h, probs, 0.5, pred);

    arr_vals(need(c, "probs", "multilabel"), want, L, "multilabel.probs");
    report("probs", probs, want, L, tol);
    report_scalar("loss", loss, json_num(need(c, "loss", "multilabel"), 0.0), tol);
    arr_vals(need(c, "dlogits", "multilabel"), want, L, "multilabel.dlogits");
    report("dlogits", dlogits, want, L, tol);
    arr_vals(need(c, "dx", "multilabel"), want, fi, "multilabel.dx");
    report("dx", dx, want, fi, tol);
    arr_vals(need(c, "grad_w", "multilabel"), want, L * fi, "multilabel.grad_w");
    report("grad_W", h->gW, want, L * fi, tol);
    arr_vals(need(c, "grad_b", "multilabel"), want, L, "multilabel.grad_b");
    report("grad_b", h->gb, want, L, tol);
    {
        const JsonValue *jp = need(c, "pred", "multilabel");
        double *gp = (double *)malloc(sizeof(double) * (size_t)L);
        double *wp = (double *)malloc(sizeof(double) * (size_t)L);
        for (i = 0; i < L; i++) {
            gp[i] = (double)pred[i];
            wp[i] = (double)json_int(json_at(jp, i), 0);
        }
        report("predict", gp, wp, L, 0.0);
        free(gp);
        free(wp);
    }
    report_scalar("param_count", (double)multilabel_head_param_count(h),
                  json_num(need(c, "param_count", "multilabel"), 0.0), 0.0);

    free(W); free(b); free(x); free(probs); free(dlogits); free(dx);
    free(targets); free(want); free(pred);
    multilabel_head_free(h);
    return 0;
}

/* ---------- MoE ---------- */
static int act_code(const char *name)
{
    if (!strcmp(name, "relu")) return FF_ACT_RELU;
    if (!strcmp(name, "gelu")) return FF_ACT_GELU;
    if (!strcmp(name, "sigmoid")) return FF_ACT_SIGMOID;
    if (!strcmp(name, "linear")) return FF_ACT_LINEAR;
    if (!strcmp(name, "silu")) return FF_ACT_SILU;
    if (!strcmp(name, "mish")) return FF_ACT_MISH;
    if (!strcmp(name, "elu")) return FF_ACT_ELU;
    if (!strcmp(name, "selu")) return FF_ACT_SELU;
    fprintf(stderr, "[compare_heads] 未知激活 %s\n", name);
    exit(2);
}

static int run_moe(const JsonValue *c)
{
    int d = json_int(need(c, "d_model", "moe"), 0);
    int dff = json_int(need(c, "d_ff", "moe"), 0);
    int E = json_int(need(c, "n_experts", "moe"), 0);
    int k = json_int(need(c, "top_k", "moe"), 0);
    int T = json_int(need(c, "T", "moe"), 0);
    int act = act_code(json_str(need(c, "activation", "moe"), "relu"));
    double tol = json_num(json_get(c, "tol"), 1e-12);
    MoE *m = moe_create(d, dff, E, k, act, T);
    double *W_gate = (double *)malloc(sizeof(double) * (size_t)E * d);
    double *b_gate = (double *)malloc(sizeof(double) * (size_t)E);
    double *x = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *dout = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *out = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *dx = (double *)malloc(sizeof(double) * (size_t)T * d);
    double *want = (double *)malloc(sizeof(double) * (size_t)T * d * 3);
    const JsonValue *jexp, *jgrads;
    int e, i, j;

    arr_vals(need(c, "w_gate", "moe"), W_gate, E * d, "moe.w_gate");
    arr_vals(need(c, "b_gate", "moe"), b_gate, E, "moe.b_gate");
    arr_vals(need(c, "x", "moe"), x, T * d, "moe.x");
    arr_vals(need(c, "dout", "moe"), dout, T * d, "moe.dout");

    jexp = need(c, "experts", "moe");
    jgrads = need(c, "expert_grads", "moe");
    if (json_array_len(jexp) != E || json_array_len(jgrads) != E) {
        fprintf(stderr, "[compare_heads] 专家数量不符\n");
        exit(2);
    }
    for (e = 0; e < E; e++) {
        FeedForward *f = moe_expert(m, e);
        const JsonValue *pe = json_at(jexp, e);
        double *W1 = (double *)malloc(sizeof(double) * (size_t)dff * d);
        double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
        double *W2 = (double *)malloc(sizeof(double) * (size_t)d * dff);
        double *b2 = (double *)malloc(sizeof(double) * (size_t)d);
        arr_vals(need(pe, "w1", "expert"), W1, dff * d, "expert.w1");
        arr_vals(need(pe, "b1", "expert"), b1, dff, "expert.b1");
        arr_vals(need(pe, "w2", "expert"), W2, d * dff, "expert.w2");
        arr_vals(need(pe, "b2", "expert"), b2, d, "expert.b2");
        ff_set_params(f, W1, b1, W2, b2);
        free(W1); free(b1); free(W2); free(b2);
    }
    moe_set_gate(m, W_gate, b_gate);
    moe_zero_grad(m);
    moe_forward(m, x, T, out);
    moe_backward(m, dout, T, dx);

    arr_vals(need(c, "out", "moe"), want, T * d, "moe.out");
    report("out", out, want, T * d, tol);
    arr_vals(need(c, "dx", "moe"), want, T * d, "moe.dx");
    report("dx", dx, want, T * d, tol);
    arr_vals(need(c, "grad_w_gate", "moe"), want, E * d, "moe.grad_w_gate");
    report("grad_W_gate", m->gW_gate, want, E * d, tol);
    arr_vals(need(c, "grad_b_gate", "moe"), want, E, "moe.grad_b_gate");
    report("grad_b_gate", m->gb_gate, want, E, tol);

    for (e = 0; e < E; e++) {
        FeedForward *f = moe_expert(m, e);
        const JsonValue *pg = json_at(jgrads, e);
        double *W1 = (double *)malloc(sizeof(double) * (size_t)dff * d);
        double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
        double *W2 = (double *)malloc(sizeof(double) * (size_t)d * dff);
        double *b2 = (double *)malloc(sizeof(double) * (size_t)d);
        char tag[32];
        arr_vals(need(pg, "w1", "expert_grad"), W1, dff * d, "expert_grad.w1");
        snprintf(tag, sizeof tag, "e%d.grad_W1", e);
        report(tag, f->grad_W1, W1, dff * d, tol);
        arr_vals(need(pg, "b1", "expert_grad"), b1, dff, "expert_grad.b1");
        snprintf(tag, sizeof tag, "e%d.grad_b1", e);
        report(tag, f->grad_b1, b1, dff, tol);
        arr_vals(need(pg, "w2", "expert_grad"), W2, d * dff, "expert_grad.w2");
        snprintf(tag, sizeof tag, "e%d.grad_W2", e);
        report(tag, f->grad_W2, W2, d * dff, tol);
        arr_vals(need(pg, "b2", "expert_grad"), b2, d, "expert_grad.b2");
        snprintf(tag, sizeof tag, "e%d.grad_b2", e);
        report(tag, f->grad_b2, b2, d, tol);
        free(W1); free(b1); free(W2); free(b2);
    }

    /* 路由：专家下标必须完全一致，权重按 tol 比 */
    {
        const JsonValue *jr = need(c, "routes", "moe");
        int bad = 0;
        for (i = 0; i < T; i++) {
            const JsonValue *rt = json_at(jr, i);
            for (j = 0; j < k; j++) {
                const JsonValue *pair = json_at(rt, j);
                int want_e = json_int(json_at(pair, 0), -1);
                double want_w = json_num(json_at(pair, 1), 0.0);
                double got_w = m->route_w[(size_t)i * k + j];
                if (m->route_e[(size_t)i * k + j] != want_e) bad++;
                if (!(fabs(got_w - want_w) <= tol)) bad++;
            }
        }
        g_arrays++;
        if (bad == 0) g_exact++; else g_failed++;
        printf("    %-22s %s  n=%d\n", "routes",
               bad == 0 ? "[OK]" : "[超差]", T * k * 2);
    }
    report_scalar("param_count", (double)moe_param_count(m),
                  json_num(need(c, "param_count", "moe"), 0.0), 0.0);

    free(W_gate); free(b_gate); free(x); free(dout); free(out); free(dx); free(want);
    moe_free(m);
    return 0;
}

int main(int argc, char **argv)
{
    char err[512];
    JsonValue *root;
    const JsonValue *cases;
    int n, i;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "用法: %s <heads_reference.json>\n", argv[0]);
        return 2;
    }
    root = json_parse_file(argv[1], err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_heads] 解析失败: %s\n", err);
        return 2;
    }
    cases = json_get(root, "cases");
    if (!cases) {
        fprintf(stderr, "[compare_heads] 缺少 cases\n");
        return 2;
    }
    n = json_array_len(cases);
    printf("[compare_heads] %d 个用例\n", n);
    for (i = 0; i < n; i++) {
        const JsonValue *c = json_at(cases, i);
        const char *kind = json_str(need(c, "kind", "case"), "?");
        printf("  [%d] %s (%s)\n", i + 1, json_str(need(c, "name", "case"), "?"), kind);
        if (!strcmp(kind, "token")) run_token(c);
        else if (!strcmp(kind, "multilabel")) run_multilabel(c);
        else if (!strcmp(kind, "moe")) run_moe(c);
        else {
            fprintf(stderr, "[compare_heads] 未知 kind %s\n", kind);
            return 2;
        }
    }
    json_free(root);
    printf("\n[compare_heads] %d 项数组/scalar，%d 项逐位相同，%d 项超差，worst=%.3e\n",
           g_arrays, g_exact, g_failed, g_worst);
    if (g_failed) {
        printf("[compare_heads] [FAIL]\n");
        return 1;
    }
    printf("[compare_heads] [OK] heads / MoE match AI.py\n");
    return 0;
}
