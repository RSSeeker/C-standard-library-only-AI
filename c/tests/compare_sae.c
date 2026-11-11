/*
 * compare_sae.c —— 稀疏自编码器 SAE 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_sae [sae_reference.json]
 *
 * 参考值由 tools/sae_reference.py 从 AI.py 现场导出；欧氏距离相关
 * 用例的 sqrt 容差用 JSON 里的 tol（默认 0，按 1e-15 生成）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../sae.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_sae] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_sae] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/sae_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static double *darr(const JsonValue *v, int n)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    return out;
}

/* 行主序展平嵌套数组（rows 个元素，每个 cols 个数字）。 */
static double *dmat(const JsonValue *v, int rows, int cols)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(rows * cols));
    if (!out) die("out of memory");
    for (int r = 0; r < rows; r++) {
        const JsonValue *row = json_at(v, r);
        for (int c = 0; c < cols; c++) out[r * cols + c] = jnum(json_at(row, c));
    }
    return out;
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
        const double d = fabs(got - want) / scale;
        if (fabs(got - want) > g_worst) g_worst = fabs(got - want);
        if (d <= tol) return;
        printf("  %-30s C=%.17g py=%.17g 相对差=%.3e [不等]\n", what, got, want, d);
        g_fail++;
    }
}

static void check_doubles(const char *what, const double *got, const double *want,
                          int n, double tol)
{
    for (int i = 0; i < n; i++) {
        char label[160];
        snprintf(label, sizeof label, "%s[%d]", what, i);
        check_num(label, got[i], want[i], tol);
    }
}

static void case_sae(const JsonValue *cs, const char *name, double tol)
{
    const int d_in = jint(need(cs, "d_in", name));
    const int d_hidden = jint(need(cs, "d_hidden", name));
    const double l1_coeff = jnum(need(cs, "l1_coeff", name));
    const int n = json_array_len(need(cs, "xs", name));
    const int wn_enc = d_hidden * d_in;
    const int wn_dec = d_in * d_hidden;
    const JsonValue *init = need(cs, "init", name);
    const JsonValue *post = need(cs, "post", name);
    const JsonValue *grads = need(cs, "grads", name);

    double *xs = dmat(need(cs, "xs", name), n, d_in);
    double *W_enc = darr(need(init, "W_enc", name), wn_enc);
    double *b_enc = darr(need(init, "b_enc", name), d_hidden);
    double *W_dec = darr(need(init, "W_dec", name), wn_dec);
    double *b_dec = darr(need(init, "b_dec", name), d_in);

    double *want_recon = dmat(need(cs, "recon", name), n, d_in);
    double *want_features = dmat(need(cs, "features", name), n, d_hidden);
    double *want_pre = dmat(need(cs, "pre", name), n, d_hidden);
    double *want_gW_enc = darr(need(grads, "gW_enc", name), wn_enc);
    double *want_gb_enc = darr(need(grads, "gb_enc", name), d_hidden);
    double *want_gW_dec = darr(need(grads, "gW_dec", name), wn_dec);
    double *want_gb_dec = darr(need(grads, "gb_dec", name), d_in);
    double *want_post_W_enc = darr(need(post, "W_enc", name), wn_enc);
    double *want_post_b_enc = darr(need(post, "b_enc", name), d_hidden);
    double *want_post_W_dec = darr(need(post, "W_dec", name), wn_dec);
    double *want_post_b_dec = darr(need(post, "b_dec", name), d_in);

    double *recon = (double *)malloc(sizeof(double) * (size_t)(n * d_in));
    double *features = (double *)malloc(sizeof(double) * (size_t)(n * d_hidden));
    double *pre = (double *)malloc(sizeof(double) * (size_t)(n * d_hidden));
    double *active = (double *)malloc(sizeof(double) * (size_t)d_hidden);
    double *encode = (double *)malloc(sizeof(double) * (size_t)d_hidden);
    double *decode = (double *)malloc(sizeof(double) * (size_t)d_in);
    if (!recon || !features || !pre || !active || !encode || !decode)
        die("out of memory");

    AiSAE *sae = ai_sae_create(d_in, d_hidden, l1_coeff);
    if (!sae) die("ai_sae_create 失败");
    if (ai_sae_load_params(sae, W_enc, b_enc, W_dec, b_dec) != 0)
        die("ai_sae_load_params 失败");
    check_num("param_count", (double)ai_sae_param_count(sae),
              (double)jint(need(cs, "param_count", name)), 0.0);

    if (ai_sae_forward(sae, xs, n, recon, features, pre) != 0)
        die("ai_sae_forward 失败");
    check_doubles("recon", recon, want_recon, n * d_in, tol);
    check_doubles("features", features, want_features, n * d_hidden, tol);
    check_doubles("pre", pre, want_pre, n * d_hidden, tol);

    {
        double loss = 0.0, mse = 0.0, l1 = 0.0;
        if (ai_sae_loss_and_backward(sae, xs, n, &loss, &mse, &l1) != 0)
            die("ai_sae_loss_and_backward 失败");
        check_num("loss", loss, jnum(need(cs, "loss", name)), tol);
        check_num("mse", mse, jnum(need(cs, "mse", name)), tol);
        check_num("l1", l1, jnum(need(cs, "l1", name)), tol);
    }
    check_doubles("gW_enc", sae->gW_enc, want_gW_enc, wn_enc, tol);
    check_doubles("gb_enc", sae->gb_enc, want_gb_enc, d_hidden, tol);
    check_doubles("gW_dec", sae->gW_dec, want_gW_dec, wn_dec, tol);
    check_doubles("gb_dec", sae->gb_dec, want_gb_dec, d_in, tol);

    if (ai_sae_sgd_step(sae, jnum(need(cs, "sgd_lr", name))) != 0)
        die("ai_sae_sgd_step 失败");
    check_doubles("post_W_enc", sae->W_enc, want_post_W_enc, wn_enc, tol);
    check_doubles("post_b_enc", sae->b_enc, want_post_b_enc, d_hidden, tol);
    check_doubles("post_W_dec", sae->W_dec, want_post_W_dec, wn_dec, tol);
    check_doubles("post_b_dec", sae->b_dec, want_post_b_dec, d_in, tol);

    {
        int dead = 0;
        double mean_active = 0.0;
        if (ai_sae_feature_stats(sae, xs, n, active, &dead, &mean_active) != 0)
            die("ai_sae_feature_stats 失败");
        check_doubles("active_fraction", active,
                      darr(need(cs, "active_fraction", name), d_hidden),
                      d_hidden, tol);
        check_num("dead_features", (double)dead,
                  (double)jint(need(cs, "dead_features", name)), 0.0);
        check_num("mean_active", mean_active,
                  jnum(need(cs, "mean_active", name)), tol);
    }

    ai_sae_encode(sae, xs, encode);
    /* 参考里的 encode/decode 是 SGD 之后的；这里同样在 sgd_step 之后算。 */
    check_doubles("encode", encode, darr(need(cs, "encode", name), d_hidden),
                  d_hidden, tol);
    ai_sae_decode(sae, encode, decode);
    check_doubles("decode", decode, darr(need(cs, "decode", name), d_in),
                  d_in, tol);

    ai_sae_free(sae);
    free(xs); free(W_enc); free(b_enc); free(W_dec); free(b_dec);
    free(want_recon); free(want_features); free(want_pre);
    free(want_gW_enc); free(want_gb_enc); free(want_gW_dec); free(want_gb_dec);
    free(want_post_W_enc); free(want_post_b_enc);
    free(want_post_W_dec); free(want_post_b_dec);
    free(recon); free(features); free(pre); free(active); free(encode); free(decode);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "sae_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_sae] 读不了 %s：%s\n"
                        "请先运行 python tools/sae_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_sae] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        printf("\n用例 %-20s kind=%s\n", name, kind);
        if (strcmp(kind, "sae") == 0) case_sae(cs, name, tol);
        else die("未知 kind");
    }
    printf("\n=== compare_sae：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
