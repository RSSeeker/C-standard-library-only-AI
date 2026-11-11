/*
 * compare_train_utils.c —— 训练辅助工具（NF4 / WS / GradScaler / MoE aux）
 * 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_train_utils [train_utils_reference.json]
 *
 * 参考值由 tools/train_utils_reference.py 从 AI.py 现场导出；默认要求双精度
 * 逐位相等（容差 0）。如果本机 libm 的 sqrt 与 CPython 有末位差异，会在这里
 * 暴露出来——那说明对拍需要显式放宽，而不是假装一致。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../train_utils.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_train_utils] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_train_utils] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/train_utils_reference.py\n", who, key);
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

static int *iarr(const JsonValue *v, int n)
{
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
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

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-30s C=%ld py=%ld [不等]\n", what, got, want);
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

static void check_ints(const char *what, const int *got, const int *want, int n)
{
    for (int i = 0; i < n; i++) {
        char label[160];
        snprintf(label, sizeof label, "%s[%d]", what, i);
        check_int(label, got[i], want[i]);
    }
}

static void case_nf4(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "x", name));
    const int block_size = jint(need(cs, "block_size", name));
    const int outer_block_size = jint(need(cs, "outer_block_size", name));
    double *x = darr(need(cs, "x", name), n);

    /* fake / codes / scales / deq */
    double *want_fake = darr(need(cs, "fake", name), n);
    double *got = (double *)malloc(sizeof(double) * (size_t)n);
    if (!got) die("out of memory");
    if (ai_nf4_fake_quantize(x, n, block_size, got) != 0) die("nf4_fake_quantize 失败");
    check_doubles("nf4_fake", got, want_fake, n, tol);

    AiNf4Quant q = ai_nf4_quantize(x, n, block_size);
    if (q.length != n) die("nf4_quantize 失败");
    const int n_blocks = (q.length + q.block_size - 1) / q.block_size;
    int *want_codes = iarr(need(cs, "codes", name), n);
    double *want_scales = darr(need(cs, "scales", name), n_blocks);
    check_ints("nf4_codes", q.codes, want_codes, n);
    check_doubles("nf4_scales", q.scales, want_scales, n_blocks, tol);

    double *want_deq = darr(need(cs, "deq", name), n);
    if (ai_nf4_dequantize(&q, got) != 0) die("nf4_dequantize 失败");
    check_doubles("nf4_deq", got, want_deq, n, tol);
    ai_nf4_quant_free(&q);

    /* double quant：codes 数量 = 一阶缩放因子数量（n_blocks），
     * outers 数量 = ceil(n_blocks / outer_block_size)。 */
    const int dq_blocks = (n_blocks + outer_block_size - 1) / outer_block_size;
    double *want_outers = darr(need(cs, "double_outers", name), dq_blocks);
    int *want_dcodes = iarr(need(cs, "double_codes", name), n_blocks);
    int *codes2 = (int *)malloc(sizeof(int) * (size_t)n_blocks);
    double *outers = (double *)malloc(sizeof(double) * (size_t)dq_blocks);
    if (!codes2 || !outers) die("out of memory");
    if (ai_double_quantize(want_scales, n_blocks, outer_block_size, codes2, outers) != 0)
        die("double_quantize 失败");
    check_ints("double_codes", codes2, want_dcodes, n_blocks);
    check_doubles("double_outers", outers, want_outers, dq_blocks, tol);

    double *want_ddeq = darr(need(cs, "double_deq", name), n_blocks);
    double *ddeq = (double *)malloc(sizeof(double) * (size_t)n_blocks);
    if (!ddeq) die("out of memory");
    if (ai_double_dequantize(want_dcodes, want_outers, n_blocks,
                             outer_block_size, ddeq) != 0)
        die("double_dequantize 失败");
    check_doubles("double_deq", ddeq, want_ddeq, n_blocks, tol);

    double *want_dfake = darr(need(cs, "double_fake", name), n);
    if (ai_nf4_fake_quantize_double(x, n, block_size, outer_block_size, got) != 0)
        die("nf4_fake_quantize_double 失败");
    check_doubles("nf4_double_fake", got, want_dfake, n, tol);

    free(codes2);
    free(outers);
    free(ddeq);
    free(want_dcodes);
    free(want_ddeq);
    free(want_codes);
    free(want_scales);
    free(want_outers);
    free(want_fake);
    free(want_deq);
    free(want_dfake);
    free(got);
    free(x);
}

static void case_ws(const JsonValue *cs, const char *name, double tol)
{
    const int fan_out = jint(need(cs, "fan_out", name));
    const int fan_in = jint(need(cs, "fan_in", name));
    const int n = jint(need(cs, "n", name));
    const double eps = jnum(need(cs, "eps", name));
    const int wn = fan_out * fan_in;

    double *w = darr(need(cs, "w", name), wn);
    double *b = darr(need(cs, "b", name), fan_out);
    double *x = darr(need(cs, "x", name), n * fan_in);
    double *dy = darr(need(cs, "dy", name), n * fan_out);
    double *want_ws = darr(need(cs, "ws", name), wn);
    double *want_mu = darr(need(cs, "mu", name), fan_out);
    double *want_sigma = darr(need(cs, "sigma", name), fan_out);
    double *want_y = darr(need(cs, "y", name), n * fan_out);
    double *want_dw = darr(need(cs, "dw", name), wn);
    double *want_db = darr(need(cs, "db", name), fan_out);
    double *want_dx = darr(need(cs, "dx", name), n * fan_in);

    double *ws = (double *)malloc(sizeof(double) * (size_t)wn);
    double *mu = (double *)malloc(sizeof(double) * (size_t)fan_out);
    double *sigma = (double *)malloc(sizeof(double) * (size_t)fan_out);
    double *y = (double *)malloc(sizeof(double) * (size_t)(n * fan_out));
    double *dw = (double *)malloc(sizeof(double) * (size_t)wn);
    double *db = (double *)malloc(sizeof(double) * (size_t)fan_out);
    double *dx = (double *)malloc(sizeof(double) * (size_t)(n * fan_in));
    if (!ws || !mu || !sigma || !y || !dw || !db || !dx) die("out of memory");

    ai_ws_forward(w, b, x, fan_out, fan_in, n, eps, y, ws, mu, sigma);
    check_doubles("ws", ws, want_ws, wn, tol);
    check_doubles("mu", mu, want_mu, fan_out, tol);
    check_doubles("sigma", sigma, want_sigma, fan_out, tol);
    check_doubles("y", y, want_y, n * fan_out, tol);

    ai_ws_backward(w, x, dy, fan_out, fan_in, n, eps, dw, db, dx);
    check_doubles("dw", dw, want_dw, wn, tol);
    check_doubles("db", db, want_db, fan_out, tol);
    check_doubles("dx", dx, want_dx, n * fan_in, tol);

    free(ws);
    free(mu);
    free(sigma);
    free(y);
    free(dw);
    free(db);
    free(dx);
    free(w);
    free(b);
    free(x);
    free(dy);
    free(want_ws);
    free(want_mu);
    free(want_sigma);
    free(want_y);
    free(want_dw);
    free(want_db);
    free(want_dx);
}

static void case_gradscaler(const JsonValue *cs, const char *name, double tol)
{
    AiGradScaler s;
    ai_grad_scaler_init(&s,
                        jnum(need(cs, "init_scale", name)),
                        jnum(need(cs, "growth_factor", name)),
                        jnum(need(cs, "backoff_factor", name)),
                        jint(need(cs, "growth_interval", name)));
    const JsonValue *steps = need(cs, "steps", name);
    const int n_steps = json_array_len(steps);
    for (int t = 0; t < n_steps; t++) {
        const JsonValue *st = json_at(steps, t);
        const double loss = jnum(need(st, "loss", name));
        const int n_grads = json_array_len(need(st, "grads", name));
        double *grads = darr(need(st, "grads", name), n_grads);
        const double want_scaled = jnum(need(st, "scaled", name));
        double *want_unscaled = darr(need(st, "unscaled", name), n_grads);
        const int want_can = jint(need(st, "can_step", name));
        const int force_inf = jint(need(st, "force_inf", name));

        const double scaled = ai_grad_scaler_scale_loss(&s, loss);
        char label[160];
        snprintf(label, sizeof label, "step%d_scaled", t);
        check_num(label, scaled, want_scaled, tol);
        ai_grad_scaler_unscale(&s, grads, n_grads);
        snprintf(label, sizeof label, "step%d_unscaled", t);
        check_doubles(label, grads, want_unscaled, n_grads, tol);
        const int can = ai_grad_scaler_update(&s, force_inf ? 1 : -1);
        snprintf(label, sizeof label, "step%d_can_step", t);
        check_int(label, can, want_can);
        free(grads);
        free(want_unscaled);
    }
    const JsonValue *want_state = need(cs, "state", name);
    check_num("final_scale", s.scale, jnum(need(want_state, "scale", name)), tol);
    check_int("final_growth_tracker", s.growth_tracker,
              jint(need(want_state, "growth_tracker", name)));
    check_int("final_steps_skipped", s.steps_skipped,
              jint(need(want_state, "steps_skipped", name)));
    check_int("final_steps_taken", s.steps_taken,
              jint(need(want_state, "steps_taken", name)));
    check_int("final_total_steps", s.total_steps,
              jint(need(want_state, "total_steps", name)));
}

static void case_moe_aux(const JsonValue *cs, const char *name, double tol)
{
    const int tokens = jint(need(cs, "tokens", name));
    const int experts = jint(need(cs, "experts", name));
    const int top_k = jint(need(cs, "top_k", name));
    const double coefficient = jnum(need(cs, "coefficient", name));
    const double want = jnum(need(cs, "value", name));
    int *topk_idx = iarr(need(cs, "topk_idx", name), tokens * top_k);
    double *gate_probs = darr(need(cs, "gate_probs", name), tokens * experts);
    const double got = ai_moe_load_balancing_loss(topk_idx, tokens, top_k,
                                                  gate_probs, experts, coefficient);
    check_num("moe_aux", got, want, tol);
    free(topk_idx);
    free(gate_probs);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "train_utils_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_train_utils] 读不了 %s：%s\n"
                        "请先运行 python tools/train_utils_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_train_utils] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        printf("\n用例 %-18s kind=%s\n", name, kind);
        if (strcmp(kind, "nf4") == 0) {
            case_nf4(cs, name, tol);
        } else if (strcmp(kind, "ws") == 0) {
            case_ws(cs, name, tol);
        } else if (strcmp(kind, "gradscaler") == 0) {
            case_gradscaler(cs, name, tol);
        } else if (strcmp(kind, "moe_aux") == 0) {
            case_moe_aux(cs, name, tol);
        } else {
            die("未知 kind");
        }
    }
    printf("\n=== compare_train_utils：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
