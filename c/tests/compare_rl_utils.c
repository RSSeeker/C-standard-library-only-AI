/*
 * compare_rl_utils.c —— RL / μP 工具与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_rl_utils [rl_utils_reference.json]
 *
 * 参考值由 tools/rl_utils_reference.py 从 AI.py 现场导出；默认要求
 * 双精度逐位相等（容差 0），sqrt 相关用例容差 1e-15。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../rl_utils.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_rl_utils] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_rl_utils] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/rl_utils_reference.py\n", who, key);
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

static void case_group_adv(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "rewards", name));
    const int group_size = jint(need(cs, "group_size", name));
    const double eps = jnum(need(cs, "eps", name));
    const int normalize = jint(need(cs, "normalize", name));
    double *rewards = darr(need(cs, "rewards", name), n);
    double *want = darr(need(cs, "out", name), n);
    double *got = (double *)malloc(sizeof(double) * (size_t)n);
    if (!got) die("out of memory");
    ai_group_relative_advantages(rewards, n, group_size, eps, normalize, got);
    check_doubles("adv", got, want, n, tol);
    free(got);
    free(want);
    free(rewards);
}

static void case_best_of_n(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "rewards", name));
    const int want_index = jint(need(cs, "index", name));
    const double want_reward = jnum(need(cs, "reward", name));
    double *rewards = darr(need(cs, "rewards", name), n);
    double got_reward = 0.0;
    const int got_index = ai_best_of_n(rewards, n, &got_reward);
    check_int("bon_index", got_index, want_index);
    check_num("bon_reward", got_reward, want_reward, tol);
    free(rewards);
}

static void case_rejection(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "rewards", name));
    (void)tol;
    const char *mode = json_str(need(cs, "mode", name), "?");
    const int want_count = jint(need(cs, "count", name));
    double *rewards = darr(need(cs, "rewards", name), n);
    int *want = iarr(need(cs, "kept", name), want_count);
    int *got = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!got) die("out of memory");
    int got_count;
    if (strcmp(mode, "threshold") == 0) {
        got_count = ai_rejection_filter_threshold(rewards, n,
                                                  jnum(need(cs, "threshold", name)),
                                                  got);
    } else {
        got_count = ai_rejection_filter_top_ratio(rewards, n,
                                                  jnum(need(cs, "ratio", name)),
                                                  got);
    }
    check_int("rej_count", got_count, want_count);
    check_ints("rej_kept", got, want, want_count);
    free(got);
    free(want);
    free(rewards);
}

static void case_self_consistency(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "samples", name));
    const int want_best = jint(need(cs, "best", name));
    const double want_conf = jnum(need(cs, "confidence", name));
    const int want_unique = jint(need(cs, "unique", name));
    int *samples = iarr(need(cs, "samples", name), n);
    int *want_keys = iarr(need(cs, "keys", name), want_unique);
    int *want_counts = iarr(need(cs, "counts", name), want_unique);
    int *keys = (int *)malloc(sizeof(int) * (size_t)n);
    int *counts = (int *)malloc(sizeof(int) * (size_t)n);
    if (!keys || !counts) die("out of memory");
    int got_unique = 0;
    double got_conf = 0.0;
    const int got_best = ai_self_consistency_int(samples, n, &got_conf,
                                                 &got_unique, keys, counts);
    check_int("sc_best", got_best, want_best);
    check_num("sc_conf", got_conf, want_conf, tol);
    check_int("sc_unique", got_unique, want_unique);
    check_ints("sc_keys", keys, want_keys, want_unique);
    check_ints("sc_counts", counts, want_counts, want_unique);
    free(keys);
    free(counts);
    free(want_keys);
    free(want_counts);
    free(samples);
}

static void case_mup_target_std(const JsonValue *cs, const char *name, double tol)
{
    const char *kind = json_str(need(cs, "mup_kind", name), "?");
    const double got = ai_mup_target_std(
        kind, jnum(need(cs, "fan_in", name)),
        jnum(need(cs, "base_width", name)), jnum(need(cs, "width", name)),
        jnum(need(cs, "scale", name)));
    check_num("mup_std", got, jnum(need(cs, "value", name)), tol);
}

static void case_mup_layer_kinds(const JsonValue *cs, const char *name, double tol)
{
    const int n = jint(need(cs, "num_layers", name));
    const JsonValue *want = need(cs, "kinds", name);
    char kinds[64];
    char label[64];
    const int got_n = ai_mup_layer_kinds(n > 64 ? 64 : n, kinds);
    (void)tol;
    check_int("kinds_n", got_n, json_array_len(want));
    for (int i = 0; i < json_array_len(want); i++) {
        const char *s = json_str(json_at(want, i), "?");
        char w = s[0];
        snprintf(label, sizeof label, "kinds[%d]", i);
        check_int(label, kinds[i], w);
    }
}

static void case_mup_lr(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "fan_ins", name));
    double *fan_ins = darr(need(cs, "fan_ins", name), n);
    double *want = darr(need(cs, "lrs", name), n);
    double *got = (double *)malloc(sizeof(double) * (size_t)n);
    if (!got) die("out of memory");
    ai_mup_lr(fan_ins, n, jnum(need(cs, "base_lr", name)),
              jnum(need(cs, "width", name)),
              jnum(need(cs, "base_width", name)), got);
    check_doubles("lrs", got, want, n, tol);
    free(got);
    free(want);
    free(fan_ins);
}

static void case_mupadam(const JsonValue *cs, const char *name, double tol)
{
    const int n = json_array_len(need(cs, "params", name));
    double *params = darr(need(cs, "params", name), n);
    double *grads = darr(need(cs, "grads", name), n);
    double *want = darr(need(cs, "out", name), n);
    double *m = (double *)calloc((size_t)n, sizeof(double));
    double *v = (double *)calloc((size_t)n, sizeof(double));
    if (!m || !v) die("out of memory");
    ai_mupadam_step(params, grads, m, v, n, jnum(need(cs, "lr", name)),
                    jnum(need(cs, "beta1", name)),
                    jnum(need(cs, "beta2", name)),
                    jnum(need(cs, "eps", name)),
                    jint(need(cs, "step_count", name)));
    check_doubles("mupadam_out", params, want, n, tol);
    free(m);
    free(v);
    free(want);
    free(grads);
    free(params);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "rl_utils_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_rl_utils] 读不了 %s：%s\n"
                        "请先运行 python tools/rl_utils_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_rl_utils] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        printf("\n用例 %-20s kind=%s\n", name, kind);
        if (strcmp(kind, "group_adv") == 0) {
            case_group_adv(cs, name, tol);
        } else if (strcmp(kind, "best_of_n") == 0) {
            case_best_of_n(cs, name, tol);
        } else if (strcmp(kind, "rejection") == 0) {
            case_rejection(cs, name, tol);
        } else if (strcmp(kind, "self_consistency") == 0) {
            case_self_consistency(cs, name, tol);
        } else if (strcmp(kind, "mup_target_std") == 0) {
            case_mup_target_std(cs, name, tol);
        } else if (strcmp(kind, "mup_layer_kinds") == 0) {
            case_mup_layer_kinds(cs, name, tol);
        } else if (strcmp(kind, "mup_lr") == 0) {
            case_mup_lr(cs, name, tol);
        } else if (strcmp(kind, "mupadam") == 0) {
            case_mupadam(cs, name, tol);
        } else {
            die("未知 kind");
        }
    }
    printf("\n=== compare_rl_utils：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
