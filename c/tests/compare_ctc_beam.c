/*
 * compare_ctc_beam.c —— C 的 CTC 前缀 beam search 与 AI.py 对拍
 *
 * 用法：./tests/compare_ctc_beam [ctc_beam_reference.json]
 *
 * tokens 必须完全一致；score / ctc_score / lm_score 按用例 tol（1e-12）比较
 * （跨工具链 log/exp 末位差不可避免）。LM 通过命名回调注入，和 Python 侧
 * 的 lm_prefer2 / lm_length_bias / lm_constant 同规则。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ctc_beam.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_ctc_beam] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_ctc_beam] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/ctc_beam_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

/* 二维数字数组 -> 扁平 double 数组（行主序），返回行数/列数 */
static double *grid_from(const JsonValue *v, int *out_t, int *out_k)
{
    const int t = json_array_len(v);
    if (t <= 0) die("期望非空二维数组");
    const int k = json_array_len(json_at(v, 0));
    if (k <= 0) die("期望非空行");
    double *flat = (double *)malloc(sizeof(double) * (size_t)t * (size_t)k);
    if (!flat) die("out of memory");
    for (int i = 0; i < t; i++) {
        const JsonValue *row = json_at(v, i);
        if (json_array_len(row) != k) die("行宽不一致");
        for (int j = 0; j < k; j++) {
            flat[(size_t)i * (size_t)k + (size_t)j] = json_num(json_at(row, j), NAN);
        }
    }
    *out_t = t;
    *out_k = k;
    return flat;
}

static int *tokens_from(const JsonValue *v, int *out_n)
{
    const int n = json_array_len(v);
    int *arr = (int *)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    if (!arr) die("out of memory");
    for (int i = 0; i < n; i++) arr[i] = json_int(json_at(v, i), -1);
    *out_n = n;
    return arr;
}

static double lm_prefer2(const int *t, int n, void *ctx)
{
    (void)ctx;
    return (n == 1 && t[0] == 2) ? 5.0 : 0.0;
}

static double lm_length_bias(const int *t, int n, void *ctx)
{
    (void)t;
    (void)ctx;
    return -0.5 * (double)n + 0.25;
}

static double lm_constant(const int *t, int n, void *ctx)
{
    (void)t;
    (void)n;
    (void)ctx;
    return 1.25;
}

static CtcLmFn lm_from_name(const char *name)
{
    if (!name || strcmp(name, "none") == 0) return NULL;
    if (strcmp(name, "prefer2") == 0) return lm_prefer2;
    if (strcmp(name, "length_bias") == 0) return lm_length_bias;
    if (strcmp(name, "constant") == 0) return lm_constant;
    fprintf(stderr, "[compare_ctc_beam] 未知 lm \"%s\"\n", name);
    exit(2);
}

static void report_double(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double d = fabs(got - want);
        if (d > g_worst) g_worst = d;
        if (d <= tol) return;
        printf("  %-22s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
        g_fail++;
    }
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "ctc_beam_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_ctc_beam] 读不了 %s：%s\n"
                        "请先运行 python tools/ctc_beam_reference.py\n", path, err);
        return 2;
    }
    {
        const JsonValue *cases = need(root, "cases", "root");
        const int n_cases = json_array_len(cases);
        printf("[compare_ctc_beam] %s：%d 个用例\n", path, n_cases);

        for (int c = 0; c < n_cases; c++) {
            const JsonValue *cs = json_at(cases, c);
            const char *name = json_str(json_get(cs, "name"), "?");
            const char *func = json_str(json_get(cs, "func"), "?");
            const double tol = json_num(json_get(cs, "tol"), 1e-12);
            const int blank = (int)json_num(need(cs, "blank", name), 0.0);
            const int beam_width = (int)json_num(need(cs, "beam_width", name), 8.0);
            const char *lm_name = json_str(json_get(cs, "lm"), "none");
            const double lm_weight = json_num(json_get(cs, "lm_weight"), 0.5);
            const double length_penalty = json_num(
                json_get(cs, "length_penalty"), 0.0);
            CtcLmFn lm = lm_from_name(lm_name);
            int want_n = 0;
            int *want = tokens_from(need(cs, "tokens", name), &want_n);
            CtcBeamResult got = {0, 0, 0.0, 0.0, 0.0};
            int rc;
            int T = 0, K = 0;
            double *grid;

            if (strcmp(func, "prefix") == 0) {
                grid = grid_from(need(cs, "log_probs", name), &T, &K);
                rc = ctc_prefix_beam_search(grid, T, K, blank, beam_width,
                                            lm, NULL, lm_weight, length_penalty,
                                            &got);
            } else if (strcmp(func, "decode") == 0) {
                grid = grid_from(need(cs, "logits", name), &T, &K);
                rc = ctc_beam_decode(grid, T, K, blank, beam_width,
                                     lm, NULL, lm_weight, length_penalty,
                                     &got);
            } else {
                fprintf(stderr, "[compare_ctc_beam] 未知 func \"%s\"\n", func);
                json_free(root);
                return 2;
            }
            free(grid);
            if (rc != 0) {
                printf("  %-22s C 返回错误 rc=%d\n", name, rc);
                g_fail++;
                free(want);
                ctc_beam_result_free(&got);
                continue;
            }
            g_checks++;
            if (got.n_tokens != want_n
                || (want_n > 0
                    && memcmp(got.tokens, want,
                              (size_t)want_n * sizeof(int)) != 0)) {
                printf("  %-22s tokens 不一致：C=[", name);
                for (int i = 0; i < got.n_tokens; i++)
                    printf("%s%d", i ? "," : "", got.tokens[i]);
                printf("] py=[");
                for (int i = 0; i < want_n; i++)
                    printf("%s%d", i ? "," : "", want[i]);
                printf("]\n");
                g_fail++;
            } else {
                g_exact++;
                report_double("score", got.score,
                              json_num(need(cs, "score", name), NAN), tol);
                report_double("ctc_score", got.ctc_score,
                              json_num(need(cs, "ctc_score", name), NAN), tol);
                report_double("lm_score", got.lm_score,
                              json_num(need(cs, "lm_score", name), NAN), tol);
            }
            printf("用例 %-24s func=%-6s tokens=", name, func);
            for (int i = 0; i < got.n_tokens; i++)
                printf("%s%d", i ? "," : "", got.tokens[i]);
            printf("\n");
            free(want);
            ctc_beam_result_free(&got);
        }

        printf("\n[compare_ctc_beam] 共 %d 项检查：逐位一致 %d，失败 %d，"
               "最大绝对差 %.3e\n", g_checks, g_exact, g_fail, g_worst);
        json_free(root);
        return (g_fail > 0) ? 1 : 0;
    }
}
