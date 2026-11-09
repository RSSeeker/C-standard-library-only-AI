/*
 * compare_text_metrics.c —— C 的文本指标与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_text_metrics [text_metrics_reference.json]
 *
 * 参考值由 tools/text_metrics_reference.py 从 AI.py 现场导出；全部用例要求
 * 双精度逐位相等（容差 0）。如果机器上 libm 的 pow 与 CPython 有末位差异，
 * 会在这里暴露出来——那说明实现对拍需要放宽，而不是假装一致。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../text_metrics.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_text_metrics] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_text_metrics] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/text_metrics_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

/* JSON 字符串数组 -> char* 数组（指针借用 JsonValue 的字符串内存，不释放） */
static const char **strs_from(const JsonValue *v, int *out_n)
{
    if (json_array_len(v) < 0) die("期望字符串数组");
    const int n = json_array_len(v);
    const char **arr = (const char **)malloc(sizeof(char *) * (size_t)(n > 0 ? n : 1));
    if (!arr) die("out of memory");
    for (int i = 0; i < n; i++) {
        const JsonValue *item = json_at(v, i);
        arr[i] = json_str(item, NULL);
        if (!arr[i]) die("字符串数组里出现非字符串元素");
    }
    *out_n = n;
    return arr;
}

static void report(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    const double d = fabs(got - want);
    if (d > g_worst) g_worst = d;
    if (d <= tol) {
        printf("  %-22s C=%.17g py=%.17g 差=%.3e [容差 %.0e 内]\n",
               what, got, want, d, tol);
        return;
    }
    printf("  %-22s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
    g_fail++;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "text_metrics_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_text_metrics] 读不了 %s：%s\n"
                        "请先运行 python tools/text_metrics_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_text_metrics] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *func = json_str(json_get(cs, "func"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        const double want = json_num(need(cs, "value", name), NAN);
        printf("\n用例 %-22s func=%s\n", name, func);

        if (strcmp(func, "meteor") == 0) {
            int n_p = 0, n_r = 0;
            const char **p = strs_from(need(cs, "pred", name), &n_p);
            const char **r = strs_from(need(cs, "ref", name), &n_r);
            const double got = tm_meteor_score(
                p, n_p, r, n_r,
                json_num(need(cs, "alpha", name), 0.9),
                json_num(need(cs, "beta", name), 3.0),
                json_num(need(cs, "gamma", name), 0.5));
            report("meteor", got, want, tol);
            free(p);
            free(r);
        } else if (strcmp(func, "meteor_full") == 0) {
            int n_p = 0, n_r = 0;
            const char **p = strs_from(need(cs, "pred", name), &n_p);
            const char **r = strs_from(need(cs, "ref", name), &n_r);
            TmSynonymGroup groups[16];
            int n_groups = 0;
            const JsonValue *jsyn = json_get(cs, "synonyms");
            if (jsyn && jsyn->kind == JSON_ARRAY) {
                const int ng = json_array_len(jsyn);
                if (ng > (int)(sizeof groups / sizeof groups[0])) die("同义词组太多");
                for (int g = 0; g < ng; g++) {
                    int n = 0;
                    const char **toks = strs_from(json_at(jsyn, g), &n);
                    groups[n_groups].tokens = toks;
                    groups[n_groups].n_tokens = n;
                    n_groups++;
                }
            }
            const TmSynonyms syn = {groups, n_groups};
            const double got = tm_meteor_score_full(
                p, n_p, r, n_r,
                json_num(need(cs, "alpha", name), 0.9),
                json_num(need(cs, "beta", name), 3.0),
                json_num(need(cs, "gamma", name), 0.5),
                (n_groups > 0) ? &syn : NULL,
                (int)json_num(need(cs, "use_stem", name), 1.0));
            report("meteor_full", got, want, tol);
            for (int g = 0; g < n_groups; g++) free((void *)groups[g].tokens);
            free(p);
            free(r);
        } else if (strcmp(func, "chrf") == 0) {
            const char *pred = json_str(need(cs, "pred", name), "");
            const char *ref = json_str(need(cs, "ref", name), "");
            const double got = tm_chrf_score(
                pred, ref,
                (int)json_num(need(cs, "max_n", name), 6.0),
                json_num(need(cs, "beta", name), 2.0));
            report("chrf", got, want, tol);
        } else if (strcmp(func, "chrf_tokens") == 0) {
            int n_p = 0, n_r = 0;
            const char **p = strs_from(need(cs, "pred", name), &n_p);
            const char **r = strs_from(need(cs, "ref", name), &n_r);
            const double got = tm_chrf_score_tokens(
                p, n_p, r, n_r,
                (int)json_num(need(cs, "max_n", name), 6.0),
                json_num(need(cs, "beta", name), 2.0));
            report("chrf_tokens", got, want, tol);
            free(p);
            free(r);
        } else if (strcmp(func, "wer") == 0) {
            const char *ref = json_str(need(cs, "ref", name), "");
            const char *hyp = json_str(need(cs, "hyp", name), "");
            report("wer", tm_word_error_rate(ref, hyp), want, tol);
        } else if (strcmp(func, "wer_tokens") == 0) {
            int n_r = 0, n_h = 0;
            const char **r = strs_from(need(cs, "ref", name), &n_r);
            const char **h = strs_from(need(cs, "hyp", name), &n_h);
            report("wer_tokens", tm_word_error_rate_tokens(r, n_r, h, n_h), want, tol);
            free(r);
            free(h);
        } else if (strcmp(func, "cer") == 0) {
            const char *ref = json_str(need(cs, "ref", name), "");
            const char *hyp = json_str(need(cs, "hyp", name), "");
            report("cer", tm_character_error_rate(ref, hyp), want, tol);
        } else if (strcmp(func, "cer_tokens") == 0) {
            int n_r = 0, n_h = 0;
            const char **r = strs_from(need(cs, "ref", name), &n_r);
            const char **h = strs_from(need(cs, "hyp", name), &n_h);
            report("cer_tokens", tm_character_error_rate_tokens(r, n_r, h, n_h), want, tol);
            free(r);
            free(h);
        } else {
            fprintf(stderr, "[compare_text_metrics] 未知 func \"%s\"（用例 %s）\n", func, name);
            json_free(root);
            return 2;
        }
    }

    json_free(root);
    printf("\n[compare_text_metrics] 共 %d 项检查：逐位一致 %d，失败 %d，最大绝对差 %.3e\n",
           g_checks, g_exact, g_fail, g_worst);
    return g_fail == 0 ? 0 : 1;
}
