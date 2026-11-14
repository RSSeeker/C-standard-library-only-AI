/*
 * compare_kv_evict.c —— KV cache 淘汰策略与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_kv_evict [kv_evict_reference.json]
 *
 * 参考值由 tools/kv_evict_reference.py 从 AI.py 现场导出；
 * 保留下标要求完全一致，K/V 数值要求逐位相等（都是输入值的拷贝）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../kv_evict.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_kv_evict] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_kv_evict] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/kv_evict_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

/* 二维数组（行主序）展平，行数 *row 列数 *col。 */
static double *darr2(const JsonValue *v, int *rows, int *cols)
{
    int r = json_array_len(v);
    int c = r > 0 ? json_array_len(json_at(v, 0)) : 0;
    double *out = (double *)malloc(sizeof(double) * (size_t)((r > 0 && c > 0) ? r * c : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < r; i++) {
        const JsonValue *row = json_at(v, i);
        for (int j = 0; j < c; j++)
            out[(size_t)i * (size_t)c + (size_t)j] = jnum(json_at(row, j));
    }
    *rows = r;
    *cols = c;
    return out;
}

static int *iarr(const JsonValue *v, int *n)
{
    int len = json_array_len(v);
    int *out = (int *)malloc(sizeof(int) * (size_t)(len ? len : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < len; i++) out[i] = jint(json_at(v, i));
    *n = len;
    return out;
}

static void check_ints(const char *what, const int *got, int ngot,
                       const int *want, int nwant)
{
    g_checks++;
    int ok = ngot == nwant;
    for (int i = 0; ok && i < nwant; i++) ok = got[i] == want[i];
    if (!ok) {
        g_fail++;
        printf("  [FAIL] %s：长度 %d vs %d\n", what, ngot, nwant);
        for (int i = 0; i < ngot && i < 16; i++) printf("    got[%d]=%d\n", i, got[i]);
        for (int i = 0; i < nwant && i < 16; i++) printf("    want[%d]=%d\n", i, want[i]);
    } else {
        g_exact++;
    }
}

static void check_nums(const char *what, const double *got, int ngot,
                       const double *want, int nwant)
{
    g_checks++;
    if (ngot != nwant) {
        g_fail++;
        printf("  [FAIL] %s：长度 %d vs %d\n", what, ngot, nwant);
        return;
    }
    for (int i = 0; i < nwant; i++) {
        double d = fabs(got[i] - want[i]);
        if (d > g_worst) g_worst = d;
        if (d != 0.0) {
            g_fail++;
            printf("  [FAIL] %s[%d]：%.17g vs %.17g（差 %.3e）\n",
                   what, i, got[i], want[i], d);
            return;
        }
    }
    g_exact++;
}

static void case_streaming(const JsonValue *cs, const char *name)
{
    int seq_len = jint(need(cs, "seq_len", name));
    int budget = jint(need(cs, "budget", name));
    int sink = jint(need(cs, "sink", name));
    int nwant = 0;
    int *want = iarr(need(cs, "keep", name), &nwant);
    int *got = (int *)malloc(sizeof(int) * (size_t)(budget ? budget : 1));
    if (!got) die("out of memory");
    int ngot = ai_streaming_kv_keep(seq_len, budget, sink, got);
    check_ints("keep", got, ngot, want, nwant);
    free(got);
    free(want);
}

static void case_scores(const JsonValue *cs, const char *name, int kind)
{
    int nq = 0, seq_len = 0;
    double *scores = darr2(need(cs, "scores", name), &nq, &seq_len);
    int budget = jint(need(cs, "budget", name));
    int sink = jint(need(cs, "sink", name));
    int nwant = 0;
    int *want = iarr(need(cs, "keep", name), &nwant);
    int *got = (int *)malloc(sizeof(int) * (size_t)(budget ? budget : 1));
    if (!got) die("out of memory");
    int ngot;
    if (kind == 0) {
        int recent = jint(need(cs, "recent", name));
        ngot = ai_h2o_kv_keep(scores, nq, seq_len, budget, sink, recent, got);
    } else {
        int obs = jint(need(cs, "obs", name));
        ngot = ai_snapkv_kv_keep(scores, nq, seq_len, budget, sink, obs, got);
    }
    check_ints("keep", got, ngot, want, nwant);
    free(got);
    free(want);
    free(scores);
}

static void case_evict(const JsonValue *cs, const char *name)
{
    int seq_len = 0, dim = 0;
    double *keys = darr2(need(cs, "keys", name), &seq_len, &dim);
    int vr = 0, vc = 0;
    double *values = darr2(need(cs, "values", name), &vr, &vc);
    int budget = jint(need(cs, "budget", name));
    const char *pol = json_str(need(cs, "policy", name), "?");
    int policy = strcmp(pol, "streaming") == 0 ? 0
                 : strcmp(pol, "h2o") == 0 ? 1 : 2;
    int sink = jint(need(cs, "sink", name));
    int recent = jint(need(cs, "recent", name));
    int obs = jint(need(cs, "obs", name));
    int nq = jint(need(cs, "nq", name));

    double *scores = NULL;
    int sr = 0, sc = 0;
    const JsonValue *sv = need(cs, "scores", name);
    int slen = json_array_len(sv);
    if (slen > 0 && nq > 0) {
        scores = darr2(sv, &sr, &sc);
        (void)sr;
        (void)sc;
    }

    int nwant = 0;
    int *want_keep = iarr(need(cs, "kept_indices", name), &nwant);
    int want_evicted = jint(need(cs, "evicted", name));
    int nk = 0, kd = 0;
    double *want_keys = darr2(need(cs, "new_keys", name), &nk, &kd);
    int nv = 0, vd = 0;
    double *want_values = darr2(need(cs, "new_values", name), &nv, &vd);

    int *got_keep = (int *)malloc(sizeof(int) * (size_t)(budget ? budget : 1));
    double *got_keys = (double *)malloc(sizeof(double) * (size_t)((budget > 0 && dim > 0) ? budget * dim : 1));
    double *got_values = (double *)malloc(sizeof(double) * (size_t)((budget > 0 && dim > 0) ? budget * dim : 1));
    if (!got_keep || !got_keys || !got_values) die("out of memory");

    int ngot = ai_kv_cache_evict(keys, values, seq_len, dim, budget,
                                 scores, nq, policy, sink, recent, obs,
                                 got_keep, got_keys, got_values);
    check_ints("kept_indices", got_keep, ngot, want_keep, nwant);
    g_checks++;
    if (seq_len - ngot != want_evicted) {
        g_fail++;
        printf("  [FAIL] evicted：%d vs %d\n", seq_len - ngot, want_evicted);
    } else {
        g_exact++;
    }
    check_nums("new_keys", got_keys, ngot * dim, want_keys, nk * kd);
    check_nums("new_values", got_values, ngot * dim, want_values, nv * vd);

    free(keys);
    free(values);
    free(scores);
    free(want_keep);
    free(want_keys);
    free(want_values);
    free(got_keep);
    free(got_keys);
    free(got_values);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "kv_evict_reference.json";
    char err[512];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) die(err);
    const JsonValue *cases = need(root, "cases", "root");
    int n = json_array_len(cases);
    printf("=== compare_kv_evict：%d 个用例（%s）===\n", n, path);
    for (int c = 0; c < n; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        printf("\n用例 %-24s kind=%s\n", name, kind);
        if (strcmp(kind, "streaming") == 0) {
            case_streaming(cs, name);
        } else if (strcmp(kind, "h2o") == 0) {
            case_scores(cs, name, 0);
        } else if (strcmp(kind, "snapkv") == 0) {
            case_scores(cs, name, 1);
        } else if (strcmp(kind, "evict") == 0) {
            case_evict(cs, name);
        } else {
            die("未知 kind");
        }
    }
    printf("\n=== compare_kv_evict：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
