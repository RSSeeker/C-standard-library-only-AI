/*
 * compare_tree_attn.c —— 树状注意力辅助与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_tree_attn [tree_attn_reference.json]
 *
 * 覆盖 build_draft_tree（parents/tokens/leaves/paths）与
 * tree_attention_bias / tree_attention_mask。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../tree_attn.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_tree_attn] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_tree_attn] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/tree_attn_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static int *iarr(const JsonValue *v, int n)
{
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
    return out;
}

static double *darr(const JsonValue *v, int n)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    return out;
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

static void check_num(const char *what, double got, double want, double tol)
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
        printf("  %-30s C=%.17g py=%.17g [不等]\n", what, got, want);
        g_fail++;
    }
}

static void case_tree(const JsonValue *cs, const char *name, double tol)
{
    const int n_seqs = json_array_len(need(cs, "seq_lens", name));
    const int n_nodes_want = jint(need(cs, "n_nodes", name));
    const int n_leaves_want = jint(need(cs, "n_leaves", name));
    const JsonValue *paths = need(cs, "paths", name);
    int *seq_lens = iarr(need(cs, "seq_lens", name), n_seqs);
    int *flat = iarr(need(cs, "flat", name), json_array_len(need(cs, "flat", name)));
    int *want_parents = iarr(need(cs, "parents", name), n_nodes_want);
    int *want_leaves = iarr(need(cs, "leaves", name), n_leaves_want);
    int *got_parents = NULL, *got_tokens = NULL, *got_leaves = NULL;
    int *got_paths = NULL, *got_path_lens = NULL;
    int got_nodes = 0, got_leaves_n = 0;

    if (ai_build_draft_tree(flat, seq_lens, n_seqs, &got_parents, &got_tokens,
                            &got_leaves, &got_nodes, &got_leaves_n,
                            &got_paths, &got_path_lens) != 0)
        die("ai_build_draft_tree 失败");
    check_int("n_nodes", got_nodes, n_nodes_want);
    check_int("n_leaves", got_leaves_n, n_leaves_want);
    check_int("root_parent", got_parents[0], -1);
    check_int("root_token", got_tokens[0], -1);
    for (int i = 1; i < n_nodes_want; i++) {
        char label[64];
        snprintf(label, sizeof label, "parents[%d]", i);
        check_int(label, got_parents[i], want_parents[i]);
        snprintf(label, sizeof label, "tokens[%d]", i);
        check_int(label, got_tokens[i],
                  json_int(json_at(need(cs, "tokens", name), i), -1));
    }
    for (int i = 0; i < n_leaves_want; i++) {
        char label[64];
        snprintf(label, sizeof label, "leaves[%d]", i);
        check_int(label, got_leaves[i], want_leaves[i]);
    }
    {
        int off = 0;
        for (int i = 0; i < n_leaves_want; i++) {
            const JsonValue *want_path = json_at(paths, i);
            const int len = json_array_len(want_path);
            check_int("path_len", got_path_lens[i], len);
            for (int k = 0; k < len; k++) {
                char label[64];
                snprintf(label, sizeof label, "path[%d][%d]", i, k);
                check_int(label, got_paths[off + k], jint(json_at(want_path, k)));
            }
            off += got_path_lens[i];
        }
    }

    /* bias / mask */
    {
        const int n = got_nodes;
        const double neg_inf = jnum(need(cs, "neg_inf", name));
        double *bias = (double *)malloc(sizeof(double) * (size_t)(n * n));
        int *mask = (int *)malloc(sizeof(int) * (size_t)(n * n));
        double *want_bias = darr(need(cs, "bias", name), n * n);
        int *want_mask = iarr(need(cs, "mask", name), n * n);
        if (!bias || !mask) die("out of memory");
        if (ai_tree_attention_bias(got_parents, n, neg_inf, bias) != 0)
            die("ai_tree_attention_bias 失败");
        for (int i = 0; i < n * n; i++) check_num("bias", bias[i], want_bias[i], tol);
        if (ai_tree_attention_mask(got_parents, n, mask) != 0)
            die("ai_tree_attention_mask 失败");
        for (int i = 0; i < n * n; i++) {
            char label[64];
            snprintf(label, sizeof label, "mask[%d]", i);
            check_int(label, mask[i], want_mask[i]);
        }
        free(bias);
        free(mask);
        free(want_bias);
        free(want_mask);
    }

    ai_draft_tree_free(got_parents, got_tokens, got_leaves, got_paths, got_path_lens);
    free(seq_lens);
    free(flat);
    free(want_parents);
    free(want_leaves);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "tree_attn_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_tree_attn] 读不了 %s：%s\n"
                        "请先运行 python tools/tree_attn_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_tree_attn] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        printf("\n用例 %-20s kind=%s\n", name, kind);
        if (strcmp(kind, "tree") == 0) case_tree(cs, name, tol);
        else die("未知 kind");
    }
    printf("\n=== compare_tree_attn：%d 项检查，逐位相同 %d，失败 %d，"
           "最大绝对差 %.3e ===\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
