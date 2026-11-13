/*
 * compare_fim.c —— FIM / T5 span corruption 与 AI.py 的对拍
 *
 * 用法：./tests/compare_fim [fim_reference.json]
 *
 * 覆盖 fim_transform / fim_target_mask / fim_batch /
 * span_corruption_mask / span_corruption_ids；错误用例只要求 C 侧返回 -1
 * （Python 侧在导出参考值时已确认会 raise，见 c/tools/fim_reference.py）。
 *
 * 随机 span 必须逐位一致：C 侧 fim.c 内置 Python random.Random 兼容
 * MT19937，seed 相同则 [start, length] 序列相同。
 */
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fim.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_fim] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key,
                             const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_fim] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/fim_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

/* seed 可能超过 32 位（用例含 2^32+1、2^40），用 double 承载再转 */
static unsigned long long jseed(const JsonValue *v)
{
    return (unsigned long long)jnum(v);
}

static int *iarr(const JsonValue *v, int n)
{
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
    return out;
}

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-34s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

/* 比较 got[0..got_n) 与 JSON 数组 want；长度不同也各记一项检查 */
static void check_ints(const char *prefix, const int *got, int got_n,
                       const JsonValue *want)
{
    const int want_n = json_array_len(want);
    char label[96];
    snprintf(label, sizeof label, "%s.len", prefix);
    check_int(label, got_n, want_n);
    for (int i = 0; i < want_n; i++) {
        snprintf(label, sizeof label, "%s[%d]", prefix, i);
        check_int(label, (i < got_n) ? got[i] : (long)LONG_MIN + 1,
                  jint(json_at(want, i)));
    }
}

/* 从用例读 transform 入参（成功用例的 mode 是字符串） */
static int mode_of(const JsonValue *cs, const char *who)
{
    const JsonValue *m = need(cs, "mode", who);
    if (m->kind == JSON_STRING) {
        const char *s = json_str(m, "");
        if (strcmp(s, "psm") == 0) return FIM_MODE_PSM;
        if (strcmp(s, "spm") == 0) return FIM_MODE_SPM;
        die("mode 只能是 \"psm\"/\"spm\"");
    }
    return json_int(m, 0); /* 错误用例里是整数 */
}

static void run_transform(const JsonValue *cs, const char *name, int want_mask)
{
    const int n_ids = json_array_len(need(cs, "ids", name));
    int *ids = iarr(need(cs, "ids", name), n_ids);
    const int vocab = jint(need(cs, "vocab_size", name));
    const int mode = mode_of(cs, name);
    const double pr = jnum(need(cs, "prefix_ratio", name));
    const double mr = jnum(need(cs, "middle_ratio", name));
    AiFimExample ex;

    if (ai_fim_transform(ids, n_ids, vocab, mode, pr, mr, &ex) != 0)
        die("ai_fim_transform 失败");

    if (!want_mask) {
        check_ints("input_ids", ex.input_ids, ex.n_input,
                   need(cs, "input_ids", name));
        check_ints("prefix", ex.prefix, ex.n_prefix, need(cs, "prefix", name));
        check_ints("middle", ex.middle, ex.n_middle, need(cs, "middle", name));
        check_ints("suffix", ex.suffix, ex.n_suffix, need(cs, "suffix", name));
        check_int("middle_start", ex.middle_start,
                  jint(need(cs, "middle_start", name)));
    } else {
        /* target_mask 用例只存掩码；transform 的逐位结果由 fim_transform
         * 用例覆盖，这里专注掩码本身（长度 = n_input-1 也一并检查）。 */
        const int mask_n = ex.n_input - 1;
        int *mask = (int *)malloc(sizeof(int) * (size_t)(mask_n > 0 ? mask_n : 1));
        if (!mask) die("out of memory");
        if (ai_fim_target_mask(&ex, mask) != 0) die("ai_fim_target_mask 失败");
        check_ints("mask", mask, mask_n, need(cs, "mask", name));
        free(mask);
    }
    ai_fim_example_free(&ex);
    free(ids);
}

static void run_batch(const JsonValue *cs, const char *name)
{
    const int n_seqs = json_array_len(need(cs, "seq_lens", name));
    int *seq_lens = iarr(need(cs, "seq_lens", name), n_seqs);
    const int flat_n = json_array_len(need(cs, "flat", name));
    int *flat = iarr(need(cs, "flat", name), flat_n);
    const int vocab = jint(need(cs, "vocab_size", name));
    const int mode = mode_of(cs, name);
    const double pr = jnum(need(cs, "prefix_ratio", name));
    const double mr = jnum(need(cs, "middle_ratio", name));
    AiFimExample *exs = NULL;
    int n_examples = 0;
    const JsonValue *want = need(cs, "examples", name);

    (void)flat_n;
    if (ai_fim_batch(flat, seq_lens, n_seqs, vocab, mode, pr, mr, &exs,
                     &n_examples) != 0)
        die("ai_fim_batch 失败");
    check_int("n_examples", n_examples, json_array_len(want));
    for (int e = 0; e < n_examples && e < json_array_len(want); e++) {
        const JsonValue *we = json_at(want, e);
        char pfx[48];
        snprintf(pfx, sizeof pfx, "ex%d.input_ids", e);
        check_ints(pfx, exs[e].input_ids, exs[e].n_input,
                   need(we, "input_ids", name));
        snprintf(pfx, sizeof pfx, "ex%d.prefix", e);
        check_ints(pfx, exs[e].prefix, exs[e].n_prefix,
                   need(we, "prefix", name));
        snprintf(pfx, sizeof pfx, "ex%d.middle", e);
        check_ints(pfx, exs[e].middle, exs[e].n_middle,
                   need(we, "middle", name));
        snprintf(pfx, sizeof pfx, "ex%d.suffix", e);
        check_ints(pfx, exs[e].suffix, exs[e].n_suffix,
                   need(we, "suffix", name));
        snprintf(pfx, sizeof pfx, "ex%d.middle_start", e);
        check_int(pfx, exs[e].middle_start, jint(need(we, "middle_start", name)));
    }
    ai_fim_batch_free(exs, n_examples);
    free(flat);
    free(seq_lens);
}

static void run_span_mask(const JsonValue *cs, const char *name)
{
    const int seq_len = jint(need(cs, "seq_len", name));
    const double ratio = jnum(need(cs, "mask_ratio", name));
    const double mean_span = jnum(need(cs, "mean_span", name));
    const unsigned long long seed = jseed(need(cs, "seed", name));
    AiFimSpan *spans = NULL;
    int n_spans = 0;
    const JsonValue *want = need(cs, "spans", name);
    const int want_n = json_array_len(want);

    if (ai_span_corruption_mask(seq_len, ratio, mean_span, seed, &spans,
                                &n_spans) != 0)
        die("ai_span_corruption_mask 失败");
    check_int("n_spans", n_spans, want_n);
    for (int i = 0; i < want_n; i++) {
        const JsonValue *ws = json_at(want, i);
        char label[64];
        if (i < n_spans) {
            snprintf(label, sizeof label, "span[%d].start", i);
            check_int(label, spans[i].start, jint(json_at(ws, 0)));
            snprintf(label, sizeof label, "span[%d].length", i);
            check_int(label, spans[i].length, jint(json_at(ws, 1)));
        } else {
            snprintf(label, sizeof label, "span[%d].start", i);
            check_int(label, LONG_MIN + 1, jint(json_at(ws, 0)));
            snprintf(label, sizeof label, "span[%d].length", i);
            check_int(label, LONG_MIN + 1, jint(json_at(ws, 1)));
        }
    }
    ai_fim_spans_free(spans);
}

static void run_span_ids(const JsonValue *cs, const char *name)
{
    const int n_ids = json_array_len(need(cs, "ids", name));
    int *ids = iarr(need(cs, "ids", name), n_ids);
    const int vocab = jint(need(cs, "vocab_size", name));
    const double ratio = jnum(need(cs, "mask_ratio", name));
    const double mean_span = jnum(need(cs, "mean_span", name));
    const unsigned long long seed = jseed(need(cs, "seed", name));
    const JsonValue *sv = need(cs, "sentinel_id", name);
    const int sentinel = (sv->kind == JSON_NULL) ? -1 : json_int(sv, -2);
    AiSpanCorruption sc;

    if (ai_span_corruption_ids(ids, n_ids, vocab, ratio, mean_span, sentinel,
                               seed, &sc) != 0)
        die("ai_span_corruption_ids 失败");
    check_ints("input_ids", sc.input_ids, sc.n_input,
               need(cs, "input_ids", name));
    check_ints("target_ids", sc.target_ids, sc.n_target,
               need(cs, "target_ids", name));
    {
        const JsonValue *want = need(cs, "spans", name);
        const int want_n = json_array_len(want);
        check_int("n_spans", sc.n_spans, want_n);
        for (int i = 0; i < want_n; i++) {
            const JsonValue *ws = json_at(want, i);
            char label[64];
            if (i < sc.n_spans) {
                snprintf(label, sizeof label, "span[%d].start", i);
                check_int(label, sc.spans[i].start, jint(json_at(ws, 0)));
                snprintf(label, sizeof label, "span[%d].length", i);
                check_int(label, sc.spans[i].length, jint(json_at(ws, 1)));
            } else {
                snprintf(label, sizeof label, "span[%d].start", i);
                check_int(label, LONG_MIN + 1, jint(json_at(ws, 0)));
            }
        }
    }
    check_int("sentinel_id", sc.sentinel_id, jint(need(cs, "sentinel", name)));
    ai_span_corruption_free(&sc);
    free(ids);
}

/* 错误用例：只要求 C 侧返回 -1（Python 侧已确认 raise） */
static void run_error(const JsonValue *cs, const char *name)
{
    const char *op = json_str(json_get(cs, "op"), "?");
    const JsonValue *a = need(cs, "args", name);
    int rc = 0;

    if (strcmp(op, "fim_transform") == 0) {
        const int n = json_array_len(need(a, "ids", name));
        int *ids = iarr(need(a, "ids", name), n);
        AiFimExample ex;
        rc = ai_fim_transform(ids, n, jint(need(a, "vocab_size", name)),
                              jint(need(a, "mode", name)),
                              jnum(need(a, "prefix_ratio", name)),
                              jnum(need(a, "middle_ratio", name)), &ex);
        if (rc == 0) ai_fim_example_free(&ex);
        free(ids);
    } else if (strcmp(op, "fim_target_mask") == 0) {
        const int n = json_array_len(need(a, "input_ids", name));
        int *ids = iarr(need(a, "input_ids", name), n);
        int *mask = (int *)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
        AiFimExample ex;
        if (!mask) die("out of memory");
        memset(&ex, 0, sizeof ex);
        ex.input_ids = ids;
        ex.n_input = n;
        ex.middle_start = jint(need(a, "middle_start", name));
        rc = ai_fim_target_mask(&ex, mask);
        free(mask);
        free(ids);
    } else if (strcmp(op, "fim_batch") == 0) {
        const int ns = json_array_len(need(a, "seq_lens", name));
        const int nf = json_array_len(need(a, "flat", name));
        int *lens = iarr(need(a, "seq_lens", name), ns);
        int *flat = iarr(need(a, "flat", name), nf);
        AiFimExample *exs = NULL;
        int got_n = 0;
        rc = ai_fim_batch(flat, lens, ns, jint(need(a, "vocab_size", name)),
                          jint(need(a, "mode", name)),
                          jnum(need(a, "prefix_ratio", name)),
                          jnum(need(a, "middle_ratio", name)), &exs, &got_n);
        if (rc == 0) ai_fim_batch_free(exs, got_n);
        free(flat);
        free(lens);
    } else if (strcmp(op, "span_corruption_mask") == 0) {
        AiFimSpan *spans = NULL;
        int got_n = 0;
        rc = ai_span_corruption_mask(jint(need(a, "seq_len", name)),
                                     jnum(need(a, "mask_ratio", name)),
                                     jnum(need(a, "mean_span", name)),
                                     jseed(need(a, "seed", name)), &spans,
                                     &got_n);
        if (rc == 0) ai_fim_spans_free(spans);
    } else if (strcmp(op, "span_corruption_ids") == 0) {
        const int n = json_array_len(need(a, "ids", name));
        int *ids = iarr(need(a, "ids", name), n);
        const JsonValue *sv = need(a, "sentinel_id", name);
        const int sentinel = (sv->kind == JSON_NULL) ? -1 : json_int(sv, -2);
        AiSpanCorruption sc;
        rc = ai_span_corruption_ids(ids, n, jint(need(a, "vocab_size", name)),
                                    jnum(need(a, "mask_ratio", name)),
                                    jnum(need(a, "mean_span", name)), sentinel,
                                    jseed(need(a, "seed", name)), &sc);
        if (rc == 0) ai_span_corruption_free(&sc);
        free(ids);
    } else {
        die("未知错误用例 op");
    }
    check_int("rc", rc, -1);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "fim_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_fim] 读不了 %s：%s\n"
                        "请先运行 python tools/fim_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_fim] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        printf("\n用例 %-28s kind=%s\n", name, kind);
        if (strcmp(kind, "fim_transform") == 0) run_transform(cs, name, 0);
        else if (strcmp(kind, "target_mask") == 0) run_transform(cs, name, 1);
        else if (strcmp(kind, "batch") == 0) run_batch(cs, name);
        else if (strcmp(kind, "span_mask") == 0) run_span_mask(cs, name);
        else if (strcmp(kind, "span_ids") == 0) run_span_ids(cs, name);
        else if (strcmp(kind, "error") == 0) run_error(cs, name);
        else die("未知 kind");
    }
    printf("\n=== compare_fim：%d 项检查，逐位相同 %d，失败 %d ===\n",
           g_checks, g_exact, g_fail);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
