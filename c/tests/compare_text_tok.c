/*
 * compare_text_tok.c —— C 侧 BPE 与 Python `BytePairTokenizer` 的逐位对拍
 *
 * 每个用例检查两遍：
 *   [A] 用 Python 学到的 merges 构造 C 分词器，比对 vocab_size、每条规则、
 *       pretokenize 的切块、encode 的 id 序列、decode 的文本、以及
 *       「从存档重建后 encode」是否一致（覆盖 bpe_create_from_merges）；
 *   [B] 用同一段语料与同一组参数在 C 里重新训练，比对学到的规则条数与每条规则
 *       （覆盖 bpe_train 的挑对次序与 min_frequency 提前停止）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../text_tok.h"

static int g_arrays = 0;
static int g_exact = 0;
static int g_bad = 0;
static double g_worst = 0.0;

static void need(const JsonValue *obj, const char *key, const char *who)
{
    if (!json_get(obj, key)) {
        fprintf(stderr, "compare_text_tok: %s 缺少字段 %s\n", who, key);
        exit(2);
    }
}

/* 记一项「逐位比较」的结果。 */
static void report_bits(const char *what, int equal, const char *detail)
{
    g_arrays += 1;
    if (equal) {
        g_exact += 1;
        printf("    %-40s [OK]  %s\n", what, detail ? detail : "逐位");
    } else {
        g_bad += 1;
        printf("    %-40s [FAIL] %s\n", what, detail ? detail : "不一致");
    }
}

/* 把 pieces 拼成 "a\x1f"b\x1f"… 便于一次比较。 */
static char *join_pieces(char **pieces, int n)
{
    size_t cap = 16, len = 0;
    char *buf = (char *)malloc(cap);
    int i;

    buf[0] = '\0';
    for (i = 0; i < n; i++) {
        size_t need_len = strlen(pieces[i]) + 2;
        if (len + need_len + 1 > cap) {
            while (len + need_len + 1 > cap) cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        if (i) buf[len++] = '\x1f';
        memcpy(buf + len, pieces[i], strlen(pieces[i]));
        len += strlen(pieces[i]);
        buf[len] = '\0';
    }
    return buf;
}

static char *join_json_strings(const JsonValue *arr)
{
    int n = json_array_len(arr);
    size_t cap = 16, len = 0;
    char *buf = (char *)malloc(cap);
    int i;

    buf[0] = '\0';
    for (i = 0; i < n; i++) {
        const char *s = json_str(json_at(arr, i), "");
        size_t sl = strlen(s);
        if (len + sl + 2 > cap) {
            while (len + sl + 2 > cap) cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        if (i) buf[len++] = '\x1f';
        memcpy(buf + len, s, sl);
        len += sl;
        buf[len] = '\0';
    }
    return buf;
}

static int ids_equal(const int *ids, int n, const JsonValue *arr, int *first_diff)
{
    int i, m = json_array_len(arr);

    *first_diff = -1;
    if (n != m) return 0;
    for (i = 0; i < n; i++) {
        int want = (int)json_num(json_at(arr, i), -1.0);
        if (ids[i] != want) {
            *first_diff = i;
            return 0;
        }
    }
    return 1;
}

static void run_case(const JsonValue *c)
{
    const char *name = json_str(json_get(c, "name"), "?");
    const char *text = json_str(json_get(c, "text"), "");
    const JsonValue *merges = json_get(c, "merges");
    const JsonValue *ids_arr = json_get(c, "ids");
    const JsonValue *want_pieces = json_get(c, "pieces");
    const char *want_decoded = json_str(json_get(c, "decoded"), "");
    int want_vocab = (int)json_num(json_get(c, "vocab_size"), -1.0);
    int vocab_arg = (int)json_num(json_get(c, "vocab_size_arg"), 256.0);
    int min_freq = (int)json_num(json_get(c, "min_frequency"), 1.0);
    int n_merges = json_array_len(merges);
    int *pairs, i, n_pieces, n_ids, diff;
    char **pieces = NULL, buf[4096];
    BpeTokenizer *t;
    char want_joined[4096], got_joined[4096];

    need(c, "merges", name);
    need(c, "ids", name);
    need(c, "pieces", name);

    printf("  [%s] vocab=%d merges=%d\n", name, want_vocab, n_merges);

    pairs = (int *)malloc((size_t)(n_merges ? n_merges : 1) * 2 * sizeof(int));
    for (i = 0; i < n_merges; i++) {
        const JsonValue *p = json_at(merges, i);
        pairs[2 * i] = (int)json_num(json_at(p, 0), -1.0);
        pairs[2 * i + 1] = (int)json_num(json_at(p, 1), -1.0);
    }

    /* ---------- [A] 用 Python 的规则重建 ---------- */
    t = bpe_create_from_merges(pairs, n_merges);
    report_bits("create_from_merges", t != NULL, t ? NULL : "返回 NULL");
    if (!t) {
        free(pairs);
        return;
    }
    snprintf(buf, sizeof(buf), "C=%d py=%d", bpe_vocab_size(t), want_vocab);
    report_bits("vocab_size", bpe_vocab_size(t) == want_vocab, buf);

    for (i = 0; i < n_merges; i++) {
        int a = -1, b = -1;
        int want_a = (int)json_num(json_at(json_at(merges, i), 0), -1.0);
        int want_b = (int)json_num(json_at(json_at(merges, i), 1), -1.0);
        bpe_merge_at(t, i, &a, &b);
        snprintf(buf, sizeof(buf), "第 %d 条 (%d,%d) vs (%d,%d)", i, a, b, want_a, want_b);
        report_bits("merge_at", a == want_a && b == want_b, buf);
    }

    n_pieces = bpe_pretokenize(text, &pieces);
    snprintf(want_joined, sizeof(want_joined), "%s", join_json_strings(want_pieces));
    if (n_pieces < 0) {
        report_bits("pretokenize", 0, "返回 -1");
    } else {
        char *got = join_pieces(pieces, n_pieces);
        snprintf(got_joined, sizeof(got_joined), "%s", got);
        free(got);
        snprintf(buf, sizeof(buf), "C=%d py=%d", n_pieces, json_array_len(want_pieces));
        report_bits("pretokenize 块数", n_pieces == json_array_len(want_pieces), buf);
        report_bits("pretokenize 内容", strcmp(got_joined, want_joined) == 0, NULL);
        bpe_pretokenize_free(pieces, n_pieces);
        pieces = NULL;
    }

    n_ids = bpe_encode(t, text, (int *)buf, 1024);   /* buf 够大：语料都是短句 */
    if (n_ids < 0) {
        report_bits("encode", 0, "返回负数");
    } else {
        int ids[1024];
        n_ids = bpe_encode(t, text, ids, 1024);
        if (!ids_equal(ids, n_ids, ids_arr, &diff)) {
            snprintf(buf, sizeof(buf), "首个不同下标 %d", diff);
            report_bits("encode", 0, buf);
        } else {
            snprintf(buf, sizeof(buf), "n=%d", n_ids);
            report_bits("encode", 1, buf);
        }
        {
            int got = bpe_decode(t, ids, n_ids, buf, (int)sizeof(buf));
            if (got < 0) {
                report_bits("decode", 0, "返回负数");
            } else {
                report_bits("decode", strcmp(buf, want_decoded) == 0, buf);
            }
        }
        /* ids_rebuilt：用同一组规则重建后 encode 必须一样 */
        {
            BpeTokenizer *t2 = bpe_create_from_merges(pairs, n_merges);
            int ids2[1024];
            int n2 = bpe_encode(t2, text, ids2, 1024);
            report_bits("重建后 encode", n2 == n_ids
                        && memcmp(ids, ids2, (size_t)n_ids * sizeof(int)) == 0, NULL);
            bpe_free(t2);
        }
    }
    bpe_free(t);

    /* ---------- [B] 在 C 里重新训练 ---------- */
    {
        BpeTokenizer *fresh = bpe_create();
        int learned = bpe_train(fresh, text, vocab_arg, min_freq);
        snprintf(buf, sizeof(buf), "C=%d py=%d", learned, n_merges);
        report_bits("train 学到的条数", learned == n_merges, buf);
        if (learned == n_merges) {
            int ok = 1;
            for (i = 0; i < n_merges; i++) {
                int a = -1, b = -1;
                bpe_merge_at(fresh, i, &a, &b);
                if (a != pairs[2 * i] || b != pairs[2 * i + 1]) {
                    ok = 0;
                    snprintf(buf, sizeof(buf), "第 %d 条不同", i);
                    break;
                }
            }
            report_bits("train 规则序列", ok, ok ? NULL : buf);
        }
        {
            int ids[1024];
            int n2 = bpe_encode(fresh, text, ids, 1024);
            int ids_ok = n2 >= 0 && ids_equal(ids, n2, ids_arr, &diff);
            report_bits("train 后 encode", ids_ok, ids_ok ? NULL : "与 Python 不同");
        }
        bpe_free(fresh);
    }

    free(pairs);
}

/* ====================== WordPiece ====================== */

static char *join_wp_tokens(const WpVocab *v)
{
    int n = wp_vocab_size(v), i;
    size_t cap = 32, len = 0;
    char *buf = (char *)malloc(cap);

    buf[0] = '\0';
    for (i = 0; i < n; i++) {
        const char *tok = wp_token(v, i);
        size_t sl = strlen(tok);
        if (len + sl + 2 > cap) {
            while (len + sl + 2 > cap) cap *= 2;
            buf = (char *)realloc(buf, cap);
        }
        if (i) buf[len++] = '\x1f';
        memcpy(buf + len, tok, sl);
        len += sl;
        buf[len] = '\0';
    }
    return buf;
}

/* 解一段 ids 并比较文本。 */
static void cmp_decode(const WpVocab *v, const int *ids, int n, const char *want, const char *what)
{
    char got[4096];
    int k = wp_decode(v, ids, n, got, (int)sizeof(got));

    if (k < 0) {
        report_bits(what, 0, "decode 返回负数");
        return;
    }
    report_bits(what, strcmp(got, want) == 0, got);
}

static void run_wp_case(const JsonValue *c)
{
    const char *name = json_str(json_get(c, "name"), "?");
    const char *mode = json_str(json_get(c, "mode"), "");
    const JsonValue *tokens = json_get(c, "tokens");
    int n_tokens = json_array_len(tokens);
    char buf[1024];
    WpVocab *v = NULL;
    const char **arr;
    int i;

    printf("  [%s] %s vocab=%d\n", name, mode, n_tokens);
    arr = (const char **)malloc((size_t)(n_tokens ? n_tokens : 1) * sizeof(char *));
    for (i = 0; i < n_tokens; i++) arr[i] = json_str(json_at(tokens, i), "");

    if (strcmp(mode, "train") == 0) {
        const char *text = json_str(json_get(c, "text"), "");
        int vocab_arg = (int)json_num(json_get(c, "vocab_size_arg"), 512.0);
        int min_freq = (int)json_num(json_get(c, "min_frequency"), 2.0);
        char err[256];

        memset(err, 0, sizeof(err));
        v = wp_train(text, vocab_arg, min_freq, err, sizeof(err));
        report_bits("wp_train", v != NULL, v ? NULL : err);
        if (!v) {
            free(arr);
            return;
        }
    } else {
        v = wp_create(arr, n_tokens, buf, sizeof(buf));
        report_bits("wp_create", v != NULL, v ? NULL : buf);
        if (!v) {
            free(arr);
            return;
        }
    }

    {
        char *got = join_wp_tokens(v);
        char *want = join_json_strings(tokens);
        report_bits("词表大小", wp_vocab_size(v) == n_tokens, NULL);
        report_bits("词表内容", strcmp(got, want) == 0, NULL);
        free(got);
        free(want);
    }

    if (strcmp(mode, "train") == 0) {
        const char *text = json_str(json_get(c, "text"), "");
        const JsonValue *ids_arr = json_get(c, "ids");
        const JsonValue *small_arr = json_get(c, "ids_small");
        int ids[4096], small[4096], n, n_small, diff;

        n = wp_encode(v, text, ids, 4096, 100);
        if (n < 0) {
            report_bits("encode", 0, "返回负数");
        } else {
            report_bits("encode", ids_equal(ids, n, ids_arr, &diff), NULL);
            cmp_decode(v, ids, n, json_str(json_get(c, "decoded"), ""), "decode");
        }
        n_small = wp_encode(v, text, small, 4096, 2);
        if (n_small < 0) {
            report_bits("encode(max_chars=2)", 0, "返回负数");
        } else {
            report_bits("encode(max_chars=2)", ids_equal(small, n_small, small_arr, &diff), NULL);
            cmp_decode(v, small, n_small, json_str(json_get(c, "decoded_small"), ""),
                       "decode(max_chars=2)");
        }
        {
            const char *second = json_str(json_get(c, "second"), "");
            int pair_ids[4096];
            int np = wp_encode_pair(v, text, second, pair_ids, 4096, 100);
            if (np < 0) {
                report_bits("encode_pair", 0, "返回负数");
            } else {
                report_bits("encode_pair", ids_equal(pair_ids, np, json_get(c, "pair"), &diff), NULL);
                cmp_decode(v, pair_ids, np, json_str(json_get(c, "pair_decoded"), ""),
                           "decode(pair)");
            }
        }
        snprintf(buf, sizeof(buf), "C=%d py=%d", v->unk_id,
                 (int)json_num(json_get(c, "unk_id"), -1.0));
        report_bits("unk_id", v->unk_id == (int)json_num(json_get(c, "unk_id"), -1.0), buf);
    } else {
        const JsonValue *encodes = json_get(c, "encodes");
        int n_enc = json_array_len(encodes);
        int ids[4096];
        int diff;

        for (i = 0; i < n_enc; i++) {
            const JsonValue *e = json_at(encodes, i);
            const char *text = json_str(json_get(e, "text"), "");
            int n = wp_encode(v, text, ids, 4096, 100);

            if (n < 0) {
                report_bits("encode(probe)", 0, "返回负数");
                continue;
            }
            report_bits("encode(probe)", ids_equal(ids, n, json_get(e, "ids"), &diff), text);
            cmp_decode(v, ids, n, json_str(json_get(e, "decoded"), ""), "decode(probe)");
        }
        {
            const JsonValue *seq = json_get(c, "decode_ids");
            int n_seq = json_array_len(seq);
            int *ids = (int *)malloc((size_t)(n_seq ? n_seq : 1) * sizeof(int));
            for (i = 0; i < n_seq; i++) ids[i] = (int)json_num(json_at(seq, i), -1.0);
            cmp_decode(v, ids, n_seq, json_str(json_get(c, "decode_text"), ""), "decode(手写 ids)");
            free(ids);
        }
    }

    wp_free(v);
    free(arr);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "text_tok_reference.json";
    char err[256];
    JsonValue *root;
    const JsonValue *cases;
    int i;

    root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "compare_text_tok: 读不了 %s（%s）\n", path, err);
        return 2;
    }
    cases = json_get(root, "cases");
    if (!cases || cases->kind != JSON_ARRAY) {
        fprintf(stderr, "compare_text_tok: %s 缺少 cases 数组\n", path);
        json_free(root);
        return 2;
    }
    printf("[compare_text_tok] BPE（Python BytePairTokenizer）对拍\n");
    for (i = 0; i < json_array_len(cases); i++) {
        run_case(json_at(cases, i));
    }
    {
        const JsonValue *wp = json_get(root, "wordpiece");
        int n_wp = wp ? json_array_len(wp) : 0;
        printf("\n[compare_text_tok] WordPiece（Python WordPieceTokenizer）对拍\n");
        for (i = 0; i < n_wp; i++) {
            run_wp_case(json_at(wp, i));
        }
    }
    json_free(root);
    printf("\n[compare_text_tok] %d 项，%d 项逐位相同，%d 项不一致，worst=%.3e\n",
           g_arrays, g_exact, g_bad, g_worst);
    if (g_bad) {
        printf("[compare_text_tok] [超出容差]\n");
        return 1;
    }
    printf("[compare_text_tok] [OK] BPE + WordPiece match AI.py\n");
    return 0;
}
