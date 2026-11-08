/*
 * test_text_tok.c —— 字节级 BPE 的自测（不依赖 Python）
 *
 * 覆盖：pretokenize 的三类切块、训练的确定性与提前停止、编解码往返、
 * 非法输入（非法 UTF-8 / 越界 id / 缓冲不足 / 前向引用规则）、存档往返。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../text_tok.h"

static int g_checks = 0;
static int g_fails = 0;

static void check(const char *name, int ok, const char *detail)
{
    g_checks += 1;
    if (ok) {
        printf("  [OK]   %-52s %s\n", name, detail ? detail : "");
    } else {
        g_fails += 1;
        printf("  [FAIL] %-52s %s\n", name, detail ? detail : "");
    }
}

/* 把 pieces 拼成以 \x1f 分隔的字符串。 */
static char *join(char **pieces, int n)
{
    size_t cap = 8, len = 0;
    char *buf = (char *)malloc(cap);
    int i;

    buf[0] = '\0';
    for (i = 0; i < n; i++) {
        size_t sl = strlen(pieces[i]);
        while (len + sl + 2 > cap) cap *= 2;
        buf = (char *)realloc(buf, cap);
        if (i) buf[len++] = '\x1f';
        memcpy(buf + len, pieces[i], sl);
        len += sl;
        buf[len] = '\0';
    }
    return buf;
}

static void test_pretokenize(void)
{
    char **pieces = NULL;
    int n;
    char *s;

    printf("[1] pretokenize\n");
    n = bpe_pretokenize("abc123  !!! 中文", &pieces);
    check("切块数", n == 5, NULL);
    if (n > 0) {
        s = join(pieces, n);
        check("三类切块内容", strcmp(s, "abc123\x1f" "  \x1f" "!!!\x1f" " \x1f" "中文") == 0, s);
        free(s);
        bpe_pretokenize_free(pieces, n);
    }
    n = bpe_pretokenize("", &pieces);
    check("空串没有任何块", n == 0, NULL);
    if (n == 0) bpe_pretokenize_free(pieces, 0);

    n = bpe_pretokenize("a\xff" "b", &pieces);
    check("非法 UTF-8 返回 -1", n == -1, NULL);
    if (n < 0) bpe_pretokenize_free(pieces, 0);
}

static void test_train(void)
{
    const char *text = "the cat sat the cat";
    BpeTokenizer *a = bpe_create();
    BpeTokenizer *b = bpe_create();
    int na = bpe_train(a, text, 262, 1);
    int nb = bpe_train(b, text, 262, 1);
    int i, same = (na == nb);

    printf("[2] 训练\n");
    check("两次训练条数相同", na == nb, NULL);
    if (same) {
        for (i = 0; i < na; i++) {
            int a1, b1, a2, b2;
            bpe_merge_at(a, i, &a1, &b1);
            bpe_merge_at(b, i, &a2, &b2);
            if (a1 != a2 || b1 != b2) same = 0;
        }
    }
    check("两次训练规则逐一相同", same, NULL);

    {
        BpeTokenizer *t = bpe_create();
        int n = bpe_train(t, text, 256, 1);          /* 目标就是基数 */
        check("vocab_size=256 时不学任何规则", n == 0 && bpe_vocab_size(t) == 256, NULL);
        n = bpe_train(t, text, 300, 1000);           /* 频次门槛太高 */
        check("min_frequency 太高时提前停止", n == 0, NULL);
        bpe_free(t);
    }
    bpe_free(a);
    bpe_free(b);
}

static void test_roundtrip(void)
{
    const char *cases[] = {
        "hello world",
        "你好，世界！",
        "a b  c\td\ne",
        "希腊 αβγ 和 привет",
        "🙂🙂"
    };
    int k;

    printf("[3] 编解码往返\n");
    for (k = 0; k < (int)(sizeof(cases) / sizeof(cases[0])); k++) {
        BpeTokenizer *t = bpe_create();
        int ids[512], n, back;
        char out[1024];
        char label[64];

        bpe_train(t, cases[k], 300, 1);
        n = bpe_encode(t, cases[k], ids, 512);
        snprintf(label, sizeof(label), "第 %d 例 encode 非负", k);
        check(label, n >= 0, NULL);
        if (n >= 0) {
            back = bpe_decode(t, ids, n, out, (int)sizeof(out));
            snprintf(label, sizeof(label), "第 %d 例 往返一致", k);
            check(label, back >= 0 && strcmp(out, cases[k]) == 0, out);
        }
        bpe_free(t);
    }
    {
        BpeTokenizer *t = bpe_create();
        int ids[8] = {0x48, 0x69};                   /* 'H' 'i' */
        char out[16];
        check("decode 正常返回字节数", bpe_decode(t, ids, 2, out, 16) == 2, out);
        check("decode 越界 id 返回 -1", bpe_decode(t, ids, 1, out, 16) >= 0
              && bpe_decode(t, (int[]){256}, 1, out, 16) == -1, NULL);
        check("decode 缓冲不足返回 -3", bpe_decode(t, ids, 2, out, 2) == -3, NULL);
        bpe_free(t);
    }
    {
        /* 0xE4 是「你」的首字节：单独解码必然不是合法 UTF-8 */
        BpeTokenizer *t = bpe_create();
        char out[16];
        check("decode 截断的多字节返回 -2",
              bpe_decode(t, (int[]){0xE4}, 1, out, 16) == -2, NULL);
        bpe_free(t);
    }
}

static void test_validation(void)
{
    int bad1[2] = {256, 97};        /* 第 0 条就引用 256（自己） */
    int bad2[4] = {97, 97, 259, 97};

    printf("[4] 非法参数\n");
    check("前向引用规则被拒", bpe_create_from_merges(bad1, 1) == NULL, NULL);
    check("隔一条再前向引用也被拒", bpe_create_from_merges(bad2, 2) == NULL, NULL);

    {
        BpeTokenizer *t = bpe_create();
        int a = -1, b = -1;
        int ids[4];
        check("merge_at 越界返回 0", bpe_merge_at(t, 0, &a, &b) == 0, NULL);
        check("encode 缓冲不足返回 -1", bpe_encode(t, "abcd", ids, 2) == -1, NULL);
        check("encode 非法 UTF-8 返回 -2", bpe_encode(t, "a\xff", ids, 4) == -2, NULL);
        check("train 的 vocab_size < 256 返回 -1", bpe_train(t, "abc", 200, 1) == -1, NULL);
        check("train 的 min_frequency < 1 返回 -1", bpe_train(t, "abc", 300, 0) == -1, NULL);
        bpe_free(t);
    }
}

static void test_save_load(void)
{
    const char *text = "banana bandana";
    const char *path = "tests/zz_text_tok_save.json";
    BpeTokenizer *t = bpe_create();
    BpeTokenizer *u;
    char err[256];
    int i, same = 1;

    printf("[5] 存档往返\n");
    bpe_train(t, text, 268, 1);
    check("save 返回 0", bpe_save(t, path) == 0, NULL);
    u = bpe_load(path, err, sizeof(err));
    check("load 成功", u != NULL, u ? NULL : err);
    if (u) {
        check("词表大小一致", bpe_vocab_size(u) == bpe_vocab_size(t), NULL);
        if (bpe_merge_count(u) != bpe_merge_count(t)) same = 0;
        for (i = 0; i < bpe_merge_count(t); i++) {
            int a1, b1, a2, b2;
            bpe_merge_at(t, i, &a1, &b1);
            bpe_merge_at(u, i, &a2, &b2);
            if (a1 != a2 || b1 != b2) same = 0;
        }
        check("规则逐一相同", same, NULL);
        {
            int ids1[64], ids2[64];
            int n1 = bpe_encode(t, text, ids1, 64);
            int n2 = bpe_encode(u, text, ids2, 64);
            check("加载后 encode 一致",
                  n1 == n2 && n1 > 0 && memcmp(ids1, ids2, (size_t)n1 * sizeof(int)) == 0,
                  NULL);
        }
        bpe_free(u);
    }
    bpe_free(t);
    remove(path);

    u = bpe_load(path, err, sizeof(err));
    check("读取不存在的文件返回 NULL", u == NULL, err);
    if (u) bpe_free(u);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[test_text_tok] 字节级 BPE\n\n");
    test_pretokenize();
    test_train();
    test_roundtrip();
    test_validation();
    test_save_load();
    printf("\n[test_text_tok] %d checks, %d failures\n", g_checks, g_fails);
    if (g_fails) {
        printf("[test_text_tok] FAILED\n");
        return 1;
    }
    printf("[test_text_tok] ALL PASS\n");
    return 0;
}
