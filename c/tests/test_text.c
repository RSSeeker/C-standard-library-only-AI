/*
 * test_text.c —— TextProcessor / 生成相关函数的自测与跨语言核对
 *
 *   ./test_text dump    打印词表、one-hot、softmax_with_temp 等结果（供 Python 侧比对）
 *   ./test_text self    基本断言
 */
#include "../text.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TEXT = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。";

static int mode_self(void)
{
    TextProcessor *tp = tp_create(TEXT);
    printf("[self] 文本长度(码点)=%d 词表大小=%d\n", utf8_length(TEXT), tp->vocab_size);

    /* 词表必须升序且去重 */
    int ok = 1;
    for (int i = 1; i < tp->vocab_size; i++)
        if (tp->chars[i] <= tp->chars[i - 1]) ok = 0;
    printf("  词表严格升序去重: %s\n", ok ? "是" : "否");

    /* one-hot 往返 */
    int bad = 0;
    for (int i = 0; i < tp->vocab_size; i++) {
        double *v = malloc((size_t)tp->vocab_size * sizeof(double));
        tp_encode_one_hot_cp(tp, tp->chars[i], v);
        double s = 0.0;
        for (int j = 0; j < tp->vocab_size; j++) {
            s += v[j];
            if (j != i && v[j] != 0.0) bad++;
        }
        if (fabs(s - 1.0) > 1e-15) bad++;
        free(v);
    }
    printf("  one-hot 往返异常数: %d\n", bad);

    /* 索引编解码往返 */
    int *idx = malloc((size_t)utf8_length(TEXT) * sizeof(int));
    int n = tp_encode_indices(tp, TEXT, idx);
    char *buf = malloc(strlen(TEXT) * 4 + 8);
    tp_decode_indices(tp, idx, n, buf, (int)strlen(TEXT) * 4 + 8);
    int roundtrip_ok = (strcmp(buf, TEXT) == 0);
    printf("  索引编解码往返一致: %s\n", roundtrip_ok ? "是" : "否");

    /* 未知字符 -> 索引 0，解码占位 '?' */
    int unknown = tp_idx_of(tp, 0x1F600);        /* 表情符号，不在词表 */
    printf("  未知字符(emoji) 查表结果: %d（应为 -1）\n", unknown);

    free(idx);
    free(buf);
    tp_free(tp);
    ok = ok && bad == 0 && roundtrip_ok && unknown == -1;
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* 一个简单的假模型：把输入按固定权重映射到 logits，用于验证生成流程能跑通 */
static void fake_forward(void *ctx, const double *x, int n_in, double *out, int n_out)
{
    (void)ctx;
    for (int j = 0; j < n_out; j++) {
        double s = 0.0;
        for (int i = 0; i < n_in; i++) s += x[i] * (double)((i * 7 + j * 13) % 11 - 5) * 0.01;
        out[j] = s;
    }
}

static int mode_dump(void)
{
    TextProcessor *tp = tp_create(TEXT);
    printf("TEXT_LEN %d\n", utf8_length(TEXT));
    printf("VOCAB %d\n", tp->vocab_size);
    printf("CHARS");
    for (int i = 0; i < tp->vocab_size; i++) printf(" %d", tp->chars[i]);
    printf("\n");

    /* 前 8 个字符的 one-hot 的 1 的位置 */
    printf("ONEHOT_POS");
    for (int i = 0; i < 8 && i < tp->vocab_size; i++) {
        double *v = malloc((size_t)tp->vocab_size * sizeof(double));
        tp_encode_one_hot_cp(tp, tp->chars[i], v);
        for (int j = 0; j < tp->vocab_size; j++) if (v[j] == 1.0) printf(" %d", j);
        free(v);
    }
    printf("\n");

    /* 索引序列（前 20 个码点） */
    int n = utf8_length(TEXT);
    int *idx = malloc((size_t)n * sizeof(int));
    int m = tp_encode_indices(tp, TEXT, idx);
    printf("INDICES");
    for (int i = 0; i < 20 && i < m; i++) printf(" %d", idx[i]);
    printf("\n");
    free(idx);

    /* softmax_with_temp：固定 logits，便于与 Python 逐位比较 */
    const double logits[5] = { 0.5, -1.25, 2.0, 0.0, 0.75 };
    for (double T = 0.0; T < 1.5; T += 0.5) {
        double tt = (T == 0.0) ? 1.0 : T;
        double probs[5];
        softmax_with_temp(logits, 5, tt, probs);
        printf("SOFTMAX T=%.1f", tt);
        for (int i = 0; i < 5; i++) printf(" %.17g", probs[i]);
        printf("\n");
    }

    /* 生成：用假模型 + 固定随机种子，两边比对同一段文本 */
    unsigned long long rng = 12345ULL;
    char out[512];
    int got = generate_text(fake_forward, NULL, tp, "君不见", 6, 3, 0.8, out, sizeof(out), &rng);
    printf("GENERATED %d %s\n", got, out);

    tp_free(tp);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "self";
    if (strcmp(mode, "self") == 0) return mode_self();
    if (strcmp(mode, "dump") == 0) return mode_dump();
    fprintf(stderr, "未知模式: %s\n", mode);
    return 2;
}
