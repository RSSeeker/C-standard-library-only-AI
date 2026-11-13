/*
 * compare_spec_decode.c —— 与 Python 版 Medusa / EAGLE 投机解码逐项对拍
 *
 * 用法：./tests/compare_spec_decode.exe spec_decode_reference.json
 *
 * 对拍内容：
 *   1) 目标模型：Python 导出的参数块按名字载入 C 的 llm_param_blocks；
 *   2) Medusa 多头 / EAGLE 草稿器：按初始权重载入；
 *   3) 每个用例比较：生成 ids 逐项相同、stats 的 drafted/accepted/target_calls 相同、
 *      accept_rate 在容差内。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../llm.h"
#include "../medusa.h"

static int g_checks = 0;
static int g_fail = 0;
static double g_max_diff = 0.0;

static void die(const char *msg)
{
    fprintf(stderr, "compare_spec_decode: %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *k, const char *who)
{
    const JsonValue *v = json_get(o, k);
    if (!v) {
        fprintf(stderr, "compare_spec_decode: %s 缺少字段 %s\n", who, k);
        exit(2);
    }
    return v;
}

static int arr_len(const JsonValue *v, const char *who)
{
    int n = json_array_len(v);
    if (n <= 0) {
        fprintf(stderr, "compare_spec_decode: %s 不是非空数组\n", who);
        exit(2);
    }
    return n;
}

static double arr_num(const JsonValue *arr, int i, const char *who)
{
    return json_num(json_at(arr, i), 0.0 / 0.0);   /* 越界时 0/0 → NaN */
}

static void check_exact(const char *name, long c, long py)
{
    g_checks++;
    if (c != py) {
        g_fail++;
        fprintf(stderr, "  %-32s C=%ld py=%ld [不等]\n", name, c, py);
    }
}

static void check_close(const char *name, double c, double py, double tol)
{
    g_checks++;
    double d = fabs(c - py);
    if (d > g_max_diff) g_max_diff = d;
    double scale = fabs(py) > 1.0 ? fabs(py) : 1.0;
    if (!(d / scale <= tol)) {
        g_fail++;
        fprintf(stderr, "  %-32s C=%.17g py=%.17g 相对差=%.3e [不等]\n",
                name, c, py, d / scale);
    }
}

static void load_target(CausalLM *m, const JsonValue *root)
{
    LLMParamBlock blocks[512];
    int nb = llm_param_blocks(m, blocks, 512);
    const JsonValue *jblocks = need(need(root, "target", "root"), "blocks", "target");
    int nj = arr_len(jblocks, "target.blocks");
    for (int j = 0; j < nj; j++) {
        const JsonValue *jb = json_at(jblocks, j);
        const char *name = json_str(need(jb, "name", "block"), "");
        const JsonValue *data = need(jb, "data", "block");
        int n = arr_len(data, "block.data");
        int found = -1;
        for (int i = 0; i < nb; i++)
            if (strcmp(blocks[i].name, name) == 0) { found = i; break; }
        if (found < 0) die("目标模型参数块名字未匹配");
        if (blocks[found].n != n) die("目标模型参数块长度不一致");
        for (int k = 0; k < n; k++)
            blocks[found].param[k] = json_num(json_at(data, k), 0.0);
    }
}

static void load_heads(MedusaHeads *h, const JsonValue *jheads)
{
    const JsonValue *ws = need(jheads, "weights", "heads");
    int k = arr_len(ws, "heads.weights");
    if (k != h->num_heads) die("heads 数量不一致");
    for (int i = 0; i < k; i++) {
        const JsonValue *jw = need(json_at(ws, i), "w", "head");
        const JsonValue *jb = need(json_at(ws, i), "b", "head");
        MLP *m = h->heads[i];
        Layer *l = &m->layers[0];
        for (int r = 0; r < l->fan_out; r++) {
            const JsonValue *row = json_at(jw, r);
            for (int c = 0; c < l->fan_in; c++)
                l->W[(size_t)r * l->fan_in + c] = json_num(json_at(row, c), 0.0);
            l->b[r] = json_num(json_at(jb, r), 0.0);
        }
    }
}

static void load_drafter(EagleDrafter *d, const JsonValue *jd)
{
    const JsonValue *Wf = need(jd, "Wf", "drafter");
    const JsonValue *We = need(jd, "We", "drafter");
    const JsonValue *bh = need(jd, "bh", "drafter");
    const JsonValue *Wh = need(jd, "Wh", "drafter");
    const JsonValue *bo = need(jd, "bo", "drafter");
    for (int j = 0; j < d->hidden; j++) {
        const JsonValue *rf = json_at(Wf, j);
        const JsonValue *re = json_at(We, j);
        for (int i = 0; i < d->d_model; i++) {
            d->Wf[(size_t)j * d->d_model + i] = json_num(json_at(rf, i), 0.0);
            d->We[(size_t)j * d->d_model + i] = json_num(json_at(re, i), 0.0);
        }
        d->bh[j] = json_num(json_at(bh, j), 0.0);
    }
    for (int j = 0; j < d->vocab_size; j++) {
        const JsonValue *rw = json_at(Wh, j);
        for (int i = 0; i < d->hidden; i++)
            d->Wh[(size_t)j * d->hidden + i] = json_num(json_at(rw, i), 0.0);
        d->bo[j] = json_num(json_at(bo, j), 0.0);
    }
}

static void load_prompt(const JsonValue *jp, int *prompt, int *n, const char *name)
{
    *n = arr_len(jp, name);
    for (int i = 0; i < *n; i++)
        prompt[i] = (int)json_num(json_at(jp, i), 0.0);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "spec_decode_reference.json";
    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "compare_spec_decode: 解析 %s 失败：%s\n", path, err);
        return 2;
    }
    const double tol = json_num(json_get(root, "tolerance"), 1e-15);
    const JsonValue *jt = need(root, "target", "root");
    int vocab = (int)json_num(need(jt, "vocab_size", "target"), 0);
    int d_model = (int)json_num(need(jt, "d_model", "target"), 0);
    int heads_n = (int)json_num(need(jt, "num_heads", "target"), 0);
    int d_ff = (int)json_num(need(jt, "d_ff", "target"), 0);
    int layers = (int)json_num(need(jt, "num_layers", "target"), 0);
    int max_len = (int)json_num(need(jt, "max_len", "target"), 0);
    double dropout = json_num(need(jt, "dropout", "target"), 0.0);

    CausalLM *target = llm_create(vocab, d_model, heads_n, d_ff, layers, max_len,
                                  dropout, 12345);
    if (!target) die("llm_create 失败");
    load_target(target, root);

    MedusaHeads *heads = medusa_heads_create(d_model, vocab, heads_n, 1);
    EagleDrafter *drafter = eagle_drafter_create(d_model, vocab, 4, 2);
    if (!heads || !drafter) die("草稿器创建失败");
    load_heads(heads, need(root, "heads", "root"));
    load_drafter(drafter, need(root, "drafter", "root"));

    const JsonValue *cases = need(root, "cases", "root");
    int nc = arr_len(cases, "cases");
    for (int ci = 0; ci < nc; ci++) {
        const JsonValue *jc = json_at(cases, ci);
        const char *name = json_str(need(jc, "name", "case"), "");
        const char *func = json_str(need(jc, "func", "case"), "");
        int prompt[4096];
        int prompt_len = 0;
        load_prompt(need(jc, "prompt", "case"), prompt, &prompt_len, "case.prompt");
        int max_new = (int)json_num(need(jc, "max_new", "case"), 0);
        const JsonValue *je = json_get(jc, "eos_id");
        int eos_id = (je && je->kind == JSON_NUMBER) ? (int)json_num(je, -1) : -1;
        int gamma = (int)json_num(json_get(jc, "gamma"), 1);
        const JsonValue *jids = need(jc, "ids", "case");
        const JsonValue *jstats = need(jc, "stats", "case");
        int py_len = arr_len(jids, "case.ids");
        int cap = py_len + 8;
        int *out = calloc((size_t)cap, sizeof(int));
        if (!out) die("calloc 失败");
        SpecStats stats;
        memset(&stats, 0, sizeof stats);
        int got;
        if (strcmp(func, "medusa") == 0)
            got = medusa_decode(target, prompt, prompt_len, max_new, eos_id, heads,
                                out, cap, &stats);
        else
            got = eagle_decode(target, prompt, prompt_len, max_new, gamma, eos_id,
                               drafter, out, cap, &stats);
        if (got < 0) {
            fprintf(stderr, "  %s: 解码失败\n", name);
            g_fail++;
            free(out);
            continue;
        }
        check_exact("case.len", got, py_len);
        for (int i = 0; i < got && i < py_len; i++)
            check_exact("case.id", out[i], (long)json_num(json_at(jids, i), 0.0));
        check_exact("case.drafted", stats.drafted,
                    (long)json_num(need(jstats, "drafted", "stats"), 0.0));
        check_exact("case.accepted", stats.accepted,
                    (long)json_num(need(jstats, "accepted", "stats"), 0.0));
        check_exact("case.target_calls", stats.target_calls,
                    (long)json_num(need(jstats, "target_calls", "stats"), 0.0));
        check_close("case.accept_rate", stats.accept_rate,
                    json_num(need(jstats, "accept_rate", "stats"), 0.0), tol);
        free(out);
    }

    printf("[compare_spec_decode] %d 项检查：失败 %d，最大绝对差 %.3e\n",
           g_checks, g_fail, g_max_diff);
    eagle_drafter_free(drafter);
    medusa_heads_free(heads);
    llm_free(target);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
