/*
 * compare_block_mgr.c —— PagedBlockManager 与 AI.py 的操作序列对拍
 *
 * 用法：./tests/compare_block_mgr [block_mgr_reference.json]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../block_mgr.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_block_mgr] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_block_mgr] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/block_mgr_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static const char *jstr(const JsonValue *v) { return json_str(v, "?"); }
static int alen(const JsonValue *v) { return json_array_len(v); }

static void check_int(const char *what, long got, long want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-30s C=%ld py=%ld [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_ints(const char *what, const int *got, int n,
                       const JsonValue *want)
{
    if (n != alen(want)) {
        g_checks++;
        g_fail++;
        printf("  %-30s 长度 C=%d py=%d [不等]\n", what, n, alen(want));
        return;
    }
    for (int i = 0; i < n; i++) {
        char label[64];
        snprintf(label, sizeof label, "%s[%d]", what, i);
        check_int(label, got[i], jint(json_at(want, i)));
    }
}

static void run_case(const JsonValue *cs)
{
    const char *name = jstr(json_get(cs, "name"));
    const int nb = jint(need(cs, "num_blocks", name));
    const int bs = jint(need(cs, "block_size", name));
    const int sc = jint(need(cs, "swap_capacity", name));
    const JsonValue *ops = need(cs, "ops", name);
    const int n_ops = alen(ops);
    AiBlockManager *m = ai_block_mgr_create(nb, bs, sc);
    int *buf = (int *)malloc(sizeof(int) * (size_t)(nb * 4 + 16));
    int *buf2 = (int *)malloc(sizeof(int) * (size_t)(nb * 4 + 16));
    if (!m || !buf || !buf2) die("out of memory");
    printf("\n用例 %s（num_blocks=%d, block_size=%d, swap_capacity=%d）\n",
           name, nb, bs, sc);
    for (int i = 0; i < n_ops; i++) {
        const JsonValue *op = json_at(ops, i);
        const char *kind = jstr(json_get(op, "op"));
        if (strcmp(kind, "new_request") == 0) {
            check_int("new_request.rid", ai_block_mgr_new_request(m),
                      jint(need(op, "rid", name)));
        } else if (strcmp(kind, "allocate") == 0) {
            int len = 0;
            int rc = ai_block_mgr_allocate(m, jint(need(op, "rid", name)),
                                           jint(need(op, "count", name)),
                                           buf, &len);
            check_int("allocate.rc", rc, 0);
            check_ints("allocate.table", buf, len, need(op, "table", name));
        } else if (strcmp(kind, "ensure") == 0) {
            int len = 0;
            int rc = ai_block_mgr_ensure(m, jint(need(op, "rid", name)),
                                         jint(need(op, "tokens", name)),
                                         buf, &len);
            check_int("ensure.rc", rc, 0);
            check_ints("ensure.table", buf, len, need(op, "table", name));
        } else if (strcmp(kind, "fork") == 0) {
            int got = ai_block_mgr_fork(m, jint(need(op, "src", name)),
                                        jint(need(op, "dst_arg", name)));
            check_int("fork.dst", got, jint(need(op, "dst", name)));
        } else if (strcmp(kind, "cow") == 0) {
            int bid = ai_block_mgr_cow(m, jint(need(op, "rid", name)),
                                       jint(need(op, "index", name)));
            check_int("cow.block", bid, jint(need(op, "block", name)));
        } else if (strcmp(kind, "free") == 0) {
            int len = 0;
            int rc;
            const JsonValue *want = need(op, "released", name);
            ai_block_mgr_table(m, jint(need(op, "rid", name)), buf, &len);
            rc = ai_block_mgr_free_request(m, jint(need(op, "rid", name)));
            check_int("free.rc", rc, 0);
            check_ints("free.released", buf, len, want);
        } else if (strcmp(kind, "swap_out") == 0) {
            int n_rel = 0;
            int rc = ai_block_mgr_swap_out(m, jint(need(op, "rid", name)),
                                           buf, &n_rel);
            check_int("swap_out.rc", rc, 0);
            check_ints("swap_out.released", buf, n_rel,
                       need(op, "released", name));
        } else if (strcmp(kind, "swap_in") == 0) {
            int n_map = 0, n2 = 0;
            int rc = ai_block_mgr_swap_in(m, jint(need(op, "rid", name)),
                                          buf, buf2, &n_map);
            check_int("swap_in.rc", rc, 0);
            check_ints("swap_in.old", buf, n_map, need(op, "old", name));
            check_ints("swap_in.new", buf2, n_map, need(op, "new", name));
            (void)n2;
        } else if (strcmp(kind, "stats") == 0) {
            AiBlockStats st;
            const JsonValue *want = need(op, "stats", name);
            if (ai_block_mgr_stats(m, &st) != 0) die("ai_block_mgr_stats 失败");
            check_int("stats.used", st.used_blocks,
                      jint(need(want, "used_blocks", name)));
            check_int("stats.free", st.free_blocks,
                      jint(need(want, "free_blocks", name)));
            check_int("stats.shared", st.shared_blocks,
                      jint(need(want, "shared_blocks", name)));
            check_int("stats.swapped", st.swapped_blocks,
                      jint(need(want, "swapped_blocks", name)));
            check_int("stats.requests", st.requests,
                      jint(need(want, "requests", name)));
            check_int("stats.cow", st.copy_on_write,
                      jint(need(want, "copy_on_write", name)));
        } else if (strcmp(kind, "tables") == 0) {
            int len = 0;
            if (ai_block_mgr_tables(m, buf, &len) != 0)
                die("ai_block_mgr_tables 失败");
            check_ints("tables.flat", buf, len, need(op, "flat", name));
        } else {
            die("未知 op");
        }
    }
    free(buf);
    free(buf2);
    ai_block_mgr_free(m);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "block_mgr_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_block_mgr] 读不了 %s：%s\n"
                        "请先运行 python tools/block_mgr_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = alen(cases);
    printf("[compare_block_mgr] %s：%d 个用例\n", path, n_cases);
    for (int c = 0; c < n_cases; c++) run_case(json_at(cases, c));
    printf("\n=== compare_block_mgr：%d 项检查，逐位相同 %d，失败 %d ===\n",
           g_checks, g_exact, g_fail);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
