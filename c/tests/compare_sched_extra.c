/*
 * compare_sched_extra.c —— WarmupLR / ReduceLROnPlateau 的跨语言对拍
 *
 * 用法（在 c/ 目录执行，先跑 python tools/sched_extra_reference.py）：
 *   ./tests/compare_sched_extra.exe sched_extra_reference.json
 *
 * 两个调度器都只操作 optimizer.lr、只用四则运算与比较，JSON 里所有用例 tol=0.0：
 * 每一步的 lr、base_scheduler 收到的 epoch 序列、以及 plateau 的
 * best/num_bad/reductions 都要求逐位（或逐值）相同。
 */
#include "../json.h"
#include "../sched.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static int g_checks;
static int g_exact;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        printf("[错误] %s 缺少字段 '%s'（请重跑 tools/sched_extra_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static double arr_at(const JsonValue *a, int i)
{
    return json_num(json_at(a, i), 0.0);
}

static void report_double(const char *what, double got, double want, double tol)
{
    const int exact = (memcmp(&got, &want, sizeof(double)) == 0);
    const double d = fabs(got - want);
    const int ok = exact || d <= tol;
    printf("    %-42s 期望 %-22.17g 实际 %-22.17g %s%s\n",
           what, want, got, ok ? "[OK]" : "[超差]", exact ? " (逐位相同)" : "");
    if (!ok) g_fail++;
    g_checks++;
    if (exact) g_exact++;
}

static void report_int(const char *what, int got, int want)
{
    const int ok = (got == want);
    printf("    %-42s 期望 %-8d 实际 %-8d %s\n", what, want, got, ok ? "[OK]" : "[超差]");
    if (!ok) g_fail++;
    g_checks++;
    if (ok) g_exact++;
}

/* C 侧的 base_scheduler 对照实现：lr = base_lr，再连续乘 epoch 次 0.5 */
typedef struct {
    double *lr;
    double base_lr;
    int epochs[64];
    int n;
} BaseCtx;

static void base_halve(void *ctx, int epoch)
{
    BaseCtx *b = (BaseCtx *)ctx;
    if (b->n < (int)(sizeof b->epochs / sizeof b->epochs[0])) b->epochs[b->n] = epoch;
    b->n++;
    double v = b->base_lr;
    for (int i = 0; i < epoch; i++) v *= 0.5;
    *b->lr = v;
}

static void run_warmup(const JsonValue *cases)
{
    const int nc = json_array_len(cases);
    for (int ci = 0; ci < nc; ci++) {
        const JsonValue *c = json_at(cases, ci);
        char who[96];
        snprintf(who, sizeof who, "warmup_cases[%d]", ci);
        const char *name = json_str(need(c, "name", who), "?");
        const char *note = json_str(json_get(c, "note"), NULL);
        const double tol = json_num(json_get(c, "tol"), 0.0);
        const double initial_lr = json_num(need(c, "initial_lr", who), 0.0);
        const int warmup_epochs = json_int(need(c, "warmup_epochs", who), 0);
        const double start_factor = json_num(need(c, "start_factor", who), 0.0);
        const JsonValue *epochs = need(c, "epochs", who);
        const JsonValue *expect = need(c, "expect_lr", who);
        const JsonValue *jbase = json_get(c, "base_lr");
        const JsonValue *expect_base = json_get(c, "expect_base_epochs");

        printf("\n[warmup %d] %s  warmup=%d start=%g initial_lr=%g\n",
               ci, name, warmup_epochs, start_factor, initial_lr);
        if (note) printf("  备注: %s\n", note);

        double lr = initial_lr;
        const char *cerr = NULL;
        WarmupLR *w = warmuplr_create(&lr, warmup_epochs, start_factor, &cerr);
        if (!w) {
            printf("  [错误] warmuplr_create 失败: %s\n", cerr ? cerr : "?");
            g_fail++;
            continue;
        }
        BaseCtx base;
        memset(&base, 0, sizeof base);
        base.lr = &lr;
        if (jbase && json_is_num(jbase)) {
            base.base_lr = json_num(jbase, 0.0);
            warmuplr_set_base(w, base_halve, &base);
        }

        const int n = json_array_len(epochs);
        for (int i = 0; i < n; i++) {
            const int ep = (int)arr_at(epochs, i);
            warmuplr_step(w, ep);
            char what[96];
            snprintf(what, sizeof what, "epoch %d -> lr", ep);
            report_double(what, lr, arr_at(expect, i), tol);
        }

        if (expect_base) {
            const int nb = json_array_len(expect_base);
            report_int("base_scheduler 调用次数", base.n, nb);
            const int lim = base.n < nb ? base.n : nb;
            for (int i = 0; i < lim; i++) {
                char what[96];
                snprintf(what, sizeof what, "base_scheduler 第 %d 次 epoch", i + 1);
                report_int(what, base.epochs[i], (int)arr_at(expect_base, i));
            }
        }
        warmuplr_free(w);
    }
}

static void run_plateau(const JsonValue *cases)
{
    const int nc = json_array_len(cases);
    for (int ci = 0; ci < nc; ci++) {
        const JsonValue *c = json_at(cases, ci);
        char who[96];
        snprintf(who, sizeof who, "plateau_cases[%d]", ci);
        const char *name = json_str(need(c, "name", who), "?");
        const char *note = json_str(json_get(c, "note"), NULL);
        const double tol = json_num(json_get(c, "tol"), 0.0);
        const double initial_lr = json_num(need(c, "initial_lr", who), 0.0);
        const double factor = json_num(need(c, "factor", who), 0.0);
        const int patience = json_int(need(c, "patience", who), 0);
        const double min_lr = json_num(need(c, "min_lr", who), 0.0);
        const char *mode_s = json_str(need(c, "mode", who), "min");
        const double threshold = json_num(need(c, "threshold", who), 0.0);
        const int max_red = json_int(need(c, "max_reductions", who), -1);
        const JsonValue *metrics = need(c, "metrics", who);
        const JsonValue *expect = need(c, "expect_lr", who);

        printf("\n[plateau %d] %s  factor=%g patience=%d min_lr=%g mode=%s threshold=%g max_red=%d\n",
               ci, name, factor, patience, min_lr, mode_s, threshold, max_red);
        if (note) printf("  备注: %s\n", note);

        PlateauMode mode;
        if (strcmp(mode_s, "min") == 0) mode = PLATEAU_MIN;
        else if (strcmp(mode_s, "max") == 0) mode = PLATEAU_MAX;
        else { printf("  [错误] 未知 mode '%s'\n", mode_s); g_fail++; continue; }

        double lr = initial_lr;
        const char *cerr = NULL;
        ReduceLROnPlateau *p = plateau_create(&lr, factor, patience, min_lr, mode,
                                              threshold, max_red, &cerr);
        if (!p) {
            printf("  [错误] plateau_create 失败: %s\n", cerr ? cerr : "?");
            g_fail++;
            continue;
        }

        const int n = json_array_len(metrics);
        const JsonValue *nan_at = json_get(c, "nan_at");
        for (int i = 0; i < n; i++) {
            int is_nan = 0;
            if (nan_at) {
                for (int k = 0; k < json_array_len(nan_at); k++)
                    if ((int)arr_at(nan_at, k) == i) is_nan = 1;
            }
            /* JSON 不能写裸 NaN：参考数据用 0.0 占位，下标放在 nan_at 里 */
            const double m = is_nan ? (double)NAN : arr_at(metrics, i);
            const double got = plateau_step(p, m);
            char what[96];
            snprintf(what, sizeof what, "metric %d (%s) -> lr", i,
                     is_nan ? "nan" : "num");
            report_double(what, got, arr_at(expect, i), tol);
        }
        report_int("has_best", plateau_has_best(p),
                   json_int(need(c, "expect_has_best", who), 0));
        report_int("num_bad", plateau_num_bad(p),
                   json_int(need(c, "expect_num_bad", who), 0));
        report_int("reductions", plateau_reductions(p),
                   json_int(need(c, "expect_reductions", who), 0));
        if (json_int(need(c, "expect_has_best", who), 0)) {
            report_double("best", plateau_best(p),
                          json_num(need(c, "expect_best", who), 0.0), tol);
        }
        plateau_free(p);
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "sched_extra_reference.json";

    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        printf("[错误] 读 %s 失败: %s\n", path, err);
        return 2;
    }
    const JsonValue *wc = need(root, "warmup_cases", "顶层");
    const JsonValue *pc = need(root, "plateau_cases", "顶层");
    printf("WarmupLR / ReduceLROnPlateau 对拍：%s（%d + %d 个用例）\n",
           path, json_array_len(wc), json_array_len(pc));

    run_warmup(wc);
    run_plateau(pc);

    printf("\n[汇总] %d 处检查中 %d 处逐位相同；%s\n",
           g_checks, g_exact, g_fail == 0 ? "全部通过" : "存在超差");
    if (g_fail == 0) printf("[OK] WarmupLR / ReduceLROnPlateau 与 AI.py 完全一致\n");
    else printf("[失败] 有 %d 处不一致\n", g_fail);

    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
