/*
 * compare_activations.c —— 把 C 的 SiLU/Mish/ELU/SELU 与 AI.py 的标量函数逐位对拍
 *
 * 用法：./tests/compare_activations [activations_reference.json]
 *       默认读当前目录的 activations_reference.json
 *
 * 规则（与项目里其它 compare_* 一致）：
 *   - 每个 (用例, 函数) 自带 tol：0.0 表示要求逐位相同；非 0 表示该批输入
 *     命中了 msvcrt vs UCRT 的 libm 1 ULP 差异（JSON 的 func_notes 写明原因）。
 *   - 只有全部数组都在各自容差内才返回 0 并打印 [OK]；任何一处超出就打印
 *     「下标 / 期望 / 实际 / 差值」并返回 1。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../activations.h"
#include "../json.h"

static int g_fail;
static int g_arrays;
static int g_exact;
static double g_worst;

typedef void (*vec_fn)(const double *, double *, int);

static const struct FuncMap {
    const char *name;
    vec_fn fn;
} FUNCS2[] = {
    {"softplus", softplus_vec},
    {"silu", silu_vec},
    {"dsilu", dsilu_vec},
    {"mish", mish_vec},
    {"dmish", dmish_vec},
    {"elu", elu_vec},
    {"delu", delu_vec},
    {"selu", selu_vec},
    {"dselu", dselu_vec},
};

#define N_FUNCS ((int)(sizeof(FUNCS2) / sizeof(FUNCS2[0])))

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_activations] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/activations_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_activations] %s 长度不符：期望 %d，JSON 里是 %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, const double *got, const double *want, int n, double tol)
{
    g_arrays++;
    int neq = 0;
    int first = -1;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            neq++;
            if (first < 0) first = i;
            double d = fabs(got[i] - want[i]);
            if (d > worst) worst = d;
        }
    }
    if (neq == 0) g_exact++;
    if (worst > g_worst) g_worst = worst;
    int ok = (worst <= tol) || (neq == 0);
    printf("  %-22s 逐位 %2d/%-3d 最大差异 %.3e 容差 %.0e %s\n",
           what, n - neq, n, worst, tol, ok ? "[OK]" : "[超出容差]");
    if (!ok) {
        g_fail++;
        int shown = 0;
        for (int i = 0; i < n && shown < 5; i++) {
            if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
                printf("      下标 %d: C=%.17g py=%.17g 差=%.3e\n",
                       i, got[i], want[i], fabs(got[i] - want[i]));
                shown++;
            }
        }
    }
}

static double tol_of(const JsonValue *tols, const char *fname)
{
    if (!tols) return 0.0;
    return json_num(json_get(tols, fname), 0.0);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "activations_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_activations] 读不了 %s：%s\n"
                        "请先运行 python tools/activations_reference.py\n", path, err);
        return 2;
    }

    const JsonValue *cases = need(root, "cases", "root");
    int n_cases = json_array_len(cases);
    printf("[compare_activations] %s：%d 个用例\n", path, n_cases);

    double *z = NULL;
    double *got = NULL;
    double *want = NULL;
    int cap = 0;

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const JsonValue *jz = need(cs, "z", name);
        int n = json_array_len(jz);
        if (n > cap) {
            free(z); free(got); free(want);
            cap = n;
            z = (double *)malloc(sizeof(double) * (size_t)cap);
            got = (double *)malloc(sizeof(double) * (size_t)cap);
            want = (double *)malloc(sizeof(double) * (size_t)cap);
            if (!z || !got || !want) { fprintf(stderr, "out of memory\n"); return 2; }
        }
        arr_vals(jz, z, n, name);
        const char *note = json_str(json_get(cs, "note"), "");
        printf("\n用例 %-16s z=%d  %s\n", name, n, note);

        const JsonValue *tols = json_get(cs, "tol");
        const JsonValue *fn_notes = json_get(cs, "func_notes");
        const JsonValue *expect = need(cs, "expect", name);

        for (int f = 0; f < N_FUNCS; f++) {
            const JsonValue *je = need(expect, FUNCS2[f].name, name);
            arr_vals(je, want, n, FUNCS2[f].name);
            FUNCS2[f].fn(z, got, n);
            report(FUNCS2[f].name, got, want, n, tol_of(tols, FUNCS2[f].name));
            if (fn_notes) {
                const char *fn_note = json_str(json_get(fn_notes, FUNCS2[f].name), NULL);
                if (fn_note) printf("      note: %s\n", fn_note);
            }
        }
    }

    printf("\n[汇总] %d 个数组，其中 %d 个逐位相同；全局最大差异 %.3e\n",
           g_arrays, g_exact, g_worst);
    if (g_fail == 0) {
        printf("[OK] SiLU/Mish/ELU/SELU 与 AI.py 在声明容差内一致\n");
    } else {
        printf("[FAIL] 有 %d 个数组超出容差\n", g_fail);
    }

    free(z);
    free(got);
    free(want);
    json_free(root);
    return g_fail == 0 ? 0 : 1;
}
