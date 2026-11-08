/*
 * compare_adapt.c — 把 C 的 LoRA 与 AI.py 的 AI.LoRA 逐位对比
 *
 * usage: ./tests/compare_adapt [adapt_reference.json]
 *
 * 容差 0：LoRA 只有乘加，没有任何超越函数，必须逐位相同。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../adapt.h"
#include "../json.h"

static int g_cases;
static int g_arrays;
static int g_exact;
static int g_fail;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_adapt] %s: missing field \"%s\"; "
                        "re-run python tools/adapt_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_adapt] %s: want %d values, json has %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    g_arrays++;
    int neq = 0, first = -1;
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            neq++;
            if (first < 0) first = i;
            double d = fabs(got[i] - want[i]);
            if (d > worst) worst = d;
        }
    }
    if (worst > g_worst) g_worst = worst;
    if (neq == 0) {
        g_exact++;
        return;
    }
    g_fail++;
    printf("  [FAIL] %-24s %d/%d differ, first #%d  C=%.17g py=%.17g  max=%.3e\n",
           what, neq, n, first, got[first], want[first], worst);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "adapt_reference.json";
    char err[512];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "[compare_adapt] cannot parse %s: %s\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int ncases = json_array_len(cases);

    printf("[compare_adapt] %s: %d cases\n", path, ncases);
    for (int ci = 0; ci < ncases; ci++) {
        const JsonValue *c = json_at(cases, ci);
        const char *name = json_str(need(c, "name", "case"), "case");
        const int fi = (int)json_num(need(c, "fan_in", name), 0);
        const int fo = (int)json_num(need(c, "fan_out", name), 0);
        const int rank = (int)json_num(need(c, "rank", name), 0);
        const double alpha = json_num(need(c, "alpha", name), 0.0);

        const int n_w = fi * fo, n_a = rank * fi, n_b = fo * rank;
        double *W = (double *)malloc(sizeof(double) * (size_t)n_w);
        double *b = (double *)malloc(sizeof(double) * (size_t)fo);
        double *A = (double *)malloc(sizeof(double) * (size_t)n_a);
        double *B = (double *)malloc(sizeof(double) * (size_t)n_b);
        double *x = (double *)malloc(sizeof(double) * (size_t)fi);
        double *dout = (double *)malloc(sizeof(double) * (size_t)fo);
        double *want = (double *)malloc(sizeof(double) * (size_t)(n_w > fo ? n_w : fo));
        double *got = (double *)malloc(sizeof(double) * (size_t)(n_w > fo ? n_w : fo));
        double *got_w = (double *)malloc(sizeof(double) * (size_t)n_w);
        double *want_w = (double *)malloc(sizeof(double) * (size_t)n_w);
        double *want_x = (double *)malloc(sizeof(double) * (size_t)fi);
        if (!W || !b || !A || !B || !x || !dout || !want || !got || !got_w || !want_w || !want_x) {
            fprintf(stderr, "[compare_adapt] out of memory\n");
            return 2;
        }

        arr_vals(need(c, "w", name), W, n_w, name);
        arr_vals(need(c, "b", name), b, fo, name);
        arr_vals(need(c, "a", name), A, n_a, name);
        arr_vals(need(c, "bmat", name), B, n_b, name);
        arr_vals(need(c, "x", name), x, fi, name);
        arr_vals(need(c, "dout", name), dout, fo, name);

        LoRA *l = lora_create(fi, fo, rank, alpha);
        lora_set_params(l, A, B);
        if (lora_trainable_params(l) != (int)json_num(need(c, "trainable", name), -1)) {
            printf("  [FAIL] %-24s trainable=%d\n", name, lora_trainable_params(l));
            g_fail++;
        }

        double *y = (double *)malloc(sizeof(double) * (size_t)fo);
        double *dx = (double *)malloc(sizeof(double) * (size_t)fi);
        lora_forward(l, W, b, x, y);
        arr_vals(need(c, "y", name), want, fo, name);
        report("forward", y, want, fo);

        lora_zero_grad(l);
        lora_backward(l, W, dout, dx);
        arr_vals(need(c, "dx", name), want_x, fi, name);
        report("dx", dx, want_x, fi);
        arr_vals(need(c, "grad_a", name), want, n_a, name);
        report("grad_a", l->gA, want, n_a);
        arr_vals(need(c, "grad_b", name), want, n_b, name);
        report("grad_b", l->gB, want, n_b);

        /* merge：W 被改写，之后 forward/backward 只走原层 */
        LoRA *m = lora_create(fi, fo, rank, alpha);
        lora_set_params(m, A, B);
        double *Wm = (double *)malloc(sizeof(double) * (size_t)n_w);
        memcpy(Wm, W, sizeof(double) * (size_t)n_w);
        lora_merge(m, Wm);
        arr_vals(need(c, "merged_w", name), want_w, n_w, name);
        report("merged_w", Wm, want_w, n_w);

        lora_forward(m, Wm, b, x, y);
        arr_vals(need(c, "y_merged", name), want, fo, name);
        report("y_merged", y, want, fo);

        lora_zero_grad(m);
        lora_backward(m, Wm, dout, dx);
        arr_vals(need(c, "dx_merged", name), want_x, fi, name);
        report("dx_merged", dx, want_x, fi);

        printf("  %-20s d=%dx%d rank=%d alpha=%.1f\n", name, fo, fi, rank, alpha);

        lora_free(l);
        lora_free(m);
        free(W);
        free(b);
        free(A);
        free(B);
        free(x);
        free(dout);
        free(want);
        free(got);
        free(got_w);
        free(want_w);
        free(want_x);
        free(y);
        free(dx);
        free(Wm);
        g_cases++;
    }

    printf("\n[compare_adapt] %d cases, %d arrays, %d bit-identical, %d failed, worst=%.3e\n",
           g_cases, g_arrays, g_exact, g_fail, g_worst);
    json_free(root);
    if (g_fail) {
        printf("[compare_adapt] FAILED\n");
        return 1;
    }
    printf("[compare_adapt] [OK] LoRA matches AI.py bit-exactly\n");
    return 0;
}
