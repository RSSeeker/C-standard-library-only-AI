/*
 * compare_ffn.c — compare C FeedForward (8 activations) against AI.py bit-exactly.
 *
 * usage: ./tests/compare_ffn [ffn_reference.json]
 *
 * Rules follow the other compare_* tools: every case carries its own tol
 * (0.0 = bit-identical required); relu/linear are pure arithmetic and must be
 * exact, the rest may differ by 1 ULP in exp/tanh/log1p (msvcrt vs UCRT).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../attention.h"
#include "../json.h"
#include "../tensor.h"

static int g_cases;
static int g_arrays;
static int g_exact;
static int g_fail;
static double g_worst;

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_ffn] %s: missing field \"%s\"; "
                        "re-run python tools/ffn_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *v, double *out, int n, const char *who)
{
    if (json_array_len(v) != n) {
        fprintf(stderr, "[compare_ffn] %s: want %d values, json has %d\n",
                who, n, json_array_len(v));
        exit(2);
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, const double *got, const double *want, int n, double tol)
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
    } else if (worst <= tol) {
        /* within tolerance, still counts as a pass */
    } else {
        g_fail++;
        printf("  [FAIL] %-28s %d/%d differ, first #%d  C=%.17g py=%.17g  max=%.3e > %.3e\n",
               what, neq, n, first, got[first], want[first], worst, tol);
    }
}

static int act_code(const char *name)
{
    static const struct { const char *name; int code; } MAP[] = {
        {"relu", FF_ACT_RELU}, {"gelu", FF_ACT_GELU}, {"sigmoid", FF_ACT_SIGMOID},
        {"linear", FF_ACT_LINEAR}, {"silu", FF_ACT_SILU}, {"mish", FF_ACT_MISH},
        {"elu", FF_ACT_ELU}, {"selu", FF_ACT_SELU},
    };
    for (size_t i = 0; i < sizeof(MAP) / sizeof(MAP[0]); i++)
        if (strcmp(MAP[i].name, name) == 0) return MAP[i].code;
    fprintf(stderr, "[compare_ffn] unknown activation \"%s\"\n", name);
    exit(2);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "ffn_reference.json";
    char err[512];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "[compare_ffn] cannot parse %s: %s\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int ncases = json_array_len(cases);

    printf("[compare_ffn] %s: %d cases\n", path, ncases);
    for (int ci = 0; ci < ncases; ci++) {
        const JsonValue *c = json_at(cases, ci);
        const JsonValue *jname = need(c, "name", "case");
        const char *name = json_str(jname, "case");
        const char *act_name = json_str(need(c, "activation", name), "relu");
        const int dm = (int)json_num(need(c, "d_model", name), 0);
        const int dff = (int)json_num(need(c, "d_ff", name), 0);
        const int T = (int)json_num(need(c, "T", name), 0);
        const double tol = json_num(need(c, "tol", name), 0.0);
        const int act = act_code(act_name);

        const int n_w1 = dff * dm, n_w2 = dm * dff, n_x = T * dm;
        double *W1 = (double *)malloc(sizeof(double) * (size_t)n_w1);
        double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
        double *W2 = (double *)malloc(sizeof(double) * (size_t)n_w2);
        double *b2 = (double *)malloc(sizeof(double) * (size_t)dm);
        double *x = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *dout = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *want_out = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *want_dx = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *got_out = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *got_dx = (double *)malloc(sizeof(double) * (size_t)n_x);
        double *want_gw1 = (double *)malloc(sizeof(double) * (size_t)n_w1);
        double *want_gb1 = (double *)malloc(sizeof(double) * (size_t)dff);
        double *want_gw2 = (double *)malloc(sizeof(double) * (size_t)n_w2);
        double *want_gb2 = (double *)malloc(sizeof(double) * (size_t)dm);
        if (!W1 || !b1 || !W2 || !b2 || !x || !dout || !want_out || !want_dx ||
            !got_out || !got_dx || !want_gw1 || !want_gb1 || !want_gw2 || !want_gb2) {
            fprintf(stderr, "[compare_ffn] out of memory\n");
            return 2;
        }

        arr_vals(need(c, "W1", name), W1, n_w1, name);
        arr_vals(need(c, "b1", name), b1, dff, name);
        arr_vals(need(c, "W2", name), W2, n_w2, name);
        arr_vals(need(c, "b2", name), b2, dm, name);
        arr_vals(need(c, "x", name), x, n_x, name);
        arr_vals(need(c, "dout", name), dout, n_x, name);
        arr_vals(need(c, "out", name), want_out, n_x, name);
        arr_vals(need(c, "dx", name), want_dx, n_x, name);
        arr_vals(need(c, "grad_W1", name), want_gw1, n_w1, name);
        arr_vals(need(c, "grad_b1", name), want_gb1, dff, name);
        arr_vals(need(c, "grad_W2", name), want_gw2, n_w2, name);
        arr_vals(need(c, "grad_b2", name), want_gb2, dm, name);

        int shape_x[2] = {T, dm};
        Tensor *tx = tensor_new(2, shape_x);
        Tensor *tout = tensor_new(2, shape_x);
        Tensor *tdout = tensor_new(2, shape_x);
        Tensor *tdx = tensor_new(2, shape_x);
        memcpy(tx->data, x, sizeof(double) * (size_t)n_x);
        memcpy(tdout->data, dout, sizeof(double) * (size_t)n_x);

        FeedForward *f = ff_create_ex(dm, dff, T, act);
        ff_set_params(f, W1, b1, W2, b2);
        ff_zero_grad(f);
        ff_forward(f, tx, tout);
        memcpy(got_out, tout->data, sizeof(double) * (size_t)n_x);
        ff_backward(f, tdout, tdx);
        memcpy(got_dx, tdx->data, sizeof(double) * (size_t)n_x);

        char what[160];
        snprintf(what, sizeof(what), "%s out", name);
        report(what, got_out, want_out, n_x, tol);
        snprintf(what, sizeof(what), "%s dx", name);
        report(what, got_dx, want_dx, n_x, tol);
        snprintf(what, sizeof(what), "%s grad_W1", name);
        report(what, f->grad_W1, want_gw1, n_w1, tol);
        snprintf(what, sizeof(what), "%s grad_b1", name);
        report(what, f->grad_b1, want_gb1, dff, tol);
        snprintf(what, sizeof(what), "%s grad_W2", name);
        report(what, f->grad_W2, want_gw2, n_w2, tol);
        snprintf(what, sizeof(what), "%s grad_b2", name);
        report(what, f->grad_b2, want_gb2, dm, tol);

        /* single-token path must agree with the batched path on a T == 1 case */
        if (T == 1) {
            double *step_out = (double *)malloc(sizeof(double) * (size_t)dm);
            if (!step_out) return 2;
            ff_forward_step(f, x, step_out);
            snprintf(what, sizeof(what), "%s forward_step", name);
            report(what, step_out, want_out, dm, tol);
            free(step_out);
        }

        /* activation accessors */
        if (ff_activation(f) != act) {
            printf("  [FAIL] %s ff_activation() = %d, want %d\n", name, ff_activation(f), act);
            g_fail++;
        }

        ff_free(f);
        tensor_free(tx);
        tensor_free(tout);
        tensor_free(tdout);
        tensor_free(tdx);
        free(W1); free(b1); free(W2); free(b2); free(x); free(dout);
        free(want_out); free(want_dx); free(got_out); free(got_dx);
        free(want_gw1); free(want_gb1); free(want_gw2); free(want_gb2);
        g_cases++;
    }
    json_free(root);

    printf("[compare_ffn] %d cases, %d arrays (%d bit-identical), worst |diff| = %.3e\n",
           g_cases, g_arrays, g_exact, g_worst);
    if (g_fail == 0) {
        printf("[OK] FeedForward 8 activations match AI.py\n");
        return 0;
    }
    printf("[FAIL] %d mismatch(es)\n", g_fail);
    return 1;
}
