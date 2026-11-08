/*
 * test_ffn.c — self-tests for the FeedForward activation support.
 *
 *  [1] relu path: hand-computed forward + mask backward (default path unchanged)
 *  [2] eight activations: scalar identity checks (gelu(0)=0, gelu'(0)=0.5,
 *      sigmoid(0)=0.5, linear identity, ...)
 *  [3] each non-relu activation: central-difference check of grad_W1/b1/W2/b2/dx
 *  [4] ff_set_activation switches the path; ff_activation reads it back
 *  [5] ff_forward_step (T == 1) agrees bit-exactly with ff_forward
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../activations.h"
#include "../attention.h"
#include "../tensor.h"

static int g_fail;

static void check(const char *name, int ok, double err, double tol)
{
    if (ok && err <= tol) {
        printf("  [ok]   %-52s err=%.3e\n", name, err);
    } else {
        printf("  [FAIL] %-52s err=%.3e tol=%.3e\n", name, err, tol);
        g_fail++;
    }
}

static unsigned long long g_rng = 0x9E3779B97F4A7C15ULL;

static double rnd_uniform(void)
{
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (double)((g_rng * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
}

static double rnd_sym(double amp) { return (rnd_uniform() * 2.0 - 1.0) * amp; }

/* loss = sum(coef[t][j] * out[t][j]) with Neumaier accumulation */
static double ff_loss(FeedForward *f, const Tensor *x, const double *coef,
                      Tensor *out, double *prod)
{
    ff_forward(f, x, out);
    const int n = out->size;
    for (int i = 0; i < n; i++) prod[i] = coef[i] * out->data[i];
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double t = s + prod[i];
        if (fabs(s) >= fabs(prod[i])) c += (s - t) + prod[i];
        else                          c += (prod[i] - t) + s;
        s = t;
    }
    return s + c;
}

static FeedForward *make_ff(int act, int dm, int dff, int T, double amp)
{
    FeedForward *f = ff_create_ex(dm, dff, T, act);
    const int nw1 = dff * dm, nw2 = dm * dff;
    double *W1 = (double *)malloc(sizeof(double) * (size_t)nw1);
    double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
    double *W2 = (double *)malloc(sizeof(double) * (size_t)nw2);
    double *b2 = (double *)malloc(sizeof(double) * (size_t)dm);
    for (int i = 0; i < nw1; i++) W1[i] = rnd_sym(amp);
    for (int i = 0; i < dff; i++) b1[i] = rnd_sym(amp);
    for (int i = 0; i < nw2; i++) W2[i] = rnd_sym(amp);
    for (int i = 0; i < dm; i++) b2[i] = rnd_sym(amp);
    ff_set_params(f, W1, b1, W2, b2);
    free(W1); free(b1); free(W2); free(b2);
    return f;
}

/* central-difference check for one activation */
static void fd_check(const char *label, int act)
{
    const int dm = 3, dff = 4, T = 2;
    const double eps = 1e-6, tol = 1e-5;
    FeedForward *f = make_ff(act, dm, dff, T, 0.7);
    const int nw1 = dff * dm, nw2 = dm * dff;
    const int nx = T * dm;

    Tensor *tx = tensor_new(2, (int[]){T, dm});
    Tensor *tout = tensor_new(2, (int[]){T, dm});
    Tensor *tdout = tensor_new(2, (int[]){T, dm});
    Tensor *tdx = tensor_new(2, (int[]){T, dm});
    double *coef = (double *)malloc(sizeof(double) * (size_t)nx);
    double *prod = (double *)malloc(sizeof(double) * (size_t)nx);
    double *W1 = (double *)malloc(sizeof(double) * (size_t)nw1);
    double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
    double *W2 = (double *)malloc(sizeof(double) * (size_t)nw2);
    double *b2 = (double *)malloc(sizeof(double) * (size_t)dm);
    double *gW1 = (double *)malloc(sizeof(double) * (size_t)nw1);
    double *gb1 = (double *)malloc(sizeof(double) * (size_t)dff);
    double *gW2 = (double *)malloc(sizeof(double) * (size_t)nw2);
    double *gb2 = (double *)malloc(sizeof(double) * (size_t)dm);

    for (int i = 0; i < nx; i++) tx->data[i] = rnd_sym(1.2);
    for (int i = 0; i < nx; i++) coef[i] = rnd_sym(1.0);
    memcpy(W1, f->W1, sizeof(double) * (size_t)nw1);
    memcpy(b1, f->b1, sizeof(double) * (size_t)dff);
    memcpy(W2, f->W2, sizeof(double) * (size_t)nw2);
    memcpy(b2, f->b2, sizeof(double) * (size_t)dm);

    /* analytic */
    ff_zero_grad(f);
    ff_loss(f, tx, coef, tout, prod);
    memcpy(tdout->data, coef, sizeof(double) * (size_t)nx);
    ff_backward(f, tdout, tdx);
    memcpy(gW1, f->grad_W1, sizeof(double) * (size_t)nw1);
    memcpy(gb1, f->grad_b1, sizeof(double) * (size_t)dff);
    memcpy(gW2, f->grad_W2, sizeof(double) * (size_t)nw2);
    memcpy(gb2, f->grad_b2, sizeof(double) * (size_t)dm);
    double *adx = (double *)malloc(sizeof(double) * (size_t)nx);
    memcpy(adx, tdx->data, sizeof(double) * (size_t)nx);

    char what[128];
    double worst = 0.0, worst_dx = 0.0;
    for (int i = 0; i < nw1; i++) {
        double old = f->W1[i];
        f->W1[i] = old + eps; double lp = ff_loss(f, tx, coef, tout, prod);
        f->W1[i] = old - eps; double lm = ff_loss(f, tx, coef, tout, prod);
        f->W1[i] = old;
        double fd = (lp - lm) / (2.0 * eps);
        worst = fmax(worst, fabs(fd - gW1[i]));
    }
    snprintf(what, sizeof(what), "%s grad_W1", label);
    check(what, 1, worst, tol);

    worst = 0.0;
    for (int i = 0; i < dff; i++) {
        double old = f->b1[i];
        f->b1[i] = old + eps; double lp = ff_loss(f, tx, coef, tout, prod);
        f->b1[i] = old - eps; double lm = ff_loss(f, tx, coef, tout, prod);
        f->b1[i] = old;
        worst = fmax(worst, fabs((lp - lm) / (2.0 * eps) - gb1[i]));
    }
    snprintf(what, sizeof(what), "%s grad_b1", label);
    check(what, 1, worst, tol);

    worst = 0.0;
    for (int i = 0; i < nw2; i++) {
        double old = f->W2[i];
        f->W2[i] = old + eps; double lp = ff_loss(f, tx, coef, tout, prod);
        f->W2[i] = old - eps; double lm = ff_loss(f, tx, coef, tout, prod);
        f->W2[i] = old;
        worst = fmax(worst, fabs((lp - lm) / (2.0 * eps) - gW2[i]));
    }
    snprintf(what, sizeof(what), "%s grad_W2", label);
    check(what, 1, worst, tol);

    worst = 0.0;
    for (int i = 0; i < dm; i++) {
        double old = f->b2[i];
        f->b2[i] = old + eps; double lp = ff_loss(f, tx, coef, tout, prod);
        f->b2[i] = old - eps; double lm = ff_loss(f, tx, coef, tout, prod);
        f->b2[i] = old;
        worst = fmax(worst, fabs((lp - lm) / (2.0 * eps) - gb2[i]));
    }
    snprintf(what, sizeof(what), "%s grad_b2", label);
    check(what, 1, worst, tol);

    for (int i = 0; i < nx; i++) {
        double old = tx->data[i];
        tx->data[i] = old + eps; double lp = ff_loss(f, tx, coef, tout, prod);
        tx->data[i] = old - eps; double lm = ff_loss(f, tx, coef, tout, prod);
        tx->data[i] = old;
        worst_dx = fmax(worst_dx, fabs((lp - lm) / (2.0 * eps) - adx[i]));
    }
    snprintf(what, sizeof(what), "%s dx", label);
    check(what, 1, worst_dx, tol);

    ff_free(f);
    tensor_free(tx); tensor_free(tout); tensor_free(tdout); tensor_free(tdx);
    free(coef); free(prod); free(adx);
    free(W1); free(b1); free(W2); free(b2);
    free(gW1); free(gb1); free(gW2); free(gb2);
}

int main(void)
{
    printf("[test_ffn] FeedForward activation\n");

    /* [1] relu default path, hand-computed: d=1, dff=1, W1=1, b1=-0.5,
     * W2=2, b2=0.3, x=1 -> h = relu(0.5) = 0.5 -> out = 1.3
     * dout = 1 -> dh = 2, grad_W1 = dh*x = 2, grad_b1 = 2, dx = 2 */
    {
        FeedForward *f = ff_create(1, 1, 1);
        double W1[1] = {1.0}, b1[1] = {-0.5}, W2[1] = {2.0}, b2[1] = {0.3};
        ff_set_params(f, W1, b1, W2, b2);
        Tensor *tx = tensor_new(2, (int[]){1, 1});
        Tensor *tout = tensor_new(2, (int[]){1, 1});
        Tensor *tdout = tensor_new(2, (int[]){1, 1});
        Tensor *tdx = tensor_new(2, (int[]){1, 1});
        tx->data[0] = 1.0;
        ff_forward(f, tx, tout);
        check("relu forward = 1.3", 1, fabs(tout->data[0] - 1.3), 0.0);
        tdout->data[0] = 1.0;
        ff_backward(f, tdout, tdx);
        check("relu grad_W1 = 2", 1, fabs(f->grad_W1[0] - 2.0), 0.0);
        check("relu grad_b1 = 2", 1, fabs(f->grad_b1[0] - 2.0), 0.0);
        check("relu dx = 2", 1, fabs(tdx->data[0] - 2.0), 0.0);
        check("ff_activation default = relu", ff_activation(f) == FF_ACT_RELU, 0.0, 0.0);
        ff_free(f);
        tensor_free(tx); tensor_free(tout); tensor_free(tdout); tensor_free(tdx);
    }

    /* [2] scalar identities (mirror AI.py _act_scalar / _dact_scalar) */
    check("gelu(0) = 0", 1, fabs(gelu_scalar(0.0)), 0.0);
    check("dgelu(0) = 0.5", 1, fabs(dgelu_scalar(0.0) - 0.5), 0.0);
    check("sigmoid(0) = 0.5", 1, fabs(sigmoid_scalar(0.0) - 0.5), 0.0);
    check("dsigmoid(0) = 0.25", 1, fabs(dsigmoid_scalar(0.0) - 0.25), 0.0);
    check("sigmoid(40) ~ 1", 1, fabs(sigmoid_scalar(40.0) - 1.0), 1e-17);
    check("sigmoid(-40) = e/(1+e)", 1,
          fabs(sigmoid_scalar(-40.0) - exp(-40.0) / (1.0 + exp(-40.0))), 0.0);
    check("sigmoid(1) = 1/(1+e^-1)", 1,
          fabs(sigmoid_scalar(1.0) - 1.0 / (1.0 + exp(-1.0))), 0.0);
    check("sigmoid(-1) = e/(1+e)", 1,
          fabs(sigmoid_scalar(-1.0) - exp(-1.0) / (1.0 + exp(-1.0))), 0.0);

    /* [3] finite differences for every non-relu activation (relu included too) */
    static const struct { const char *name; int act; } ACTS[] = {
        {"relu", FF_ACT_RELU}, {"gelu", FF_ACT_GELU}, {"sigmoid", FF_ACT_SIGMOID},
        {"linear", FF_ACT_LINEAR}, {"silu", FF_ACT_SILU}, {"mish", FF_ACT_MISH},
        {"elu", FF_ACT_ELU}, {"selu", FF_ACT_SELU},
    };
    for (size_t i = 0; i < sizeof(ACTS) / sizeof(ACTS[0]); i++)
        fd_check(ACTS[i].name, ACTS[i].act);

    /* [4] switch activation at runtime */
    {
        FeedForward *f = make_ff(FF_ACT_RELU, 2, 3, 1, 0.8);
        Tensor *tx = tensor_new(2, (int[]){1, 2});
        Tensor *t1 = tensor_new(2, (int[]){1, 2});
        Tensor *t2 = tensor_new(2, (int[]){1, 2});
        tx->data[0] = -1.0; tx->data[1] = 1.0;
        ff_forward(f, tx, t1);
        double relu_out = t1->data[0];
        ff_set_activation(f, FF_ACT_GELU);
        check("ff_activation after set = gelu", ff_activation(f) == FF_ACT_GELU, 0.0, 0.0);
        ff_forward(f, tx, t2);
        check("gelu output differs from relu", fabs(t2->data[0] - relu_out) > 0.0, 0.0, 0.0);
        ff_free(f);
        tensor_free(tx); tensor_free(t1); tensor_free(t2);
    }

    /* [5] forward_step vs forward on T == 1, every activation, bit-exact */
    for (size_t i = 0; i < sizeof(ACTS) / sizeof(ACTS[0]); i++) {
        FeedForward *f = make_ff(ACTS[i].act, 3, 5, 1, 0.9);
        Tensor *tx = tensor_new(2, (int[]){1, 3});
        Tensor *tout = tensor_new(2, (int[]){1, 3});
        for (int j = 0; j < 3; j++) tx->data[j] = rnd_sym(1.4);
        ff_forward(f, tx, tout);
        double step[3];
        ff_forward_step(f, tx->data, step);
        int same = 1;
        for (int j = 0; j < 3; j++)
            if (memcmp(&step[j], &tout->data[j], sizeof(double)) != 0) same = 0;
        char what[96];
        snprintf(what, sizeof(what), "%s forward_step == forward (T=1)", ACTS[i].name);
        check(what, same, 0.0, 0.0);
        ff_free(f);
        tensor_free(tx); tensor_free(tout);
    }

    printf("[test_ffn] %s (failures: %d)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
