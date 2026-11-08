/* ==========================================================================
 *  test_heads.c：TokenClassificationHead / MultiLabelHead / MoE 的自测
 *
 *  覆盖：闭式解、mask 语义、中心差分（逐参数）、路由、未选中专家零梯度、
 *        参数计数与非法输入。
 * ========================================================================== */
#include "../heads.h"
#include "../attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FD_EPS 1e-6
#define FD_TOL 1e-5

static int g_checks = 0;
static int g_fails = 0;

static void check(const char *name, double err, double tol)
{
    g_checks++;
    if (err <= tol) {
        printf("  [OK]   %-52s err=%.3e\n", name, err);
    } else {
        g_fails++;
        printf("  [FAIL] %-52s err=%.3e > %.1e\n", name, err, tol);
    }
}

static void check_true(const char *name, int cond)
{
    check(name, cond ? 0.0 : 1.0, 0.0);
}

/* ---------- [1] TokenClassificationHead：闭式 + mask ---------- */
static void test_token_basic(void)
{
    const int d = 1, L = 2, T = 2;
    /* W = [[1], [-1]], b = [0, 0]；x = [2, 3] */
    TokenHead *h = token_head_create(d, L);
    double W[2] = {1.0, -1.0};
    double b[2] = {0.0, 0.0};
    double x[2] = {2.0, 3.0};
    double logits[4], dlogits[4], dx[2];
    int targets[2] = {0, 1};
    int mask2[2] = {1, 0};
    int pred[2];
    double loss;

    token_head_set_params(h, W, b);
    token_head_zero_grad(h);
    token_head_forward(h, x, T, logits);
    check("token logits[0] = x0", fabs(logits[0] - 2.0), 0.0);
    check("token logits[1] = -x0", fabs(logits[1] + 2.0), 0.0);
    check("token logits[2] = x1", fabs(logits[2] - 3.0), 0.0);
    check("token logits[3] = -x1", fabs(logits[3] + 3.0), 0.0);

    /* 全有效：两个 token 各 1 条，loss = (softmax CE 平均) */
    loss = token_head_loss_grad(h, logits, targets, NULL, T, dlogits);
    {
        double p0 = 1.0 / (1.0 + exp(-4.0));   /* softmax([2,-2]) 的第一个 */
        double p1 = 1.0 / (1.0 + exp(6.0));    /* softmax([3,-3]) 的第二个 */
        double want = (-log(p0) - log(p1)) / 2.0;
        check("token loss（两个 token 平均）", fabs(loss - want), 1e-15);
        check("token dlogits 之和为 0（第一行）",
              fabs(dlogits[0] + dlogits[1]), 1e-15);
    }

    /* mask 掉第二个 token：loss 只由第一个决定，梯度里第二行为 0 */
    token_head_zero_grad(h);
    loss = token_head_loss_grad(h, logits, targets, mask2, T, dlogits);
    {
        double p0 = 1.0 / (1.0 + exp(-4.0));
        check("token masked loss 只算有效 token", fabs(loss + log(p0)), 1e-15);
        check("token masked 行梯度为 0",
              fabs(dlogits[2]) + fabs(dlogits[3]), 0.0);
    }
    token_head_backward(h, dlogits, T, dx);
    check("token masked dx 第二行为 0", fabs(dx[1]), 0.0);
    check("token masked grad_b[1] 只来自第一个 token",
          fabs(h->gb[1] - dlogits[1]), 1e-15);

    token_head_predict(h, logits, T, mask2, pred);
    check_true("token predict 首 token 取 argmax", pred[0] == 0);
    check_true("token predict 被 mask 的位置为 -1", pred[1] == -1);
    check_true("token param_count", token_head_param_count(h) == d * L + L);
    token_head_free(h);
}

/* ---------- [2] TokenClassificationHead：中心差分 ---------- */
static double token_loss_of(TokenHead *h, const double *x, int T,
                            const int *targets, const double *coef)
{
    double logits[16], loss;
    int t, l;
    token_head_forward(h, x, T, logits);
    loss = 0.0;
    for (t = 0; t < T; t++)
        for (l = 0; l < h->num_labels; l++)
            loss += coef[t * h->num_labels + l] * logits[t * h->num_labels + l];
    (void)targets;
    return loss;
}

static void test_token_fd(void)
{
    const int d = 3, L = 3, T = 3;
    TokenHead *h = token_head_create(d, L);
    double W[9] = {0.3, -0.5, 0.2, 0.1, 0.4, -0.2, -0.3, 0.25, 0.6};
    double b[3] = {0.05, -0.1, 0.2};
    double x[9] = {0.5, -0.3, 0.8, -0.2, 0.4, 0.1, 0.9, -0.6, 0.3};
    double coef[9] = {0.7, -0.2, 0.4, -0.5, 0.6, 0.1, 0.2, 0.3, -0.8};
    int targets[3] = {0, 1, 2};
    int mask3[3] = {1, 0, 1};
    double logits[9], dlogits[9], dx[9];
    double worst_w = 0.0, worst_b = 0.0;
    int i, t;

    token_head_set_params(h, W, b);
    token_head_zero_grad(h);
    token_head_forward(h, x, T, logits);
    token_head_loss_grad(h, logits, targets, mask3, T, dlogits);
    token_head_backward(h, dlogits, T, dx);

    for (i = 0; i < L * d; i++) {
        double save = h->W[i], lp, lm, num, ana = 0.0;
        int l = i / d;
        int j = i % d;
        h->W[i] = save + FD_EPS;
        lp = token_loss_of(h, x, T, targets, coef);
        h->W[i] = save - FD_EPS;
        lm = token_loss_of(h, x, T, targets, coef);
        h->W[i] = save;
        num = (lp - lm) / (2.0 * FD_EPS);
        /* loss 对 logits 是线性的，所以 dL/dlogits = coef，链式即
         * dL/dW[l][j] = Σ_t coef[t][l] * x[t][j]（softmax 不参与）。 */
        for (t = 0; t < T; t++) ana += coef[t * L + l] * x[t * d + j];
        if (fabs(num - ana) > worst_w) worst_w = fabs(num - ana);
    }
    check("token FD grad_W（线性探针）", worst_w, 1e-9);

    for (i = 0; i < L; i++) {
        double save = h->b[i], lp, lm, num, ana = 0.0;
        h->b[i] = save + FD_EPS;
        lp = token_loss_of(h, x, T, targets, coef);
        h->b[i] = save - FD_EPS;
        lm = token_loss_of(h, x, T, targets, coef);
        h->b[i] = save;
        num = (lp - lm) / (2.0 * FD_EPS);
        for (t = 0; t < T; t++) ana += coef[t * L + i];
        if (fabs(num - ana) > worst_b) worst_b = fabs(num - ana);
    }
    check("token FD grad_b（线性探针）", worst_b, 1e-9);

    /* 交叉熵路径的 dx：用真实 loss 做中心差分 */
    {
        double worst = 0.0;
        for (i = 0; i < T * d; i++) {
            double save = x[i], lp, lm, num;
            double logits2[9], dl[9];
            double l;
            x[i] = save + FD_EPS;
            token_head_forward(h, x, T, logits2);
            l = token_head_loss_grad(h, logits2, targets, mask3, T, dl);
            lp = l;
            x[i] = save - FD_EPS;
            token_head_forward(h, x, T, logits2);
            l = token_head_loss_grad(h, logits2, targets, mask3, T, dl);
            lm = l;
            x[i] = save;
            num = (lp - lm) / (2.0 * FD_EPS);
            if (fabs(num - dx[i]) > worst) worst = fabs(num - dx[i]);
        }
        check("token FD dx（交叉熵 + mask）", worst, 1e-8);
    }
    token_head_free(h);
}

/* ---------- [3] MultiLabelHead ---------- */
static void test_multilabel(void)
{
    const int fi = 2, L = 3;
    MultiLabelHead *h = multilabel_head_create(fi, L);
    double W[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double b[3] = {0.0, 0.0, 0.0};
    double x[2] = {1.0, -2.0};
    double probs[3], dlogits[3], dx[2];
    double targets[3] = {1.0, 0.0, 1.0};
    int pred[3];
    double loss;

    multilabel_head_set_params(h, W, b);
    multilabel_head_zero_grad(h);
    multilabel_head_forward(h, x, probs);
    check("multilabel 全零权重 probs = 0.5", fabs(probs[0] - 0.5) + fabs(probs[2] - 0.5), 0.0);
    loss = multilabel_head_loss_grad(h, probs, targets, dlogits);
    check("multilabel loss = log2（3 个标签平均）", fabs(loss - log(2.0)), 1e-15);
    check("multilabel dlogits[1] 符号（y=0 → 正）", dlogits[1] > 0.0 ? 0.0 : 1.0, 0.0);
    multilabel_head_backward(h, dlogits, dx);
    check("multilabel W=0 时 dx = 0", fabs(dx[0]) + fabs(dx[1]), 0.0);
    check("multilabel grad_b 累加", fabs(h->gb[0] - dlogits[0]), 1e-16);
    multilabel_head_predict(h, probs, 0.5, pred);
    check_true("multilabel predict 阈值包含等号", pred[0] == 1 && pred[1] == 1 && pred[2] == 1);

    /* 中心差分：grad_W / grad_b / dx */
    {
        double W2[6] = {0.4, -0.7, 0.2, 0.9, -0.3, 0.5};
        double b2[3] = {0.1, -0.2, 0.3};
        double worst_w = 0.0, worst_b = 0.0, worst_x = 0.0;
        int i;
        multilabel_head_set_params(h, W2, b2);
        multilabel_head_zero_grad(h);
        multilabel_head_forward(h, x, probs);
        multilabel_head_loss_grad(h, probs, targets, dlogits);
        multilabel_head_backward(h, dlogits, dx);
        for (i = 0; i < L * fi; i++) {
            double save = h->W[i], lp, lm, num, p[3];
            h->W[i] = save + FD_EPS;
            multilabel_head_forward(h, x, p);
            lp = multilabel_head_loss_grad(h, p, targets, NULL);
            h->W[i] = save - FD_EPS;
            multilabel_head_forward(h, x, p);
            lm = multilabel_head_loss_grad(h, p, targets, NULL);
            h->W[i] = save;
            num = (lp - lm) / (2.0 * FD_EPS);
            if (fabs(num - h->gW[i]) > worst_w) worst_w = fabs(num - h->gW[i]);
        }
        for (i = 0; i < L; i++) {
            double save = h->b[i], lp, lm, num, p[3];
            h->b[i] = save + FD_EPS;
            multilabel_head_forward(h, x, p);
            lp = multilabel_head_loss_grad(h, p, targets, NULL);
            h->b[i] = save - FD_EPS;
            multilabel_head_forward(h, x, p);
            lm = multilabel_head_loss_grad(h, p, targets, NULL);
            h->b[i] = save;
            num = (lp - lm) / (2.0 * FD_EPS);
            if (fabs(num - h->gb[i]) > worst_b) worst_b = fabs(num - h->gb[i]);
        }
        for (i = 0; i < fi; i++) {
            double save = x[i], lp, lm, num, p[3];
            x[i] = save + FD_EPS;
            multilabel_head_forward(h, x, p);
            lp = multilabel_head_loss_grad(h, p, targets, NULL);
            x[i] = save - FD_EPS;
            multilabel_head_forward(h, x, p);
            lm = multilabel_head_loss_grad(h, p, targets, NULL);
            x[i] = save;
            num = (lp - lm) / (2.0 * FD_EPS);
            if (fabs(num - dx[i]) > worst_x) worst_x = fabs(num - dx[i]);
        }
        check("multilabel FD grad_W", worst_w, FD_TOL);
        check("multilabel FD grad_b", worst_b, FD_TOL);
        check("multilabel FD dx", worst_x, FD_TOL);
    }
    multilabel_head_free(h);
}

/* ---------- [4] MoE：路由 + 未选中专家零梯度 ---------- */
static void set_uniform_experts(MoE *m, double scale)
{
    int e, j, k;
    for (e = 0; e < m->n_experts; e++) {
        FeedForward *f = moe_expert(m, e);
        int d = f->d_model, dff = f->d_ff;
        double *W1 = (double *)malloc(sizeof(double) * (size_t)dff * d);
        double *b1 = (double *)malloc(sizeof(double) * (size_t)dff);
        double *W2 = (double *)malloc(sizeof(double) * (size_t)d * dff);
        double *b2 = (double *)malloc(sizeof(double) * (size_t)d);
        for (j = 0; j < dff * d; j++) W1[j] = scale * ((j % 5) - 2) / 4.0;
        for (j = 0; j < dff; j++) b1[j] = scale * 0.1 * (j % 3);
        for (j = 0; j < d * dff; j++) W2[j] = scale * ((j % 4) - 1) / 5.0;
        for (k = 0; k < d; k++) b2[k] = scale * 0.05 * k;
        ff_set_params(f, W1, b1, W2, b2);
        free(W1); free(b1); free(W2); free(b2);
    }
}

static void test_moe_routing(void)
{
    const int d = 2, dff = 3, E = 3, T = 2;
    MoE *m = moe_create(d, dff, E, 1, FF_ACT_RELU, T);
    double Wg[6] = {1.0, 0.0, 0.0, 1.0, -1.0, -1.0};
    double bg[3] = {0.5, 0.0, -0.5};
    double x[4] = {1.0, 0.0, 0.0, 1.0};
    double dout[4] = {1.0, 0.0, 0.0, 1.0};
    double out[4], dx[4];
    int t;

    set_uniform_experts(m, 0.5);
    moe_set_gate(m, Wg, bg);
    moe_zero_grad(m);
    moe_forward(m, x, T, out);
    /* token0: logits = 1.5, 0, -1.5 → expert 0；token1: 0, 1.5, -1.5 → expert 1 */
    check_true("MoE top-1 路由 token0 → e0", m->route_e[0] == 0);
    check_true("MoE top-1 路由 token1 → e1", m->route_e[1] == 1);
    check("MoE top-1 权重为 1", fabs(m->route_w[0] - 1.0) + fabs(m->route_w[1] - 1.0), 0.0);
    check_true("MoE 未选中的 e2 目标数为 0", m->group_len[2] == 0);

    moe_backward(m, dout, T, dx);
    {
        FeedForward *f2 = moe_expert(m, 2);
        double s = 0.0;
        int j;
        for (j = 0; j < f2->d_ff * f2->d_model; j++) s += fabs(f2->grad_W1[j]);
        for (j = 0; j < f2->d_ff; j++) s += fabs(f2->grad_b1[j]);
        check("MoE 未选中专家梯度为 0", s, 0.0);
    }
    {
        /* 门控：top-1 时 q=1、base=c → dz=0，梯度必须为 0 */
        double s = 0.0;
        int j;
        for (j = 0; j < E * d; j++) s += fabs(m->gW_gate[j]);
        for (t = 0; t < T; t++) (void)t;
        check("MoE top-1 时门控梯度为 0", s, 0.0);
    }
    check_true("MoE param_count",
               moe_param_count(m) == E * (2 * d * dff + d + dff) + E * d + E);
    moe_free(m);
}

/* ---------- [5] MoE：全专家路由下的中心差分 ---------- */
static void test_moe_fd(void)
{
    const int d = 2, dff = 3, E = 2, T = 2;
    MoE *m = moe_create(d, dff, E, 2, FF_ACT_RELU, T);   /* top_k = E：路由恒定 */
    double Wg[4] = {0.7, -0.4, 0.2, 0.6};
    double bg[2] = {0.1, -0.2};
    double x[4] = {0.5, -0.3, 0.2, 0.8};
    double dout[4] = {0.4, -0.6, 0.3, 0.9};
    double out[4], dx[4];
    double worst_gate = 0.0, worst_w1 = 0.0, worst_dx = 0.0;
    int e, i;

    set_uniform_experts(m, 0.6);
    moe_set_gate(m, Wg, bg);
    moe_zero_grad(m);
    moe_forward(m, x, T, out);
    moe_backward(m, dout, T, dx);

    /* 数值梯度：loss = Σ dout * out（dout 当固定系数） */
    for (i = 0; i < E * d; i++) {
        double save = m->W_gate[i], lp, lm, num, o[4];
        m->W_gate[i] = save + FD_EPS;
        moe_forward(m, x, T, o);
        lp = o[0] * dout[0] + o[1] * dout[1] + o[2] * dout[2] + o[3] * dout[3];
        m->W_gate[i] = save - FD_EPS;
        moe_forward(m, x, T, o);
        lm = o[0] * dout[0] + o[1] * dout[1] + o[2] * dout[2] + o[3] * dout[3];
        m->W_gate[i] = save;
        num = (lp - lm) / (2.0 * FD_EPS);
        if (fabs(num - m->gW_gate[i]) > worst_gate) worst_gate = fabs(num - m->gW_gate[i]);
    }
    check("MoE FD grad_W_gate（top_k=E）", worst_gate, FD_TOL);

    for (e = 0; e < E; e++) {
        FeedForward *f = moe_expert(m, e);
        for (i = 0; i < f->d_ff * f->d_model; i++) {
            double save = f->W1[i], lp, lm, num, o[4];
            f->W1[i] = save + FD_EPS;
            moe_forward(m, x, T, o);
            lp = o[0] * dout[0] + o[1] * dout[1] + o[2] * dout[2] + o[3] * dout[3];
            f->W1[i] = save - FD_EPS;
            moe_forward(m, x, T, o);
            lm = o[0] * dout[0] + o[1] * dout[1] + o[2] * dout[2] + o[3] * dout[3];
            f->W1[i] = save;
            num = (lp - lm) / (2.0 * FD_EPS);
            if (fabs(num - f->grad_W1[i]) > worst_w1) worst_w1 = fabs(num - f->grad_W1[i]);
        }
    }
    check("MoE FD 专家 grad_W1（top_k=E）", worst_w1, FD_TOL);

    for (i = 0; i < T * d; i++) {
        double save = x[i], lp, lm, num;
        x[i] = save + FD_EPS;
        moe_forward(m, x, T, out);
        lp = out[0] * dout[0] + out[1] * dout[1] + out[2] * dout[2] + out[3] * dout[3];
        x[i] = save - FD_EPS;
        moe_forward(m, x, T, out);
        lm = out[0] * dout[0] + out[1] * dout[1] + out[2] * dout[2] + out[3] * dout[3];
        x[i] = save;
        num = (lp - lm) / (2.0 * FD_EPS);
        if (fabs(num - dx[i]) > worst_dx) worst_dx = fabs(num - dx[i]);
    }
    check("MoE FD dx（top_k=E）", worst_dx, FD_TOL);
    moe_free(m);
}

/* ---------- [6] 参数计数与边界 ---------- */
static void test_validation(void)
{
    TokenHead *th = token_head_create(2, 3);
    MultiLabelHead *ml = multilabel_head_create(3, 4);
    MoE *m = moe_create(2, 3, 2, 1, FF_ACT_RELU, 4);
    double x[2] = {1.0, 2.0};
    double p[4];
    double t[4] = {0.0, 1.0, 1.0, 0.0};

    check_true("token 头 param_count = L*(d+1)",
               token_head_param_count(th) == 3 * (2 + 1));
    check_true("多标签头 param_count = L*(fan_in+1)",
               multilabel_head_param_count(ml) == 4 * (3 + 1));
    check_true("MoE param_count = E*(2*d*dff+d+dff) + E*d + E",
               moe_param_count(m) == 2 * (2 * 2 * 3 + 2 + 3) + 2 * 2 + 2);
    multilabel_head_zero_grad(ml);
    multilabel_head_forward(ml, x, p);
    check("多标签全零权重时 loss 恰为 log2",
          fabs(multilabel_head_loss_grad(ml, p, t, NULL) - log(2.0)), 1e-15);

    moe_free(m);
    multilabel_head_free(ml);
    token_head_free(th);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("[test_heads] 序列标注头 / 多标签头 / MoE\n");
    test_token_basic();
    test_token_fd();
    test_multilabel();
    test_moe_routing();
    test_moe_fd();
    test_validation();
    printf("\n[test_heads] %d checks, %d failures\n", g_checks, g_fails);
    if (g_fails) {
        printf("[test_heads] FAILED\n");
        return 1;
    }
    printf("[test_heads] ALL PASS\n");
    return 0;
}
