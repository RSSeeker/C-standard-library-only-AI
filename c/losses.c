/*
 * losses.c —— 额外损失函数（focal / bce / bce_with_logits / huber）
 *
 * 每个函数都严格照抄 AI.py 的运算顺序与分组，保证与 Python 逐位一致
 * （c/loss_reference.json 由 c/tools/loss_reference.py 从 AI.py 导出）。
 * 例如 bce 的损失累加必须是 -(y*log(q) + (1-y)*log(1-q))，而不是先拆成
 * 两项再相加——浮点加法不满足结合律，改了就会差最后几位。
 */
#include "losses.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* 统一的报错出口：参数非法一律立刻退出，不静默截断 */
static void die(const char *who, const char *msg)
{
    fprintf(stderr, "losses: %s: %s\n", who, msg);
    exit(1);
}

static void check_len(const char *who, int n)
{
    if (n <= 0) die(who, "长度必须 > 0");
}

double focal_loss_grad(const double *pred, int k, int target_idx,
                       double gamma, double alpha, double *grad)
{
    const char *who = "focal_loss";
    if (!pred) die(who, "pred 为空");
    check_len(who, k);
    if (!(gamma >= 0.0)) die(who, "gamma 必须 >= 0");
    if (target_idx < 0 || target_idx >= k) die(who, "target_idx 越界");

    const double a_t = (alpha < 0.0) ? 1.0 : alpha;
    double p = pred[target_idx];
    if (!(p == p) || p < 0.0 || p > 1.0) die(who, "pred 必须是概率 [0,1]");
    if (p < 1e-15) p = 1e-15;
    if (p > 1.0) p = 1.0;

    if (grad) {
        for (int j = 0; j < k; j++) grad[j] = 0.0;
    }

    const double rest = 1.0 - p;
    if (rest <= 0.0) {
        /* p_t 恰为 1：loss 与梯度都是 0 */
        return 0.0;
    }

    const double log_p = log(p);
    const double p_gamma = pow(rest, gamma);
    const double loss = -a_t * p_gamma * log_p;

    if (!grad) return loss;

    /* A = a_t·[γ·p_t·(1-p_t)^(γ-1)·log p_t − (1-p_t)^γ] */
    const double term = (gamma > 0.0)
        ? gamma * p * pow(rest, gamma - 1.0) * log_p
        : 0.0;
    const double a = a_t * (term - p_gamma);
    for (int j = 0; j < k; j++) grad[j] = -a * pred[j];
    grad[target_idx] += a;
    return loss;
}

double bce_loss_grad(const double *pred, const double *target, int n,
                     double eps, double *grad)
{
    const char *who = "bce_loss";
    if (!pred || !target) die(who, "输入为空");
    check_len(who, n);
    if (!(eps > 0.0) || !(eps < 0.5)) die(who, "eps 必须在 (0, 0.5) 内");

    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        const double y = target[i];
        const double p = pred[i];
        if (!(p >= 0.0) || p > 1.0) die(who, "pred 必须是概率 [0,1]");
        double q = p < eps ? eps : p;
        if (q > 1.0 - eps) q = 1.0 - eps;
        loss += -(y * log(q) + (1.0 - y) * log(1.0 - q));
        if (grad) grad[i] = (q - y) / (q * (1.0 - q)) / (double)n;
    }
    return loss / (double)n;
}

double bce_with_logits_loss_grad(const double *logits, const double *target,
                                 int n, double *grad)
{
    const char *who = "bce_with_logits_loss";
    if (!logits || !target) die(who, "输入为空");
    check_len(who, n);

    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        const double y = target[i];
        const double z = logits[i];
        loss += fmax(z, 0.0) - z * y + log1p(exp(-fabs(z)));
        if (grad) {
            double s;
            if (z >= 0.0) {
                s = 1.0 / (1.0 + exp(-z));
            } else {
                const double e = exp(z);
                s = e / (1.0 + e);
            }
            grad[i] = (s - y) / (double)n;
        }
    }
    return loss / (double)n;
}

double huber_loss_grad(const double *pred, const double *target, int n,
                       double delta, double *grad)
{
    const char *who = "huber_loss";
    if (!pred || !target) die(who, "输入为空");
    check_len(who, n);
    if (!(delta > 0.0)) die(who, "delta 必须 > 0");

    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        const double e = pred[i] - target[i];
        if (fabs(e) <= delta) {
            loss += 0.5 * e * e;
            if (grad) grad[i] = e / (double)n;
        } else {
            loss += delta * (fabs(e) - 0.5 * delta);
            if (grad) grad[i] = ((e > 0.0) ? delta : -delta) / (double)n;
        }
    }
    return loss / (double)n;
}

double smooth_l1_loss_grad(const double *pred, const double *target, int n,
                           double delta, double *grad)
{
    return huber_loss_grad(pred, target, n, delta, grad);
}
