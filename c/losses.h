/*
 * losses.h —— 额外损失函数（W1-A / W1-B 的 A 档 C 移植）
 *
 * 与 Python 版一一对应，逐位一致（参考 c/loss_reference.json）：
 *   AI.focal_loss              -> focal_loss_grad
 *   AI.bce_loss                -> bce_loss_grad
 *   AI.bce_with_logits_loss    -> bce_with_logits_loss_grad
 *   AI.huber_loss / smooth_l1  -> huber_loss_grad / smooth_l1_loss_grad
 *
 * 约定与 Python 相同：
 *   - 返回 loss；grad 非 NULL 时写出对 pred（或 logits）的梯度；
 *   - loss 都是**逐元素平均**（focal 是单样本单类，没有平均）；
 *   - 非法输入（维度、gamma/delta/eps 越界、概率越界）直接 ai_die。
 */
#ifndef AI_LOSSES_H
#define AI_LOSSES_H

/* Focal Loss：FL = -alpha_t·(1-p_t)^gamma·log(p_t)
 *
 * pred 是 softmax 概率（长度 k），target_idx 是类别下标；
 * alpha < 0 表示不做 alpha 加权（对应 Python 的 alpha=None）。
 * grad 是 softmax+focal 的联合梯度（长度 k）：dz_j = A·(p_j - delta_tj)。
 * p_t 被夹到 [1e-15, 1]，p_t == 1 时 loss 与梯度都是 0。 */
double focal_loss_grad(const double *pred, int k, int target_idx,
                       double gamma, double alpha, double *grad);

/* 二元交叉熵（pred 是概率，长度 n；target 是 0/1，长度 n）。
 * loss = -mean(y·log q + (1-y)·log(1-q))，q = clamp(p, eps, 1-eps)；
 * grad[i] = (q_i - y_i) / (q_i·(1-q_i)) / n。 */
double bce_loss_grad(const double *pred, const double *target, int n,
                     double eps, double *grad);

/* 带 logits 的二元交叉熵（数值稳定）。
 * loss = mean(max(z,0) - z·y + log1p(exp(-|z|)))；
 * grad[i] = (sigmoid(z_i) - y_i) / n。 */
double bce_with_logits_loss_grad(const double *logits, const double *target,
                                 int n, double *grad);

/* Huber（Smooth L1）：|e| <= delta 用 0.5e²，否则 delta(|e|-0.5delta)；
 * loss 取逐元素平均，grad[i] = (e 或 ±delta) / n。 */
double huber_loss_grad(const double *pred, const double *target, int n,
                       double delta, double *grad);

/* Smooth L1 = Huber 的别名（与 Python 的 smooth_l1_loss 一致） */
double smooth_l1_loss_grad(const double *pred, const double *target, int n,
                           double delta, double *grad);

#endif /* AI_LOSSES_H */
