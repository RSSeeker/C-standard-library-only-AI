/*
 * offline_rl.c —— IQL / CQL 的 C17 实现（手写反向 + Adam，零依赖）
 */
#include "offline_rl.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================ 通用小工具 ============================ */

static void *orl_xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) {
        fprintf(stderr, "[offline_rl] out of memory\n");
        exit(1);
    }
    return p;
}

static void orl_softmax(const double *logits, int n, double *probs)
{
    double m = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > m) m = logits[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(logits[i] - m);
        s += probs[i];
    }
    if (!(s > 0.0) || !isfinite(s)) {
        /* 极端输入兜底：均匀分布 */
        for (int i = 0; i < n; i++) probs[i] = 1.0 / (double)n;
        return;
    }
    for (int i = 0; i < n; i++) probs[i] /= s;
}

static double orl_logsumexp(const double *row, int n)
{
    double m = row[0];
    for (int i = 1; i < n; i++)
        if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(row[i] - m);
    return m + log(s);
}

static int orl_argmax(const double *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static void orl_soft_update(MLP *target, const MLP *source, double tau)
{
    if (target->n_layers != source->n_layers) return;
    for (int i = 0; i < target->n_layers; i++) {
        Layer *tl = &target->layers[i];
        const Layer *sl = &source->layers[i];
        if (tl->fan_in != sl->fan_in || tl->fan_out != sl->fan_out) continue;
        for (int j = 0; j < tl->fan_out * tl->fan_in; j++)
            tl->W[j] = tau * tl->W[j] + (1.0 - tau) * sl->W[j];
        for (int j = 0; j < tl->fan_out; j++)
            tl->b[j] = tau * tl->b[j] + (1.0 - tau) * sl->b[j];
    }
}

/* ---------- 梯度累加器：把逐样本累积的梯度求平均后写回 MLP ---------- */

typedef struct {
    int n_layers;
    double **gw;
    double **gb;
} GradAcc;

static GradAcc acc_create(const MLP *m)
{
    GradAcc a;
    a.n_layers = m->n_layers;
    a.gw = (double **)orl_xcalloc((size_t)a.n_layers, sizeof(double *));
    a.gb = (double **)orl_xcalloc((size_t)a.n_layers, sizeof(double *));
    for (int i = 0; i < a.n_layers; i++) {
        const Layer *l = &m->layers[i];
        a.gw[i] = (double *)orl_xcalloc((size_t)(l->fan_out * l->fan_in), sizeof(double));
        a.gb[i] = (double *)orl_xcalloc((size_t)l->fan_out, sizeof(double));
    }
    return a;
}

static void acc_zero_from(GradAcc *a, const MLP *m)
{
    for (int i = 0; i < a->n_layers; i++) {
        const Layer *l = &m->layers[i];
        memset(a->gw[i], 0, (size_t)(l->fan_out * l->fan_in) * sizeof(double));
        memset(a->gb[i], 0, (size_t)l->fan_out * sizeof(double));
    }
}

static void acc_add(GradAcc *a, const MLP *m)
{
    for (int i = 0; i < a->n_layers; i++) {
        const Layer *l = &m->layers[i];
        const int nw = l->fan_out * l->fan_in;
        for (int j = 0; j < nw; j++) a->gw[i][j] += l->gW[j];
        for (int j = 0; j < l->fan_out; j++) a->gb[i][j] += l->gb[j];
    }
}

/* 把累加器求平均后写回模型梯度，供 optimizer_step 使用 */
static void acc_average_into(const GradAcc *a, MLP *m, int n)
{
    const double inv = 1.0 / (double)n;
    for (int i = 0; i < a->n_layers; i++) {
        Layer *l = &m->layers[i];
        const int nw = l->fan_out * l->fan_in;
        for (int j = 0; j < nw; j++) l->gW[j] = a->gw[i][j] * inv;
        for (int j = 0; j < l->fan_out; j++) l->gb[j] = a->gb[i][j] * inv;
    }
}

static void acc_free(GradAcc *a)
{
    for (int i = 0; i < a->n_layers; i++) {
        free(a->gw[i]);
        free(a->gb[i]);
    }
    free(a->gw);
    free(a->gb);
}

/* ---------- 网络构建 ---------- */

/* sizes = [state_dim, hidden..., out_dim]；hidden 用 relu，输出线性。
 * 返回 NULL 表示参数非法。 */
static MLP *orl_make_mlp(int state_dim, const int *hidden, int n_hidden,
                         int out_dim)
{
    const int n_sizes = n_hidden + 2;
    int *sizes = (int *)orl_xcalloc((size_t)n_sizes, sizeof(int));
    Activation *acts = (Activation *)orl_xcalloc((size_t)(n_sizes - 1), sizeof(Activation));
    sizes[0] = state_dim;
    for (int i = 0; i < n_hidden; i++) sizes[i + 1] = hidden[i];
    sizes[n_sizes - 1] = out_dim;
    for (int i = 0; i < n_sizes - 1; i++)
        acts[i] = (i == n_sizes - 2) ? ACT_LINEAR : ACT_RELU;
    MLP *m = mlp_create(sizes, acts, n_sizes);
    free(sizes);
    free(acts);
    return m;
}

static int orl_valid_hidden(const int *hidden, int n_hidden)
{
    if (n_hidden < 0) return 0;
    if (n_hidden > 0 && !hidden) return 0;
    for (int i = 0; i < n_hidden; i++)
        if (hidden[i] <= 0) return 0;
    return 1;
}

/* ============================ IQL ============================ */

IqlAgent *iql_create(int state_dim, int num_actions, const int *hidden,
                     int n_hidden, double actor_lr, double critic_lr,
                     double tau, double beta, double gamma, double target_tau)
{
    if (state_dim <= 0 || num_actions <= 0) return NULL;
    if (!orl_valid_hidden(hidden, n_hidden)) return NULL;
    if (!(actor_lr > 0.0) || !(critic_lr > 0.0)) return NULL;
    if (!(tau > 0.0 && tau < 1.0)) return NULL;
    if (!(beta > 0.0)) return NULL;
    if (!(gamma >= 0.0 && gamma <= 1.0)) return NULL;
    if (!(target_tau > 0.0 && target_tau < 1.0)) return NULL;

    IqlAgent *a = (IqlAgent *)orl_xcalloc(1, sizeof(IqlAgent));
    a->state_dim = state_dim;
    a->num_actions = num_actions;
    a->tau = tau;
    a->beta = beta;
    a->gamma = gamma;
    a->target_tau = target_tau;

    a->q = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    a->q_target = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    a->v = orl_make_mlp(state_dim, hidden, n_hidden, 1);
    a->v_target = orl_make_mlp(state_dim, hidden, n_hidden, 1);
    a->policy = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    if (!a->q || !a->q_target || !a->v || !a->v_target || !a->policy) {
        iql_free(a);
        return NULL;
    }
    /* 目标网络初始化为在线网络副本 */
    for (int i = 0; i < a->q->n_layers; i++) {
        Layer *tl = &a->q_target->layers[i];
        const Layer *sl = &a->q->layers[i];
        memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
        memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
    }
    for (int i = 0; i < a->v->n_layers; i++) {
        Layer *tl = &a->v_target->layers[i];
        const Layer *sl = &a->v->layers[i];
        memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
        memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
    }
    optimizer_init(&a->q_opt, OPT_ADAM, critic_lr);
    optimizer_init(&a->v_opt, OPT_ADAM, critic_lr);
    optimizer_init(&a->policy_opt, OPT_ADAM, actor_lr);
    return a;
}

void iql_free(IqlAgent *a)
{
    if (!a) return;
    mlp_free(a->q);
    mlp_free(a->q_target);
    mlp_free(a->v);
    mlp_free(a->v_target);
    mlp_free(a->policy);
    free(a);
}

MLP *iql_net_q(IqlAgent *a) { return a ? a->q : NULL; }
MLP *iql_net_v(IqlAgent *a) { return a ? a->v : NULL; }
MLP *iql_net_policy(IqlAgent *a) { return a ? a->policy : NULL; }

void iql_q(const IqlAgent *a, const double *state, double *q_logits)
{
    mlp_forward(a->q, state, q_logits);
}

double iql_v(const IqlAgent *a, const double *state)
{
    double out = 0.0;
    mlp_forward(a->v, state, &out);
    return out;
}

void iql_probs(const IqlAgent *a, const double *state, double *probs)
{
    double *logits = (double *)orl_xcalloc((size_t)a->num_actions, sizeof(double));
    mlp_forward(a->policy, state, logits);
    orl_softmax(logits, a->num_actions, probs);
    free(logits);
}

int iql_act(const IqlAgent *a, const double *state)
{
    double *logits = (double *)orl_xcalloc((size_t)a->num_actions, sizeof(double));
    mlp_forward(a->policy, state, logits);
    int best = orl_argmax(logits, a->num_actions);
    free(logits);
    return best;
}

int iql_param_count(const IqlAgent *a)
{
    int total = 0;
    MLP *nets[5] = {a->q, a->q_target, a->v, a->v_target, a->policy};
    for (int k = 0; k < 5; k++)
        for (int i = 0; i < nets[k]->n_layers; i++)
            total += nets[k]->layers[i].fan_out * nets[k]->layers[i].fan_in
                     + nets[k]->layers[i].fan_out;
    return total;
}

int iql_update(IqlAgent *a, const double *states, const int *actions,
               const double *rewards, const double *next_states,
               const int *dones, int n, double losses[3])
{
    if (!a || !states || !actions || !rewards || !next_states || !dones) return -1;
    if (n < 1) return -1;
    const int sd = a->state_dim;
    const int na = a->num_actions;
    for (int i = 0; i < n; i++)
        if (actions[i] < 0 || actions[i] >= na) return -1;

    double *qs = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *dq = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *probs = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *dlogits = (double *)orl_xcalloc((size_t)na, sizeof(double));

    GradAcc q_acc = acc_create(a->q);
    GradAcc v_acc = acc_create(a->v);
    GradAcc pi_acc = acc_create(a->policy);
    acc_zero_from(&q_acc, a->q);
    acc_zero_from(&v_acc, a->v);
    acc_zero_from(&pi_acc, a->policy);

    double q_loss = 0.0, v_loss = 0.0, pi_loss = 0.0;
    const double n_d = (double)n;

    for (int i = 0; i < n; i++) {
        const double *s = states + (size_t)i * sd;
        const double *sp = next_states + (size_t)i * sd;
        const int act = actions[i];
        const double r = rewards[i];
        const int d = dones[i] ? 1 : 0;

        mlp_forward(a->q, s, qs);
        const double q_a = qs[act];
        const double v = iql_v(a, s);

        /* V 的 expectile 回归 */
        const double diff = q_a - v;
        const double w = (diff > 0.0) ? a->tau : (1.0 - a->tau);
        v_loss += w * diff * diff / n_d;
        const double dv = -2.0 * w * diff / n_d;
        mlp_zero_grad(a->v);
        mlp_backward(a->v, &dv);
        acc_add(&v_acc, a->v);

        /* Q 的 TD 目标用 V_target(s')（IQL 不做 max） */
        double y = r;
        if (!d) {
            double vt = 0.0;
            mlp_forward(a->v_target, sp, &vt);
            y = r + a->gamma * vt;
        }
        q_loss += (q_a - y) * (q_a - y) / n_d;
        for (int j = 0; j < na; j++) dq[j] = 0.0;
        dq[act] = 2.0 * (q_a - y) / n_d;
        mlp_zero_grad(a->q);
        mlp_backward(a->q, dq);
        acc_add(&q_acc, a->q);

        /* 策略 AWR（优势 = Q(s,a) - V(s)） */
        const double adv = q_a - v;
        const double weight = exp(fmin(a->beta * adv, 30.0));
        mlp_forward(a->policy, s, probs);
        orl_softmax(probs, na, probs);
        const double pa = fmax(probs[act], 1e-12);
        pi_loss += -weight * log(pa) / n_d;
        for (int j = 0; j < na; j++) {
            const double target = (j == act) ? 1.0 : 0.0;
            dlogits[j] = weight * (probs[j] - target) / n_d;
        }
        mlp_zero_grad(a->policy);
        mlp_backward(a->policy, dlogits);
        acc_add(&pi_acc, a->policy);
    }

    acc_average_into(&v_acc, a->v, n);
    optimizer_step(a->v, &a->v_opt);
    acc_average_into(&q_acc, a->q, n);
    optimizer_step(a->q, &a->q_opt);
    acc_average_into(&pi_acc, a->policy, n);
    optimizer_step(a->policy, &a->policy_opt);
    orl_soft_update(a->q_target, a->q, a->target_tau);
    orl_soft_update(a->v_target, a->v, a->target_tau);

    if (losses) {
        losses[0] = q_loss;
        losses[1] = v_loss;
        losses[2] = pi_loss;
    }

    acc_free(&q_acc);
    acc_free(&v_acc);
    acc_free(&pi_acc);
    free(qs);
    free(dq);
    free(probs);
    free(dlogits);
    return 0;
}

/* ============================ CQL ============================ */

CqlAgent *cql_create(int state_dim, int num_actions, const int *hidden,
                     int n_hidden, double actor_lr, double critic_lr,
                     double alpha, double beta, double gamma, double target_tau)
{
    if (state_dim <= 0 || num_actions <= 0) return NULL;
    if (!orl_valid_hidden(hidden, n_hidden)) return NULL;
    if (!(actor_lr > 0.0) || !(critic_lr > 0.0)) return NULL;
    if (!(alpha >= 0.0)) return NULL;
    if (!(beta > 0.0)) return NULL;
    if (!(gamma >= 0.0 && gamma <= 1.0)) return NULL;
    if (!(target_tau > 0.0 && target_tau < 1.0)) return NULL;

    CqlAgent *a = (CqlAgent *)orl_xcalloc(1, sizeof(CqlAgent));
    a->state_dim = state_dim;
    a->num_actions = num_actions;
    a->alpha = alpha;
    a->beta = beta;
    a->gamma = gamma;
    a->target_tau = target_tau;

    a->q = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    a->q_target = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    a->policy = orl_make_mlp(state_dim, hidden, n_hidden, num_actions);
    if (!a->q || !a->q_target || !a->policy) {
        cql_free(a);
        return NULL;
    }
    for (int i = 0; i < a->q->n_layers; i++) {
        Layer *tl = &a->q_target->layers[i];
        const Layer *sl = &a->q->layers[i];
        memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
        memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
    }
    optimizer_init(&a->q_opt, OPT_ADAM, critic_lr);
    optimizer_init(&a->policy_opt, OPT_ADAM, actor_lr);
    return a;
}

void cql_free(CqlAgent *a)
{
    if (!a) return;
    mlp_free(a->q);
    mlp_free(a->q_target);
    mlp_free(a->policy);
    free(a);
}

MLP *cql_net_q(CqlAgent *a) { return a ? a->q : NULL; }
MLP *cql_net_policy(CqlAgent *a) { return a ? a->policy : NULL; }

void cql_q(const CqlAgent *a, const double *state, double *q_logits)
{
    mlp_forward(a->q, state, q_logits);
}

void cql_probs(const CqlAgent *a, const double *state, double *probs)
{
    double *logits = (double *)orl_xcalloc((size_t)a->num_actions, sizeof(double));
    mlp_forward(a->policy, state, logits);
    orl_softmax(logits, a->num_actions, probs);
    free(logits);
}

int cql_act(const CqlAgent *a, const double *state)
{
    double *logits = (double *)orl_xcalloc((size_t)a->num_actions, sizeof(double));
    mlp_forward(a->q, state, logits);
    int best = orl_argmax(logits, a->num_actions);
    free(logits);
    return best;
}

int cql_param_count(const CqlAgent *a)
{
    int total = 0;
    MLP *nets[3] = {a->q, a->q_target, a->policy};
    for (int k = 0; k < 3; k++)
        for (int i = 0; i < nets[k]->n_layers; i++)
            total += nets[k]->layers[i].fan_out * nets[k]->layers[i].fan_in
                     + nets[k]->layers[i].fan_out;
    return total;
}

int cql_update(CqlAgent *a, const double *states, const int *actions,
               const double *rewards, const double *next_states,
               const int *dones, int n, double losses[2])
{
    if (!a || !states || !actions || !rewards || !next_states || !dones) return -1;
    if (n < 1) return -1;
    const int sd = a->state_dim;
    const int na = a->num_actions;
    for (int i = 0; i < n; i++)
        if (actions[i] < 0 || actions[i] >= na) return -1;

    double *qs = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *dq = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *probs_q = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *probs = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *dlogits = (double *)orl_xcalloc((size_t)na, sizeof(double));
    double *next_q = (double *)orl_xcalloc((size_t)na, sizeof(double));

    GradAcc q_acc = acc_create(a->q);
    GradAcc pi_acc = acc_create(a->policy);
    acc_zero_from(&q_acc, a->q);
    acc_zero_from(&pi_acc, a->policy);

    double q_loss = 0.0, pi_loss = 0.0;
    const double n_d = (double)n;

    for (int i = 0; i < n; i++) {
        const double *s = states + (size_t)i * sd;
        const double *sp = next_states + (size_t)i * sd;
        const int act = actions[i];
        const double r = rewards[i];
        const int d = dones[i] ? 1 : 0;

        mlp_forward(a->q, s, qs);
        const double q_a = qs[act];
        double y = r;
        if (!d) {
            mlp_forward(a->q_target, sp, next_q);
            double mx = next_q[0];
            for (int j = 1; j < na; j++)
                if (next_q[j] > mx) mx = next_q[j];
            y = r + a->gamma * mx;
        }
        orl_softmax(qs, na, probs_q);
        const double lse = orl_logsumexp(qs, na);
        q_loss += (q_a - y) * (q_a - y) / n_d
                  + a->alpha * (lse - q_a) / n_d;
        for (int j = 0; j < na; j++) {
            const double onehot = (j == act) ? 1.0 : 0.0;
            dq[j] = a->alpha * (probs_q[j] - onehot) / n_d;
        }
        dq[act] += 2.0 * (q_a - y) / n_d;
        mlp_zero_grad(a->q);
        mlp_backward(a->q, dq);
        acc_add(&q_acc, a->q);

        /* 策略 AWR（优势 = Q(s,a) - logsumexp Q） */
        const double adv = q_a - lse;
        const double weight = exp(fmin(a->beta * adv, 30.0));
        mlp_forward(a->policy, s, probs);
        orl_softmax(probs, na, probs);
        const double pa = fmax(probs[act], 1e-12);
        pi_loss += -weight * log(pa) / n_d;
        for (int j = 0; j < na; j++) {
            const double target = (j == act) ? 1.0 : 0.0;
            dlogits[j] = weight * (probs[j] - target) / n_d;
        }
        mlp_zero_grad(a->policy);
        mlp_backward(a->policy, dlogits);
        acc_add(&pi_acc, a->policy);
    }

    acc_average_into(&q_acc, a->q, n);
    optimizer_step(a->q, &a->q_opt);
    acc_average_into(&pi_acc, a->policy, n);
    optimizer_step(a->policy, &a->policy_opt);
    orl_soft_update(a->q_target, a->q, a->target_tau);

    if (losses) {
        losses[0] = q_loss;
        losses[1] = pi_loss;
    }

    acc_free(&q_acc);
    acc_free(&pi_acc);
    free(qs);
    free(dq);
    free(probs_q);
    free(probs);
    free(dlogits);
    free(next_q);
    return 0;
}
