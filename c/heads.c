/* ==========================================================================
 *  heads.c：TokenClassificationHead / MultiLabelHead / MoE 的 C17 实现
 *
 *  与 Python 逐位一致的关键点都写在注释里：
 *    - 内积用 dot_neumaier（对应 sum(map(_mul, ...))）；
 *    - softmax 分母用 sum_neumaier；
 *    - 其余累加（loss、梯度、专家加权求和）一律朴素 +=。
 * ========================================================================== */
#include "heads.h"
#include "attention.h"
#include "numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 *  内部助手
 * -------------------------------------------------------------------------- */
static void *hd_malloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "[heads] 内存分配失败\n");
        exit(1);
    }
    return p;
}

static void *hd_calloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "[heads] 内存分配失败\n");
        exit(1);
    }
    return p;
}

static void hd_die(const char *where, const char *msg)
{
    fprintf(stderr, "[heads] %s: %s\n", where, msg);
    exit(1);
}

static double hd_sigmoid(double z)
{
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    {
        double e = exp(z);
        return e / (1.0 + e);
    }
}

/* 与 Python _softmax_row 同构：max 平移 → exp → sum_neumaier → 逐项除 */
static void hd_softmax(const double *v, int n, double *out)
{
    int i;
    double mx = v[0], tot;
    for (i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    for (i = 0; i < n; i++) out[i] = exp(v[i] - mx);
    tot = sum_neumaier(out, n);
    for (i = 0; i < n; i++) out[i] = out[i] / tot;
}

/* --------------------------------------------------------------------------
 *  TokenClassificationHead
 * -------------------------------------------------------------------------- */
TokenHead *token_head_create(int d_model, int num_labels)
{
    TokenHead *h;
    if (d_model < 1 || num_labels < 1)
        hd_die("token_head_create", "d_model / num_labels 必须 >= 1");
    h = (TokenHead *)hd_calloc(1, sizeof(TokenHead));
    h->d_model = d_model;
    h->num_labels = num_labels;
    h->W = (double *)hd_calloc((size_t)num_labels * (size_t)d_model, sizeof(double));
    h->b = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->gW = (double *)hd_calloc((size_t)num_labels * (size_t)d_model, sizeof(double));
    h->gb = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->cap = 0;
    h->x = NULL;
    h->T = 0;
    return h;
}

void token_head_free(TokenHead *h)
{
    if (!h) return;
    free(h->W);
    free(h->b);
    free(h->gW);
    free(h->gb);
    free(h->x);
    free(h);
}

void token_head_zero_grad(TokenHead *h)
{
    memset(h->gW, 0, (size_t)h->num_labels * (size_t)h->d_model * sizeof(double));
    memset(h->gb, 0, (size_t)h->num_labels * sizeof(double));
}

void token_head_set_params(TokenHead *h, const double *W, const double *b)
{
    if (!W || !b) hd_die("token_head_set_params", "W / b 不能为空");
    memcpy(h->W, W, (size_t)h->num_labels * (size_t)h->d_model * sizeof(double));
    memcpy(h->b, b, (size_t)h->num_labels * sizeof(double));
}

int token_head_param_count(const TokenHead *h)
{
    return h->num_labels * h->d_model + h->num_labels;
}

static void token_head_reserve(TokenHead *h, int T)
{
    if (T <= h->cap) return;
    free(h->x);
    h->x = (double *)hd_malloc((size_t)T * (size_t)h->d_model * sizeof(double));
    h->cap = T;
}

void token_head_forward(TokenHead *h, const double *x, int T, double *logits)
{
    int t, l, j;
    if (T < 1) hd_die("token_head_forward", "T 必须 >= 1");
    if (!x || !logits) hd_die("token_head_forward", "x / logits 不能为空");
    token_head_reserve(h, T);
    memcpy(h->x, x, (size_t)T * (size_t)h->d_model * sizeof(double));
    h->T = T;
    for (t = 0; t < T; t++) {
        const double *xt = h->x + (size_t)t * h->d_model;
        for (l = 0; l < h->num_labels; l++) {
            double acc = dot_neumaier(h->W + (size_t)l * h->d_model, xt, h->d_model);
            logits[(size_t)t * h->num_labels + l] = acc + h->b[l];
        }
    }
    (void)j;
}

double token_head_loss_grad(TokenHead *h, const double *logits, const int *targets,
                            const int *mask, int T, double *dlogits)
{
    int t, l, n = 0;
    double loss = 0.0;
    double *probs;
    if (T < 1) hd_die("token_head_loss_grad", "T 必须 >= 1");
    if (!logits || !targets) hd_die("token_head_loss_grad", "logits / targets 不能为空");
    for (t = 0; t < T; t++) if (!mask || mask[t]) n++;
    if (n == 0) hd_die("token_head_loss_grad", "mask 全为 0，没有可训练的 token");
    probs = (double *)hd_malloc((size_t)h->num_labels * sizeof(double));
    if (dlogits)
        memset(dlogits, 0, (size_t)T * (size_t)h->num_labels * sizeof(double));
    for (t = 0; t < T; t++) {
        int target = targets[t];
        if (target < 0 || target >= h->num_labels)
            hd_die("token_head_loss_grad", "标签越界");
        hd_softmax(logits + (size_t)t * h->num_labels, h->num_labels, probs);
        if (mask && !mask[t]) continue;
        {
            double p = probs[target];
            if (p > 1e-12) loss += -log(p);
            else loss += -log(1e-12);
        }
        if (dlogits) {
            double *row = dlogits + (size_t)t * h->num_labels;
            for (l = 0; l < h->num_labels; l++) row[l] = probs[l] / (double)n;
            row[target] -= 1.0 / (double)n;
        }
    }
    free(probs);
    return loss / (double)n;
}

void token_head_backward(TokenHead *h, const double *dlogits, int T, double *dx)
{
    int t, l, j;
    if (h->T < 1) hd_die("token_head_backward", "必须先 forward");
    if (T != h->T) hd_die("token_head_backward", "T 与 forward 不一致");
    if (!dlogits) hd_die("token_head_backward", "dlogits 不能为空");
    if (dx) memset(dx, 0, (size_t)T * (size_t)h->d_model * sizeof(double));
    for (t = 0; t < T; t++) {
        const double *drow = dlogits + (size_t)t * h->num_labels;
        const double *xt = h->x + (size_t)t * h->d_model;
        for (l = 0; l < h->num_labels; l++) {
            double d = drow[l];
            double *gw;
            if (d == 0.0) continue;
            h->gb[l] += d;
            gw = h->gW + (size_t)l * h->d_model;
            for (j = 0; j < h->d_model; j++) gw[j] += d * xt[j];
            if (dx) {
                double *dxt = dx + (size_t)t * h->d_model;
                const double *wl = h->W + (size_t)l * h->d_model;
                for (j = 0; j < h->d_model; j++) dxt[j] += d * wl[j];
            }
        }
    }
}

void token_head_predict(const TokenHead *h, const double *logits, int T,
                        const int *mask, int *out)
{
    int t, l;
    for (t = 0; t < T; t++) {
        if (mask && !mask[t]) { out[t] = -1; continue; }
        {
            const double *row = logits + (size_t)t * h->num_labels;
            int best = 0;
            for (l = 1; l < h->num_labels; l++) if (row[l] > row[best]) best = l;
            out[t] = best;
        }
    }
}

/* --------------------------------------------------------------------------
 *  MultiLabelHead
 * -------------------------------------------------------------------------- */
MultiLabelHead *multilabel_head_create(int fan_in, int num_labels)
{
    MultiLabelHead *h;
    if (fan_in < 1 || num_labels < 1)
        hd_die("multilabel_head_create", "fan_in / num_labels 必须 >= 1");
    h = (MultiLabelHead *)hd_calloc(1, sizeof(MultiLabelHead));
    h->fan_in = fan_in;
    h->num_labels = num_labels;
    h->W = (double *)hd_calloc((size_t)num_labels * (size_t)fan_in, sizeof(double));
    h->b = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->gW = (double *)hd_calloc((size_t)num_labels * (size_t)fan_in, sizeof(double));
    h->gb = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->x = (double *)hd_calloc((size_t)fan_in, sizeof(double));
    h->logits = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->probs = (double *)hd_calloc((size_t)num_labels, sizeof(double));
    h->ready = 0;
    return h;
}

void multilabel_head_free(MultiLabelHead *h)
{
    if (!h) return;
    free(h->W);
    free(h->b);
    free(h->gW);
    free(h->gb);
    free(h->x);
    free(h->logits);
    free(h->probs);
    free(h);
}

void multilabel_head_zero_grad(MultiLabelHead *h)
{
    memset(h->gW, 0, (size_t)h->num_labels * (size_t)h->fan_in * sizeof(double));
    memset(h->gb, 0, (size_t)h->num_labels * sizeof(double));
}

void multilabel_head_set_params(MultiLabelHead *h, const double *W, const double *b)
{
    if (!W || !b) hd_die("multilabel_head_set_params", "W / b 不能为空");
    memcpy(h->W, W, (size_t)h->num_labels * (size_t)h->fan_in * sizeof(double));
    memcpy(h->b, b, (size_t)h->num_labels * sizeof(double));
}

int multilabel_head_param_count(const MultiLabelHead *h)
{
    return h->num_labels * h->fan_in + h->num_labels;
}

void multilabel_head_forward(MultiLabelHead *h, const double *x, double *probs)
{
    int l;
    if (!x) hd_die("multilabel_head_forward", "x 不能为空");
    memcpy(h->x, x, (size_t)h->fan_in * sizeof(double));
    for (l = 0; l < h->num_labels; l++) {
        h->logits[l] = dot_neumaier(h->W + (size_t)l * h->fan_in, h->x, h->fan_in)
                       + h->b[l];
        h->probs[l] = hd_sigmoid(h->logits[l]);
    }
    h->ready = 1;
    if (probs) memcpy(probs, h->probs, (size_t)h->num_labels * sizeof(double));
}

double multilabel_head_loss_grad(MultiLabelHead *h, const double *probs,
                                 const double *targets, double *dlogits)
{
    int l;
    double loss = 0.0;
    if (!h->ready) hd_die("multilabel_head_loss_grad", "必须先 forward");
    if (!probs || !targets) hd_die("multilabel_head_loss_grad", "probs / targets 不能为空");
    for (l = 0; l < h->num_labels; l++) {
        double y = targets[l];
        double z = h->logits[l];
        if (y != 0.0 && y != 1.0)
            hd_die("multilabel_head_loss_grad", "标签必须是 0 / 1");
        loss += fmax(z, 0.0) - z * y + log1p(exp(-fabs(z)));
        if (dlogits) dlogits[l] = (probs[l] - y) / (double)h->num_labels;
    }
    return loss / (double)h->num_labels;
}

void multilabel_head_backward(MultiLabelHead *h, const double *dlogits, double *dx)
{
    int l, j;
    if (!h->ready) hd_die("multilabel_head_backward", "必须先 forward");
    if (!dlogits) hd_die("multilabel_head_backward", "dlogits 不能为空");
    if (dx) memset(dx, 0, (size_t)h->fan_in * sizeof(double));
    for (l = 0; l < h->num_labels; l++) {
        double d = dlogits[l];
        double *gw;
        if (d == 0.0) continue;
        h->gb[l] += d;
        gw = h->gW + (size_t)l * h->fan_in;
        for (j = 0; j < h->fan_in; j++) {
            gw[j] += d * h->x[j];
            if (dx) dx[j] += d * h->W[(size_t)l * h->fan_in + j];
        }
    }
}

void multilabel_head_predict(const MultiLabelHead *h, const double *probs,
                             double threshold, int *out)
{
    int l;
    const double *p = probs ? probs : h->probs;
    if (!(threshold > 0.0 && threshold < 1.0))
        hd_die("multilabel_head_predict", "threshold 必须在 (0, 1) 内");
    for (l = 0; l < h->num_labels; l++) out[l] = (p[l] >= threshold) ? 1 : 0;
}

/* --------------------------------------------------------------------------
 *  MoE
 * -------------------------------------------------------------------------- */
MoE *moe_create(int d_model, int d_ff, int n_experts, int top_k, int act, int max_seq)
{
    MoE *m;
    int e;
    if (d_model < 1 || d_ff < 1 || n_experts < 1 || top_k < 1)
        hd_die("moe_create", "d_model / d_ff / n_experts / top_k 必须 >= 1");
    if (top_k > n_experts) hd_die("moe_create", "top_k 不能大于 n_experts");
    if (max_seq < 1) hd_die("moe_create", "max_seq 必须 >= 1");
    if (act < FF_ACT_RELU || act > FF_ACT_SELU) hd_die("moe_create", "激活码非法");
    m = (MoE *)hd_calloc(1, sizeof(MoE));
    m->d_model = d_model;
    m->d_ff = d_ff;
    m->n_experts = n_experts;
    m->top_k = top_k;
    m->act = act;
    m->max_seq = max_seq;
    m->experts = (FeedForward **)hd_calloc((size_t)n_experts, sizeof(FeedForward *));
    for (e = 0; e < n_experts; e++)
        m->experts[e] = ff_create_ex(d_model, d_ff, max_seq, act);
    m->W_gate = (double *)hd_calloc((size_t)n_experts * (size_t)d_model, sizeof(double));
    m->b_gate = (double *)hd_calloc((size_t)n_experts, sizeof(double));
    m->gW_gate = (double *)hd_calloc((size_t)n_experts * (size_t)d_model, sizeof(double));
    m->gb_gate = (double *)hd_calloc((size_t)n_experts, sizeof(double));
    m->x = (double *)hd_calloc((size_t)max_seq * (size_t)d_model, sizeof(double));
    m->route_e = (int *)hd_calloc((size_t)max_seq * (size_t)top_k, sizeof(int));
    m->route_w = (double *)hd_calloc((size_t)max_seq * (size_t)top_k, sizeof(double));
    m->group_idx = (int *)hd_calloc((size_t)n_experts * (size_t)max_seq, sizeof(int));
    m->group_w = (double *)hd_calloc((size_t)n_experts * (size_t)max_seq, sizeof(double));
    m->group_len = (int *)hd_calloc((size_t)n_experts, sizeof(int));
    m->group_pos = (int *)hd_calloc((size_t)n_experts * (size_t)max_seq, sizeof(int));
    m->group_out = (double *)hd_calloc((size_t)n_experts * (size_t)max_seq *
                                       (size_t)d_model, sizeof(double));
    m->ready = 0;
    return m;
}

void moe_free(MoE *m)
{
    int e;
    if (!m) return;
    for (e = 0; e < m->n_experts; e++) ff_free((FeedForward *)m->experts[e]);
    free(m->experts);
    free(m->W_gate);
    free(m->b_gate);
    free(m->gW_gate);
    free(m->gb_gate);
    free(m->x);
    free(m->route_e);
    free(m->route_w);
    free(m->group_idx);
    free(m->group_w);
    free(m->group_len);
    free(m->group_pos);
    free(m->group_out);
    free(m);
}

void moe_zero_grad(MoE *m)
{
    int e;
    for (e = 0; e < m->n_experts; e++) ff_zero_grad((FeedForward *)m->experts[e]);
    memset(m->gW_gate, 0, (size_t)m->n_experts * (size_t)m->d_model * sizeof(double));
    memset(m->gb_gate, 0, (size_t)m->n_experts * sizeof(double));
}

int moe_param_count(const MoE *m)
{
    int per = 0;
    per = ff_param_count((const FeedForward *)m->experts[0]);
    return per * m->n_experts + m->n_experts * m->d_model + m->n_experts;
}

FeedForward *moe_expert(const MoE *m, int e)
{
    if (e < 0 || e >= m->n_experts) hd_die("moe_expert", "专家下标越界");
    return (FeedForward *)m->experts[e];
}

void moe_set_gate(MoE *m, const double *W_gate, const double *b_gate)
{
    if (!W_gate || !b_gate) hd_die("moe_set_gate", "W_gate / b_gate 不能为空");
    memcpy(m->W_gate, W_gate,
           (size_t)m->n_experts * (size_t)m->d_model * sizeof(double));
    memcpy(m->b_gate, b_gate, (size_t)m->n_experts * sizeof(double));
}

void moe_forward(MoE *m, const double *x, int T, double *out)
{
    int t, e, k, j;
    double *gate_logits;
    if (T < 1 || T > m->max_seq)
        hd_die("moe_forward", "T 必须在 [1, max_seq] 内");
    if (!x || !out) hd_die("moe_forward", "x / out 不能为空");
    memcpy(m->x, x, (size_t)T * (size_t)m->d_model * sizeof(double));
    m->T = T;
    memset(out, 0, (size_t)T * (size_t)m->d_model * sizeof(double));
    for (e = 0; e < m->n_experts; e++) m->group_len[e] = 0;
    for (e = 0; e < m->n_experts; e++)
        for (t = 0; t < T; t++)
            m->group_pos[(size_t)e * m->max_seq + t] = -1;

    gate_logits = (double *)hd_malloc((size_t)m->n_experts * sizeof(double));

    /* 1) 路由：按 logit 降序取 top-k（并列时下标小的在前，与 Python sorted 一致） */
    for (t = 0; t < T; t++) {
        const double *xt = m->x + (size_t)t * m->d_model;
        int *order = (int *)hd_malloc((size_t)m->n_experts * sizeof(int));
        double *picked = (double *)hd_malloc((size_t)m->top_k * sizeof(double));
        for (e = 0; e < m->n_experts; e++)
            gate_logits[e] = dot_neumaier(m->W_gate + (size_t)e * m->d_model, xt,
                                          m->d_model) + m->b_gate[e];
        for (e = 0; e < m->n_experts; e++) order[e] = e;
        /* 稳定选择排序：严格大于才交换 */
        for (k = 0; k < m->top_k; k++) {
            int best = k;
            for (j = k + 1; j < m->n_experts; j++)
                if (gate_logits[order[j]] > gate_logits[order[best]]) best = j;
            if (best != k) {
                int tmp = order[k];
                order[k] = order[best];
                order[best] = tmp;
            }
        }
        for (k = 0; k < m->top_k; k++) picked[k] = gate_logits[order[k]];
        hd_softmax(picked, m->top_k, m->route_w + (size_t)t * m->top_k);
        for (k = 0; k < m->top_k; k++) {
            int ex = order[k];
            m->route_e[(size_t)t * m->top_k + k] = ex;
            {
                int pos = m->group_len[ex];
                m->group_idx[(size_t)ex * m->max_seq + pos] = t;
                m->group_w[(size_t)ex * m->max_seq + pos] =
                    m->route_w[(size_t)t * m->top_k + k];
                m->group_pos[(size_t)ex * m->max_seq + t] = pos;
                m->group_len[ex] = pos + 1;
            }
        }
        free(order);
        free(picked);
    }
    free(gate_logits);

    /* 2) 逐专家前向，再按路由权重累加 */
    for (e = 0; e < m->n_experts; e++) {
        int len = m->group_len[e];
        Tensor *tin, *tout;
        int shape[2];
        if (len <= 0) continue;
        shape[0] = len;
        shape[1] = m->d_model;
        tin = tensor_new(2, shape);
        tout = tensor_new(2, shape);
        for (k = 0; k < len; k++) {
            int tt = m->group_idx[(size_t)e * m->max_seq + k];
            for (j = 0; j < m->d_model; j++)
                tin->data[(size_t)k * m->d_model + j] =
                    m->x[(size_t)tt * m->d_model + j];
        }
        ff_forward((FeedForward *)m->experts[e], tin, tout);
        for (k = 0; k < len; k++) {
            int tt = m->group_idx[(size_t)e * m->max_seq + k];
            double w = m->group_w[(size_t)e * m->max_seq + k];
            for (j = 0; j < m->d_model; j++) {
                double v = tout->data[(size_t)k * m->d_model + j];
                m->group_out[((size_t)e * m->max_seq + k) * m->d_model + j] = v;
                out[(size_t)tt * m->d_model + j] += w * v;
            }
        }
        tensor_free(tin);
        tensor_free(tout);
    }
    m->ready = 1;
}

void moe_backward(MoE *m, const double *dout, int T, double *dx)
{
    int t, e, k, j;
    double *c;
    if (!m->ready) hd_die("moe_backward", "必须先 forward");
    if (T != m->T) hd_die("moe_backward", "T 与 forward 不一致");
    if (!dout) hd_die("moe_backward", "dout 不能为空");
    if (dx) memset(dx, 0, (size_t)T * (size_t)m->d_model * sizeof(double));

    /* 1) 专家路径：dgroup[k] = w * dout_t，专家反向后把 dx 累加回 token */
    for (e = 0; e < m->n_experts; e++) {
        int len = m->group_len[e];
        Tensor *tdout, *tdx;
        int shape[2];
        if (len <= 0) continue;
        shape[0] = len;
        shape[1] = m->d_model;
        tdout = tensor_new(2, shape);
        tdx = tensor_new(2, shape);
        for (k = 0; k < len; k++) {
            int tt = m->group_idx[(size_t)e * m->max_seq + k];
            double w = m->group_w[(size_t)e * m->max_seq + k];
            for (j = 0; j < m->d_model; j++)
                tdout->data[(size_t)k * m->d_model + j] =
                    w * dout[(size_t)tt * m->d_model + j];
        }
        ff_backward((FeedForward *)m->experts[e], tdout, tdx);
        if (dx) {
            for (k = 0; k < len; k++) {
                int tt = m->group_idx[(size_t)e * m->max_seq + k];
                for (j = 0; j < m->d_model; j++)
                    dx[(size_t)tt * m->d_model + j] += tdx->data[(size_t)k * m->d_model + j];
            }
        }
        tensor_free(tdout);
        tensor_free(tdx);
    }

    /* 2) 门控路径：q 只在被选中的 top-k 上做 softmax，dz = q_k (c_k - base) */
    c = (double *)hd_malloc((size_t)m->top_k * sizeof(double));
    for (t = 0; t < T; t++) {
        const double *dt = dout + (size_t)t * m->d_model;
        const double *xt = m->x + (size_t)t * m->d_model;
        double base = 0.0;
        for (k = 0; k < m->top_k; k++) {
            int ex = m->route_e[(size_t)t * m->top_k + k];
            int pos = m->group_pos[(size_t)ex * m->max_seq + t];
            const double *gout = m->group_out +
                ((size_t)ex * m->max_seq + (size_t)pos) * m->d_model;
            c[k] = dot_neumaier(dt, gout, m->d_model);
            base += m->route_w[(size_t)t * m->top_k + k] * c[k];
        }
        for (k = 0; k < m->top_k; k++) {
            int ex = m->route_e[(size_t)t * m->top_k + k];
            double q = m->route_w[(size_t)t * m->top_k + k];
            double dz = q * (c[k] - base);
            double *gw = m->gW_gate + (size_t)ex * m->d_model;
            m->gb_gate[ex] += dz;
            for (j = 0; j < m->d_model; j++) {
                gw[j] += dz * xt[j];
                if (dx) dx[(size_t)t * m->d_model + j] += dz * m->W_gate[(size_t)ex * m->d_model + j];
            }
        }
    }
    free(c);
}
