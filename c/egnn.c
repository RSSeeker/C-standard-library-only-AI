/*
 * egnn.c —— E(3) 等变图神经网络（EGNN）的 C 实现
 *
 * 与 Python 版 AI.py 逐项对应：
 *   边特征 e_ij = phi_e([h_i, h_j, ||x_i - x_j||^2])
 *   边权重 w_ij = phi_x(e_ij)
 *   m_i = sum_{j != i} e_ij
 *   h_i' = h_i + phi_h([h_i, m_i])
 *   x_i' = x_i + sum_{j != i} w_ij * (x_i - x_j)
 *
 * 所有 MLP 用 mlp_forward / mlp_backward_with_input 调用，反向梯度
 * 累积到各层 gW/gb；更新用普通 SGD（与 Python 的 _e3_apply_sgd 一致）。
 */
#include "egnn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================ 小工具 ============================ */

static long mlp_param_count(const MLP *m)
{
    long total = 0;
    for (int i = 0; i < m->n_layers; i++) {
        const Layer *l = &m->layers[i];
        total += (long)l->fan_out * l->fan_in + l->fan_out;
    }
    return total;
}

static void mlp_sgd_step(MLP *m, double lr)
{
    for (int i = 0; i < m->n_layers; i++) {
        Layer *l = &m->layers[i];
        int nw = l->fan_out * l->fan_in;
        for (int k = 0; k < nw; k++) l->W[k] -= lr * l->gW[k];
        for (int j = 0; j < l->fan_out; j++) l->b[j] -= lr * l->gb[j];
    }
}

/* 预留 double 缓冲：需求小于当前容量时不动。返回 0 成功。 */
static int reserve_double(double **p, size_t *cap, size_t need)
{
    if (need <= *cap) return 0;
    double *q = (double *)realloc(*p, need * sizeof(double));
    if (!q) return -1;
    /* 新追加部分清零，避免读到未初始化内存 */
    memset(q + *cap, 0, (need - *cap) * sizeof(double));
    *p = q;
    *cap = need;
    return 0;
}

static int reserve_int(int **p, size_t *cap, size_t need)
{
    if (need <= *cap) return 0;
    int *q = (int *)realloc(*p, need * sizeof(int));
    if (!q) return -1;
    *p = q;
    *cap = need;
    return 0;
}

/* ============================ EGNNLayer ============================ */

struct EgnnLayer {
    int d_model;
    int d_ff;

    MLP *phi_e; /* [2d+1, d_ff, d_ff] relu */
    MLP *phi_x; /* [d_ff, 1] linear */
    MLP *phi_h; /* [d+d_ff, d_ff, d] relu */

    /* 前向缓存 */
    int n;
    int n_pairs;
    double *h;      /* n*d_model */
    double *x;      /* n*3 */
    int *pi, *pj;   /* n_pairs */
    double *d2;     /* n_pairs */
    double *edge;   /* n_pairs*(2d+1) */
    double *e_rows; /* n_pairs*d_ff */
    double *w_rows; /* n_pairs */
    double *m;      /* n*d_ff */
    double *node_in;/* n*(d+d_ff) */
    double *dh_rows;/* n*d_model */

    /* 反向临时 */
    double *de;     /* n_pairs*d_ff（phi_x 对边特征的输入梯度） */
    double *scratch;/* max(d+d_ff, 2d+1, d_ff) */

    size_t cap_h, cap_x, cap_pi, cap_pj, cap_d2, cap_edge, cap_e;
    size_t cap_w, cap_m, cap_node, cap_dh, cap_de, cap_scratch;
};

static void layer_free(EgnnLayer *l)
{
    if (!l) return;
    if (l->phi_e) mlp_free(l->phi_e);
    if (l->phi_x) mlp_free(l->phi_x);
    if (l->phi_h) mlp_free(l->phi_h);
    free(l->h); free(l->x); free(l->pi); free(l->pj); free(l->d2);
    free(l->edge); free(l->e_rows); free(l->w_rows); free(l->m);
    free(l->node_in); free(l->dh_rows); free(l->de); free(l->scratch);
    free(l);
}

static int reserve_layer(EgnnLayer *l, int n)
{
    int d = l->d_model, ff = l->d_ff;
    size_t pairs = (size_t)n * (size_t)(n - 1);
    size_t edge_dim = (size_t)(2 * d + 1);
    size_t scratch_dim = edge_dim;
    if ((size_t)(d + ff) > scratch_dim) scratch_dim = (size_t)(d + ff);
    if ((size_t)ff > scratch_dim) scratch_dim = (size_t)ff;

    if (reserve_double(&l->h, &l->cap_h, (size_t)n * d)) return -1;
    if (reserve_double(&l->x, &l->cap_x, (size_t)n * 3)) return -1;
    if (reserve_int(&l->pi, &l->cap_pi, pairs ? pairs : 1)) return -1;
    if (reserve_int(&l->pj, &l->cap_pj, pairs ? pairs : 1)) return -1;
    if (reserve_double(&l->d2, &l->cap_d2, pairs ? pairs : 1)) return -1;
    if (reserve_double(&l->edge, &l->cap_edge, (pairs ? pairs : 1) * edge_dim)) return -1;
    if (reserve_double(&l->e_rows, &l->cap_e, (pairs ? pairs : 1) * (size_t)ff)) return -1;
    if (reserve_double(&l->w_rows, &l->cap_w, pairs ? pairs : 1)) return -1;
    if (reserve_double(&l->m, &l->cap_m, (size_t)n * ff)) return -1;
    if (reserve_double(&l->node_in, &l->cap_node, (size_t)n * (size_t)(d + ff))) return -1;
    if (reserve_double(&l->dh_rows, &l->cap_dh, (size_t)n * d)) return -1;
    if (reserve_double(&l->de, &l->cap_de, (pairs ? pairs : 1) * (size_t)ff)) return -1;
    if (reserve_double(&l->scratch, &l->cap_scratch, scratch_dim)) return -1;

    l->n = n;
    l->n_pairs = (int)pairs;
    return 0;
}

EgnnLayer *egnn_layer_create(int d_model, int d_ff)
{
    if (d_model < 1) return NULL;
    if (d_ff <= 0) d_ff = 2 * d_model;

    EgnnLayer *l = (EgnnLayer *)calloc(1, sizeof(EgnnLayer));
    if (!l) return NULL;
    l->d_model = d_model;
    l->d_ff = d_ff;

    int e_sizes[3] = {2 * d_model + 1, d_ff, d_ff};
    Activation e_acts[2] = {ACT_RELU, ACT_RELU};
    int x_sizes[2] = {d_ff, 1};
    Activation x_acts[1] = {ACT_LINEAR};
    int h_sizes[3] = {d_model + d_ff, d_ff, d_model};
    Activation h_acts[2] = {ACT_RELU, ACT_RELU};

    l->phi_e = mlp_create(e_sizes, e_acts, 3);
    l->phi_x = mlp_create(x_sizes, x_acts, 2);
    l->phi_h = mlp_create(h_sizes, h_acts, 3);
    if (!l->phi_e || !l->phi_x || !l->phi_h) {
        layer_free(l);
        return NULL;
    }
    return l;
}

void egnn_layer_free(EgnnLayer *l) { layer_free(l); }

void egnn_layer_zero_grad(EgnnLayer *l)
{
    if (!l) return;
    mlp_zero_grad(l->phi_e);
    mlp_zero_grad(l->phi_x);
    mlp_zero_grad(l->phi_h);
}

void egnn_layer_sgd_step(EgnnLayer *l, double lr)
{
    if (!l) return;
    mlp_sgd_step(l->phi_e, lr);
    mlp_sgd_step(l->phi_x, lr);
    mlp_sgd_step(l->phi_h, lr);
}

long egnn_layer_param_count(const EgnnLayer *l)
{
    if (!l) return 0;
    return mlp_param_count(l->phi_e) + mlp_param_count(l->phi_x)
         + mlp_param_count(l->phi_h);
}

int egnn_layer_forward(EgnnLayer *l,
                       const double *h, const double *x, int n,
                       double *h_out, double *x_out)
{
    if (!l || !h || !x || !h_out || !x_out || n < 1) return -1;
    if (reserve_layer(l, n) != 0) return -1;

    const int d = l->d_model;
    const int ff = l->d_ff;
    const int edge_dim = 2 * d + 1;

    memcpy(l->h, h, (size_t)n * d * sizeof(double));
    memcpy(l->x, x, (size_t)n * 3 * sizeof(double));

    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            l->pi[idx] = i;
            l->pj[idx] = j;
            idx++;
        }
    }

    /* 边特征 + phi_e / phi_x */
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p], j = l->pj[p];
        double diff[3];
        double d2 = 0.0;
        for (int k = 0; k < 3; k++) {
            diff[k] = x[(size_t)i * 3 + k] - x[(size_t)j * 3 + k];
            d2 += diff[k] * diff[k];
        }
        l->d2[p] = d2;
        double *e = l->edge + (size_t)p * edge_dim;
        for (int k = 0; k < d; k++) {
            e[k] = h[(size_t)i * d + k];
            e[d + k] = h[(size_t)j * d + k];
        }
        e[2 * d] = d2;
        mlp_forward(l->phi_e, e, l->e_rows + (size_t)p * ff);
        mlp_forward(l->phi_x, l->e_rows + (size_t)p * ff, l->w_rows + p);
    }

    /* m_i = sum_j e_ij */
    memset(l->m, 0, (size_t)n * ff * sizeof(double));
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p];
        const double *e = l->e_rows + (size_t)p * ff;
        double *mi = l->m + (size_t)i * ff;
        for (int k = 0; k < ff; k++) mi[k] += e[k];
    }

    /* h' 与 node_in 缓存 */
    for (int i = 0; i < n; i++) {
        double *ni = l->node_in + (size_t)i * (d + ff);
        for (int k = 0; k < d; k++) ni[k] = h[(size_t)i * d + k];
        for (int k = 0; k < ff; k++) ni[d + k] = l->m[(size_t)i * ff + k];
        double *dhr = l->dh_rows + (size_t)i * d;
        mlp_forward(l->phi_h, ni, dhr);
        for (int k = 0; k < d; k++)
            h_out[(size_t)i * d + k] = h[(size_t)i * d + k] + dhr[k];
    }

    /* x' = x + sum_j w_ij (x_i - x_j) */
    memcpy(x_out, x, (size_t)n * 3 * sizeof(double));
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p], j = l->pj[p];
        double w = l->w_rows[p];
        for (int k = 0; k < 3; k++) {
            double diff = x[(size_t)i * 3 + k] - x[(size_t)j * 3 + k];
            x_out[(size_t)i * 3 + k] += w * diff;
        }
    }
    return 0;
}

int egnn_layer_backward(EgnnLayer *l,
                        const double *dh_new, const double *dx_new,
                        double *dh, double *dx)
{
    if (!l || !dh_new || !dx_new || !dh || !dx || l->n < 1) return -1;
    if (!l->h || !l->x || !l->de) return -1;

    const int d = l->d_model;
    const int ff = l->d_ff;
    const int n = l->n;
    const int edge_dim = 2 * d + 1;

    memcpy(dh, dh_new, (size_t)n * d * sizeof(double));
    memcpy(dx, dx_new, (size_t)n * 3 * sizeof(double));

    /* ---------- phi_h：node_in = [h_i, m_i] ---------- */
    memset(l->m, 0, (size_t)n * ff * sizeof(double));
    for (int i = 0; i < n; i++) {
        /* 逐行重跑前向，恢复该行对应的中间缓存（MLP 内部只保留最后一次前向） */
        mlp_forward(l->phi_h, l->node_in + (size_t)i * (d + ff), l->scratch);
        mlp_backward_with_input(l->phi_h, dh_new + (size_t)i * d, l->scratch);
        for (int k = 0; k < d; k++) dh[(size_t)i * d + k] += l->scratch[k];
        for (int k = 0; k < ff; k++) l->m[(size_t)i * ff + k] += l->scratch[d + k];
    }

    /* ---------- phi_x：w_ij 梯度 + 坐标梯度 ---------- */
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p], j = l->pj[p];
        double d_w = 0.0;
        for (int k = 0; k < 3; k++) {
            double diff = l->x[(size_t)i * 3 + k] - l->x[(size_t)j * 3 + k];
            d_w += diff * dx_new[(size_t)i * 3 + k];
        }
        mlp_forward(l->phi_x, l->e_rows + (size_t)p * ff, l->scratch);
        mlp_backward_with_input(l->phi_x, &d_w, l->scratch);
        memcpy(l->de + (size_t)p * ff, l->scratch, (size_t)ff * sizeof(double));

        double w = l->w_rows[p];
        for (int k = 0; k < 3; k++) {
            double g = dx_new[(size_t)i * 3 + k];
            dx[(size_t)i * 3 + k] += w * g;
            dx[(size_t)j * 3 + k] -= w * g;
        }
    }

    /* de_ij = dm_i（m_i = sum_j e_ij）+ phi_x 的输入梯度 */
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p];
        double *de = l->de + (size_t)p * ff;
        const double *dm = l->m + (size_t)i * ff;
        for (int k = 0; k < ff; k++) de[k] += dm[k];
    }

    /* ---------- phi_e：edge = [h_i, h_j, d2] ---------- */
    for (int p = 0; p < l->n_pairs; p++) {
        int i = l->pi[p], j = l->pj[p];
        mlp_forward(l->phi_e, l->edge + (size_t)p * edge_dim, l->scratch);
        mlp_backward_with_input(l->phi_e, l->de + (size_t)p * ff, l->scratch);
        for (int k = 0; k < d; k++) {
            dh[(size_t)i * d + k] += l->scratch[k];
            dh[(size_t)j * d + k] += l->scratch[d + k];
        }
        double dd2 = l->scratch[2 * d];
        for (int k = 0; k < 3; k++) {
            double diff = l->x[(size_t)i * 3 + k] - l->x[(size_t)j * 3 + k];
            dx[(size_t)i * 3 + k] += 2.0 * dd2 * diff;
            dx[(size_t)j * 3 + k] -= 2.0 * dd2 * diff;
        }
    }
    return 0;
}

MLP *egnn_layer_phi_e(EgnnLayer *l) { return l ? l->phi_e : NULL; }
MLP *egnn_layer_phi_x(EgnnLayer *l) { return l ? l->phi_x : NULL; }
MLP *egnn_layer_phi_h(EgnnLayer *l) { return l ? l->phi_h : NULL; }

/* ============================ EGNNModel ============================ */

struct EgnnModel {
    int d_model;
    int d_ff;
    int num_layers;
    EgnnLayer **layers;
    MLP *readout;

    int n;
    double *h;    /* n*d_model 最终层缓存 */
    double *x;    /* n*3 */
    double *mean; /* d_model */

    double *h_a, *h_b;   /* n*d_model 乒乓 */
    double *x_a, *x_b;   /* n*3 乒乓 */
    double *dh_a, *dh_b; /* 反向 dh 乒乓 */
    double *dx_a, *dx_b; /* 反向 dx 乒乓 */
    double *scratch;     /* n*d_model（反向 d_mean 展开） */
    size_t cap_h, cap_x, cap_mean;
    size_t cap_h_a, cap_h_b, cap_x_a, cap_x_b;
    size_t cap_dh_a, cap_dh_b, cap_dx_a, cap_dx_b;
    size_t cap_scratch;
};

static void model_free(EgnnModel *m)
{
    if (!m) return;
    if (m->layers) {
        for (int i = 0; i < m->num_layers; i++) egnn_layer_free(m->layers[i]);
        free(m->layers);
    }
    if (m->readout) mlp_free(m->readout);
    free(m->h); free(m->x); free(m->mean);
    free(m->h_a); free(m->h_b); free(m->x_a); free(m->x_b);
    free(m->dh_a); free(m->dh_b); free(m->dx_a); free(m->dx_b);
    free(m->scratch);
    free(m);
}

EgnnModel *egnn_model_create(int d_model, int num_layers, int d_ff)
{
    if (d_model < 1 || num_layers < 1) return NULL;
    if (d_ff <= 0) d_ff = 2 * d_model;

    EgnnModel *m = (EgnnModel *)calloc(1, sizeof(EgnnModel));
    if (!m) return NULL;
    m->d_model = d_model;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->layers = (EgnnLayer **)calloc((size_t)num_layers, sizeof(EgnnLayer *));
    if (!m->layers) { model_free(m); return NULL; }
    for (int i = 0; i < num_layers; i++) {
        m->layers[i] = egnn_layer_create(d_model, d_ff);
        if (!m->layers[i]) { model_free(m); return NULL; }
    }
    int r_sizes[3] = {d_model, d_ff, 1};
    Activation r_acts[2] = {ACT_RELU, ACT_LINEAR};
    m->readout = mlp_create(r_sizes, r_acts, 3);
    if (!m->readout) { model_free(m); return NULL; }
    return m;
}

void egnn_model_free(EgnnModel *m) { model_free(m); }

void egnn_model_zero_grad(EgnnModel *m)
{
    if (!m) return;
    for (int i = 0; i < m->num_layers; i++) egnn_layer_zero_grad(m->layers[i]);
    mlp_zero_grad(m->readout);
}

void egnn_model_sgd_step(EgnnModel *m, double lr)
{
    if (!m) return;
    for (int i = 0; i < m->num_layers; i++) egnn_layer_sgd_step(m->layers[i], lr);
    mlp_sgd_step(m->readout, lr);
}

long egnn_model_param_count(const EgnnModel *m)
{
    if (!m) return 0;
    long total = mlp_param_count(m->readout);
    for (int i = 0; i < m->num_layers; i++)
        total += egnn_layer_param_count(m->layers[i]);
    return total;
}

static int reserve_model(EgnnModel *m, int n)
{
    int d = m->d_model;
    if (reserve_double(&m->h, &m->cap_h, (size_t)n * d)) return -1;
    if (reserve_double(&m->x, &m->cap_x, (size_t)n * 3)) return -1;
    if (reserve_double(&m->mean, &m->cap_mean, (size_t)d)) return -1;
    if (reserve_double(&m->h_a, &m->cap_h_a, (size_t)n * d)) return -1;
    if (reserve_double(&m->h_b, &m->cap_h_b, (size_t)n * d)) return -1;
    if (reserve_double(&m->x_a, &m->cap_x_a, (size_t)n * 3)) return -1;
    if (reserve_double(&m->x_b, &m->cap_x_b, (size_t)n * 3)) return -1;
    if (reserve_double(&m->dh_a, &m->cap_dh_a, (size_t)n * d)) return -1;
    if (reserve_double(&m->dh_b, &m->cap_dh_b, (size_t)n * d)) return -1;
    if (reserve_double(&m->dx_a, &m->cap_dx_a, (size_t)n * 3)) return -1;
    if (reserve_double(&m->dx_b, &m->cap_dx_b, (size_t)n * 3)) return -1;
    if (reserve_double(&m->scratch, &m->cap_scratch, (size_t)d)) return -1;
    m->n = n;
    return 0;
}

int egnn_model_forward(EgnnModel *m,
                       const double *h, const double *x, int n,
                       double *h_out, double *x_out, double *scalar)
{
    if (!m || !h || !x || !h_out || !x_out || !scalar || n < 1) return -1;
    if (reserve_model(m, n) != 0) return -1;

    const int d = m->d_model;
    double *h_cur = m->h_a, *h_nxt = m->h_b;
    double *x_cur = m->x_a, *x_nxt = m->x_b;
    memcpy(h_cur, h, (size_t)n * d * sizeof(double));
    memcpy(x_cur, x, (size_t)n * 3 * sizeof(double));

    for (int li = 0; li < m->num_layers; li++) {
        if (egnn_layer_forward(m->layers[li], h_cur, x_cur, n, h_nxt, x_nxt) != 0)
            return -1;
        double *th = h_cur; h_cur = h_nxt; h_nxt = th;
        double *tx = x_cur; x_cur = x_nxt; x_nxt = tx;
    }
    memcpy(m->h, h_cur, (size_t)n * d * sizeof(double));
    memcpy(m->x, x_cur, (size_t)n * 3 * sizeof(double));
    memcpy(h_out, h_cur, (size_t)n * d * sizeof(double));
    memcpy(x_out, x_cur, (size_t)n * 3 * sizeof(double));

    for (int k = 0; k < d; k++) {
        double s = 0.0;
        for (int i = 0; i < n; i++) s += h_cur[(size_t)i * d + k];
        m->mean[k] = s / (double)n;
    }
    double ro = 0.0;
    mlp_forward(m->readout, m->mean, &ro);
    *scalar = ro;
    return 0;
}

double egnn_model_loss_and_backward(EgnnModel *m,
                                    const double *h, const double *x, int n,
                                    double target)
{
    if (!m || !h || !x || n < 1) return NAN;

    double scalar = 0.0;
    /* forward 需要调用方提供输出缓冲；训练/对拍频率低，临时分配即可。 */
    double *h_out = (double *)malloc((size_t)n * m->d_model * sizeof(double));
    double *x_out = (double *)malloc((size_t)n * 3 * sizeof(double));
    if (!h_out || !x_out) { free(h_out); free(x_out); return NAN; }
    int rc = egnn_model_forward(m, h, x, n, h_out, x_out, &scalar);
    free(h_out);
    free(x_out);
    if (rc != 0) return NAN;

    double diff = scalar - target;
    double loss = diff * diff;

    double d_out = 2.0 * diff;
    mlp_backward_with_input(m->readout, &d_out, m->scratch);

    for (int i = 0; i < n; i++)
        for (int k = 0; k < m->d_model; k++)
            m->dh_a[(size_t)i * m->d_model + k] = m->scratch[k] / (double)n;

    memset(m->dx_a, 0, (size_t)n * 3 * sizeof(double));

    double *dh_cur = m->dh_a, *dh_nxt = m->dh_b;
    double *dx_cur = m->dx_a, *dx_nxt = m->dx_b;
    for (int li = m->num_layers - 1; li >= 0; li--) {
        if (egnn_layer_backward(m->layers[li], dh_cur, dx_cur, dh_nxt, dx_nxt) != 0)
            return NAN;
        double *th = dh_cur; dh_cur = dh_nxt; dh_nxt = th;
        double *tx = dx_cur; dx_cur = dx_nxt; dx_nxt = tx;
    }
    return loss;
}

EgnnLayer *egnn_model_layer_mut(EgnnModel *m, int idx)
{
    if (!m || idx < 0 || idx >= m->num_layers) return NULL;
    return m->layers[idx];
}

MLP *egnn_model_readout_mut(EgnnModel *m) { return m ? m->readout : NULL; }

int egnn_model_d_model(const EgnnModel *m) { return m ? m->d_model : 0; }
int egnn_model_num_layers(const EgnnModel *m) { return m ? m->num_layers : 0; }
