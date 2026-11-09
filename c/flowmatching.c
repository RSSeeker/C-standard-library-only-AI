/* flowmatching.c —— Flow Matching / Rectified Flow（速度场 + Euler/Heun 采样） */
#include "flowmatching.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FlowMatching *flowmatching_create(int data_dim, int hidden_dim, int t_dim)
{
    if (data_dim < 1 || hidden_dim < 1 || t_dim < 1) return NULL;
    int td = (t_dim % 2 == 0) ? t_dim : t_dim + 1;
    FlowMatching *m = (FlowMatching *)calloc(1, sizeof(FlowMatching));
    if (!m) return NULL;
    m->data_dim = data_dim;
    m->hidden_dim = hidden_dim;
    m->t_dim = td;
    int sizes[4] = {data_dim + td, hidden_dim, hidden_dim, data_dim};
    Activation acts[3] = {ACT_RELU, ACT_RELU, ACT_LINEAR};
    m->net = mlp_create(sizes, acts, 4);
    if (!m->net) {
        free(m);
        return NULL;
    }
    return m;
}

void flowmatching_free(FlowMatching *m)
{
    if (!m) return;
    if (m->net) mlp_free(m->net);
    free(m);
}

void flowmatching_time_features(double t, int t_dim, double *out)
{
    for (int i = 0; i < t_dim / 2; i++) {
        double f = pow(2.0, (double)i);
        out[2 * i] = sin(M_PI * f * t);
        out[2 * i + 1] = cos(M_PI * f * t);
    }
}

void flowmatching_interpolate(const FlowMatching *m, const double *x0,
                              const double *noise, double t, double *out)
{
    for (int i = 0; i < m->data_dim; i++)
        out[i] = (1.0 - t) * x0[i] + t * noise[i];
}

void flowmatching_target(const FlowMatching *m, const double *x0,
                         const double *noise, double *out)
{
    for (int i = 0; i < m->data_dim; i++) out[i] = noise[i] - x0[i];
}

void flowmatching_velocity(FlowMatching *m, const double *x_t, double t, double *out)
{
    int dim = m->data_dim + m->t_dim;
    double *feat = (double *)malloc((size_t)dim * sizeof(double));
    if (!feat) return;
    memcpy(feat, x_t, (size_t)m->data_dim * sizeof(double));
    flowmatching_time_features(t, m->t_dim, feat + m->data_dim);
    mlp_forward(m->net, feat, out);
    free(feat);
}

double flowmatching_train_step(FlowMatching *m, const double *x0, double t,
                               const double *noise)
{
    int n = m->data_dim;
    double *x_t = (double *)calloc((size_t)n, sizeof(double));
    double *pred = (double *)malloc((size_t)n * sizeof(double));
    double *tgt = (double *)malloc((size_t)n * sizeof(double));
    double *d_pred = (double *)malloc((size_t)n * sizeof(double));
    if (!x_t || !pred || !tgt || !d_pred) {
        free(x_t); free(pred); free(tgt); free(d_pred);
        return 0.0;
    }
    flowmatching_interpolate(m, x0, noise, t, x_t);
    flowmatching_velocity(m, x_t, t, pred);
    flowmatching_target(m, x0, noise, tgt);
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double d = pred[i] - tgt[i];
        loss += d * d;
        d_pred[i] = 2.0 * d / (double)n;
    }
    loss /= (double)n;
    mlp_backward(m->net, d_pred);
    free(x_t); free(pred); free(tgt); free(d_pred);
    return loss;
}

void flowmatching_zero_grad(FlowMatching *m)
{
    if (!m || !m->net) return;
    for (int i = 0; i < m->net->n_layers; i++) {
        Layer *l = &m->net->layers[i];
        memset(l->gW, 0, (size_t)(l->fan_in * l->fan_out) * sizeof(double));
        memset(l->gb, 0, (size_t)l->fan_out * sizeof(double));
    }
}

void flowmatching_sgd_step(FlowMatching *m, double lr)
{
    if (!m || !m->net) return;
    for (int i = 0; i < m->net->n_layers; i++) {
        Layer *l = &m->net->layers[i];
        for (int r = 0; r < l->fan_out; r++) {
            for (int c = 0; c < l->fan_in; c++) {
                size_t idx = (size_t)r * l->fan_in + c;
                l->W[idx] -= lr * l->gW[idx];
            }
            l->b[r] -= lr * l->gb[r];
        }
    }
    flowmatching_zero_grad(m);
}

void flowmatching_sample(FlowMatching *m, const double *x_init, int steps,
                         int solver, double *out)
{
    int n = m->data_dim;
    if (steps < 1) return;
    double dt = 1.0 / (double)steps;
    double *x = (double *)calloc((size_t)n, sizeof(double));
    double *v1 = (double *)calloc((size_t)n, sizeof(double));
    double *mid = (double *)calloc((size_t)n, sizeof(double));
    double *v2 = (double *)calloc((size_t)n, sizeof(double));
    if (!x || !v1 || !mid || !v2) {
        free(x); free(v1); free(mid); free(v2);
        return;
    }
    memcpy(x, x_init, (size_t)n * sizeof(double));
    for (int k = 0; k < steps; k++) {
        double t = 1.0 - (double)k * dt;
        flowmatching_velocity(m, x, t, v1);
        if (solver == 0) {
            for (int i = 0; i < n; i++) x[i] -= dt * v1[i];
        } else {
            for (int i = 0; i < n; i++) mid[i] = x[i] - dt * v1[i];
            double t_next = t - dt;
            if (t_next < 0.0) t_next = 0.0;
            flowmatching_velocity(m, mid, t_next, v2);
            for (int i = 0; i < n; i++)
                x[i] -= 0.5 * dt * (v1[i] + v2[i]);
        }
    }
    memcpy(out, x, (size_t)n * sizeof(double));
    free(x); free(v1); free(mid); free(v2);
}

int flowmatching_param_count(const FlowMatching *m)
{
    if (!m || !m->net) return 0;
    int total = 0;
    for (int i = 0; i < m->net->n_layers; i++) {
        total += m->net->layers[i].fan_in * m->net->layers[i].fan_out
                 + m->net->layers[i].fan_out;
    }
    return total;
}
