/*
 * egnn.h —— E(3) 等变图神经网络（EGNN）的 C 移植
 *
 * 与 Python 版 AI.py 的对应关系：
 *   EGNNLayer -> EgnnLayer：phi_e / phi_x / phi_h 三张 MLP
 *   EGNNModel -> EgnnModel：N 层 + 均值池化 readout（标量输出）
 *
 * 数据约定：
 *   h: n × d_model 行主序；x: n × 3 行主序（每个节点的三维坐标）。
 *   前向返回 h'（n × d_model）、x'（n × 3），模型整体输出标量。
 *
 * 数值对齐：所有前向/反向公式与 Python 版逐项对应，反向使用
 * mlp_backward_with_input 取每张 MLP 的输入梯度。
 */
#ifndef EGNN_H
#define EGNN_H

#include "ai.h"

typedef struct EgnnLayer EgnnLayer;

/* d_model >= 1；d_ff <= 0 时取 2*d_model。参数非法返回 NULL。 */
EgnnLayer *egnn_layer_create(int d_model, int d_ff);
void egnn_layer_free(EgnnLayer *l);
void egnn_layer_zero_grad(EgnnLayer *l);
void egnn_layer_sgd_step(EgnnLayer *l, double lr);
long egnn_layer_param_count(const EgnnLayer *l);

/* 返回 0 成功，-1 失败（参数/内存）。h_out、x_out 由调用方分配。 */
int egnn_layer_forward(EgnnLayer *l,
                       const double *h, const double *x, int n,
                       double *h_out, double *x_out);

/* 从 (dh_new, dx_new) 反传，输出 (dh, dx)；梯度累积到三张 MLP。
 * 调用前需 egnn_layer_zero_grad（或在 batch 内累积）。 */
int egnn_layer_backward(EgnnLayer *l,
                        const double *dh_new, const double *dx_new,
                        double *dh, double *dx);

/* 访问器（供对拍脚本载入权重） */
MLP *egnn_layer_phi_e(EgnnLayer *l);
MLP *egnn_layer_phi_x(EgnnLayer *l);
MLP *egnn_layer_phi_h(EgnnLayer *l);

typedef struct EgnnModel EgnnModel;

/* d_model >= 1；num_layers >= 1；d_ff <= 0 时取 2*d_model。 */
EgnnModel *egnn_model_create(int d_model, int num_layers, int d_ff);
void egnn_model_free(EgnnModel *m);
void egnn_model_zero_grad(EgnnModel *m);
void egnn_model_sgd_step(EgnnModel *m, double lr);
long egnn_model_param_count(const EgnnModel *m);

/* 返回 0 成功，-1 失败。scalar 为标量读出。 */
int egnn_model_forward(EgnnModel *m,
                       const double *h, const double *x, int n,
                       double *h_out, double *x_out, double *scalar);

/* MSE(scalar, target) 并反向，返回损失。 */
double egnn_model_loss_and_backward(EgnnModel *m,
                                    const double *h, const double *x, int n,
                                    double target);

/* 访问器（供对拍脚本载入权重） */
EgnnLayer *egnn_model_layer_mut(EgnnModel *m, int idx);
MLP *egnn_model_readout_mut(EgnnModel *m);
int egnn_model_d_model(const EgnnModel *m);
int egnn_model_num_layers(const EgnnModel *m);

#endif /* EGNN_H */
