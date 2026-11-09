/* flowmatching.h —— Flow Matching / Rectified Flow（速度场 + Euler/Heun 采样）
 *
 * 与 Python AI.py 的 FlowMatching 对应：直线插值 x_t=(1-t)x0+t·noise，
 * 目标速度 v*=noise-x0，用三层 MLP（relu/relu/linear）拟合速度场。
 * C 侧只保留确定性的前向/反向与采样入口（RNG 由调用方提供初值，避免跨语言
 * RNG 差异）。时间特征与 Python _flow_time_features 同构。
 */
#ifndef FLOWMATCHING_H
#define FLOWMATCHING_H

#include "ai.h"

typedef struct FlowMatching {
    int data_dim;
    int hidden_dim;
    int t_dim;          /* 偶数（奇数输入会 +1，与 Python 一致） */
    MLP *net;           /* [data_dim+t_dim, hidden, hidden, data_dim] */
    int step_count;
} FlowMatching;

FlowMatching *flowmatching_create(int data_dim, int hidden_dim, int t_dim);
void flowmatching_free(FlowMatching *m);

/* 时间特征：sin(pi*2^i*t), cos(pi*2^i*t)，共 t_dim 个。 */
void flowmatching_time_features(double t, int t_dim, double *out);

/* 直线插值 / 目标速度（out 长度 data_dim）。 */
void flowmatching_interpolate(const FlowMatching *m, const double *x0,
                              const double *noise, double t, double *out);
void flowmatching_target(const FlowMatching *m, const double *x0,
                         const double *noise, double *out);

/* 速度场前向（缓存供反向使用）。 */
void flowmatching_velocity(FlowMatching *m, const double *x_t, double t, double *out);

/* 一步训练：显式给定 t 与 noise（不采样），累积梯度并返回 MSE。 */
double flowmatching_train_step(FlowMatching *m, const double *x0, double t,
                               const double *noise);

void flowmatching_zero_grad(FlowMatching *m);
/* 对累积梯度做 SGD 更新，然后清零梯度缓冲。 */
void flowmatching_sgd_step(FlowMatching *m, double lr);

/* 从给定初始点 x_init 积分到 t=0：solver=0 Euler、非 0 Heun。 */
void flowmatching_sample(FlowMatching *m, const double *x_init, int steps,
                         int solver, double *out);

int flowmatching_param_count(const FlowMatching *m);

#endif /* FLOWMATCHING_H */
