/*
 * kan.h —— Kolmogorov-Arnold Network（B 样条边函数）的 C 实现
 *
 * 与 Python 版 AI.py 的 KANLayer / KAN 数值一致：
 *   y_i = b_i + sum_j sum_b c[i][j][b] * B_b(x_j)
 * B_b 为均匀节点上的 Cox-de Boor B 样条基函数；反向按基函数值与
 * 导数精确累积梯度。参数不做随机初始化（由调用方 load 注入，
 * 便于与参考实现逐位对拍）。
 */
#ifndef AI_C_KAN_H
#define AI_C_KAN_H

typedef struct AiKANLayer {
    int in_dim;
    int out_dim;
    int num_grid;
    int spline_order;
    int num_basis;
    double grid_range;

    double *coeff;       /* out_dim * in_dim * num_basis */
    double *biases;      /* out_dim */
    double *grad_coeff;  /* 同 coeff；NULL 表示尚未反向 */
    double *grad_biases; /* out_dim 或 NULL */

    double *cache_basis; /* n * in_dim * num_basis（前向基函数值） */
    double *cache_ders;  /* 同形状（基函数导数） */
    int cache_n;
} AiKANLayer;

typedef struct AiKAN {
    int n_layers;
    int *sizes;              /* n_layers + 1 个维度 */
    AiKANLayer **layers;
} AiKAN;

#ifdef __cplusplus
extern "C" {
#endif

/* 创建 / 释放（参数全 0，由调用方注入）。成功返回 0，失败 -1。 */
int ai_kan_layer_create(AiKANLayer **out, int in_dim, int out_dim,
                        int num_grid, int spline_order, double grid_range);
void ai_kan_layer_free(AiKANLayer *layer);
int ai_kan_layer_param_count(const AiKANLayer *layer);

/* 前向：xs 为 n 行 in_dim 列（行主序），out 为 n 行 out_dim 列。 */
int ai_kan_layer_forward(AiKANLayer *layer, const double *xs, int n,
                         double *out);
/* 反向：dx 与 xs 同形；梯度累加到 grad_coeff/grad_biases。 */
int ai_kan_layer_backward(AiKANLayer *layer, const double *dout, int n,
                          double *dx);
void ai_kan_layer_zero_grad(AiKANLayer *layer);
int ai_kan_layer_sgd_step(AiKANLayer *layer, double lr);

int ai_kan_create(AiKAN **out, const int *sizes, int n_sizes,
                  int num_grid, int spline_order, double grid_range);
void ai_kan_free(AiKAN *kan);
int ai_kan_param_count(const AiKAN *kan);

int ai_kan_forward_batch(AiKAN *kan, const double *xs, int n, double *out);
int ai_kan_backward_batch(AiKAN *kan, const double *dout, int n, double *dx);
/* 等价于 Python KAN.loss_and_backward：MSE=sum(diff^2)/n。 */
int ai_kan_loss_and_backward(AiKAN *kan, const double *xs, const double *ys,
                             int n, double *out_loss, double *out_dx);
void ai_kan_zero_grad(AiKAN *kan);
int ai_kan_sgd_step(AiKAN *kan, double lr);

#ifdef __cplusplus
}
#endif

#endif /* AI_C_KAN_H */
