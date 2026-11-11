/*
 * train_utils.h —— 训练辅助工具：NF4 量化、Weight Standardization、
 * 动态 GradScaler、MoE 负载均衡损失
 *
 * 对应 Python 版 AI.py 的：
 *   - NF4_LEVELS / nf4_quantize / nf4_dequantize / nf4_fake_quantize /
 *     double_quantize / double_dequantize / nf4_fake_quantize_double
 *   - WeightStandardization
 *   - GradScaler
 *   - MoE.load_balancing_loss
 *
 * 全部接口为纯计算、不分配全局状态；数组一律用行主序扁平 double。
 */
#ifndef AI_TRAIN_UTILS_H
#define AI_TRAIN_UTILS_H

#define AI_NF4_LEVELS_COUNT 16

extern const double ai_nf4_levels[AI_NF4_LEVELS_COUNT];

/* ---------- NF4 量化 ---------- */

typedef struct {
    int *codes;       /* 每个元素的 0–15 级编号（长度 length） */
    double *scales;   /* 每个 block 的 absmax 缩放因子 */
    int block_size;
    int length;
} AiNf4Quant;

/* 成功时 codes/scales 由调用方用 ai_nf4_quant_free 释放；失败返回 length<=0 的空结构。 */
AiNf4Quant ai_nf4_quantize(const double *values, int n, int block_size);
void ai_nf4_quant_free(AiNf4Quant *q);

/* 还原为扁平浮点；返回 0 成功，-1 参数不合法。 */
int ai_nf4_dequantize(const AiNf4Quant *q, double *out);

/* 量化后立即反量化（形状不变，扁平）。 */
int ai_nf4_fake_quantize(const double *values, int n, int block_size, double *out);

/* 双重量化：对一阶缩放因子再做 8-bit 均匀量化。 */
int ai_double_quantize(const double *scales, int n, int block_size,
                       int *codes, double *outers);
int ai_double_dequantize(const int *codes, const double *outers, int n,
                         int block_size, double *out);

/* NF4 + 双重量化的伪量化。 */
int ai_nf4_fake_quantize_double(const double *values, int n, int block_size,
                                int outer_block_size, double *out);

/* ---------- Weight Standardization ---------- */

/* 计算标准化后的权重 ws 以及每行的均值 mu、标准差 sigma。 */
void ai_ws_effective(const double *w, int fan_out, int fan_in, double eps,
                     double *ws, double *mu, double *sigma);

/* 前向：y = ws·x + b，x 为 n 行 fan_in 列，y 为 n 行 fan_out 列。 */
void ai_ws_forward(const double *w, const double *b, const double *x,
                   int fan_out, int fan_in, int n, double eps, double *y,
                   double *ws, double *mu, double *sigma);

/* 反向：给定 dY（n 行 fan_out 列）计算 dW、db、dX。 */
void ai_ws_backward(const double *w, const double *x, const double *dy,
                    int fan_out, int fan_in, int n, double eps,
                    double *dw, double *db, double *dx);

/* ---------- 动态 GradScaler ---------- */

typedef struct {
    double scale;
    double growth_factor;
    double backoff_factor;
    int growth_interval;
    int growth_tracker;
    int found_inf;      /* 上一次 unscale 是否检测到 inf/nan */
    int steps_skipped;
    int steps_taken;
    int total_steps;
} AiGradScaler;

void ai_grad_scaler_init(AiGradScaler *s, double init_scale,
                         double growth_factor, double backoff_factor,
                         int growth_interval);

double ai_grad_scaler_scale_loss(AiGradScaler *s, double loss);

/* 把梯度除回 scale 并检测 inf/nan；返回 1 表示发现非有限值。 */
int ai_grad_scaler_unscale(AiGradScaler *s, double *grads, int n);

/* 更新 scale；found_inf_override 传 -1 表示使用 unscale 记录的状态。
 * 返回 1 表示本步可执行 optimizer.step()。 */
int ai_grad_scaler_update(AiGradScaler *s, int found_inf_override);

/* ---------- MoE 负载均衡损失 ---------- */

/* topk_idx 为 tokens×top_k 行主序专家编号；
 * gate_probs 为 tokens×experts 行主序的全专家门概率。 */
double ai_moe_load_balancing_loss(const int *topk_idx, int tokens, int top_k,
                                  const double *gate_probs, int experts,
                                  double coefficient);

#endif /* AI_TRAIN_UTILS_H */
