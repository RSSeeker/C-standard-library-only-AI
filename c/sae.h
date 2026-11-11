/*
 * sae.h —— 稀疏自编码器（Sparse Autoencoder）
 *
 * 对应 Python 版 AI.py 的 SAE：
 *   编码 f = relu(W_enc x + b_enc)，解码 x_hat = W_dec f + b_dec
 *   损失 = 重建 MSE + l1_coeff * mean(|f|)
 *
 * 纯计算、无全局状态；权重用行主序扁平 double：
 *   W_enc: d_hidden x d_in
 *   W_dec: d_in x d_hidden
 */
#ifndef AI_SAE_H
#define AI_SAE_H

typedef struct {
    int d_in;
    int d_hidden;
    double l1_coeff;

    double *W_enc; /* d_hidden x d_in */
    double *b_enc; /* d_hidden */
    double *W_dec; /* d_in x d_hidden */
    double *b_dec; /* d_in */

    double *gW_enc;
    double *gb_enc;
    double *gW_dec;
    double *gb_dec;
} AiSAE;

/* 创建参数全 0 的 SAE（初始化请用 ai_sae_load_params 或调用方自行填值）。 */
AiSAE *ai_sae_create(int d_in, int d_hidden, double l1_coeff);
void ai_sae_free(AiSAE *sae);

/* 载入权重（数组长度按结构体注释）；返回 0 成功，-1 参数不合法。 */
int ai_sae_load_params(AiSAE *sae, const double *W_enc, const double *b_enc,
                       const double *W_dec, const double *b_dec);

void ai_sae_zero_grad(AiSAE *sae);

/* 单样本：features 长度 d_hidden，recon 长度 d_in。 */
void ai_sae_encode(const AiSAE *sae, const double *x, double *features);
void ai_sae_decode(const AiSAE *sae, const double *features, double *recon);

/* 批量前向：xs 为 n x d_in 行主序；recon n x d_in，features/pre n x d_hidden。 */
int ai_sae_forward(const AiSAE *sae, const double *xs, int n, double *recon,
                   double *features, double *pre);

/* 重建 MSE + L1 稀疏损失，并累积梯度；返回 0 成功，非 0 参数不合法。 */
int ai_sae_loss_and_backward(AiSAE *sae, const double *xs, int n,
                             double *out_loss, double *out_mse,
                             double *out_l1);

/* SGD 一步（更新后清零梯度）。 */
int ai_sae_sgd_step(AiSAE *sae, double lr);

/* 每个隐藏特征的激活比例；dead = 从未激活的特征数；mean_active 为平均激活数/样本。 */
int ai_sae_feature_stats(const AiSAE *sae, const double *xs, int n,
                         double *active_fraction, int *out_dead,
                         double *out_mean_active);

int ai_sae_param_count(const AiSAE *sae);

#endif /* AI_SAE_H */
