#ifndef GAN_VARIANTS_H
#define GAN_VARIANTS_H

/* GAN 变体族辅助函数（同步 Python AI.py 的纯数值实现）
 *
 * 所有图像都按通道主序展平：`k * H * W + i * W + j`。
 * 函数返回 0 表示成功，-1 表示参数非法（Python 侧对应 ValueError）。
 * 需要梯度的函数把梯度写入调用方提供的缓冲区；缓冲区为 NULL 时跳过。
 */

/* fade_in：out = (1-alpha)*upsample2x(low) + alpha*high
 * low: c*lh*lw；high/out: c*hh*hw，且 hh==2*lh、hw==2*lw。 */
int gv_fade_in(const double *low, const double *high, int c, int lh, int lw,
               double alpha, double *out);

/* progressive_alpha：min(1, max(0, step/total))，step/total 为非负整数。 */
int gv_progressive_alpha(int step, int total, double *alpha);

/* progressive_stages：把递增分辨率依次展开成 (low, high, alpha) 三元组。
 * 返回阶段数；out 容量不足或输入非法返回 -1。 */
int gv_progressive_stages(const int *resolutions, int n, int fade_steps,
                          int *low_out, int *high_out, double *alpha_out,
                          int cap);

/* adain_style：按通道标准化后用 style 的 (scale, bias) 调制。
 * features/out: c*h*w；style: 2*c（前半 scale、后半 bias）；
 * stats_mean/stats_std 可为 NULL。 */
int gv_adain_style(const double *features, int c, int h, int w,
                   const double *style, double eps,
                   double *out, double *stats_mean, double *stats_std);

/* truncation_trick：out = w_avg + psi*(w - w_avg)，psi 在 [0,1]。 */
int gv_truncation_trick(const double *w, const double *w_avg, int n,
                        double psi, double *out);

/* cycle_consistency_loss：lambda*(L1(a,a_rec) + L1(b,b_rec))；
 * 可选输出两侧 wrt 第一个图像的梯度：sign(a-a_rec)*lambda/n 与
 * sign(b-b_rec)*lambda/n，其中 n=c*h*w。 */
int gv_cycle_consistency_loss(const double *a, const double *a_rec,
                              const double *b, const double *b_rec,
                              int c, int h, int w, double lambda_cycle,
                              double *loss, double *grad_a, double *grad_b);

/* identity_mapping_loss：lambda*(L1(a,g_ba) + L1(b,g_ab))，不除元素数。 */
int gv_identity_mapping_loss(const double *a, const double *g_ba,
                             const double *b, const double *g_ab,
                             int c, int h, int w, double lambda_identity,
                             double *loss);

/* patchgan_targets：real 非 0 时填 1，否则填 0，网格 h*w。 */
int gv_patchgan_targets(int h, int w, int real, double *out);

/* pix2pix_g_loss：BCE(D(fake),1) + lambda_l1 * mean|fake-real|。
 * scores 是 ns 个裸 patch 分数；fake/real 为 c*h*w。 */
int gv_pix2pix_g_loss(const double *scores, int ns, const double *fake,
                      const double *real, int c, int h, int w,
                      double lambda_l1, double *loss);

/* R1/R2 罚项：0.5 * lam * mean(||g||^2)，grads 为 n_samples*grad_dim 的扁平数组。 */
double gv_r1_penalty(const double *grads, int n_samples, int grad_dim, double lam);
double gv_r2_penalty(const double *grads, int n_samples, int grad_dim, double lam);

#endif /* GAN_VARIANTS_H */
