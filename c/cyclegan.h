#ifndef CYCLEGAN_H
#define CYCLEGAN_H

#include "pix2pix.h"

/* ======================================================================
 *  CycleGAN 训练步（双向循环一致性 + identity 约束）
 *
 *  对应 Python AI.py 的 train_cyclegan_step：两个生成器 g_ab / g_ba 与两个
 *  判别器 d_a / d_b（复用 Pix2Pix 的 PatchDiscriminator / Pix2PixGenerator）。
 *
 *  一步流程：
 *    1) 判别器阶段：D_A 对 real_a 目标 1、对 fake_a 目标 0；
 *                   D_B 对 real_b 目标 1、对 fake_b 目标 0；各自 SGD 更新。
 *    2) 生成器阶段：对抗（D_A(fake_a)、D_B(fake_b) 目标 1）
 *                 + 循环一致（G_BA(G_AB(a)) ≈ a、G_AB(G_BA(b)) ≈ b）
 *                 + identity（G_AB(b) ≈ b、G_BA(a) ≈ a）；G_AB / G_BA SGD 更新。
 *
 *  像素约定：通道主序，image[k*H*W + r*W + c]。
 * ========================================================================== */

typedef struct {
    double d_loss;
    double g_loss;
    double adv_loss;
    double cycle_loss;
    double identity_loss;
} CycleGanStep;

/* 一步训练：先 D 后 G。成功返回 0。 */
int cyclegan_train_step(Pix2PixGenerator *g_ab, Pix2PixGenerator *g_ba,
                        PatchDiscriminator *d_a, PatchDiscriminator *d_b,
                        const double *real_a, const double *real_b,
                        int channels, int height, int width,
                        double lambda_cycle, double lambda_identity,
                        double lr_g, double lr_d,
                        CycleGanStep *out);

#endif /* CYCLEGAN_H */
