#ifndef PROGRESSIVE_H
#define PROGRESSIVE_H

#include "ai.h"
#include "layers_conv.h"
#include "tensor.h"

/* ProGAN 风格逐分辨率增长生成器（前向推理，与 Python AI.ProgressiveGenerator 对齐）
 *
 * 初始分辨率：z -> fc(relu) -> (c0, base, base) -> 1x1 RGB 头；
 * 此后每次 grow：ConvTranspose2d(k=4,s=2,p=1) -> ReLU -> 1x1 RGB 头。
 * `alpha<1` 时只用 `fade_in(上一分辨率输出, 当前分辨率输出, alpha)` 混合。
 */
typedef struct {
    int latent_dim;
    int base;
    int out_channels;
    int n_stages;
    int *channels;

    MLP *fc;                    /* [latent_dim, channels[0]*base*base], relu */
    ConvTranspose2d **convs;    /* n_stages-1 个 */
    Conv2d **rgb_heads;         /* n_stages 个 */

    double *fc_out;             /* channels[0]*base*base */
} ProgressiveGenerator;

ProgressiveGenerator *pg_create(int latent_dim, const int *channels, int n_stages,
                                int base, int out_channels);
void pg_free(ProgressiveGenerator *g);

/* 输出形状：out_channels, side, side（side = base * 2^(n_stages-1)） */
void pg_out_shape(const ProgressiveGenerator *g, int *c, int *side);

/* 追加一个分辨率块（分辨率翻倍，通道数沿用最后一阶）；返回新的 side。 */
int pg_grow(ProgressiveGenerator *g);

int pg_param_count(const ProgressiveGenerator *g);

/* z 长度 latent_dim；alpha∈[0,1]；out 长度 out_channels*side*side。 */
int pg_forward(ProgressiveGenerator *g, const double *z, double alpha, double *out);

#endif /* PROGRESSIVE_H */
