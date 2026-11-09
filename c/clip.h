#ifndef CLIP_H
#define CLIP_H

/* ======================================================================
 *  CLIP 双塔：对称 InfoNCE 损失 / 余弦相似度矩阵 / 检索准确率 / 线性投影头
 *
 *  对应 Python AI.py 的 clip_similarity / clip_loss /
 *  contrastive_retrieval_accuracy / CLIPProjector。
 *
 *  数据约定：嵌入按行主序展平，row i 的第 k 维在 i*dim + k。
 * ========================================================================== */

/* 余弦相似度矩阵：imgs (n x d)、txts (n_txt x d)，结果写入 sim (n x n_txt)。
 * 全零向量返回 -1。 */
int cm_clip_similarity(const double *imgs, int n, int d,
                       const double *txts, int n_txt, int d2,
                       double *sim);

/* 对称 InfoNCE：返回损失；g_img / g_txt 可传 NULL。
 * 约定第 i 张图配第 i 段文本（n_img 必须等于 n_txt）。 */
double cm_clip_loss(const double *imgs, int n, int d,
                    const double *txts, int d2,
                    double temperature, double *g_img, double *g_txt);

/* 图文检索准确率：i2t / t2i 写入指针（可 NULL）。返回 0 成功。 */
int cm_clip_retrieval_accuracy(const double *imgs, int n, int d,
                               const double *txts, int n_txt, int d2,
                               double *i2t, double *t2i);

/* 线性投影头：img(W_img,b_img) -> embed、txt(W_txt,b_txt) -> embed。 */
typedef struct {
    int image_dim;
    int text_dim;
    int embed_dim;
    double temperature;
    double *W_img;  /* embed_dim x image_dim，行主序 */
    double *b_img;
    double *W_txt;  /* embed_dim x text_dim，行主序 */
    double *b_txt;
    /* 最近一次 project 的输入缓存（供 loss_and_grads 复用） */
    const double *img_x;
    int img_n;
    const double *txt_x;
    int txt_n;
} ClipProjector;

ClipProjector *cm_projector_create(int image_dim, int text_dim, int embed_dim,
                                   double temperature);
void cm_projector_free(ClipProjector *p);
int cm_projector_param_count(const ClipProjector *p);

/* 从外部灌入权重（跨语言对拍）：W_img 长度 embed*image_dim，W_txt 长度 embed*text_dim。 */
int cm_projector_set_weights(ClipProjector *p,
                             const double *w_img, const double *b_img,
                             const double *w_txt, const double *b_txt);

/* 投影：img_emb (n x embed)、txt_emb (n_txt x embed)，按行主序写出。 */
int cm_projector_project(ClipProjector *p,
                         const double *imgs, int n_imgs,
                         const double *txts, int n_txts,
                         double *img_emb, double *txt_emb);

/* 投影后算 clip_loss，只返回损失。 */
double cm_projector_loss(ClipProjector *p,
                         const double *imgs, int n, const double *txts);

/* 投影响应 + clip_loss + 解析梯度。grads 可传 NULL（只算损失）。 */
int cm_projector_loss_and_grads(ClipProjector *p,
                                const double *imgs, int n,
                                const double *txts,
                                double *gW_img, double *gb_img,
                                double *gW_txt, double *gb_txt,
                                double *loss);

/* 按 SGD 应用一次梯度。 */
int cm_projector_apply_sgd(ClipProjector *p, double lr,
                           const double *gW_img, const double *gb_img,
                           const double *gW_txt, const double *gb_txt);

#endif /* CLIP_H */
