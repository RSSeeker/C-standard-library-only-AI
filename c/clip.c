/*
 * clip.c —— CLIP 双塔：对称 InfoNCE / 余弦相似度 / 检索准确率 / 线性投影头
 *
 * 对应 Python AI.py 的 clip_similarity / clip_loss /
 * contrastive_retrieval_accuracy / CLIPProjector。
 */
#include "clip.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cm_rows_ok(const double *x, int n, int d)
{
    if (!x || n <= 0 || d <= 0) return 0;
    for (int i = 0; i < n * d; i++) {
        if (!isfinite(x[i])) return 0;
    }
    return 1;
}

/* 行归一化；范数 <= 0 返回 -1。out 可为 NULL（只取范数）。 */
static int cm_unit_row(const double *row, int d, double *out, double *norm_out)
{
    double s = 0.0;
    for (int k = 0; k < d; k++) s += row[k] * row[k];
    double n = sqrt(s);
    if (!(n > 0.0)) return -1;
    if (out) {
        for (int k = 0; k < d; k++) out[k] = row[k] / n;
    }
    if (norm_out) *norm_out = n;
    return 0;
}

static void cm_softmax_row(double *row, int n)
{
    if (n <= 0) return;
    double mx = row[0];
    for (int i = 1; i < n; i++) {
        if (row[i] > mx) mx = row[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        row[i] = exp(row[i] - mx);
        sum += row[i];
    }
    if (sum > 0.0) {
        for (int i = 0; i < n; i++) row[i] /= sum;
    }
}

int cm_clip_similarity(const double *imgs, int n, int d,
                       const double *txts, int n_txt, int d2,
                       double *sim)
{
    if (!cm_rows_ok(imgs, n, d) || !cm_rows_ok(txts, n_txt, d2) || !sim) return -1;
    if (d != d2) return -1;
    double *ui = (double *)malloc((size_t)n * d * sizeof(double));
    double *ut = (double *)malloc((size_t)n_txt * d * sizeof(double));
    if (!ui || !ut) {
        free(ui);
        free(ut);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (cm_unit_row(imgs + (size_t)i * d, d, ui + (size_t)i * d, NULL) != 0) {
            free(ui);
            free(ut);
            return -1;
        }
    }
    for (int j = 0; j < n_txt; j++) {
        if (cm_unit_row(txts + (size_t)j * d, d, ut + (size_t)j * d, NULL) != 0) {
            free(ui);
            free(ut);
            return -1;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_txt; j++) {
            double s = 0.0;
            for (int k = 0; k < d; k++) {
                s += ui[(size_t)i * d + k] * ut[(size_t)j * d + k];
            }
            sim[(size_t)i * n_txt + j] = s;
        }
    }
    free(ui);
    free(ut);
    return 0;
}

double cm_clip_loss(const double *imgs, int n, int d,
                    const double *txts, int d2,
                    double temperature, double *g_img, double *g_txt)
{
    if (!cm_rows_ok(imgs, n, d) || !cm_rows_ok(txts, n, d2)) return NAN;
    if (d != d2 || !(temperature > 0.0)) return NAN;

    const double tau = temperature;
    double *ui = (double *)malloc((size_t)n * d * sizeof(double));
    double *ut = (double *)malloc((size_t)n * d * sizeof(double));
    double *ni = (double *)malloc((size_t)n * sizeof(double));
    double *nt = (double *)malloc((size_t)n * sizeof(double));
    double *s = (double *)malloc((size_t)n * n * sizeof(double));
    double *p = (double *)malloc((size_t)n * n * sizeof(double));
    double *q = (double *)malloc((size_t)n * n * sizeof(double));
    double *ds = (double *)malloc((size_t)n * n * sizeof(double));
    double *dui = (double *)malloc((size_t)n * d * sizeof(double));
    double *dut = (double *)malloc((size_t)n * d * sizeof(double));
    int ok = ui && ut && ni && nt && s && p && q && ds && dui && dut;
    double loss = NAN;
    if (!ok) goto done;

    for (int i = 0; i < n; i++) {
        if (cm_unit_row(imgs + (size_t)i * d, d, ui + (size_t)i * d, ni + i) != 0)
            goto done;
        if (cm_unit_row(txts + (size_t)i * d, d, ut + (size_t)i * d, nt + i) != 0)
            goto done;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++) {
                dot += ui[(size_t)i * d + k] * ut[(size_t)j * d + k];
            }
            s[(size_t)i * n + j] = dot / tau;
        }
    }
    memcpy(p, s, (size_t)n * n * sizeof(double));
    memcpy(q, s, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) cm_softmax_row(p + (size_t)i * n, n);
    /* 列 softmax：先取转置行，再归一化写回 q 的列。 */
    for (int j = 0; j < n; j++) {
        double mx = q[j];
        for (int i = 1; i < n; i++) {
            double v = q[(size_t)i * n + j];
            if (v > mx) mx = v;
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double v = exp(q[(size_t)i * n + j] - mx);
            q[(size_t)i * n + j] = v;
            sum += v;
        }
        if (sum > 0.0) {
            for (int i = 0; i < n; i++) q[(size_t)i * n + j] /= sum;
        }
    }
    loss = 0.0;
    for (int i = 0; i < n; i++) {
        double pii = p[(size_t)i * n + i];
        if (pii < 1e-300) pii = 1e-300;
        loss += -log(pii);
    }
    for (int j = 0; j < n; j++) {
        double qjj = q[(size_t)j * n + j];
        if (qjj < 1e-300) qjj = 1e-300;
        loss += -log(qjj);
    }
    loss /= (2.0 * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double eye = (i == j) ? 1.0 : 0.0;
            ds[(size_t)i * n + j] =
                0.5 * ((p[(size_t)i * n + j] - eye) / n +
                       (q[(size_t)i * n + j] - eye) / n);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += ds[(size_t)i * n + j] * ut[(size_t)j * d + k];
            }
            dui[(size_t)i * d + k] = acc / tau;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < d; k++) {
            double acc = 0.0;
            for (int i = 0; i < n; i++) {
                acc += ds[(size_t)i * n + j] * ui[(size_t)i * d + k];
            }
            dut[(size_t)j * d + k] = acc / tau;
        }
    }
    if (g_img) {
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++) {
                dot += dui[(size_t)i * d + k] * ui[(size_t)i * d + k];
            }
            for (int k = 0; k < d; k++) {
                g_img[(size_t)i * d + k] =
                    (dui[(size_t)i * d + k] - ui[(size_t)i * d + k] * dot) / ni[i];
            }
        }
    }
    if (g_txt) {
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++) {
                dot += dut[(size_t)j * d + k] * ut[(size_t)j * d + k];
            }
            for (int k = 0; k < d; k++) {
                g_txt[(size_t)j * d + k] =
                    (dut[(size_t)j * d + k] - ut[(size_t)j * d + k] * dot) / nt[j];
            }
        }
    }
done:
    free(ui);
    free(ut);
    free(ni);
    free(nt);
    free(s);
    free(p);
    free(q);
    free(ds);
    free(dui);
    free(dut);
    return loss;
}

int cm_clip_retrieval_accuracy(const double *imgs, int n, int d,
                               const double *txts, int n_txt, int d2,
                               double *i2t, double *t2i)
{
    if (n != n_txt) return -1;
    double *sim = (double *)malloc((size_t)n * n * sizeof(double));
    if (!sim) return -1;
    if (cm_clip_similarity(imgs, n, d, txts, n_txt, d2, sim) != 0) {
        free(sim);
        return -1;
    }
    int hit_i = 0, hit_t = 0;
    for (int i = 0; i < n; i++) {
        int best = 0;
        for (int j = 1; j < n; j++) {
            if (sim[(size_t)i * n + j] > sim[(size_t)i * n + best]) best = j;
        }
        if (best == i) hit_i++;
    }
    for (int j = 0; j < n; j++) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (sim[(size_t)i * n + j] > sim[(size_t)best * n + j]) best = i;
        }
        if (best == j) hit_t++;
    }
    if (i2t) *i2t = (double)hit_i / n;
    if (t2i) *t2i = (double)hit_t / n;
    free(sim);
    return 0;
}

/* ---------------- 线性投影头 ---------------- */

ClipProjector *cm_projector_create(int image_dim, int text_dim, int embed_dim,
                                   double temperature)
{
    if (image_dim < 1 || text_dim < 1 || embed_dim < 1) return NULL;
    if (!(temperature > 0.0)) return NULL;
    ClipProjector *p = (ClipProjector *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->image_dim = image_dim;
    p->text_dim = text_dim;
    p->embed_dim = embed_dim;
    p->temperature = temperature;
    p->W_img = (double *)calloc((size_t)embed_dim * image_dim, sizeof(double));
    p->b_img = (double *)calloc((size_t)embed_dim, sizeof(double));
    p->W_txt = (double *)calloc((size_t)embed_dim * text_dim, sizeof(double));
    p->b_txt = (double *)calloc((size_t)embed_dim, sizeof(double));
    if (!p->W_img || !p->b_img || !p->W_txt || !p->b_txt) {
        cm_projector_free(p);
        return NULL;
    }
    return p;
}

void cm_projector_free(ClipProjector *p)
{
    if (!p) return;
    free(p->W_img);
    free(p->b_img);
    free(p->W_txt);
    free(p->b_txt);
    free(p);
}

int cm_projector_param_count(const ClipProjector *p)
{
    if (!p) return -1;
    return p->embed_dim * p->image_dim + p->embed_dim +
           p->embed_dim * p->text_dim + p->embed_dim;
}

int cm_projector_set_weights(ClipProjector *p,
                             const double *w_img, const double *b_img,
                             const double *w_txt, const double *b_txt)
{
    if (!p || !w_img || !b_img || !w_txt || !b_txt) return -1;
    memcpy(p->W_img, w_img,
           (size_t)p->embed_dim * p->image_dim * sizeof(double));
    memcpy(p->b_img, b_img, (size_t)p->embed_dim * sizeof(double));
    memcpy(p->W_txt, w_txt,
           (size_t)p->embed_dim * p->text_dim * sizeof(double));
    memcpy(p->b_txt, b_txt, (size_t)p->embed_dim * sizeof(double));
    return 0;
}

static int cm_project_rows(const ClipProjector *p,
                           const double *x, int n, int dim, int is_img,
                           double *out)
{
    const double *W = is_img ? p->W_img : p->W_txt;
    const double *b = is_img ? p->b_img : p->b_txt;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p->embed_dim; j++) {
            double acc = b[j];
            for (int k = 0; k < dim; k++) {
                acc += W[(size_t)j * dim + k] * x[(size_t)i * dim + k];
            }
            out[(size_t)i * p->embed_dim + j] = acc;
        }
    }
    return 0;
}

int cm_projector_project(ClipProjector *p,
                         const double *imgs, int n_imgs,
                         const double *txts, int n_txts,
                         double *img_emb, double *txt_emb)
{
    if (!p || !imgs || !txts || !img_emb || !txt_emb) return -1;
    if (!cm_rows_ok(imgs, n_imgs, p->image_dim) ||
        !cm_rows_ok(txts, n_txts, p->text_dim))
        return -1;
    if (cm_project_rows(p, imgs, n_imgs, p->image_dim, 1, img_emb) != 0) return -1;
    if (cm_project_rows(p, txts, n_txts, p->text_dim, 0, txt_emb) != 0) return -1;
    p->img_x = imgs;
    p->img_n = n_imgs;
    p->txt_x = txts;
    p->txt_n = n_txts;
    return 0;
}

double cm_projector_loss(ClipProjector *p,
                         const double *imgs, int n, const double *txts)
{
    if (!p || n <= 0) return NAN;
    int e = p->embed_dim;
    double *ie = (double *)malloc((size_t)n * e * sizeof(double));
    double *te = (double *)malloc((size_t)n * e * sizeof(double));
    double loss = NAN;
    if (!ie || !te) goto done;
    if (cm_projector_project(p, imgs, n, txts, n, ie, te) != 0) goto done;
    loss = cm_clip_loss(ie, n, e, te, e, p->temperature, NULL, NULL);
done:
    free(ie);
    free(te);
    return loss;
}

int cm_projector_loss_and_grads(ClipProjector *p,
                                const double *imgs, int n,
                                const double *txts,
                                double *gW_img, double *gb_img,
                                double *gW_txt, double *gb_txt,
                                double *loss)
{
    if (!p || n <= 0) return -1;
    int e = p->embed_dim;
    double *ie = (double *)malloc((size_t)n * e * sizeof(double));
    double *te = (double *)malloc((size_t)n * e * sizeof(double));
    double *gi = (double *)malloc((size_t)n * e * sizeof(double));
    double *gt = (double *)malloc((size_t)n * e * sizeof(double));
    int ret = -1;
    if (!ie || !te || !gi || !gt) goto done;
    if (cm_projector_project(p, imgs, n, txts, n, ie, te) != 0) goto done;
    double l = cm_clip_loss(ie, n, e, te, e, p->temperature, gi, gt);
    if (isnan(l)) goto done;
    memset(gW_img, 0, (size_t)e * p->image_dim * sizeof(double));
    memset(gb_img, 0, (size_t)e * sizeof(double));
    memset(gW_txt, 0, (size_t)e * p->text_dim * sizeof(double));
    memset(gb_txt, 0, (size_t)e * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < e; j++) {
            double gj = gi[(size_t)i * e + j];
            gb_img[j] += gj;
            for (int k = 0; k < p->image_dim; k++) {
                gW_img[(size_t)j * p->image_dim + k] +=
                    gj * imgs[(size_t)i * p->image_dim + k];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < e; j++) {
            double gj = gt[(size_t)i * e + j];
            gb_txt[j] += gj;
            for (int k = 0; k < p->text_dim; k++) {
                gW_txt[(size_t)j * p->text_dim + k] +=
                    gj * txts[(size_t)i * p->text_dim + k];
            }
        }
    }
    if (loss) *loss = l;
    ret = 0;
done:
    free(ie);
    free(te);
    free(gi);
    free(gt);
    return ret;
}

int cm_projector_apply_sgd(ClipProjector *p, double lr,
                           const double *gW_img, const double *gb_img,
                           const double *gW_txt, const double *gb_txt)
{
    if (!p || !(lr > 0.0) || !gW_img || !gb_img || !gW_txt || !gb_txt) return -1;
    for (int i = 0; i < p->embed_dim * p->image_dim; i++) {
        p->W_img[i] -= lr * gW_img[i];
    }
    for (int i = 0; i < p->embed_dim; i++) p->b_img[i] -= lr * gb_img[i];
    for (int i = 0; i < p->embed_dim * p->text_dim; i++) {
        p->W_txt[i] -= lr * gW_txt[i];
    }
    for (int i = 0; i < p->embed_dim; i++) p->b_txt[i] -= lr * gb_txt[i];
    return 0;
}
