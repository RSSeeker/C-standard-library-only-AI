/*
 * compare_clip.c —— CLIP 双塔对拍：投影头 / 损失 / 梯度 / 相似度 / 检索
 *
 * 读取 tools/clip_reference.py 生成的 JSON，与 C 实现逐项比较。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../clip.h"
#include "../json.h"

static int g_checks = 0;
static int g_fails = 0;
static int g_exact = 0;
static double g_maxdiff = 0.0;
static const char *g_func = "";

static void fail(const char *what, double expected, double got)
{
    g_fails++;
    g_checks++;
    printf("FAIL [%s] %s: expected %.17g got %.17g\n", g_func, what, expected, got);
}

static void check_num(const char *what, double expected, double got, double tol)
{
    double diff = fabs(expected - got);
    if (diff > g_maxdiff) g_maxdiff = diff;
    if (diff <= tol) {
        g_exact++;
        g_checks++;
        return;
    }
    fail(what, expected, got);
}

static double *read_nums(const JsonValue *v, int *n_out)
{
    if (!v || v->kind != JSON_ARRAY) {
        if (n_out) *n_out = 0;
        return NULL;
    }
    int n = json_array_len(v);
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!out) {
        if (n_out) *n_out = 0;
        return NULL;
    }
    for (int i = 0; i < n; i++) out[i] = json_num(json_at(v, i), NAN);
    if (n_out) *n_out = n;
    return out;
}

static void check_array(const char *what, const JsonValue *expected, int n,
                        const double *got, double tol)
{
    double *exp = read_nums(expected, NULL);
    if (!exp) {
        fail(what, 1.0, 0.0);
        return;
    }
    for (int i = 0; i < n; i++) {
        char label[160];
        snprintf(label, sizeof(label), "%s[%d]", what, i);
        check_num(label, exp[i], got[i], tol);
    }
    free(exp);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "clip_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "parse %s failed: %s\n", path, err);
        return 2;
    }

    const JsonValue *model = json_get(root, "model");
    int image_dim = json_int(json_get(model, "image_dim"), 0);
    int text_dim = json_int(json_get(model, "text_dim"), 0);
    int embed_dim = json_int(json_get(model, "embed_dim"), 0);
    double temperature = json_num(json_get(model, "temperature"), 0.0);
    const JsonValue *weights = json_get(root, "weights");
    int nW_img = 0, nW_txt = 0;
    double *W_img = read_nums(json_get(weights, "W_img"), &nW_img);
    double *b_img = read_nums(json_get(weights, "b_img"), NULL);
    double *W_txt = read_nums(json_get(weights, "W_txt"), &nW_txt);
    double *b_txt = read_nums(json_get(weights, "b_txt"), NULL);
    if (!W_img || !b_img || !W_txt || !b_txt ||
        nW_img != embed_dim * image_dim || nW_txt != embed_dim * text_dim) {
        fprintf(stderr, "bad weights in %s\n", path);
        return 2;
    }

    ClipProjector *p = cm_projector_create(image_dim, text_dim, embed_dim, temperature);
    if (!p) {
        fprintf(stderr, "create failed\n");
        return 2;
    }
    if (cm_projector_set_weights(p, W_img, b_img, W_txt, b_txt) != 0) {
        fprintf(stderr, "set weights failed\n");
        return 2;
    }

    g_func = "meta";
    check_num("param_count", (double)json_int(json_get(root, "param_count"), 0),
              (double)cm_projector_param_count(p), 0.0);

    const JsonValue *inputs = json_get(root, "inputs");
    int n_img = 0, n_txt = 0;
    double *image_feats = read_nums(json_get(inputs, "image_feats"), &n_img);
    double *text_feats = read_nums(json_get(inputs, "text_feats"), &n_txt);
    int rows_img = n_img / image_dim;
    int rows_txt = n_txt / text_dim;

    g_func = "project";
    double *img_emb = (double *)malloc((size_t)rows_img * embed_dim * sizeof(double));
    double *txt_emb = (double *)malloc((size_t)rows_txt * embed_dim * sizeof(double));
    if (cm_projector_project(p, image_feats, rows_img, text_feats, rows_txt,
                             img_emb, txt_emb) != 0) {
        fprintf(stderr, "project failed\n");
        return 2;
    }
    const JsonValue *proj = json_get(root, "project");
    check_array("img_emb", json_get(proj, "img_emb"), rows_img * embed_dim, img_emb, 1e-12);
    check_array("txt_emb", json_get(proj, "txt_emb"), rows_txt * embed_dim, txt_emb, 1e-12);

    g_func = "loss";
    check_num("loss", json_num(json_get(root, "loss"), NAN),
              cm_projector_loss(p, image_feats, rows_img, text_feats), 1e-12);

    g_func = "grads";
    double *gW_img = (double *)calloc((size_t)embed_dim * image_dim, sizeof(double));
    double *gb_img = (double *)calloc((size_t)embed_dim, sizeof(double));
    double *gW_txt = (double *)calloc((size_t)embed_dim * text_dim, sizeof(double));
    double *gb_txt = (double *)calloc((size_t)embed_dim, sizeof(double));
    double loss2 = NAN;
    if (cm_projector_loss_and_grads(p, image_feats, rows_img, text_feats,
                                    gW_img, gb_img, gW_txt, gb_txt, &loss2) != 0) {
        fprintf(stderr, "loss_and_grads failed\n");
        return 2;
    }
    check_num("loss2", json_num(json_get(root, "loss2"), NAN), loss2, 1e-12);
    const JsonValue *grads = json_get(root, "grads");
    check_array("gW_img", json_get(grads, "W_img"), embed_dim * image_dim, gW_img, 1e-12);
    check_array("gb_img", json_get(grads, "b_img"), embed_dim, gb_img, 1e-12);
    check_array("gW_txt", json_get(grads, "W_txt"), embed_dim * text_dim, gW_txt, 1e-12);
    check_array("gb_txt", json_get(grads, "b_txt"), embed_dim, gb_txt, 1e-12);

    g_func = "apply_sgd";
    double lr = json_num(json_get(root, "lr"), 0.0);
    if (cm_projector_apply_sgd(p, lr, gW_img, gb_img, gW_txt, gb_txt) != 0) {
        fprintf(stderr, "apply_sgd failed\n");
        return 2;
    }
    const JsonValue *after = json_get(root, "after");
    check_array("after_W_img", json_get(after, "W_img"), embed_dim * image_dim, p->W_img, 1e-12);
    check_array("after_b_img", json_get(after, "b_img"), embed_dim, p->b_img, 1e-12);
    check_array("after_W_txt", json_get(after, "W_txt"), embed_dim * text_dim, p->W_txt, 1e-12);
    check_array("after_b_txt", json_get(after, "b_txt"), embed_dim, p->b_txt, 1e-12);

    const JsonValue *sim = json_get(root, "similarity");
    double *sim_imgs = read_nums(json_get(sim, "imgs"), NULL);
    double *sim_txts = read_nums(json_get(sim, "txts"), NULL);
    int n_img_rows = json_int(json_get(sim, "n_img"), 0);
    int n_txt_rows = json_int(json_get(sim, "n_txt"), 0);
    int d_sim = json_int(json_get(sim, "d_img"), 0);
    double *got_sim = (double *)malloc((size_t)n_img_rows * n_txt_rows * sizeof(double));
    g_func = "similarity";
    if (cm_clip_similarity(sim_imgs, n_img_rows, d_sim, sim_txts, n_txt_rows, d_sim,
                           got_sim) != 0) {
        fprintf(stderr, "similarity failed\n");
        return 2;
    }
    check_array("sim", json_get(sim, "sim"), n_img_rows * n_txt_rows, got_sim, 1e-12);

    const JsonValue *cl = json_get(root, "clip_loss");
    int cl_n = 0;
    double *cl_imgs = read_nums(json_get(cl, "imgs"), NULL);
    double *cl_txts = read_nums(json_get(cl, "txts"), NULL);
    int d_cl = json_int(json_get(cl, "d"), 0);
    int n_cl = n_img_rows;
    double *cl_gimg = (double *)calloc((size_t)n_cl * d_cl, sizeof(double));
    double *cl_gtxt = (double *)calloc((size_t)n_cl * d_cl, sizeof(double));
    g_func = "clip_loss";
    double cl_loss = cm_clip_loss(cl_imgs, n_cl, d_cl, cl_txts, d_cl,
                                  json_num(json_get(root, "temperature"), 0.07),
                                  cl_gimg, cl_gtxt);
    check_num("loss", json_num(json_get(cl, "loss"), NAN), cl_loss, 1e-12);
    check_array("g_img", json_get(cl, "g_img"), n_cl * d_cl, cl_gimg, 1e-12);
    check_array("g_txt", json_get(cl, "g_txt"), n_cl * d_cl, cl_gtxt, 1e-12);

    g_func = "retrieval";
    const JsonValue *ret = json_get(root, "retrieval");
    double i2t = 0.0, t2i = 0.0;
    if (cm_clip_retrieval_accuracy(cl_imgs, n_cl, d_cl, cl_txts, n_cl, d_cl,
                                   &i2t, &t2i) != 0) {
        fprintf(stderr, "retrieval failed\n");
        return 2;
    }
    check_num("i2t", json_num(json_get(ret, "i2t"), NAN), i2t, 1e-15);
    check_num("t2i", json_num(json_get(ret, "t2i"), NAN), t2i, 1e-15);

    printf("CHECK %d: exact %d, fail %d, maxdiff %.3e\n",
           g_checks, g_exact, g_fails, g_maxdiff);
    cm_projector_free(p);
    free(W_img);
    free(b_img);
    free(W_txt);
    free(b_txt);
    free(image_feats);
    free(text_feats);
    free(img_emb);
    free(txt_emb);
    free(gW_img);
    free(gb_img);
    free(gW_txt);
    free(gb_txt);
    free(sim_imgs);
    free(sim_txts);
    free(got_sim);
    free(cl_imgs);
    free(cl_txts);
    free(cl_gimg);
    free(cl_gtxt);
    json_free(root);
    return g_fails == 0 ? 0 : 1;
}
