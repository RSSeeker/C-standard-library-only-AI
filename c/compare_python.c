/*
 * compare_python.c —— 读取 tools/c_reference.py 导出的参考数据，
 * 用 C 实现做同样的计算，逐项比对数值差异。
 *
 * 用法:  compare_python <reference.txt>
 * 退出码: 0 = 全部在容差内，1 = 存在超差项
 */
#include "../c/ai.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static double tol = 1e-12;

typedef struct {
    int n_layers;
    int *sizes;
    int *acts;
    double *W, *b;
    int n_samples, in_dim, out_dim, steps;
    double lr;
    double *X, *Y, *out_ref;
    double *gradW_ref, *gradb_ref;
    double *W_after_ref, *b_after_ref;
    int nW, nb;
} Ref;

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "oom\n"); exit(2); }
    return p;
}

static int expect_tag(FILE *f, const char *tag)
{
    char buf[64];
    if (fscanf(f, "%63s", buf) != 1) { fprintf(stderr, "读取 %s 失败(EOF)\n", tag); exit(2); }
    if (strcmp(buf, tag) != 0) { fprintf(stderr, "标签不匹配: 期望 %s 实得 %s\n", tag, buf); exit(2); }
    return 0;
}

/* 读 double 必须用 %s + strtod，不能用 fscanf("%lf")：
 * msvcrt 的 scanf("%lf") 对某些 17 位十进制串会解析错 1 ULP
 * （实测 "-0.01863766757260517" -> ...168，而 strtod / Python 的 float() 都是 ...172），
 * 而 reference.txt 正是 Python 用 repr() 写出的最短往返表示 ——
 * 用 fscanf 读会产生"幻影差异"（之前被 1e-12 的容差盖住了）。
 * 注意 -D__USE_MINGW_ANSI_STDIO=1 不能修这个问题。 */
static double read_one_double(FILE *f)
{
    char buf[64];
    if (fscanf(f, "%63s", buf) != 1) { fprintf(stderr, "读 double 失败\n"); exit(2); }
    return strtod(buf, NULL);
}

static void read_doubles(FILE *f, double *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = read_one_double(f);
}

static void read_ints(FILE *f, int *dst, int n)
{
    for (int i = 0; i < n; i++)
        if (fscanf(f, "%d", &dst[i]) != 1) { fprintf(stderr, "读 int 失败\n"); exit(2); }
}

static void load_ref(const char *path, Ref *r)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "打不开 %s\n", path); exit(2); }
    expect_tag(f, "AI_REF");
    int version; if (fscanf(f, "%d", &version) != 1) exit(2);

    expect_tag(f, "n_layers"); if (fscanf(f, "%d", &r->n_layers) != 1) exit(2);
    r->sizes = xmalloc((size_t)(r->n_layers + 1) * sizeof(int));
    r->acts  = xmalloc((size_t)r->n_layers * sizeof(int));
    expect_tag(f, "sizes"); read_ints(f, r->sizes, r->n_layers + 1);
    expect_tag(f, "acts");  read_ints(f, r->acts, r->n_layers);

    r->nW = 0; r->nb = 0;
    for (int i = 0; i < r->n_layers; i++) {
        r->nW += r->sizes[i] * r->sizes[i + 1];
        r->nb += r->sizes[i + 1];
    }
    r->W = xmalloc((size_t)r->nW * sizeof(double));
    r->b = xmalloc((size_t)r->nb * sizeof(double));
    expect_tag(f, "W"); read_doubles(f, r->W, r->nW);
    expect_tag(f, "b"); read_doubles(f, r->b, r->nb);

    expect_tag(f, "n_samples"); if (fscanf(f, "%d", &r->n_samples) != 1) exit(2);
    expect_tag(f, "in_dim");    if (fscanf(f, "%d", &r->in_dim) != 1) exit(2);
    expect_tag(f, "out_dim");   if (fscanf(f, "%d", &r->out_dim) != 1) exit(2);
    expect_tag(f, "steps");     if (fscanf(f, "%d", &r->steps) != 1) exit(2);
    expect_tag(f, "lr");        r->lr = read_one_double(f);

    int nX = r->n_samples * r->in_dim, nY = r->n_samples * r->out_dim;
    r->X = xmalloc((size_t)nX * sizeof(double));
    r->Y = xmalloc((size_t)nY * sizeof(double));
    r->out_ref = xmalloc((size_t)nY * sizeof(double));
    r->gradW_ref = xmalloc((size_t)r->nW * sizeof(double));
    r->gradb_ref = xmalloc((size_t)r->nb * sizeof(double));
    r->W_after_ref = xmalloc((size_t)r->nW * sizeof(double));
    r->b_after_ref = xmalloc((size_t)r->nb * sizeof(double));

    expect_tag(f, "X");        read_doubles(f, r->X, nX);
    expect_tag(f, "Y");        read_doubles(f, r->Y, nY);
    expect_tag(f, "out");      read_doubles(f, r->out_ref, nY);
    expect_tag(f, "gradW");    read_doubles(f, r->gradW_ref, r->nW);
    expect_tag(f, "gradb");    read_doubles(f, r->gradb_ref, r->nb);
    expect_tag(f, "W_after");  read_doubles(f, r->W_after_ref, r->nW);
    expect_tag(f, "b_after");  read_doubles(f, r->b_after_ref, r->nb);
    fclose(f);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    double worst = 0.0;
    int at = -1;
    for (int i = 0; i < n; i++) {
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    int ok = worst <= tol;
    if (!ok) failures++;
    printf("  %-28s 最大差异 %.3e  %s", what, worst,
           ok ? "[OK]" : "[超出容差]");
    if (!ok) printf("  (index %d: C=%.17g py=%.17g)", at, got[at], want[at]);
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "reference.txt";
    if (argc > 2) tol = atof(argv[2]);

    Ref r;
    memset(&r, 0, sizeof(r));
    load_ref(path, &r);
    printf("参考数据: %s\n", path);
    printf("模型 %d 层, 尺寸", r.n_layers);
    for (int i = 0; i <= r.n_layers; i++) printf(" %d", r.sizes[i]);
    printf(" ; %d 样本; %d 步 Adam(lr=%.3g)\n\n", r.n_samples, r.steps, r.lr);

    /* ---------- 1. 用同一份权重建模型 ---------- */
    Activation *acts = xmalloc((size_t)r.n_layers * sizeof(Activation));
    for (int i = 0; i < r.n_layers; i++) acts[i] = (Activation)r.acts[i];
    MLP *m = mlp_create(r.sizes, acts, r.n_layers + 1);

    int off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(l->W, r.W + off, (size_t)l->fan_out * l->fan_in * sizeof(double));
        off += l->fan_out * l->fan_in;
    }
    off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(l->b, r.b + off, (size_t)l->fan_out * sizeof(double));
        off += l->fan_out;
    }

    printf("1) 前向传播（%d 个样本）\n", r.n_samples);
    double *out = xmalloc((size_t)r.n_samples * r.out_dim * sizeof(double));
    for (int i = 0; i < r.n_samples; i++)
        mlp_forward(m, r.X + (size_t)i * r.in_dim, out + (size_t)i * r.out_dim);
    report("forward output", out, r.out_ref, r.n_samples * r.out_dim);

    printf("\n2) 梯度累加（%d 个样本，未除以 N）\n", r.n_samples);
    mlp_zero_grad(m);
    for (int i = 0; i < r.n_samples; i++) {
        double *pred = xmalloc((size_t)r.out_dim * sizeof(double));
        double *grad = xmalloc((size_t)r.out_dim * sizeof(double));
        mlp_forward(m, r.X + (size_t)i * r.in_dim, pred);
        mse_grad(pred, r.Y + (size_t)i * r.out_dim, r.out_dim, grad);
        mlp_backward(m, grad);
        free(pred); free(grad);
    }
    double *gw = xmalloc((size_t)r.nW * sizeof(double));
    double *gb = xmalloc((size_t)r.nb * sizeof(double));
    off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(gw + off, l->gW, (size_t)l->fan_out * l->fan_in * sizeof(double));
        off += l->fan_out * l->fan_in;
    }
    off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(gb + off, l->gb, (size_t)l->fan_out * sizeof(double));
        off += l->fan_out;
    }
    report("accumulated grad W", gw, r.gradW_ref, r.nW);
    report("accumulated grad b", gb, r.gradb_ref, r.nb);

    printf("\n3) 训练 %d 步全批量 Adam(lr=%.3g)\n", r.steps, r.lr);
    Optimizer opt;
    optimizer_init(&opt, OPT_ADAM, r.lr);
    for (int step = 0; step < r.steps; step++) {
        mlp_zero_grad(m);
        for (int i = 0; i < r.n_samples; i++) {
            double *pred = xmalloc((size_t)r.out_dim * sizeof(double));
            double *grad = xmalloc((size_t)r.out_dim * sizeof(double));
            mlp_forward(m, r.X + (size_t)i * r.in_dim, pred);
            mse_grad(pred, r.Y + (size_t)i * r.out_dim, r.out_dim, grad);
            mlp_backward(m, grad);
            free(pred); free(grad);
        }
        /* 平均梯度：直接按 batch 大小缩放 */
        double inv = 1.0 / (double)r.n_samples;
        for (int i = 0; i < r.n_layers; i++) {
            Layer *l = &m->layers[i];
            for (int k = 0; k < l->fan_out * l->fan_in; k++) l->gW[k] *= inv;
            for (int j = 0; j < l->fan_out; j++) l->gb[j] *= inv;
        }
        optimizer_step(m, &opt);
    }
    double *W_after = xmalloc((size_t)r.nW * sizeof(double));
    double *b_after = xmalloc((size_t)r.nb * sizeof(double));
    off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(W_after + off, l->W, (size_t)l->fan_out * l->fan_in * sizeof(double));
        off += l->fan_out * l->fan_in;
    }
    off = 0;
    for (int i = 0; i < r.n_layers; i++) {
        Layer *l = &m->layers[i];
        memcpy(b_after + off, l->b, (size_t)l->fan_out * sizeof(double));
        off += l->fan_out;
    }
    report("weights after training", W_after, r.W_after_ref, r.nW);
    report("biases  after training", b_after, r.b_after_ref, r.nb);

    printf("\n结论: %s（容差 %.1e）\n", failures ? "存在超差项" : "C 与 Python 数值一致", tol);

    mlp_free(m);
    free(acts); free(out); free(gw); free(gb); free(W_after); free(b_after);
    free(r.sizes); free(r.acts); free(r.W); free(r.b); free(r.X); free(r.Y);
    free(r.out_ref); free(r.gradW_ref); free(r.gradb_ref);
    free(r.W_after_ref); free(r.b_after_ref);
    return failures ? 1 : 0;
}
