/*
 * compare_conv.c —— 读取 tools/conv_reference.py 的参考数据，
 * 用 C 实现重算 Conv2d / MaxPool2d / Flatten 并逐项比对（默认要求逐位相同）。
 *
 * 用法: compare_conv [conv_reference.json] [容差]
 */
#include "../json.h"
#include "../layers_conv.h"
#include "../tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static double g_tol = 0.0;

static const JsonValue *need(const JsonValue *obj, const char *key, const char *who)
{
    const JsonValue *v = json_get(obj, key);
    if (!v) {
        fprintf(stderr, "参考数据缺少字段 %s.%s（请重新运行 tools/conv_reference.py）\n", who, key);
        exit(2);
    }
    return v;
}

static void arr_vals(const JsonValue *v, double *dst, int n)
{
    for (int i = 0; i < n; i++) dst[i] = json_num(json_at(v, i), 0.0);
}

static void report(const char *what, const double *got, const double *want, int n)
{
    double worst = 0.0;
    int at = -1, exact = 1;
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) exact = 0;
        double d = fabs(got[i] - want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    int ok = (worst <= g_tol);
    if (!ok) g_fail++;
    printf("    %-26s 最大差异 %.3e  %s%s\n", what, worst,
           ok ? "[OK]" : "[超出容差]", exact ? "  (逐位相同)" : "");
    if (!ok) printf("        index %d: C=%.17g py=%.17g\n", at, got[at], want[at]);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "conv_reference.json";
    if (argc > 2) g_tol = atof(argv[2]);

    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "打不开/解析失败 %s: %s\n", path, err);
        fprintf(stderr, "请先运行: python tools/conv_reference.py %s\n", path);
        return 1;
    }
    printf("参考数据: %s（容差 %.1e）\n\n", path, g_tol);

    /* ===================== Conv2d ===================== */
    const JsonValue *convs = need(root, "conv", "root");
    int n_conv = json_array_len(convs);
    for (int ci = 0; ci < n_conv; ci++) {
        const JsonValue *c = json_at(convs, ci);
        int C = json_int(need(c, "C", "conv"), 0);
        int OC = json_int(need(c, "OC", "conv"), 0);
        int K = json_int(need(c, "K", "conv"), 0);
        int S = json_int(need(c, "S", "conv"), 0);
        int P = json_int(need(c, "P", "conv"), 0);
        int H = json_int(need(c, "H", "conv"), 0);
        int W = json_int(need(c, "W", "conv"), 0);
        int ho = json_int(need(c, "ho", "conv"), 0);
        int wo = json_int(need(c, "wo", "conv"), 0);

        printf("Conv2d[%d] C=%d OC=%d K=%d S=%d P=%d  %dx%d -> %dx%d\n",
               ci, C, OC, K, S, P, H, W, ho, wo);

        Conv2d *conv = conv2d_create(C, OC, K, S, P);
        int nW = OC * C * K * K;
        double *w = malloc((size_t)nW * sizeof(double));
        double *b = malloc((size_t)OC * sizeof(double));
        arr_vals(need(c, "weights", "conv"), w, nW);
        arr_vals(need(c, "b", "conv"), b, OC);
        conv2d_set_weights(conv, w, b);

        int xs[3] = { C, H, W };
        int os[3] = { OC, ho, wo };
        Tensor *x = tensor_new(3, xs), *out = tensor_new(3, os);
        Tensor *dout = tensor_new(3, os), *dx = tensor_new(3, xs);
        arr_vals(need(c, "x", "conv"), x->data, C * H * W);
        arr_vals(need(c, "dout", "conv"), dout->data, OC * ho * wo);

        /* 期望值缓冲：一次分配成各段最大长度，避免"按小尺寸分配、按大尺寸写入"越界 */
        int n_out = OC * ho * wo, n_in = C * H * W;
        int cap = n_out > n_in ? n_out : n_in;
        if (nW > cap) cap = nW;
        double *want = malloc((size_t)cap * sizeof(double));

        conv2d_forward(conv, x, out);
        arr_vals(need(c, "out", "conv"), want, n_out);
        report("forward", out->data, want, n_out);

        conv2d_backward(conv, dout, dx);
        arr_vals(need(c, "dx", "conv"), want, n_in);
        report("backward dx", dx->data, want, n_in);

        double *got_w = malloc((size_t)nW * sizeof(double));
        double *got_b = malloc((size_t)OC * sizeof(double));
        /* grad_w / grad_b 直接读结构体（头文件里是公开字段 gW / gb） */
        {
            int k = 0;
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < C; ic++)
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++)
                            got_w[k++] = conv->gW[((oc * C + ic) * K + kh) * K + kw];
        }
        for (int oc = 0; oc < OC; oc++) got_b[oc] = conv->gb[oc];
        arr_vals(need(c, "grad_w", "conv"), want, nW);
        report("grad_w", got_w, want, nW);
        arr_vals(need(c, "grad_b", "conv"), want, OC);
        report("grad_b", got_b, want, OC);

        free(w); free(b); free(want); free(got_w); free(got_b);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        conv2d_free(conv);
    }

    /* ===================== MaxPool2d ===================== */
    const JsonValue *pools = need(root, "pool", "root");
    int n_pool = json_array_len(pools);
    for (int pi = 0; pi < n_pool; pi++) {
        const JsonValue *p = json_at(pools, pi);
        int C = json_int(need(p, "C", "pool"), 0);
        int K = json_int(need(p, "K", "pool"), 0);
        int S = json_int(need(p, "S", "pool"), 0);
        int H = json_int(need(p, "H", "pool"), 0);
        int W = json_int(need(p, "W", "pool"), 0);
        int ho = json_int(need(p, "ho", "pool"), 0);
        int wo = json_int(need(p, "wo", "pool"), 0);
        printf("\nMaxPool2d[%d] K=%d S=%d  %dx%d -> %dx%d\n", pi, K, S, H, W, ho, wo);

        MaxPool2d *pool = maxpool2d_create(K, S);
        int xs[3] = { C, H, W };
        int os[3] = { C, ho, wo };
        Tensor *x = tensor_new(3, xs), *out = tensor_new(3, os);
        Tensor *dout = tensor_new(3, os), *dx = tensor_new(3, xs);
        arr_vals(need(p, "x", "pool"), x->data, C * H * W);
        arr_vals(need(p, "dout", "pool"), dout->data, C * ho * wo);

        int n_out = C * ho * wo, n_in = C * H * W;
        int cap = n_out > n_in ? n_out : n_in;
        double *want = malloc((size_t)cap * sizeof(double));

        maxpool2d_forward(pool, x, out);
        arr_vals(need(p, "out", "pool"), want, n_out);
        report("forward", out->data, want, n_out);

        maxpool2d_backward(pool, dout, dx);
        arr_vals(need(p, "dx", "pool"), want, n_in);
        report("backward dx", dx->data, want, n_in);

        free(want);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        maxpool2d_free(pool);
    }

    /* ===================== Flatten ===================== */
    {
        const JsonValue *f = need(root, "flatten", "root");
        int C = json_int(need(f, "C", "flatten"), 0);
        int H = json_int(need(f, "H", "flatten"), 0);
        int W = json_int(need(f, "W", "flatten"), 0);
        printf("\nFlatten (%d,%d,%d) -> (%d,)\n", C, H, W, C * H * W);

        Flatten *fl = flatten_create();
        int xs[3] = { C, H, W };
        int os[1] = { C * H * W };
        Tensor *x = tensor_new(3, xs), *out = tensor_new(1, os);
        Tensor *dout = tensor_new(1, os), *dx = tensor_new(3, xs);
        arr_vals(need(f, "x", "flatten"), x->data, C * H * W);
        arr_vals(need(f, "dout", "flatten"), dout->data, C * H * W);

        flatten_forward(fl, x, out);
        double *want = malloc((size_t)C * H * W * sizeof(double));
        arr_vals(need(f, "out", "flatten"), want, C * H * W);
        report("forward", out->data, want, C * H * W);

        flatten_backward(fl, dout, dx);
        arr_vals(need(f, "dx", "flatten"), want, C * H * W);
        report("backward dx", dx->data, want, C * H * W);

        free(want);
        tensor_free(x); tensor_free(out); tensor_free(dout); tensor_free(dx);
        flatten_free(fl);
    }

    json_free(root);
    printf("\n结论: %s（容差 %.1e）\n", g_fail ? "存在超差项" : "C 与 Python 数值一致", g_tol);
    return g_fail ? 1 : 0;
}
