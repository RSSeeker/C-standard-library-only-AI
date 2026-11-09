/*
 * compare_onnx_mlp.c —— C 的 MLP→ONNX 导出/解析/执行与 AI.py 对拍
 *
 * 用法：./tests/compare_onnx_mlp [onnx_mlp_reference.json]
 *
 * 每个用例：
 *  1) export 用例：C 导出的字节与 Python 导出的 hex 逐字节比较（必须相同）；
 *  2) 解析 Python 字节：结构摘要（ir_version/opset/producer/节点序列/初始化器
 *     名称与维度/输入输出签名）逐项比较；
 *  3) 执行：输出按用例容差比较（纯 relu/linear 为 0，含 exp/sigmoid 为 1e-12）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ai.h"
#include "../json.h"
#include "../onnx_mlp.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_onnx_mlp] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_onnx_mlp] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/onnx_mlp_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int act_from_name(const char *name)
{
    if (strcmp(name, "linear") == 0) return ACT_LINEAR;
    if (strcmp(name, "relu") == 0) return ACT_RELU;
    if (strcmp(name, "leaky_relu") == 0) return ACT_LEAKY_RELU;
    if (strcmp(name, "softmax") == 0) return ACT_SOFTMAX;
    return -1;
}

static const char *expected_op_name(const char *act)
{
    if (strcmp(act, "linear") == 0) return "Identity";
    if (strcmp(act, "relu") == 0) return "Relu";
    if (strcmp(act, "leaky_relu") == 0) return "LeakyRelu";
    if (strcmp(act, "softmax") == 0) return "Softmax";
    if (strcmp(act, "sigmoid") == 0) return "Sigmoid";
    if (strcmp(act, "tanh") == 0) return "Tanh";
    return "?";
}

static void check_int(const char *what, int got, int want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-30s C=%d py=%d [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_str(const char *what, const char *got, const char *want)
{
    g_checks++;
    if (strcmp(got, want) == 0) g_exact++;
    else {
        printf("  %-30s C=\"%s\" py=\"%s\" [不等]\n", what, got, want);
        g_fail++;
    }
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double d = fabs(got - want);
        if (d > g_worst) g_worst = d;
        if (d <= tol) return;
        printf("  %-30s C=%.17g py=%.17g 差=%.3e [不等]\n",
               what, got, want, d);
        g_fail++;
    }
}

/* hex 字符串 -> 字节（malloc） */
static unsigned char *hex_decode(const char *s, size_t *out_len)
{
    const size_t n = strlen(s);
    unsigned char *buf;
    if (n % 2 != 0) die("hex 长度必须是偶数");
    buf = (unsigned char *)malloc(n / 2 ? n / 2 : 1);
    if (!buf) die("out of memory");
    for (size_t i = 0; i < n / 2; i++) {
        int hi = 0, lo = 0, k;
        k = s[i * 2];
        if (k >= '0' && k <= '9') hi = k - '0';
        else if (k >= 'a' && k <= 'f') hi = k - 'a' + 10;
        else die("非法 hex");
        k = s[i * 2 + 1];
        if (k >= '0' && k <= '9') lo = k - '0';
        else if (k >= 'a' && k <= 'f') lo = k - 'a' + 10;
        else die("非法 hex");
        buf[i] = (unsigned char)((hi << 4) | lo);
    }
    *out_len = n / 2;
    return buf;
}

static double *matrix_from(const JsonValue *v, int *out_rows, int *out_cols)
{
    const int rows = json_array_len(v);
    const int cols = rows > 0 ? json_array_len(json_at(v, 0)) : 0;
    double *m;
    if (rows < 0 || cols < 0) die("期望二维数字数组");
    m = (double *)malloc(sizeof(double) * (size_t)(rows * cols > 0 ? rows * cols : 1));
    if (!m) die("out of memory");
    for (int i = 0; i < rows; i++) {
        const JsonValue *row = json_at(v, i);
        if (json_array_len(row) != cols) die("行宽不一致");
        for (int j = 0; j < cols; j++)
            m[(size_t)i * cols + j] = json_num(json_at(row, j), NAN);
    }
    *out_rows = rows;
    *out_cols = cols;
    return m;
}

static void build_mlp(const JsonValue *cs, MLP **out)
{
    const JsonValue *sizes_v = need(cs, "layer_sizes", "case");
    const JsonValue *acts_v = need(cs, "activations", "case");
    const int L = json_array_len(sizes_v) - 1;
    int *sizes = (int *)malloc(sizeof(int) * (size_t)(L + 1));
    Activation *acts = (Activation *)malloc(sizeof(Activation) * (size_t)L);
    if (!sizes || !acts) die("out of memory");
    for (int i = 0; i <= L; i++) sizes[i] = json_int(json_at(sizes_v, i), 0);
    for (int i = 0; i < L; i++) {
        const int a = act_from_name(json_str(json_at(acts_v, i), "?"));
        if (a < 0) die("C 侧不支持的激活（该用例不应要求 C 导出）");
        acts[i] = (Activation)a;
    }
    *out = mlp_create(sizes, acts, L + 1);
    if (!*out) die("mlp_create 失败");
    {
        const JsonValue *weights = need(cs, "weights", "case");
        for (int k = 0; k < L; k++) {
            const JsonValue *wl = json_at(weights, k);
            const JsonValue *wv = need(wl, "w", "layer");
            const JsonValue *bv = need(wl, "b", "layer");
            Layer *layer = &(*out)->layers[k];
            for (int r = 0; r < layer->fan_out; r++) {
                const JsonValue *row = json_at(wv, r);
                for (int c = 0; c < layer->fan_in; c++)
                    layer->W[(size_t)r * layer->fan_in + c] =
                        json_num(json_at(row, c), NAN);
            }
            for (int r = 0; r < layer->fan_out; r++)
                layer->b[r] = json_num(json_at(bv, r), NAN);
        }
    }
    free(sizes);
    free(acts);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "onnx_mlp_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_onnx_mlp] 读不了 %s：%s\n"
                        "请先运行 python tools/onnx_mlp_reference.py\n", path, err);
        return 2;
    }
    {
        const JsonValue *cases = need(root, "cases", "root");
        const int n_cases = json_array_len(cases);
        printf("[compare_onnx_mlp] %s：%d 个用例\n", path, n_cases);

        for (int ci = 0; ci < n_cases; ci++) {
            const JsonValue *cs = json_at(cases, ci);
            const char *name = json_str(json_get(cs, "name"), "?");
            const JsonValue *export_v = json_get(cs, "export");
            const int want_export = export_v
                ? (export_v->kind == JSON_BOOL ? export_v->boolean : 1) : 1;
            const double tol = json_num(json_get(cs, "tolerance"), 0.0);
            const int L = json_array_len(need(cs, "layer_sizes", name)) - 1;
            unsigned char *bytes = NULL;
            size_t bytes_len = 0;
            int have_own_bytes = 0;

            printf("\n用例 %-24s L=%d export=%d\n", name, L, want_export);

            if (want_export) {
                MLP *net = NULL;
                OnnxMlpBytes own = {NULL, 0};
                unsigned char *py_bytes;
                size_t py_len;
                const char *hex = json_str(need(cs, "export_hex", name), "");
                build_mlp(cs, &net);
                if (onnx_export_mlp(net, &own) != 0) {
                    printf("  C 导出返回错误\n");
                    g_fail++;
                    mlp_free(net);
                    continue;
                }
                py_bytes = hex_decode(hex, &py_len);
                g_checks++;
                if (own.len == py_len && py_len > 0
                    && memcmp(own.data, py_bytes, py_len) == 0) {
                    g_exact++;
                    printf("  导出字节：与 Python 逐字节一致（%zu 字节）\n", py_len);
                } else {
                    size_t diff = 0;
                    printf("  导出字节：不一致（C=%zu py=%zu）\n", own.len, py_len);
                    while (diff < own.len && diff < py_len
                           && own.data[diff] == py_bytes[diff]) diff++;
                    printf("    首处差异 @ %zu：C=%02x py=%02x\n", diff,
                           diff < own.len ? own.data[diff] : 0,
                           diff < py_len ? py_bytes[diff] : 0);
                    g_fail++;
                }
                bytes = py_bytes;
                bytes_len = py_len;
                have_own_bytes = 1;
                onnx_mlp_bytes_free(&own);
                mlp_free(net);
            } else {
                const char *hex = json_str(need(cs, "export_hex", name), "");
                bytes = hex_decode(hex, &bytes_len);
                have_own_bytes = 1;
            }

            {
                OnnxMlpGraph g;
                if (onnx_mlp_parse(bytes, bytes_len, &g) != 0) {
                    printf("  C 解析失败\n");
                    g_fail++;
                    free(bytes);
                    continue;
                }
                check_int("ir_version", g.ir_version,
                          json_int(need(cs, "ir_version", name), -1));
                check_int("opset", g.opset,
                          json_int(need(cs, "opset", name), -1));
                check_str("producer", g.producer,
                          json_str(need(cs, "producer", name), "?"));
                check_str("graph_name", g.graph_name,
                          json_str(need(cs, "graph_name", name), "?"));
                check_int("n_nodes", g.n_nodes, 2 * L);
                check_int("n_tensors", g.n_tensors,
                          json_int(need(cs, "n_tensors", name), -1));
                for (int k = 0; k < L; k++) {
                    const JsonValue *acts_v = need(cs, "activations", name);
                    const char *want_act = expected_op_name(
                        json_str(json_at(acts_v, k), "?"));
                    check_str("gemm op", g.nodes[2 * k].op_type, "Gemm");
                    check_str("act op", g.nodes[2 * k + 1].op_type, want_act);
                    check_int("gemm transB", g.nodes[2 * k].has_transb
                              ? g.nodes[2 * k].transb : -99, 1);
                }
                {
                    const JsonValue *tensors = need(cs, "tensors", name);
                    for (int k = 0; k < g.n_tensors; k++) {
                        const JsonValue *tv = json_at(tensors, k);
                        check_str("tensor name", g.tensors[k].name,
                                  json_str(json_at(tv, 0), "?"));
                        check_int("tensor dims0", g.tensors[k].n_dims > 0
                                  ? g.tensors[k].dims[0] : -99,
                                  json_int(json_at(json_at(tv, 1), 0), -1));
                        if (g.tensors[k].n_dims > 1)
                            check_int("tensor dims1", g.tensors[k].dims[1],
                                      json_int(json_at(json_at(tv, 1), 1), -1));
                    }
                }
                check_str("input name", g.input_name,
                          json_str(need(cs, "input_name", name), "?"));
                check_str("output name", g.output_name,
                          json_str(need(cs, "output_name", name), "?"));
                {
                    /* 执行：rows 与 forward 矩阵 */
                    int n_rows = 0, in_dim = 0;
                    double *rows = matrix_from(need(cs, "rows", name),
                                               &n_rows, &in_dim);
                    int want_rows = 0, want_cols = 0;
                    double *want = matrix_from(need(cs, "forward", name),
                                               &want_rows, &want_cols);
                    double *out = NULL;
                    int out_cols = 0;
                    if (onnx_mlp_forward(&g, rows, n_rows, &out, &out_cols) != 0) {
                        printf("  C 执行失败\n");
                        g_fail++;
                    } else {
                        check_int("forward rows", n_rows, want_rows);
                        check_int("forward cols", out_cols, want_cols);
                        for (int i = 0; i < n_rows; i++)
                            for (int j = 0; j < out_cols; j++)
                                check_num("forward",
                                          out[(size_t)i * out_cols + j],
                                          want[(size_t)i * want_cols + j], tol);
                        free(out);
                    }
                    free(rows);
                    free(want);
                }
                onnx_mlp_graph_free(&g);
            }
            if (have_own_bytes) free(bytes);
        }

        printf("\n[compare_onnx_mlp] 共 %d 项检查：逐位一致 %d，失败 %d，"
               "最大绝对差 %.3e\n", g_checks, g_exact, g_fail, g_worst);
        json_free(root);
        return (g_fail > 0) ? 1 : 0;
    }
}
