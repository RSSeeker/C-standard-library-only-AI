/*
 * test_json.c —— JSON 存档互通测试
 *
 * 三个模式：
 *   ./test_json self     C 自测：造模型 -> 存 JSON -> 读回 -> 逐位比对 + 前向一致
 *   ./test_json write    写出 c_model.json 与 c_forward.txt（供 Python 读取验证）
 *   ./test_json check    读入 py_model.json 与 py_forward.txt（由 Python 写出）并比对
 *
 * 与 tools/json_interop.py 配合完成双向互通验证。
 */
#include "../ai.h"
#include "../json.h"
#include "../model_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 用确定性公式造权重，便于两边核对（不依赖 RNG） */
static void fill_deterministic(MLP *m)
{
    for (int li = 0; li < m->n_layers; li++) {
        Layer *l = &m->layers[li];
        for (int j = 0; j < l->fan_out; j++) {
            for (int i = 0; i < l->fan_in; i++)
                l->W[(size_t)j * l->fan_in + i] =
                    0.01 * (double)((li + 1) * 1000 + j * 100 + i) - 0.5;
            l->b[j] = 0.001 * (double)((li + 1) * 100 + j) - 0.05;
        }
    }
}

static const double TEST_X[4] = { 0.25, -0.5, 0.75, 1.0 };

static MLP *make_test_model(void)
{
    const int sizes[] = { 4, 5, 3 };
    const Activation acts[] = { ACT_LEAKY_RELU, ACT_LINEAR };
    MLP *m = mlp_create(sizes, acts, 3);
    fill_deterministic(m);
    return m;
}

static int same_model(const MLP *a, const MLP *b, double tol)
{
    if (a->n_layers != b->n_layers) return 0;
    double worst = 0.0;
    for (int li = 0; li < a->n_layers; li++) {
        const Layer *la = &a->layers[li], *lb = &b->layers[li];
        if (la->fan_in != lb->fan_in || la->fan_out != lb->fan_out || la->act != lb->act) return 0;
        for (int k = 0; k < la->fan_out * la->fan_in; k++) {
            double d = fabs(la->W[k] - lb->W[k]);
            if (d > worst) worst = d;
        }
        for (int j = 0; j < la->fan_out; j++) {
            double d = fabs(la->b[j] - lb->b[j]);
            if (d > worst) worst = d;
        }
    }
    printf("  权重/偏置最大差异: %.3e\n", worst);
    return worst <= tol;
}

static int mode_self(void)
{
    printf("[self] C 自测：存 -> 读 -> 比对\n");
    MLP *m = make_test_model();
    double out1[3];
    mlp_forward(m, TEST_X, out1);

    if (mlp_save_json(m, "self_model.json") != 0) return 1;
    MLP *m2 = mlp_load_json("self_model.json");
    if (!m2) return 1;

    int ok = same_model(m, m2, 0.0);
    double out2[3];
    mlp_forward(m2, TEST_X, out2);
    printf("  forward: [%.17g %.17g %.17g]\n", out2[0], out2[1], out2[2]);
    printf("  前向输出逐位一致: %s\n", (out1[0] == out2[0] && out1[1] == out2[1] && out1[2] == out2[2]) ? "是" : "否");
    mlp_free(m);
    mlp_free(m2);
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int mode_write(void)
{
    printf("[write] 写出 c_model.json / c_forward.txt 供 Python 验证\n");
    MLP *m = make_test_model();
    if (mlp_save_json(m, "c_model.json") != 0) return 1;

    double out[3];
    mlp_forward(m, TEST_X, out);
    FILE *f = fopen("c_forward.txt", "w");
    if (!f) { fprintf(stderr, "写 c_forward.txt 失败\n"); return 1; }
    for (int i = 0; i < 4; i++) fprintf(f, "%.17g ", TEST_X[i]);
    fprintf(f, "\n");
    for (int i = 0; i < 3; i++) fprintf(f, "%.17g ", out[i]);
    fprintf(f, "\n");
    fclose(f);
    printf("  forward = [%.17g %.17g %.17g]\n", out[0], out[1], out[2]);
    mlp_free(m);
    return 0;
}

static int mode_check(void)
{
    printf("[check] 读入 py_model.json / py_forward.txt（Python 写出）并比对\n");
    MLP *m = mlp_load_json("py_model.json");
    if (!m) return 1;

    FILE *f = fopen("py_forward.txt", "r");
    if (!f) { fprintf(stderr, "打不开 py_forward.txt（先跑 python tools/json_interop.py write）\n"); mlp_free(m); return 1; }
    double x[4], expect[3];
    for (int i = 0; i < 4; i++) if (fscanf(f, "%lf", &x[i]) != 1) { fclose(f); return 1; }
    for (int i = 0; i < 3; i++) if (fscanf(f, "%lf", &expect[i]) != 1) { fclose(f); return 1; }
    fclose(f);

    double got[3];
    mlp_forward(m, x, got);
    double worst = 0.0;
    for (int i = 0; i < 3; i++) {
        double d = fabs(got[i] - expect[i]);
        if (d > worst) worst = d;
    }
    printf("  C 前向    = [%.17g %.17g %.17g]\n", got[0], got[1], got[2]);
    printf("  Python 前向 = [%.17g %.17g %.17g]\n", expect[0], expect[1], expect[2]);
    printf("  最大差异 = %.3e\n", worst);
    mlp_free(m);
    int ok = worst <= 1e-12;
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *mode = (argc > 1) ? argv[1] : "self";
    int rc;
    if (strcmp(mode, "self") == 0)       rc = mode_self();
    else if (strcmp(mode, "write") == 0) rc = mode_write();
    else if (strcmp(mode, "check") == 0) rc = mode_check();
    else { fprintf(stderr, "未知模式: %s（可用 self/write/check）\n", mode); return 2; }
    printf("退出码 %d\n", rc);
    return rc;
}
