/*
 * compare_flowmatching.c —— C 侧 Flow Matching 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_flowmatching [flowmatching_reference.json]
 *
 * 参考用例按顺序执行：velocity/interpolate/target → train_step（损失+梯度）
 * → sgd_after（更新后权重+速度场）→ sample_euler/sample_heun。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../flowmatching.h"
#include "../json.h"

static int g_checks;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_flowmatching] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_flowmatching] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/flowmatching_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static double jnum(const JsonValue *v) { return json_num(v, 0.0); }
static int jint(const JsonValue *v) { return json_int(v, 0); }
static const char *jstr(const JsonValue *v) { return json_str(v, ""); }

static double *darr(const JsonValue *v, int *n_out)
{
    int n = v ? v->n_items : 0;
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    if (n_out) *n_out = n;
    return out;
}

static void check(const char *name, int idx, double got, double want, double tol)
{
    g_checks++;
    double diff = fabs(got - want);
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    double rel = diff / scale;
    if (diff > g_worst) g_worst = diff;
    if (rel > tol) {
        g_fail++;
        printf("  %-28s [%d] C=%.17g py=%.17g rel=%.3e [不等]\n",
               name, idx, got, want, rel);
    }
}

static void load_layers(FlowMatching *m, const JsonValue *layers, const char *who)
{
    int n = json_array_len(layers);
    if (n != m->net->n_layers) die("层数不一致");
    for (int i = 0; i < n; i++) {
        const JsonValue *lv = json_at(layers, i);
        const JsonValue *wv = need(lv, "w", who);
        const JsonValue *bv = need(lv, "b", who);
        Layer *l = &m->net->layers[i];
        for (int r = 0; r < l->fan_out; r++) {
            const JsonValue *row = json_at(wv, r);
            for (int c = 0; c < l->fan_in; c++)
                l->W[(size_t)r * l->fan_in + c] = jnum(json_at(row, c));
        }
        for (int r = 0; r < l->fan_out; r++) l->b[r] = jnum(json_at(bv, r));
    }
}

static void check_layers(FlowMatching *m, const JsonValue *layers, const char *who)
{
    int n = json_array_len(layers);
    if (n != m->net->n_layers) die("层数不一致");
    for (int i = 0; i < n; i++) {
        const JsonValue *lv = json_at(layers, i);
        const JsonValue *wv = need(lv, "w", who);
        const JsonValue *bv = need(lv, "b", who);
        Layer *l = &m->net->layers[i];
        for (int r = 0; r < l->fan_out; r++) {
            const JsonValue *row = json_at(wv, r);
            for (int c = 0; c < l->fan_in; c++)
                check(who, r * l->fan_in + c,
                      l->W[(size_t)r * l->fan_in + c], jnum(json_at(row, c)), 1e-12);
        }
        for (int r = 0; r < l->fan_out; r++)
            check(who, 0, l->b[r], jnum(json_at(bv, r)), 1e-12);
    }
}

static void check_vec(const char *name, const double *got, const JsonValue *want,
                      int n)
{
    if (want == NULL || want->n_items != n) die("期望向量长度不一致");
    for (int i = 0; i < n; i++) check(name, i, got[i], jnum(json_at(want, i)), 1e-12);
}

static void run_case(FlowMatching *m, const JsonValue *c)
{
    const char *name = jstr(need(c, "name", "case"));
    const char *func = jstr(need(c, "func", name));

    if (strcmp(func, "velocity") == 0) {
        int xn = 0;
        double *x = darr(need(c, "x_t", name), &xn);
        double *got = (double *)malloc(sizeof(double) * (size_t)xn);
        flowmatching_velocity(m, x, jnum(need(c, "t", name)), got);
        check_vec(name, got, need(c, "expected", name), xn);
        free(x); free(got);
    } else if (strcmp(func, "interpolate") == 0 ||
               strcmp(func, "target") == 0) {
        int n = 0;
        double *x0 = darr(need(c, "x0", name), &n);
        double *noise = darr(need(c, "noise", name), &n);
        double *got = (double *)malloc(sizeof(double) * (size_t)n);
        if (strcmp(func, "interpolate") == 0) {
            flowmatching_interpolate(m, x0, noise, jnum(need(c, "t", name)), got);
        } else {
            flowmatching_target(m, x0, noise, got);
        }
        check_vec(name, got, need(c, "expected", name), n);
        free(x0); free(noise); free(got);
    } else if (strcmp(func, "train_step") == 0) {
        int n = 0;
        double *x0 = darr(need(c, "x0", name), &n);
        double *noise = darr(need(c, "noise", name), &n);
        flowmatching_zero_grad(m);
        double loss = flowmatching_train_step(m, x0, jnum(need(c, "t", name)), noise);
        check(name, 0, loss, jnum(need(c, "expected_loss", name)), 1e-10);
        const JsonValue *grads = need(c, "expected_grads", name);
        for (int i = 0; i < m->net->n_layers; i++) {
            const JsonValue *gv = json_at(grads, i);
            if (!gv || gv->kind == JSON_NULL) continue;
            const JsonValue *wv = need(gv, "w", name);
            const JsonValue *bv = need(gv, "b", name);
            Layer *l = &m->net->layers[i];
            for (int r = 0; r < l->fan_out; r++) {
                const JsonValue *row = json_at(wv, r);
                for (int cc = 0; cc < l->fan_in; cc++)
                    check(name, r * l->fan_in + cc,
                          l->gW[(size_t)r * l->fan_in + cc],
                          jnum(json_at(row, cc)), 1e-10);
            }
            for (int r = 0; r < l->fan_out; r++)
                check(name, 0, l->gb[r], jnum(json_at(bv, r)), 1e-10);
        }
        free(x0); free(noise);
    } else if (strcmp(func, "sgd") == 0) {
        flowmatching_sgd_step(m, jnum(need(c, "lr", name)));
        check_layers(m, need(c, "expected_layers", name), name);
        static const double xt[3] = {0.3, -0.2, 0.5};
        double got[3];
        flowmatching_velocity(m, xt, 0.25, got);
        check_vec(name, got, need(c, "expected_velocity", name), 3);
    } else if (strcmp(func, "sample") == 0) {
        int n = 0;
        double *x0 = darr(need(c, "x_init", name), &n);
        double *got = (double *)malloc(sizeof(double) * (size_t)n);
        int solver = strcmp(jstr(need(c, "solver", name)), "heun") == 0 ? 1 : 0;
        flowmatching_sample(m, x0, jint(need(c, "steps", name)), solver, got);
        check_vec(name, got, need(c, "expected", name), n);
        free(x0); free(got);
    } else {
        die("未知 func");
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "flowmatching_reference.json";
    char err[256] = {0};
    JsonValue *doc = json_parse_file(path, err, sizeof err);
    if (!doc) {
        fprintf(stderr, "[compare_flowmatching] JSON 解析失败：%s\n", err);
        return 2;
    }
    const JsonValue *model = need(doc, "model", "doc");
    int data_dim = jint(need(model, "data_dim", "model"));
    int hidden = jint(need(model, "hidden_dim", "model"));
    int t_dim = jint(need(model, "t_dim", "model"));
    FlowMatching *m = flowmatching_create(data_dim, hidden, t_dim);
    if (!m) die("flowmatching_create 失败");
    load_layers(m, need(model, "layers", "model"), "model");
    int want_params = jint(need(model, "param_count", "model"));
    g_checks++;
    if (flowmatching_param_count(m) != want_params) {
        g_fail++;
        printf("  param_count C=%d py=%d [不等]\n",
               flowmatching_param_count(m), want_params);
    }

    const JsonValue *cases = need(doc, "cases", "doc");
    for (int i = 0; i < cases->n_items; i++) run_case(m, json_at(cases, i));

    printf("[compare_flowmatching] %d 项检查：失败 %d，最大绝对差 %.3e\n",
           g_checks, g_fail, g_worst);
    flowmatching_free(m);
    json_free(doc);
    return g_fail == 0 ? 0 : 1;
}
