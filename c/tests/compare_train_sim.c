/*
 * compare_train_sim.c —— 分布式训练模拟 + 低精度舍入对拍
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../parallel.h"
#include "../quant.h"

static int g_checks = 0;
static int g_fails = 0;
static int g_exact = 0;
static double g_maxdiff = 0.0;
static const char *g_func = "";

static void check_num(const char *what, double expected, double got, double tol)
{
    double diff = fabs(expected - got);
    if (diff > g_maxdiff) g_maxdiff = diff;
    if (diff <= tol) {
        g_exact++;
        g_checks++;
        return;
    }
    g_fails++;
    g_checks++;
    printf("FAIL [%s] %s: expected %.17g got %.17g\n", g_func, what, expected, got);
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
        g_fails++;
        g_checks++;
        printf("FAIL [%s] %s: missing expected\n", g_func, what);
        return;
    }
    for (int i = 0; i < n; i++) {
        char label[160];
        snprintf(label, sizeof(label), "%s[%d]", what, i);
        check_num(label, exp[i], got[i], tol);
    }
    free(exp);
}


static void flatten_into(const JsonValue *v, double **buf, int *n, int *cap)
{
    if (!v) return;
    if (v->kind == JSON_ARRAY) {
        for (int i = 0; i < json_array_len(v); i++) flatten_into(json_at(v, i), buf, n, cap);
        return;
    }
    if (v->kind != JSON_NUMBER) return;
    if (*n >= *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *buf = (double *)realloc(*buf, (size_t)(*cap) * sizeof(double));
    }
    (*buf)[(*n)++] = v->num;
}

static double *read_flat(const JsonValue *v, int *n_out)
{
    double *buf = NULL;
    int n = 0, cap = 0;
    flatten_into(v, &buf, &n, &cap);
    if (n_out) *n_out = n;
    return buf;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "train_sim_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "parse %s failed: %s\n", path, err);
        return 2;
    }

    /* ---------- FSDP 分片 / all-gather ---------- */
    g_func = "fsdp";
    const JsonValue *fsdp = json_get(root, "fsdp");
    int n_flat = 0;
    double *flat = read_nums(json_get(fsdp, "flat"), &n_flat);
    int world = json_int(json_get(fsdp, "world"), 0);
    double *out = (double *)malloc((size_t)n_flat * sizeof(double));
    int *sizes = (int *)malloc((size_t)world * sizeof(int));
    if (par_fsdp_shard_flat(flat, n_flat, world, out, sizes) != 0) {
        fprintf(stderr, "fsdp shard failed\n");
        return 2;
    }
    const JsonValue *shards = json_get(fsdp, "shards");
    int n_shards = json_array_len(shards);
    int gs[16], ge[16];
    for (int r = 0; r < n_shards; r++) {
        const JsonValue *sh = json_at(shards, r);
        gs[r] = json_int(json_at(sh, 0), 0);
        ge[r] = json_int(json_at(sh, 1), 0);
        check_num("shard_size", (double)(ge[r] - gs[r]), (double)sizes[r], 0.0);
    }
    double *gathered = (double *)malloc((size_t)n_flat * sizeof(double));
    if (par_fsdp_all_gather(out, gs, ge, n_shards, gathered) != 0) {
        fprintf(stderr, "fsdp gather failed\n");
        return 2;
    }
    check_array("gather", json_get(fsdp, "gather"), n_flat, gathered, 1e-15);

    /* ---------- ZeRO 更新 ---------- */
    g_func = "zero";
    const JsonValue *z = json_get(root, "zero");
    int nz = 0;
    double *zp = read_nums(json_get(z, "params"), &nz);
    double *zg = read_nums(json_get(z, "grads"), NULL);
    int zw = json_int(json_get(z, "world"), 0);
    double zlr = json_num(json_get(z, "lr"), 0.0);
    double *zout = (double *)malloc((size_t)nz * sizeof(double));
    int us[16], ue[16];
    if (par_zero_step(zp, zg, nz, zlr, zw, zout, us, ue) != 0) {
        fprintf(stderr, "zero failed\n");
        return 2;
    }
    check_array("out", json_get(z, "out"), nz, zout, 1e-15);
    const JsonValue *updates = json_get(z, "updates");
    for (int r = 0; r < json_array_len(updates); r++) {
        const JsonValue *u = json_at(updates, r);
        check_num("upd_start", json_num(json_at(u, 0), NAN), (double)us[r], 0.0);
        check_num("upd_end", json_num(json_at(u, 1), NAN), (double)ue[r], 0.0);
    }

    /* ---------- FSDP reduce-scatter ---------- */
    g_func = "reduce_scatter";
    const JsonValue *rs = json_get(root, "reduce_scatter");
    int rw = json_int(json_get(rs, "world"), 0);
    int rn = json_int(json_get(rs, "n"), 0);
    int nrgs = 0;
    double *rank_grads = read_flat(json_get(rs, "rank_grads"), &nrgs);
    double *rsout = (double *)malloc((size_t)rn * sizeof(double));
    if (par_fsdp_reduce_scatter(rank_grads, rw, rn, rsout) != 0) {
        fprintf(stderr, "reduce_scatter failed\n");
        return 2;
    }
    check_array("out", json_get(rs, "out"), rn, rsout, 1e-15);
    (void)nrgs;

    /* ---------- 张量并行 ---------- */
    g_func = "tp";
    const JsonValue *tp = json_get(root, "tp");
    int rows = json_int(json_get(tp, "rows"), 0);
    int cols = json_int(json_get(tp, "cols"), 0);
    double *W = read_nums(json_get(tp, "W"), NULL);
    double *bias = read_nums(json_get(tp, "bias"), NULL);
    double *x = read_nums(json_get(tp, "x"), NULL);
    int has_bias = json_int(json_get(tp, "has_bias"), 0);
    int tpw = json_int(json_get(tp, "world"), 0);
    double *full = (double *)malloc((size_t)rows * sizeof(double));
    double *shard = (double *)malloc((size_t)rows * (size_t)tpw * sizeof(double));
    int *ssizes = (int *)malloc((size_t)tpw * sizeof(int));
    if (par_tensor_parallel(W, rows, cols, bias, has_bias, x, tpw, 0,
                            full, shard, ssizes) != 0) {
        fprintf(stderr, "tp column failed\n");
        return 2;
    }
    check_array("col_full", json_get(tp, "col_full"), rows, full, 1e-15);
    check_array("col_shard", json_get(tp, "col_full"), rows, shard, 1e-15);
    const JsonValue *col_sizes = json_get(tp, "col_sizes");
    for (int r = 0; r < json_array_len(col_sizes); r++) {
        check_num("col_size", json_num(json_at(col_sizes, r), NAN),
                  (double)ssizes[r], 0.0);
    }
    int rww = json_int(json_get(tp, "row_world"), 0);
    double *rowfull = (double *)malloc((size_t)rows * sizeof(double));
    double *rowshard = (double *)malloc((size_t)rows * (size_t)rww * sizeof(double));
    int *rowsizes = (int *)malloc((size_t)rww * sizeof(int));
    if (par_tensor_parallel(W, rows, cols, NULL, 0, x, rww, 1,
                            rowfull, rowshard, rowsizes) != 0) {
        fprintf(stderr, "tp row failed\n");
        return 2;
    }
    check_array("row_full", json_get(tp, "row_full"), rows, rowfull, 1e-15);
    check_array("row_shards", json_get(tp, "row_shards"), rows * rww, rowshard, 1e-15);

    /* ---------- 序列并行 LayerNorm ---------- */
    g_func = "sp_ln";
    const JsonValue *sp = json_get(root, "sp_ln");
    int tokens = json_int(json_get(sp, "tokens"), 0);
    int d = json_int(json_get(sp, "d"), 0);
    double eps = json_num(json_get(sp, "eps"), 0.0);
    int nx = 0;
    double *xseq = read_nums(json_get(sp, "x"), &nx);
    double *spout = (double *)malloc((size_t)nx * sizeof(double));
    if (par_sequence_parallel_layer_norm(xseq, tokens, d, eps, spout) != 0) {
        fprintf(stderr, "sp ln failed\n");
        return 2;
    }
    check_array("out", json_get(sp, "out"), nx, spout, 1e-15);

    /* ---------- 低精度舍入 ---------- */
    g_func = "round";
    const JsonValue *rd = json_get(root, "round");
    int nv = 0;
    double *vals = read_nums(json_get(rd, "values"), &nv);
    double *tmp = (double *)malloc((size_t)nv * sizeof(double));
    if (quant_fp16_round(vals, nv, tmp) != 0) return 2;
    check_array("fp16", json_get(rd, "fp16"), nv, tmp, 1e-15);
    if (quant_bf16_round(vals, nv, tmp) != 0) return 2;
    check_array("bf16", json_get(rd, "bf16"), nv, tmp, 1e-15);
    if (quant_fp8_round(vals, nv, 0, tmp) != 0) return 2;
    check_array("fp8_e4m3", json_get(rd, "fp8_e4m3"), nv, tmp, 1e-15);
    if (quant_fp8_round(vals, nv, 1, tmp) != 0) return 2;
    check_array("fp8_e5m2", json_get(rd, "fp8_e5m2"), nv, tmp, 1e-15);

    /* ---------- 伪量化 ---------- */
    g_func = "fake_quantize";
    const JsonValue *fq = json_get(root, "fq");
    const JsonValue *auto_ = json_get(fq, "auto");
    int na = 0;
    double *avals = read_nums(json_get(auto_, "values"), &na);
    double *aout = (double *)malloc((size_t)na * sizeof(double));
    if (quant_fake_quantize(avals, na, json_int(json_get(auto_, "bits"), 8),
                            1.0, 0, aout) != 0) {
        fprintf(stderr, "fq auto failed\n");
        return 2;
    }
    check_array("auto_out", json_get(auto_, "out"), na, aout, 1e-15);
    const JsonValue *sc = json_get(fq, "scaled");
    int ns = 0;
    double *svals = read_nums(json_get(sc, "values"), &ns);
    double *sout = (double *)malloc((size_t)ns * sizeof(double));
    if (quant_fake_quantize(svals, ns, json_int(json_get(sc, "bits"), 8),
                            json_num(json_get(sc, "scale"), 0.0), 1, sout) != 0) {
        fprintf(stderr, "fq scaled failed\n");
        return 2;
    }
    check_array("scaled_out", json_get(sc, "out"), ns, sout, 1e-15);

    printf("CHECK %d: exact %d, fail %d, maxdiff %.3e\n",
           g_checks, g_exact, g_fails, g_maxdiff);

    free(flat);
    free(out);
    free(sizes);
    free(gathered);
    free(zp);
    free(zg);
    free(zout);
    free(rank_grads);
    free(rsout);
    free(W);
    free(bias);
    free(x);
    free(full);
    free(shard);
    free(ssizes);
    free(rowfull);
    free(rowshard);
    free(rowsizes);
    free(xseq);
    free(spout);
    free(vals);
    free(tmp);
    free(avals);
    free(aout);
    free(svals);
    free(sout);
    json_free(root);
    return g_fails == 0 ? 0 : 1;
}
