/*
 * test_rnn.c —— RNN / LSTM / GRU 移植的自测
 *
 * 覆盖：
 *   [1] 步进一致性：一次 zero_grad + 逐步 backward  ==  各步单独 zero_grad 后求和
 *       （三种 cell 各测一遍；每步返回的 dx / dh_prev / dc_prev 也一起比）
 *   [2] restore_cache 等价性：把 step_cache 内容备份、用别的序列覆盖缓存后再还原，
 *       backward 的结果必须与重新 forward 后 backward 逐位一致（三种多层网络）
 *   [3] 有限差分校验：三种 cell 的全部参数（W_ih / W_hh / b_ih / b_hh），
 *       中心差分 eps = 1e-6，阈值 1e-5
 *   [4] 多层 forward / backward：与手工的多层链（逐层 cell）和手工 BPTT 循环逐位一致
 *   [5] 优化器：SGD / Adam 一步的结果与手工更新逐位一致
 *
 * 编译（在 D:\vscode\AI_C\c 下）：
 *   gcc -std=c17 -O2 -Wall -Wextra -o tests\test_rnn.exe tests\test_rnn.c rnn.c tensor.c -lm
 * 全部通过返回 0。
 */
#include "../rnn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 测试规模 */
enum { T = 6, INS = 3, HS = 4, LAYERS = 2 };

static int g_fail = 0;
static unsigned long long g_rng = 0x243F6A8885A308D3ULL;

/* ------------------------------ 小工具 ------------------------------ */

static double urand(void)
{
    unsigned long long x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return (double)((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

/* 用 [-a, a] 的伪随机值填充（确定性，便于复现） */
static void fill_rand(double *v, int n, double a)
{
    for (int i = 0; i < n; i++) v[i] = (2.0 * urand() - 1.0) * a;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "test_rnn: out of memory\n"); exit(1); }
    return p;
}

static double max_abs_diff(const double *a, const double *b, int n)
{
    double mx = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* 逐位比较：要求 max|Δ| == 0 */
static void check_bits(const char *tag, const double *a, const double *b, int n)
{
    double mx = max_abs_diff(a, b, n);
    int ok = (mx == 0.0);
    printf("    %-52s max|Δ| = %-10.3e %s\n", tag, mx,
           ok ? "[逐位相同]" : "[FAIL 不是逐位相同]");
    if (!ok) g_fail++;
}

/* 数值比较（有限差分用） */
static void check_tol(const char *tag, const double *analytic, const double *numeric, int n,
                      double tol)
{
    double mx = max_abs_diff(analytic, numeric, n);
    int ok = (mx <= tol);
    printf("    %-52s max|Δ| = %-10.3e %s\n", tag, mx, ok ? "[OK]" : "[FAIL 超过阈值]");
    if (!ok) g_fail++;
}

/* ---------------------- 梯度集合（4 个参数块） ---------------------- */

typedef struct {
    double *buf[4];   /* W_ih / W_hh / b_ih / b_hh */
    int n[4];
} GradPack;

static const char *const GRAD_NAME[4] = { "grad_W_ih", "grad_W_hh", "grad_b_ih", "grad_b_hh" };

static GradPack pack_rnn(RNNCell *c)
{
    GradPack g;
    g.buf[0] = c->grad_W_ih; g.n[0] = c->hidden_size * c->input_size;
    g.buf[1] = c->grad_W_hh; g.n[1] = c->hidden_size * c->hidden_size;
    g.buf[2] = c->grad_b_ih; g.n[2] = c->hidden_size;
    g.buf[3] = c->grad_b_hh; g.n[3] = c->hidden_size;
    return g;
}

static GradPack pack_lstm(LSTMCell *c)
{
    int gs = c->gate_size;
    GradPack g;
    g.buf[0] = c->grad_W_ih; g.n[0] = gs * c->input_size;
    g.buf[1] = c->grad_W_hh; g.n[1] = gs * c->hidden_size;
    g.buf[2] = c->grad_b_ih; g.n[2] = gs;
    g.buf[3] = c->grad_b_hh; g.n[3] = gs;
    return g;
}

static GradPack pack_gru(GRUCell *c)
{
    int gs = c->gate_size;
    GradPack g;
    g.buf[0] = c->grad_W_ih; g.n[0] = gs * c->input_size;
    g.buf[1] = c->grad_W_hh; g.n[1] = gs * c->hidden_size;
    g.buf[2] = c->grad_b_ih; g.n[2] = gs;
    g.buf[3] = c->grad_b_hh; g.n[3] = gs;
    return g;
}

static GradPack pack_dup(const GradPack *src)
{
    GradPack d;
    for (int k = 0; k < 4; k++) {
        d.n[k] = src->n[k];
        d.buf[k] = xmalloc((size_t)d.n[k] * sizeof(double));
        memcpy(d.buf[k], src->buf[k], (size_t)d.n[k] * sizeof(double));
    }
    return d;
}

/* 清零副本（累加用） */
static GradPack pack_dst(const GradPack *src)
{
    GradPack d;
    for (int k = 0; k < 4; k++) {
        d.n[k] = src->n[k];
        d.buf[k] = xmalloc((size_t)d.n[k] * sizeof(double));
        memset(d.buf[k], 0, (size_t)d.n[k] * sizeof(double));
    }
    return d;
}

static void pack_accum(GradPack *acc, const GradPack *g)
{
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < g->n[k]; i++) acc->buf[k][i] += g->buf[k][i];
}

static void pack_free(GradPack *g)
{
    for (int k = 0; k < 4; k++) free(g->buf[k]);
}

static void pack_check_bits(const char *tag, const GradPack *a, const GradPack *b)
{
    for (int k = 0; k < 4; k++) {
        char t[144];
        snprintf(t, sizeof(t), "%s: %s", tag, GRAD_NAME[k]);
        check_bits(t, a->buf[k], b->buf[k], a->n[k]);
    }
}

/* ------------------------ 参数初始化工具 ------------------------ */

/* rows × ins 的 W_ih、rows × hs 的 W_hh、rows 长的 b_ih / b_hh 都填确定性随机值 */
static void rand_params(double **W_ih, double **W_hh, double **b_ih, double **b_hh,
                        int rows, int ins, int hs)
{
    *W_ih = xmalloc((size_t)rows * ins * sizeof(double));
    *W_hh = xmalloc((size_t)rows * hs * sizeof(double));
    *b_ih = xmalloc((size_t)rows * sizeof(double));
    *b_hh = xmalloc((size_t)rows * sizeof(double));
    fill_rand(*W_ih, rows * ins, 0.5);
    fill_rand(*W_hh, rows * hs, 0.5);
    fill_rand(*b_ih, rows, 0.5);
    fill_rand(*b_hh, rows, 0.5);
}

static void free_params(double *W_ih, double *W_hh, double *b_ih, double *b_hh)
{
    free(W_ih); free(W_hh); free(b_ih); free(b_hh);
}

/* 把 net 的所有层权重灌成确定性随机值（用于多层测试） */
static void rand_net_weights(void *net_ptr, int kind)
{
    for (int l = 0; l < LAYERS; l++) {
        if (kind == 0) {
            RNNCell *c = rnn_layer((RNN *)net_ptr, l);
            double *W_ih, *W_hh, *b_ih, *b_hh;
            rand_params(&W_ih, &W_hh, &b_ih, &b_hh, c->hidden_size, c->input_size,
                        c->hidden_size);
            rnn_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);
            free_params(W_ih, W_hh, b_ih, b_hh);
        } else if (kind == 1) {
            LSTMCell *c = lstm_layer((LSTM *)net_ptr, l);
            double *W_ih, *W_hh, *b_ih, *b_hh;
            rand_params(&W_ih, &W_hh, &b_ih, &b_hh, c->gate_size, c->input_size,
                        c->hidden_size);
            lstm_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);
            free_params(W_ih, W_hh, b_ih, b_hh);
        } else {
            GRUCell *c = gru_layer((GRU *)net_ptr, l);
            double *W_ih, *W_hh, *b_ih, *b_hh;
            rand_params(&W_ih, &W_hh, &b_ih, &b_hh, c->gate_size, c->input_size,
                        c->hidden_size);
            gru_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);
            free_params(W_ih, W_hh, b_ih, b_hh);
        }
    }
}

/* =====================================================================
 * [1] 单步 cell：一次 zero_grad + 逐步 backward == 各步隔离梯度之和
 * ===================================================================== */

static void test_rnn_cell_accum(void)
{
    RNNCell *c = rnn_cell_create(INS, HS);
    double *W_ih, *W_hh, *b_ih, *b_hh;
    rand_params(&W_ih, &W_hh, &b_ih, &b_hh, HS, INS, HS);
    rnn_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

    double *X = xmalloc((size_t)T * INS * sizeof(double));
    double *DH = xmalloc((size_t)T * HS * sizeof(double));
    fill_rand(X, T * INS, 1.0);
    fill_rand(DH, T * HS, 1.0);

    /* 前向 + 每步快照（等价于 Python 的 step_cache[t]） */
    int rec = rnn_cell_cache_size(c);
    double *snaps = xmalloc((size_t)T * rec * sizeof(double));
    const double *h = NULL;
    for (int t = 0; t < T; t++) {
        h = rnn_cell_forward(c, X + t * INS, h);
        rnn_cell_cache_snapshot(c, snaps + (size_t)t * rec);
    }

    double *dxA = xmalloc((size_t)T * INS * sizeof(double));
    double *dhA = xmalloc((size_t)T * HS * sizeof(double));
    double *dhIn = xmalloc((size_t)T * HS * sizeof(double));   /* A 里每步实际用的 dh */
    double dx[INS], dhp[HS], *dh_cur = xmalloc((size_t)HS * sizeof(double));

    /* A：一次 zero_grad，逐步反向（时间步逆序，dh 链式传递） */
    rnn_cell_zero_grad(c);
    memcpy(dh_cur, DH + (T - 1) * HS, (size_t)HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        memcpy(dhIn + (size_t)t * HS, dh_cur, (size_t)HS * sizeof(double));
        rnn_cell_cache_restore(c, snaps + (size_t)t * rec);
        rnn_cell_backward(c, dh_cur, dx, dhp);
        memcpy(dxA + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhA + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
    }
    GradPack g = pack_rnn(c);
    GradPack accA = pack_dup(&g);

    /* B：每步单独 zero_grad（用 A 里同一步的 dh 输入），最后把各步梯度加起来 */
    GradPack accB = pack_dst(&g);
    double *dxB = xmalloc((size_t)T * INS * sizeof(double));
    double *dhB = xmalloc((size_t)T * HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        rnn_cell_zero_grad(c);
        rnn_cell_cache_restore(c, snaps + (size_t)t * rec);
        rnn_cell_backward(c, dhIn + (size_t)t * HS, dx, dhp);
        memcpy(dxB + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhB + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        GradPack gt = pack_rnn(c);
        pack_accum(&accB, &gt);
    }

    pack_check_bits("RNNCell", &accA, &accB);
    check_bits("RNNCell: 每步 dx", dxA, dxB, T * INS);
    check_bits("RNNCell: 每步 dh_prev", dhA, dhB, T * HS);

    pack_free(&accA); pack_free(&accB);
    free(W_ih); free(W_hh); free(b_ih); free(b_hh);
    free(X); free(DH); free(snaps); free(dxA); free(dhA); free(dxB); free(dhB); free(dh_cur);
    rnn_cell_free(c);
}

static void test_lstm_cell_accum(void)
{
    LSTMCell *c = lstm_cell_create(INS, HS);
    double *W_ih, *W_hh, *b_ih, *b_hh;
    rand_params(&W_ih, &W_hh, &b_ih, &b_hh, 4 * HS, INS, HS);
    lstm_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

    double *X = xmalloc((size_t)T * INS * sizeof(double));
    double *DH = xmalloc((size_t)T * HS * sizeof(double));
    double *DC = xmalloc((size_t)T * HS * sizeof(double));
    fill_rand(X, T * INS, 1.0);
    fill_rand(DH, T * HS, 1.0);
    fill_rand(DC, T * HS, 1.0);

    int rec = lstm_cell_cache_size(c);
    double *snaps = xmalloc((size_t)T * rec * sizeof(double));
    const double *h = NULL, *cs = NULL;
    for (int t = 0; t < T; t++) {
        h = lstm_cell_forward(c, X + t * INS, h, cs);
        cs = c->c;                       /* 下一步的 c_prev（forward 内部先拷贝再覆盖） */
        lstm_cell_cache_snapshot(c, snaps + (size_t)t * rec);
    }

    double dx[INS], dhp[HS], dcp[HS];
    double *dxA = xmalloc((size_t)T * INS * sizeof(double));
    double *dhA = xmalloc((size_t)T * HS * sizeof(double));
    double *dcA = xmalloc((size_t)T * HS * sizeof(double));
    double *dhIn = xmalloc((size_t)T * HS * sizeof(double));   /* A 里每步实际用的 dh */
    double *dcIn = xmalloc((size_t)T * HS * sizeof(double));   /* A 里每步实际用的 dc_next */
    double *dh_cur = xmalloc((size_t)HS * sizeof(double));
    double *dc_cur = xmalloc((size_t)HS * sizeof(double));

    /* A：一次 zero_grad，逐步反向（dh / dc 都链式传递） */
    lstm_cell_zero_grad(c);
    memset(dh_cur, 0, (size_t)HS * sizeof(double));
    memset(dc_cur, 0, (size_t)HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        for (int j = 0; j < HS; j++) dh_cur[j] += DH[t * HS + j];   /* 每步输出都有损失 */
        memcpy(dhIn + (size_t)t * HS, dh_cur, (size_t)HS * sizeof(double));
        memcpy(dcIn + (size_t)t * HS, dc_cur, (size_t)HS * sizeof(double));
        lstm_cell_cache_restore(c, snaps + (size_t)t * rec);
        lstm_cell_backward(c, dh_cur, dc_cur, dx, dhp, dcp);
        memcpy(dxA + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhA + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        memcpy(dcA + (size_t)t * HS, dcp, (size_t)HS * sizeof(double));
        memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
        memcpy(dc_cur, dcp, (size_t)HS * sizeof(double));
    }
    GradPack g = pack_lstm(c);
    GradPack accA = pack_dup(&g);

    /* B：每步单独 zero_grad 后求和 */
    GradPack accB = pack_dst(&g);
    double *dxB = xmalloc((size_t)T * INS * sizeof(double));
    double *dhB = xmalloc((size_t)T * HS * sizeof(double));
    double *dcB = xmalloc((size_t)T * HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        lstm_cell_zero_grad(c);
        lstm_cell_cache_restore(c, snaps + (size_t)t * rec);
        lstm_cell_backward(c, dhIn + (size_t)t * HS, dcIn + (size_t)t * HS, dx, dhp, dcp);
        memcpy(dxB + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhB + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        memcpy(dcB + (size_t)t * HS, dcp, (size_t)HS * sizeof(double));
        GradPack gt = pack_lstm(c);
        pack_accum(&accB, &gt);
    }

    pack_check_bits("LSTMCell", &accA, &accB);
    check_bits("LSTMCell: 每步 dx", dxA, dxB, T * INS);
    check_bits("LSTMCell: 每步 dh_prev", dhA, dhB, T * HS);
    check_bits("LSTMCell: 每步 dc_prev", dcA, dcB, T * HS);

    pack_free(&accA); pack_free(&accB);
    free(W_ih); free(W_hh); free(b_ih); free(b_hh);
    free(X); free(DH); free(DC); free(snaps);
    free(dxA); free(dhA); free(dcA); free(dxB); free(dhB); free(dcB);
    free(dhIn); free(dcIn); free(dh_cur); free(dc_cur);
    lstm_cell_free(c);
}

static void test_gru_cell_accum(void)
{
    GRUCell *c = gru_cell_create(INS, HS);
    double *W_ih, *W_hh, *b_ih, *b_hh;
    rand_params(&W_ih, &W_hh, &b_ih, &b_hh, 3 * HS, INS, HS);
    gru_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

    double *X = xmalloc((size_t)T * INS * sizeof(double));
    double *DH = xmalloc((size_t)T * HS * sizeof(double));
    fill_rand(X, T * INS, 1.0);
    fill_rand(DH, T * HS, 1.0);

    int rec = gru_cell_cache_size(c);
    double *snaps = xmalloc((size_t)T * rec * sizeof(double));
    const double *h = NULL;
    for (int t = 0; t < T; t++) {
        h = gru_cell_forward(c, X + t * INS, h);
        gru_cell_cache_snapshot(c, snaps + (size_t)t * rec);
    }

    double dx[INS], dhp[HS], *dh_cur = xmalloc((size_t)HS * sizeof(double));
    double *dxA = xmalloc((size_t)T * INS * sizeof(double));
    double *dhA = xmalloc((size_t)T * HS * sizeof(double));
    double *dhIn = xmalloc((size_t)T * HS * sizeof(double));   /* A 里每步实际用的 dh */

    gru_cell_zero_grad(c);
    memcpy(dh_cur, DH + (T - 1) * HS, (size_t)HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        memcpy(dhIn + (size_t)t * HS, dh_cur, (size_t)HS * sizeof(double));
        gru_cell_cache_restore(c, snaps + (size_t)t * rec);
        gru_cell_backward(c, dh_cur, dx, dhp);
        memcpy(dxA + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhA + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
    }
    GradPack g = pack_gru(c);
    GradPack accA = pack_dup(&g);

    GradPack accB = pack_dst(&g);
    double *dxB = xmalloc((size_t)T * INS * sizeof(double));
    double *dhB = xmalloc((size_t)T * HS * sizeof(double));
    for (int t = T - 1; t >= 0; t--) {
        gru_cell_zero_grad(c);
        gru_cell_cache_restore(c, snaps + (size_t)t * rec);
        gru_cell_backward(c, dhIn + (size_t)t * HS, dx, dhp);
        memcpy(dxB + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
        memcpy(dhB + (size_t)t * HS, dhp, (size_t)HS * sizeof(double));
        GradPack gt = pack_gru(c);
        pack_accum(&accB, &gt);
    }

    pack_check_bits("GRUCell", &accA, &accB);
    check_bits("GRUCell: 每步 dx", dxA, dxB, T * INS);
    check_bits("GRUCell: 每步 dh_prev", dhA, dhB, T * HS);

    pack_free(&accA); pack_free(&accB);
    free(W_ih); free(W_hh); free(b_ih); free(b_hh);
    free(X); free(DH); free(snaps); free(dxA); free(dhA); free(dxB); free(dhB);
    free(dhIn); free(dh_cur);
    gru_cell_free(c);
}

/* =====================================================================
 * [2] restore_cache 等价性（多层）：备份 step_cache -> 用别的序列覆盖缓存
 *     -> 还原 -> backward，必须与"重新 forward 后 backward"逐位一致
 * ===================================================================== */

/* 备份 / 还原 step_cache 的实际数据（只动内容，不换缓冲，避免所有权问题） */
typedef struct {
    double *rec;
    size_t n;
    int cached_T;
} CacheSnap;

static CacheSnap cache_save(RNNStepCache *sc)
{
    CacheSnap s;
    s.n = (size_t)sc->max_T * (size_t)sc->rec_stride;
    s.rec = xmalloc(s.n * sizeof(double));
    memcpy(s.rec, sc->rec, s.n * sizeof(double));
    s.cached_T = sc->cached_T;
    return s;
}

static void cache_load(RNNStepCache *sc, const CacheSnap *s)
{
    memcpy(sc->rec, s->rec, s->n * sizeof(double));
    sc->cached_T = s->cached_T;
}

static void test_rnn_net_restore(void)
{
    RNN *net = rnn_create(INS, HS, LAYERS);
    rand_net_weights(net, 0);

    int xs[2] = { T, INS };
    int os[2] = { T, HS };
    int hs2[2] = { LAYERS, HS };
    Tensor *x = tensor_new(2, xs), *x2 = tensor_new(2, xs);
    Tensor *out = tensor_new(2, os), *out2 = tensor_new(2, os);
    Tensor *hl = tensor_new(2, hs2), *dhl = tensor_new(2, hs2);
    Tensor *dxA = tensor_new(2, xs), *dxB = tensor_new(2, xs);
    fill_rand(x->data, T * INS, 1.0);
    fill_rand(x2->data, T * INS, 1.0);
    fill_rand(dhl->data, LAYERS * HS, 1.0);

    /* 路径 A：forward 之后直接 backward（用的是刚写好的 step_cache） */
    rnn_forward(net, x, out, hl);
    rnn_zero_grad(net);
    rnn_backward(net, dhl, dxA);
    GradPack ref[LAYERS];
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_rnn(rnn_layer(net, l));
        ref[l] = pack_dup(&cur);
    }

    /* 备份 step_cache，然后用另一条序列把 cell 内部缓存 + 快照区全部覆盖 */
    CacheSnap snap = cache_save(&net->step_cache);
    rnn_forward(net, x2, out2, NULL);

    /* 路径 B：还原快照后 backward */
    cache_load(&net->step_cache, &snap);
    rnn_zero_grad(net);
    rnn_backward(net, dhl, dxB);
    printf("    RNN: T=%d, layers=%d, cached_T=%d\n", T, LAYERS, rnn_cached_steps(net));
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_rnn(rnn_layer(net, l));
        char tag[64];
        snprintf(tag, sizeof(tag), "RNN layer %d", l);
        pack_check_bits(tag, &ref[l], &cur);
        pack_free(&ref[l]);
    }
    check_bits("RNN: dx_seq（恢复缓存 vs 重新 forward）", dxA->data, dxB->data, T * INS);

    free(snap.rec);
    tensor_free(x); tensor_free(x2); tensor_free(out); tensor_free(out2);
    tensor_free(hl); tensor_free(dhl); tensor_free(dxA); tensor_free(dxB);
    rnn_free(net);
}

static void test_lstm_net_restore(void)
{
    LSTM *net = lstm_create(INS, HS, LAYERS);
    rand_net_weights(net, 1);

    int xs[2] = { T, INS };
    int os[2] = { T, HS };
    int hs2[2] = { LAYERS, HS };
    Tensor *x = tensor_new(2, xs), *x2 = tensor_new(2, xs);
    Tensor *out = tensor_new(2, os), *out2 = tensor_new(2, os);
    Tensor *hl = tensor_new(2, hs2), *cl = tensor_new(2, hs2);
    Tensor *dhl = tensor_new(2, hs2);
    Tensor *dxA = tensor_new(2, xs), *dxB = tensor_new(2, xs);
    fill_rand(x->data, T * INS, 1.0);
    fill_rand(x2->data, T * INS, 1.0);
    fill_rand(dhl->data, LAYERS * HS, 1.0);

    lstm_forward(net, x, out, hl, cl);
    lstm_zero_grad(net);
    lstm_backward(net, dhl, dxA);
    GradPack ref[LAYERS];
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_lstm(lstm_layer(net, l));
        ref[l] = pack_dup(&cur);
    }

    CacheSnap snap = cache_save(&net->step_cache);
    lstm_forward(net, x2, out2, NULL, NULL);

    cache_load(&net->step_cache, &snap);
    lstm_zero_grad(net);
    lstm_backward(net, dhl, dxB);
    printf("    LSTM: T=%d, layers=%d, cached_T=%d\n", T, LAYERS, lstm_cached_steps(net));
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_lstm(lstm_layer(net, l));
        char tag[64];
        snprintf(tag, sizeof(tag), "LSTM layer %d", l);
        pack_check_bits(tag, &ref[l], &cur);
        pack_free(&ref[l]);
    }
    check_bits("LSTM: dx_seq（恢复缓存 vs 重新 forward）", dxA->data, dxB->data, T * INS);

    free(snap.rec);
    tensor_free(x); tensor_free(x2); tensor_free(out); tensor_free(out2);
    tensor_free(hl); tensor_free(cl); tensor_free(dhl); tensor_free(dxA); tensor_free(dxB);
    lstm_free(net);
}

static void test_gru_net_restore(void)
{
    GRU *net = gru_create(INS, HS, LAYERS);
    rand_net_weights(net, 2);

    int xs[2] = { T, INS };
    int os[2] = { T, HS };
    int hs2[2] = { LAYERS, HS };
    Tensor *x = tensor_new(2, xs), *x2 = tensor_new(2, xs);
    Tensor *out = tensor_new(2, os), *out2 = tensor_new(2, os);
    Tensor *hl = tensor_new(2, hs2), *dhl = tensor_new(2, hs2);
    Tensor *dxA = tensor_new(2, xs), *dxB = tensor_new(2, xs);
    fill_rand(x->data, T * INS, 1.0);
    fill_rand(x2->data, T * INS, 1.0);
    fill_rand(dhl->data, LAYERS * HS, 1.0);

    gru_forward(net, x, out, hl);
    gru_zero_grad(net);
    gru_backward(net, dhl, dxA);
    GradPack ref[LAYERS];
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_gru(gru_layer(net, l));
        ref[l] = pack_dup(&cur);
    }

    CacheSnap snap = cache_save(&net->step_cache);
    gru_forward(net, x2, out2, NULL);

    cache_load(&net->step_cache, &snap);
    gru_zero_grad(net);
    gru_backward(net, dhl, dxB);
    printf("    GRU: T=%d, layers=%d, cached_T=%d\n", T, LAYERS, gru_cached_steps(net));
    for (int l = 0; l < LAYERS; l++) {
        GradPack cur = pack_gru(gru_layer(net, l));
        char tag[64];
        snprintf(tag, sizeof(tag), "GRU layer %d", l);
        pack_check_bits(tag, &ref[l], &cur);
        pack_free(&ref[l]);
    }
    check_bits("GRU: dx_seq（恢复缓存 vs 重新 forward）", dxA->data, dxB->data, T * INS);

    free(snap.rec);
    tensor_free(x); tensor_free(x2); tensor_free(out); tensor_free(out2);
    tensor_free(hl); tensor_free(dhl); tensor_free(dxA); tensor_free(dxB);
    gru_free(net);
}

/* =====================================================================
 * [3] 有限差分校验：L = Σ_t w_t · h_t，对全部参数做中心差分
 * ===================================================================== */

static double g_eps = 1e-6;

/* 三种 cell 的损失函数（用文件级上下文，避免为测试引入回调框架） */
static RNNCell *g_rnn_cell;
static LSTMCell *g_lstm_cell;
static GRUCell *g_gru_cell;
static const double *g_fd_X;
static const double *g_fd_W;

static double fd_loss_rnn(void)
{
    const double *h = NULL;
    double L = 0.0;
    for (int t = 0; t < T; t++) {
        h = rnn_cell_forward(g_rnn_cell, g_fd_X + t * INS, h);
        for (int j = 0; j < HS; j++) L += g_fd_W[t * HS + j] * h[j];
    }
    return L;
}

static double fd_loss_lstm(void)
{
    const double *h = NULL, *cs = NULL;
    double L = 0.0;
    for (int t = 0; t < T; t++) {
        h = lstm_cell_forward(g_lstm_cell, g_fd_X + t * INS, h, cs);
        cs = g_lstm_cell->c;
        for (int j = 0; j < HS; j++) L += g_fd_W[t * HS + j] * h[j];
    }
    return L;
}

static double fd_loss_gru(void)
{
    const double *h = NULL;
    double L = 0.0;
    for (int t = 0; t < T; t++) {
        h = gru_cell_forward(g_gru_cell, g_fd_X + t * INS, h);
        for (int j = 0; j < HS; j++) L += g_fd_W[t * HS + j] * h[j];
    }
    return L;
}

/* 对 param[0..n) 做中心差分，与 analytic 比较 */
static void fd_check(const char *tag, double *param, int n, const double *analytic,
                     double (*loss)(void), double tol)
{
    double *num = xmalloc((size_t)n * sizeof(double));
    for (int k = 0; k < n; k++) {
        double orig = param[k];
        param[k] = orig + g_eps;
        double lp = loss();
        param[k] = orig - g_eps;
        double lm = loss();
        param[k] = orig;                       /* 逐位还原 */
        num[k] = (lp - lm) / (2.0 * g_eps);
    }
    check_tol(tag, analytic, num, n, tol);
    free(num);
}

static void test_finite_diff(void)
{
    double *X = xmalloc((size_t)T * INS * sizeof(double));
    double *Wt = xmalloc((size_t)T * HS * sizeof(double));
    fill_rand(X, T * INS, 1.0);
    fill_rand(Wt, T * HS, 1.0);
    g_fd_X = X;
    g_fd_W = Wt;

    /* ---------- RNNCell ---------- */
    {
        RNNCell *c = rnn_cell_create(INS, HS);
        g_rnn_cell = c;
        double *W_ih, *W_hh, *b_ih, *b_hh;
        rand_params(&W_ih, &W_hh, &b_ih, &b_hh, HS, INS, HS);
        rnn_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

        int rec = rnn_cell_cache_size(c);
        double *snaps = xmalloc((size_t)T * rec * sizeof(double));
        const double *h = NULL;
        for (int t = 0; t < T; t++) {
            h = rnn_cell_forward(c, X + t * INS, h);
            rnn_cell_cache_snapshot(c, snaps + (size_t)t * rec);
        }
        /* 解析梯度：dL/dh_t = W_t + 从 t+1 回传的部分 */
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double dhp[HS];
        rnn_cell_zero_grad(c);
        memcpy(dh_cur, Wt + (T - 1) * HS, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            rnn_cell_cache_restore(c, snaps + (size_t)t * rec);
            rnn_cell_backward(c, dh_cur, NULL, dhp);
            if (t > 0) for (int j = 0; j < HS; j++) dh_cur[j] = Wt[(t - 1) * HS + j] + dhp[j];
        }
        fd_check("RNNCell: grad_W_ih（有限差分）", c->W_ih, HS * INS, c->grad_W_ih, fd_loss_rnn, 1e-5);
        fd_check("RNNCell: grad_W_hh（有限差分）", c->W_hh, HS * HS, c->grad_W_hh, fd_loss_rnn, 1e-5);
        fd_check("RNNCell: grad_b_ih（有限差分）", c->b_ih, HS, c->grad_b_ih, fd_loss_rnn, 1e-5);
        fd_check("RNNCell: grad_b_hh（有限差分）", c->b_hh, HS, c->grad_b_hh, fd_loss_rnn, 1e-5);

        free(dh_cur); free(snaps);
        free(W_ih); free(W_hh); free(b_ih); free(b_hh);
        rnn_cell_free(c);
    }

    /* ---------- LSTMCell ---------- */
    {
        LSTMCell *c = lstm_cell_create(INS, HS);
        g_lstm_cell = c;
        int gs = c->gate_size;
        double *W_ih, *W_hh, *b_ih, *b_hh;
        rand_params(&W_ih, &W_hh, &b_ih, &b_hh, gs, INS, HS);
        lstm_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

        int rec = lstm_cell_cache_size(c);
        double *snaps = xmalloc((size_t)T * rec * sizeof(double));
        const double *h = NULL, *cs = NULL;
        for (int t = 0; t < T; t++) {
            h = lstm_cell_forward(c, X + t * INS, h, cs);
            cs = c->c;
            lstm_cell_cache_snapshot(c, snaps + (size_t)t * rec);
        }
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double *dc_cur = xmalloc((size_t)HS * sizeof(double));
        double dhp[HS], dcp[HS];
        lstm_cell_zero_grad(c);
        memcpy(dh_cur, Wt + (T - 1) * HS, (size_t)HS * sizeof(double));
        memset(dc_cur, 0, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            lstm_cell_cache_restore(c, snaps + (size_t)t * rec);
            lstm_cell_backward(c, dh_cur, dc_cur, NULL, dhp, dcp);
            if (t > 0) for (int j = 0; j < HS; j++) dh_cur[j] = Wt[(t - 1) * HS + j] + dhp[j];
            memcpy(dc_cur, dcp, (size_t)HS * sizeof(double));
        }
        fd_check("LSTMCell: grad_W_ih（有限差分）", c->W_ih, gs * INS, c->grad_W_ih, fd_loss_lstm, 1e-5);
        fd_check("LSTMCell: grad_W_hh（有限差分）", c->W_hh, gs * HS, c->grad_W_hh, fd_loss_lstm, 1e-5);
        fd_check("LSTMCell: grad_b_ih（有限差分）", c->b_ih, gs, c->grad_b_ih, fd_loss_lstm, 1e-5);
        fd_check("LSTMCell: grad_b_hh（有限差分）", c->b_hh, gs, c->grad_b_hh, fd_loss_lstm, 1e-5);

        free(dh_cur); free(dc_cur); free(snaps);
        free(W_ih); free(W_hh); free(b_ih); free(b_hh);
        lstm_cell_free(c);
    }

    /* ---------- GRUCell ---------- */
    {
        GRUCell *c = gru_cell_create(INS, HS);
        g_gru_cell = c;
        int gs = c->gate_size;
        double *W_ih, *W_hh, *b_ih, *b_hh;
        rand_params(&W_ih, &W_hh, &b_ih, &b_hh, gs, INS, HS);
        gru_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

        int rec = gru_cell_cache_size(c);
        double *snaps = xmalloc((size_t)T * rec * sizeof(double));
        const double *h = NULL;
        for (int t = 0; t < T; t++) {
            h = gru_cell_forward(c, X + t * INS, h);
            gru_cell_cache_snapshot(c, snaps + (size_t)t * rec);
        }
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double dhp[HS];
        gru_cell_zero_grad(c);
        memcpy(dh_cur, Wt + (T - 1) * HS, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            gru_cell_cache_restore(c, snaps + (size_t)t * rec);
            gru_cell_backward(c, dh_cur, NULL, dhp);
            if (t > 0) for (int j = 0; j < HS; j++) dh_cur[j] = Wt[(t - 1) * HS + j] + dhp[j];
        }
        fd_check("GRUCell: grad_W_ih（有限差分）", c->W_ih, gs * INS, c->grad_W_ih, fd_loss_gru, 1e-5);
        fd_check("GRUCell: grad_W_hh（有限差分）", c->W_hh, gs * HS, c->grad_W_hh, fd_loss_gru, 1e-5);
        fd_check("GRUCell: grad_b_ih（有限差分）", c->b_ih, gs, c->grad_b_ih, fd_loss_gru, 1e-5);
        fd_check("GRUCell: grad_b_hh（有限差分）", c->b_hh, gs, c->grad_b_hh, fd_loss_gru, 1e-5);

        free(dh_cur); free(snaps);
        free(W_ih); free(W_hh); free(b_ih); free(b_hh);
        gru_cell_free(c);
    }

    free(X); free(Wt);
}

/* =====================================================================
 * [4] 多层 forward / backward：与手工多层链 + 手工 BPTT 循环逐位一致
 * ===================================================================== */

static void test_net_vs_manual(void)
{
    int xs[2] = { T, INS };
    int os[2] = { T, HS };
    int hs2[2] = { LAYERS, HS };
    Tensor *x = tensor_new(2, xs);
    Tensor *out = tensor_new(2, os);
    Tensor *hl = tensor_new(2, hs2), *cl = tensor_new(2, hs2);
    Tensor *dhl = tensor_new(2, hs2);
    Tensor *dxA = tensor_new(2, xs), *dxB = tensor_new(2, xs);
    fill_rand(x->data, T * INS, 1.0);
    fill_rand(dhl->data, LAYERS * HS, 1.0);

    /* ---------- RNN ---------- */
    {
        RNN *net = rnn_create(INS, HS, LAYERS);
        rand_net_weights(net, 0);
        rnn_forward(net, x, out, hl);

        /* 手工多层链：每层一个独立 cell，权重与 net 相同 */
        RNNCell *man[LAYERS];
        for (int l = 0; l < LAYERS; l++) {
            man[l] = rnn_cell_create((l == 0) ? INS : HS, HS);
            rnn_cell_set_params(man[l], net->cells[l].W_ih, net->cells[l].W_hh,
                                net->cells[l].b_ih, net->cells[l].b_hh);
        }
        double *man_out = xmalloc((size_t)T * HS * sizeof(double));
        for (int t = 0; t < T; t++) {
            const double *lin = x->data + (size_t)t * INS;
            for (int l = 0; l < LAYERS; l++) lin = rnn_cell_forward(man[l], lin, man[l]->h);
            memcpy(man_out + (size_t)t * HS, lin, (size_t)HS * sizeof(double));
        }
        check_bits("RNN: out_seq（net vs 手工多层链）", out->data, man_out, T * HS);
        for (int l = 0; l < LAYERS; l++) {
            char tag[64];
            snprintf(tag, sizeof(tag), "RNN: h_last layer %d (net vs manual chain)", l);
            check_bits(tag, hl->data + (size_t)l * HS, man[l]->h, HS);
        }

        /* 手工 BPTT（完全照 Python 示例的循环写法，单层） */
        rnn_zero_grad(net);
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double dx[INS], dhp[HS];
        memcpy(dh_cur, dhl->data + (size_t)(LAYERS - 1) * HS, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            rnn_restore_cache(net, t, LAYERS - 1);
            rnn_cell_backward(rnn_layer(net, LAYERS - 1), dh_cur, dx, dhp);
            memcpy(dxB->data + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
            memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
        }
        GradPack cur = pack_rnn(rnn_layer(net, LAYERS - 1));
        GradPack accA = pack_dup(&cur);
        rnn_zero_grad(net);
        rnn_backward(net, dhl, dxA);
        cur = pack_rnn(rnn_layer(net, LAYERS - 1));
        GradPack accB = pack_dup(&cur);
        pack_check_bits("RNN 顶层: rnn_backward vs 手工 BPTT 循环", &accA, &accB);
        if (LAYERS == 1)
            check_bits("RNN: dx_seq（rnn_backward vs 手工循环）", dxA->data, dxB->data, T * INS);
        pack_free(&accA); pack_free(&accB);
        free(dh_cur); free(man_out);
        for (int l = 0; l < LAYERS; l++) rnn_cell_free(man[l]);
        rnn_free(net);
    }

    /* ---------- LSTM ---------- */
    {
        LSTM *net = lstm_create(INS, HS, LAYERS);
        rand_net_weights(net, 1);
        lstm_forward(net, x, out, hl, cl);

        LSTMCell *man[LAYERS];
        for (int l = 0; l < LAYERS; l++) {
            man[l] = lstm_cell_create((l == 0) ? INS : HS, HS);
            lstm_cell_set_params(man[l], net->cells[l].W_ih, net->cells[l].W_hh,
                                 net->cells[l].b_ih, net->cells[l].b_hh);
        }
        double *man_out = xmalloc((size_t)T * HS * sizeof(double));
        for (int t = 0; t < T; t++) {
            const double *lin = x->data + (size_t)t * INS;
            for (int l = 0; l < LAYERS; l++)
                lin = lstm_cell_forward(man[l], lin, man[l]->h, man[l]->c);
            memcpy(man_out + (size_t)t * HS, lin, (size_t)HS * sizeof(double));
        }
        check_bits("LSTM: out_seq（net vs 手工多层链）", out->data, man_out, T * HS);
        for (int l = 0; l < LAYERS; l++) {
            char tag[64];
            snprintf(tag, sizeof(tag), "LSTM: h_last layer %d (net vs manual chain)", l);
            check_bits(tag, hl->data + (size_t)l * HS, man[l]->h, HS);
            snprintf(tag, sizeof(tag), "LSTM: c_last layer %d (net vs manual chain)", l);
            check_bits(tag, cl->data + (size_t)l * HS, man[l]->c, HS);
        }

        lstm_zero_grad(net);
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double *dc_cur = xmalloc((size_t)HS * sizeof(double));
        double dx[INS], dhp[HS], dcp[HS];
        memcpy(dh_cur, dhl->data + (size_t)(LAYERS - 1) * HS, (size_t)HS * sizeof(double));
        memset(dc_cur, 0, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            lstm_restore_cache(net, t, LAYERS - 1);
            lstm_cell_backward(lstm_layer(net, LAYERS - 1), dh_cur, dc_cur, dx, dhp, dcp);
            memcpy(dxB->data + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
            memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
            memcpy(dc_cur, dcp, (size_t)HS * sizeof(double));
        }
        GradPack cur = pack_lstm(lstm_layer(net, LAYERS - 1));
        GradPack accA = pack_dup(&cur);
        lstm_zero_grad(net);
        lstm_backward(net, dhl, dxA);
        cur = pack_lstm(lstm_layer(net, LAYERS - 1));
        GradPack accB = pack_dup(&cur);
        pack_check_bits("LSTM 顶层: lstm_backward vs 手工 BPTT 循环", &accA, &accB);
        if (LAYERS == 1)
            check_bits("LSTM: dx_seq（lstm_backward vs 手工循环）", dxA->data, dxB->data, T * INS);
        pack_free(&accA); pack_free(&accB);
        free(dh_cur); free(dc_cur); free(man_out);
        for (int l = 0; l < LAYERS; l++) lstm_cell_free(man[l]);
        lstm_free(net);
    }

    /* ---------- GRU ---------- */
    {
        GRU *net = gru_create(INS, HS, LAYERS);
        rand_net_weights(net, 2);
        gru_forward(net, x, out, hl);

        GRUCell *man[LAYERS];
        for (int l = 0; l < LAYERS; l++) {
            man[l] = gru_cell_create((l == 0) ? INS : HS, HS);
            gru_cell_set_params(man[l], net->cells[l].W_ih, net->cells[l].W_hh,
                                net->cells[l].b_ih, net->cells[l].b_hh);
        }
        double *man_out = xmalloc((size_t)T * HS * sizeof(double));
        for (int t = 0; t < T; t++) {
            const double *lin = x->data + (size_t)t * INS;
            for (int l = 0; l < LAYERS; l++) lin = gru_cell_forward(man[l], lin, man[l]->h);
            memcpy(man_out + (size_t)t * HS, lin, (size_t)HS * sizeof(double));
        }
        check_bits("GRU: out_seq（net vs 手工多层链）", out->data, man_out, T * HS);
        for (int l = 0; l < LAYERS; l++) {
            char tag[64];
            snprintf(tag, sizeof(tag), "GRU: h_last layer %d (net vs manual chain)", l);
            check_bits(tag, hl->data + (size_t)l * HS, man[l]->h, HS);
        }

        gru_zero_grad(net);
        double *dh_cur = xmalloc((size_t)HS * sizeof(double));
        double dx[INS], dhp[HS];
        memcpy(dh_cur, dhl->data + (size_t)(LAYERS - 1) * HS, (size_t)HS * sizeof(double));
        for (int t = T - 1; t >= 0; t--) {
            gru_restore_cache(net, t, LAYERS - 1);
            gru_cell_backward(gru_layer(net, LAYERS - 1), dh_cur, dx, dhp);
            memcpy(dxB->data + (size_t)t * INS, dx, (size_t)INS * sizeof(double));
            memcpy(dh_cur, dhp, (size_t)HS * sizeof(double));
        }
        GradPack cur = pack_gru(gru_layer(net, LAYERS - 1));
        GradPack accA = pack_dup(&cur);
        gru_zero_grad(net);
        gru_backward(net, dhl, dxA);
        cur = pack_gru(gru_layer(net, LAYERS - 1));
        GradPack accB = pack_dup(&cur);
        pack_check_bits("GRU 顶层: gru_backward vs 手工 BPTT 循环", &accA, &accB);
        if (LAYERS == 1)
            check_bits("GRU: dx_seq（gru_backward vs 手工循环）", dxA->data, dxB->data, T * INS);
        pack_free(&accA); pack_free(&accB);
        free(dh_cur); free(man_out);
        for (int l = 0; l < LAYERS; l++) gru_cell_free(man[l]);
        gru_free(net);
    }

    tensor_free(x); tensor_free(out); tensor_free(hl); tensor_free(cl);
    tensor_free(dhl); tensor_free(dxA); tensor_free(dxB);
}

/* =====================================================================
 * [5] 优化器：与手工更新逐位一致
 * ===================================================================== */

static void opt_check(const char *tag, OptKind kind)
{
    const double lr = 0.03;
    RNN *net = rnn_create(INS, HS, 1);
    RNNCell *c = rnn_layer(net, 0);
    double *W_ih, *W_hh, *b_ih, *b_hh;
    rand_params(&W_ih, &W_hh, &b_ih, &b_hh, HS, INS, HS);
    rnn_cell_set_params(c, W_ih, W_hh, b_ih, b_hh);

    int xs[2] = { T, INS };
    int hs2[2] = { 1, HS };
    Tensor *x = tensor_new(2, xs), *hl = tensor_new(2, hs2);
    Tensor *dhl = tensor_new(2, hs2), *dx = tensor_new(2, xs);
    fill_rand(x->data, T * INS, 1.0);
    fill_rand(dhl->data, HS, 1.0);

    rnn_forward(net, x, NULL, hl);
    rnn_zero_grad(net);
    rnn_backward(net, dhl, dx);

    /* 手工更新（Adam 的 m / v 从 0 开始，step = 1，与 ai.c 的公式一致） */
    double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    double c1 = 1.0 - b1, c2 = 1.0 - b2, o1 = 1.0 - b1, o2 = 1.0 - b2;
    double *p[4] = { c->W_ih, c->W_hh, c->b_ih, c->b_hh };
    double *g[4] = { c->grad_W_ih, c->grad_W_hh, c->grad_b_ih, c->grad_b_hh };
    int n[4] = { HS * INS, HS * HS, HS, HS };
    double *ref[4];
    for (int k = 0; k < 4; k++) {
        ref[k] = xmalloc((size_t)n[k] * sizeof(double));
        memcpy(ref[k], p[k], (size_t)n[k] * sizeof(double));
        for (int i = 0; i < n[k]; i++) {
            if (kind == OPT_ADAM) {
                double m = o1 * g[k][i];
                double v = o2 * g[k][i] * g[k][i];
                ref[k][i] -= lr * (m / c1) / (sqrt(v / c2) + eps);
            } else {
                ref[k][i] -= lr * g[k][i];
            }
        }
    }

    Optimizer o;
    rnn_optimizer_init(&o, kind, lr);
    rnn_optimizer_step(net, &o);

    for (int k = 0; k < 4; k++) {
        char t[144];
        snprintf(t, sizeof(t), "%s: %s", tag, GRAD_NAME[k] + 5 /* 去掉 "grad_" */);
        check_bits(t, p[k], ref[k], n[k]);
        free(ref[k]);
    }

    free(W_ih); free(W_hh); free(b_ih); free(b_hh);
    tensor_free(x); tensor_free(hl); tensor_free(dhl); tensor_free(dx);
    rnn_free(net);
}

/* ================================ main ================================ */

static int mode_self(void)
{
    printf("test_rnn —— RNNCell/RNN、LSTMCell/LSTM、GRUCell/GRU 移植自测\n");
    printf("规模：T=%d, input_size=%d, hidden_size=%d, layers=%d\n", T, INS, HS, LAYERS);

    printf("\n[1] 步进一致性：一次 zero_grad + 逐步 backward == 各步隔离梯度之和\n");
    test_rnn_cell_accum();
    test_lstm_cell_accum();
    test_gru_cell_accum();

    printf("\n[2] restore_cache 等价性（缓存被覆盖后还原 vs 重新 forward）\n");
    test_rnn_net_restore();
    test_lstm_net_restore();
    test_gru_net_restore();

    printf("\n[3] 有限差分校验（中心差分 eps=%.0e，阈值 1e-5）\n", g_eps);
    test_finite_diff();

    printf("\n[4] 多层 forward/backward vs 手工多层链 / 手工 BPTT 循环\n");
    test_net_vs_manual();

    printf("\n[5] 优化器：SGD / Adam 一步 vs 手工更新\n");
    opt_check("RNN SGD", OPT_SGD);
    opt_check("RNN Adam", OPT_ADAM);

    printf("\n  -> %s\n", g_fail == 0 ? "PASS（全部通过）" : "FAIL（存在失败项）");
    return g_fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    /* 约定与项目里其它自测一致：./tests/test_rnn [self] */
    const char *mode = (argc > 1) ? argv[1] : "self";
    if (strcmp(mode, "self") == 0) return mode_self();
    fprintf(stderr, "未知模式: %s（可用: self）\n", mode);
    return 2;
}
