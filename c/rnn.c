/*
 * rnn.c —— RNN / LSTM / GRU 家族（含 BPTT 用的 step cache 机制）
 *
 * 移植自 AI.py 的 RNNCell / RNN、LSTMCell / LSTM、GRUCell / GRU。
 * 数值上逐条对齐 Python 的运算顺序（详见 rnn.h 顶部的约定）：
 *   - 前向预激活从偏置开始累加，再按 i 升序累加 W_ih 行 / W_hh 行；
 *   - GRU 的 _linear 分段（z_ih / z_hh 各自从偏置起累加，最后才相加）；
 *   - 反向的 dx / dh_prev 是显式索引循环，只有 GRU 用 Neumaier 补偿求和
 *     （Python 写的是 sum(generator)，CPython 3.12+ 对浮点用 Neumaier）；
 *   - 梯度一律原地累加。
 *
 * 内存：全部缓冲在 create / reserve 时一次性分配好，forward / backward /
 * optimizer_step 全程不 malloc（项目硬要求，check_noalloc 会检查）。
 */
#include "rnn.h"
#include "numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================ 基础工具 ============================ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "rnn: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "rnn: out of memory (%zu x %zu bytes)\n", count, size);
        exit(1);
    }
    return p;
}

static double *dalloc(int n)
{
    return xcalloc((size_t)(n > 0 ? n : 1), sizeof(double));
}

/* 形状 / 长度 / 指针不合法一律报错退出，不静默截断（项目约定） */
static void rnn_fail(const char *who, const char *msg)
{
    fprintf(stderr, "%s: %s\n", who, msg);
    exit(1);
}

static void require_ptr(const void *p, const char *who, const char *what)
{
    if (!p) {
        char buf[160];
        snprintf(buf, sizeof(buf), "%s 不能为 NULL", what);
        rnn_fail(who, buf);
    }
}

/* Σ_j W[j*stride + col] * d[j]，对齐 Python 的
 *   sum(self.W_ih[j][i] * d_all_gates[j] for j in range(n))
 * 求和方式与 numeric.h 的 dot_neumaier 相同（Neumaier 补偿求和），只是权重按列取。 */
static double sum_col_neumaier(const double *W, int stride, int col, const double *d, int n)
{
    double s = 0.0, c = 0.0;
    for (int j = 0; j < n; j++) {
        double x = W[(size_t)j * (size_t)stride + (size_t)col] * d[j];
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* sigmoid：Python 是 1.0 / (1.0 + math.exp(-x)) */
static double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/* 随机初始化（仅 C 侧独立使用；跨语言对齐请用 set_params 灌权重）。
 * 与 ai.c 的同名工具一致：xorshift64 + Box-Muller，分布与 Python 的
 * random.gauss(0, std) 相同，但随机序列不同。 */
static unsigned long long rng_next(unsigned long long *s)
{
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(unsigned long long *s)
{
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

static double rng_gauss(unsigned long long *s)
{
    double u1 = rng_uniform(s);
    double u2 = rng_uniform(s);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* ============================== RNNCell ============================== */
/*
 * h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)
 * 前向缓存字段顺序（= Python 的 _CACHE_FIELDS）：x, h_prev, z
 */
RNNCell *rnn_cell_create(int input_size, int hidden_size)
{
    if (input_size <= 0 || hidden_size <= 0)
        rnn_fail("rnn_cell_create", "input_size / hidden_size 必须为正");
    RNNCell *c = xmalloc(sizeof(RNNCell));
    memset(c, 0, sizeof(*c));
    c->input_size = input_size;
    c->hidden_size = hidden_size;

    int ins = input_size, hs = hidden_size;
    size_t nw_ih = (size_t)hs * (size_t)ins;
    size_t nw_hh = (size_t)hs * (size_t)hs;

    c->W_ih = dalloc((int)nw_ih);
    c->W_hh = dalloc((int)nw_hh);
    c->b_ih = dalloc(hs);
    c->b_hh = dalloc(hs);

    c->grad_W_ih = dalloc((int)nw_ih);
    c->grad_W_hh = dalloc((int)nw_hh);
    c->grad_b_ih = dalloc(hs);
    c->grad_b_hh = dalloc(hs);

    c->m_W_ih = dalloc((int)nw_ih);  c->v_W_ih = dalloc((int)nw_ih);
    c->m_W_hh = dalloc((int)nw_hh);  c->v_W_hh = dalloc((int)nw_hh);
    c->m_b_ih = dalloc(hs);          c->v_b_ih = dalloc(hs);
    c->m_b_hh = dalloc(hs);          c->v_b_hh = dalloc(hs);

    c->x = dalloc(ins);
    c->h_prev = dalloc(hs);
    c->z = dalloc(hs);
    c->h = dalloc(hs);
    c->scratch = dalloc(hs);

    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    rnn_cell_init_params(c, &seed);
    return c;
}

void rnn_cell_init_params(RNNCell *c, unsigned long long *rng_state)
{
    double std_ih = sqrt(1.0 / (double)c->input_size);
    double std_hh = sqrt(1.0 / (double)c->hidden_size);
    for (int j = 0; j < c->hidden_size; j++) {
        for (int i = 0; i < c->input_size; i++)
            c->W_ih[(size_t)j * c->input_size + i] = rng_gauss(rng_state) * std_ih;
        for (int i = 0; i < c->hidden_size; i++)
            c->W_hh[(size_t)j * c->hidden_size + i] = rng_gauss(rng_state) * std_hh;
        c->b_ih[j] = 0.0;
        c->b_hh[j] = 0.0;
    }
}

void rnn_cell_free(RNNCell *c)
{
    if (!c) return;
    free(c->W_ih); free(c->W_hh); free(c->b_ih); free(c->b_hh);
    free(c->grad_W_ih); free(c->grad_W_hh); free(c->grad_b_ih); free(c->grad_b_hh);
    free(c->m_W_ih); free(c->v_W_ih); free(c->m_W_hh); free(c->v_W_hh);
    free(c->m_b_ih); free(c->v_b_ih); free(c->m_b_hh); free(c->v_b_hh);
    free(c->x); free(c->h_prev); free(c->z); free(c->h); free(c->scratch);
    memset(c, 0, sizeof(*c));
}

void rnn_cell_set_params(RNNCell *c, const double *W_ih, const double *W_hh,
                         const double *b_ih, const double *b_hh)
{
    const char *who = "rnn_cell_set_params";
    require_ptr(W_ih, who, "W_ih");
    require_ptr(W_hh, who, "W_hh");
    memcpy(c->W_ih, W_ih, (size_t)c->hidden_size * c->input_size * sizeof(double));
    memcpy(c->W_hh, W_hh, (size_t)c->hidden_size * c->hidden_size * sizeof(double));
    if (b_ih) memcpy(c->b_ih, b_ih, (size_t)c->hidden_size * sizeof(double));
    else      memset(c->b_ih, 0, (size_t)c->hidden_size * sizeof(double));
    if (b_hh) memcpy(c->b_hh, b_hh, (size_t)c->hidden_size * sizeof(double));
    else      memset(c->b_hh, 0, (size_t)c->hidden_size * sizeof(double));
}

void rnn_cell_zero_grad(RNNCell *c)
{
    size_t nw_ih = (size_t)c->hidden_size * (size_t)c->input_size;
    size_t nw_hh = (size_t)c->hidden_size * (size_t)c->hidden_size;
    memset(c->grad_W_ih, 0, nw_ih * sizeof(double));
    memset(c->grad_W_hh, 0, nw_hh * sizeof(double));
    memset(c->grad_b_ih, 0, (size_t)c->hidden_size * sizeof(double));
    memset(c->grad_b_hh, 0, (size_t)c->hidden_size * sizeof(double));
}

const double *rnn_cell_forward(RNNCell *c, const double *x, const double *h_prev)
{
    const char *who = "rnn_cell_forward";
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(x, who, "x");

    if (x != c->x) memcpy(c->x, x, (size_t)ins * sizeof(double));
    if (h_prev && h_prev != c->h_prev) memcpy(c->h_prev, h_prev, (size_t)hs * sizeof(double));
    else if (!h_prev) memset(c->h_prev, 0, (size_t)hs * sizeof(double));

    for (int j = 0; j < hs; j++) {
        double s = c->b_ih[j] + c->b_hh[j];
        for (int i = 0; i < ins; i++) s += c->W_ih[(size_t)j * ins + i] * c->x[i];
        for (int i = 0; i < hs; i++)  s += c->W_hh[(size_t)j * hs + i] * c->h_prev[i];
        c->z[j] = s;
    }
    for (int j = 0; j < hs; j++) c->h[j] = tanh(c->z[j]);
    return c->h;
}

void rnn_cell_backward(RNNCell *c, const double *dh, double *dx, double *dh_prev)
{
    const char *who = "rnn_cell_backward";
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(dh, who, "dh");

    /* tanh 导数: 1 - tanh(z)^2（Python 写的是 math.tanh(zj) ** 2，走 libm 的 pow） */
    for (int j = 0; j < hs; j++) {
        double t = tanh(c->z[j]);
        c->scratch[j] = dh[j] * (1.0 - pow(t, 2.0));
    }
    const double *dz = c->scratch;

    /* dx：for i: s = 0; for j: s += W_ih[j][i] * dz[j] */
    if (dx) {
        for (int i = 0; i < ins; i++) {
            double s = 0.0;
            for (int j = 0; j < hs; j++) s += c->W_ih[(size_t)j * ins + i] * dz[j];
            dx[i] = s;
        }
    }
    /* dh_prev：同上，用 W_hh */
    if (dh_prev) {
        for (int i = 0; i < hs; i++) {
            double s = 0.0;
            for (int j = 0; j < hs; j++) s += c->W_hh[(size_t)j * hs + i] * dz[j];
            dh_prev[i] = s;
        }
    }

    /* 梯度原地累加（契约见 rnn.h：调用方不要再手工加一遍） */
    for (int j = 0; j < hs; j++) {
        c->grad_b_ih[j] += dz[j];
        c->grad_b_hh[j] += dz[j];
        for (int i = 0; i < ins; i++)
            c->grad_W_ih[(size_t)j * ins + i] += dz[j] * c->x[i];
        for (int i = 0; i < hs; i++)
            c->grad_W_hh[(size_t)j * hs + i] += dz[j] * c->h_prev[i];
    }
}

void rnn_cell_get_grads(RNNCell *c, double **grad_W_ih, double **grad_W_hh,
                        double **grad_b_ih, double **grad_b_hh)
{
    /* 返回内部缓冲本身（同 Python 的 get_grads()）：
     * 已经包含全部时间步的累加结果，调用方不要再往回加 */
    if (grad_W_ih) *grad_W_ih = c->grad_W_ih;
    if (grad_W_hh) *grad_W_hh = c->grad_W_hh;
    if (grad_b_ih) *grad_b_ih = c->grad_b_ih;
    if (grad_b_hh) *grad_b_hh = c->grad_b_hh;
}

int rnn_cell_param_count(const RNNCell *c)
{
    return c->hidden_size * c->input_size + c->hidden_size * c->hidden_size
         + 2 * c->hidden_size;
}

int rnn_cell_cache_size(const RNNCell *c)
{
    return c->input_size + 2 * c->hidden_size;   /* x, h_prev, z */
}

void rnn_cell_cache_snapshot(const RNNCell *c, double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "rnn_cell_cache_snapshot", "rec");
    memcpy(rec, c->x, (size_t)ins * sizeof(double));            rec += ins;
    memcpy(rec, c->h_prev, (size_t)hs * sizeof(double));        rec += hs;
    memcpy(rec, c->z, (size_t)hs * sizeof(double));
}

void rnn_cell_cache_restore(RNNCell *c, const double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "rnn_cell_cache_restore", "rec");
    memcpy(c->x, rec, (size_t)ins * sizeof(double));            rec += ins;
    memcpy(c->h_prev, rec, (size_t)hs * sizeof(double));        rec += hs;
    memcpy(c->z, rec, (size_t)hs * sizeof(double));
}

/* ============================== LSTMCell ============================== */
/*
 * gates = [i | f | g | o]，每个 hidden_size 行
 *   i = σ(...)  f = σ(...)  g = tanh(...)  o = σ(...)
 *   c = f ⊙ c_prev + i ⊙ g      h = o ⊙ tanh(c)
 * 前向缓存字段顺序（= Python 的 _CACHE_FIELDS）：x, h_prev, c_prev, i, f, g, o, c
 */
LSTMCell *lstm_cell_create(int input_size, int hidden_size)
{
    if (input_size <= 0 || hidden_size <= 0)
        rnn_fail("lstm_cell_create", "input_size / hidden_size 必须为正");
    LSTMCell *c = xmalloc(sizeof(LSTMCell));
    memset(c, 0, sizeof(*c));
    c->input_size = input_size;
    c->hidden_size = hidden_size;
    c->gate_size = 4 * hidden_size;

    int ins = input_size, hs = hidden_size, gs = c->gate_size;
    size_t nw_ih = (size_t)gs * (size_t)ins;
    size_t nw_hh = (size_t)gs * (size_t)hs;

    c->W_ih = dalloc((int)nw_ih);
    c->W_hh = dalloc((int)nw_hh);
    c->b_ih = dalloc(gs);
    c->b_hh = dalloc(gs);

    c->grad_W_ih = dalloc((int)nw_ih);
    c->grad_W_hh = dalloc((int)nw_hh);
    c->grad_b_ih = dalloc(gs);
    c->grad_b_hh = dalloc(gs);

    c->m_W_ih = dalloc((int)nw_ih);  c->v_W_ih = dalloc((int)nw_ih);
    c->m_W_hh = dalloc((int)nw_hh);  c->v_W_hh = dalloc((int)nw_hh);
    c->m_b_ih = dalloc(gs);          c->v_b_ih = dalloc(gs);
    c->m_b_hh = dalloc(gs);          c->v_b_hh = dalloc(gs);

    c->x = dalloc(ins);
    c->h_prev = dalloc(hs);
    c->c_prev = dalloc(hs);
    c->i_gate = dalloc(hs);
    c->f_gate = dalloc(hs);
    c->g_gate = dalloc(hs);
    c->o_gate = dalloc(hs);
    c->c = dalloc(hs);
    c->h = dalloc(hs);

    c->gates = dalloc(gs);
    c->d_gates = dalloc(gs);
    c->dc = dalloc(hs);

    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    lstm_cell_init_params(c, &seed);
    return c;
}

void lstm_cell_init_params(LSTMCell *c, unsigned long long *rng_state)
{
    double std_ih = sqrt(1.0 / (double)c->input_size);
    double std_hh = sqrt(1.0 / (double)c->hidden_size);
    for (int j = 0; j < c->gate_size; j++) {
        for (int i = 0; i < c->input_size; i++)
            c->W_ih[(size_t)j * c->input_size + i] = rng_gauss(rng_state) * std_ih;
        for (int i = 0; i < c->hidden_size; i++)
            c->W_hh[(size_t)j * c->hidden_size + i] = rng_gauss(rng_state) * std_hh;
        c->b_ih[j] = 0.0;
        c->b_hh[j] = 0.0;
    }
}

void lstm_cell_free(LSTMCell *c)
{
    if (!c) return;
    free(c->W_ih); free(c->W_hh); free(c->b_ih); free(c->b_hh);
    free(c->grad_W_ih); free(c->grad_W_hh); free(c->grad_b_ih); free(c->grad_b_hh);
    free(c->m_W_ih); free(c->v_W_ih); free(c->m_W_hh); free(c->v_W_hh);
    free(c->m_b_ih); free(c->v_b_ih); free(c->m_b_hh); free(c->v_b_hh);
    free(c->x); free(c->h_prev); free(c->c_prev);
    free(c->i_gate); free(c->f_gate); free(c->g_gate); free(c->o_gate);
    free(c->c); free(c->h); free(c->gates); free(c->d_gates); free(c->dc);
    memset(c, 0, sizeof(*c));
}

void lstm_cell_set_params(LSTMCell *c, const double *W_ih, const double *W_hh,
                          const double *b_ih, const double *b_hh)
{
    const char *who = "lstm_cell_set_params";
    require_ptr(W_ih, who, "W_ih");
    require_ptr(W_hh, who, "W_hh");
    memcpy(c->W_ih, W_ih, (size_t)c->gate_size * c->input_size * sizeof(double));
    memcpy(c->W_hh, W_hh, (size_t)c->gate_size * c->hidden_size * sizeof(double));
    if (b_ih) memcpy(c->b_ih, b_ih, (size_t)c->gate_size * sizeof(double));
    else      memset(c->b_ih, 0, (size_t)c->gate_size * sizeof(double));
    if (b_hh) memcpy(c->b_hh, b_hh, (size_t)c->gate_size * sizeof(double));
    else      memset(c->b_hh, 0, (size_t)c->gate_size * sizeof(double));
}

void lstm_cell_zero_grad(LSTMCell *c)
{
    memset(c->grad_W_ih, 0, (size_t)c->gate_size * c->input_size * sizeof(double));
    memset(c->grad_W_hh, 0, (size_t)c->gate_size * c->hidden_size * sizeof(double));
    memset(c->grad_b_ih, 0, (size_t)c->gate_size * sizeof(double));
    memset(c->grad_b_hh, 0, (size_t)c->gate_size * sizeof(double));
}

const double *lstm_cell_forward(LSTMCell *c, const double *x,
                                const double *h_prev, const double *c_prev)
{
    const char *who = "lstm_cell_forward";
    int ins = c->input_size, hs = c->hidden_size, gs = c->gate_size;
    require_ptr(x, who, "x");

    if (x != c->x) memcpy(c->x, x, (size_t)ins * sizeof(double));
    if (h_prev && h_prev != c->h_prev) memcpy(c->h_prev, h_prev, (size_t)hs * sizeof(double));
    else if (!h_prev) memset(c->h_prev, 0, (size_t)hs * sizeof(double));
    if (c_prev && c_prev != c->c_prev) memcpy(c->c_prev, c_prev, (size_t)hs * sizeof(double));
    else if (!c_prev) memset(c->c_prev, 0, (size_t)hs * sizeof(double));

    /* 预激活：gates = W_ih·x + b_ih + W_hh·h_prev + b_hh */
    for (int j = 0; j < gs; j++) {
        double s = c->b_ih[j] + c->b_hh[j];
        for (int i = 0; i < ins; i++) s += c->W_ih[(size_t)j * ins + i] * c->x[i];
        for (int i = 0; i < hs; i++)  s += c->W_hh[(size_t)j * hs + i] * c->h_prev[i];
        c->gates[j] = s;
    }
    /* 分解四个门：[i | f | g | o] */
    for (int j = 0; j < hs; j++) {
        c->i_gate[j] = sigmoid(c->gates[j]);
        c->f_gate[j] = sigmoid(c->gates[hs + j]);
        c->g_gate[j] = tanh(c->gates[2 * hs + j]);
        c->o_gate[j] = sigmoid(c->gates[3 * hs + j]);
    }
    /* c = f * c_prev + i * g */
    for (int j = 0; j < hs; j++)
        c->c[j] = c->f_gate[j] * c->c_prev[j] + c->i_gate[j] * c->g_gate[j];
    /* h = o * tanh(c) */
    for (int j = 0; j < hs; j++)
        c->h[j] = c->o_gate[j] * tanh(c->c[j]);
    return c->h;
}

void lstm_cell_backward(LSTMCell *c, const double *dh, const double *dc_next,
                        double *dx, double *dh_prev, double *dc_prev)
{
    const char *who = "lstm_cell_backward";
    int ins = c->input_size, hs = c->hidden_size, gs = c->gate_size;
    require_ptr(dh, who, "dh");

    double *dg = c->d_gates;
    double *dc = c->dc;

    /* dc = dh * o * (1 - tanh(c)^2) + dc_next */
    for (int j = 0; j < hs; j++) {
        double tc = tanh(c->c[j]);
        dc[j] = dh[j] * c->o_gate[j] * (1.0 - tc * tc) + (dc_next ? dc_next[j] : 0.0);
    }
    /* 各门梯度（di | df | dg | do） */
    for (int j = 0; j < hs; j++) {
        dg[j]              = dc[j] * c->g_gate[j] * c->i_gate[j] * (1.0 - c->i_gate[j]);
        dg[hs + j]         = dc[j] * c->c_prev[j] * c->f_gate[j] * (1.0 - c->f_gate[j]);
        dg[2 * hs + j]     = dc[j] * c->i_gate[j] * (1.0 - c->g_gate[j] * c->g_gate[j]);
        dg[3 * hs + j]     = dh[j] * tanh(c->c[j]) * c->o_gate[j] * (1.0 - c->o_gate[j]);
    }

    /* 梯度原地累加 */
    for (int j = 0; j < gs; j++) {
        c->grad_b_ih[j] += dg[j];
        c->grad_b_hh[j] += dg[j];
        for (int i = 0; i < ins; i++)
            c->grad_W_ih[(size_t)j * ins + i] += dg[j] * c->x[i];
        for (int i = 0; i < hs; i++)
            c->grad_W_hh[(size_t)j * hs + i] += dg[j] * c->h_prev[i];
    }

    /* dx：for i: s = 0; for j: s += W_ih[j][i] * dg[j] */
    if (dx) {
        for (int i = 0; i < ins; i++) {
            double s = 0.0;
            for (int j = 0; j < gs; j++) s += c->W_ih[(size_t)j * ins + i] * dg[j];
            dx[i] = s;
        }
    }
    /* dh_prev：用 W_hh */
    if (dh_prev) {
        for (int i = 0; i < hs; i++) {
            double s = 0.0;
            for (int j = 0; j < gs; j++) s += c->W_hh[(size_t)j * hs + i] * dg[j];
            dh_prev[i] = s;
        }
    }
    /* dc_prev = dc * f */
    if (dc_prev) {
        for (int j = 0; j < hs; j++) dc_prev[j] = dc[j] * c->f_gate[j];
    }
}

void lstm_cell_get_grads(LSTMCell *c, double **grad_W_ih, double **grad_W_hh,
                         double **grad_b_ih, double **grad_b_hh)
{
    if (grad_W_ih) *grad_W_ih = c->grad_W_ih;
    if (grad_W_hh) *grad_W_hh = c->grad_W_hh;
    if (grad_b_ih) *grad_b_ih = c->grad_b_ih;
    if (grad_b_hh) *grad_b_hh = c->grad_b_hh;
}

int lstm_cell_param_count(const LSTMCell *c)
{
    return c->gate_size * c->input_size + c->gate_size * c->hidden_size
         + 2 * c->gate_size;
}

int lstm_cell_cache_size(const LSTMCell *c)
{
    return c->input_size + 7 * c->hidden_size;   /* x, h_prev, c_prev, i, f, g, o, c */
}

void lstm_cell_cache_snapshot(const LSTMCell *c, double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "lstm_cell_cache_snapshot", "rec");
    memcpy(rec, c->x, (size_t)ins * sizeof(double));        rec += ins;
    memcpy(rec, c->h_prev, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->c_prev, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->i_gate, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->f_gate, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->g_gate, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->o_gate, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->c, (size_t)hs * sizeof(double));
}

void lstm_cell_cache_restore(LSTMCell *c, const double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "lstm_cell_cache_restore", "rec");
    memcpy(c->x, rec, (size_t)ins * sizeof(double));        rec += ins;
    memcpy(c->h_prev, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->c_prev, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->i_gate, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->f_gate, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->g_gate, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->o_gate, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->c, rec, (size_t)hs * sizeof(double));
}

/* ============================== GRUCell ============================== */
/*
 * z = σ(W_iz·x + b_iz + W_hz·h + b_hz)     update gate
 * r = σ(W_ir·x + b_ir + W_hr·h + b_hr)     reset gate
 * n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))
 * h' = (1 - z) ⊙ n + z ⊙ h
 * 前向缓存字段顺序（= Python 的 _CACHE_FIELDS）：
 *     x, h_prev, z, r, n_ih, n_hh, z_a, r_a, n_val
 */
GRUCell *gru_cell_create(int input_size, int hidden_size)
{
    if (input_size <= 0 || hidden_size <= 0)
        rnn_fail("gru_cell_create", "input_size / hidden_size 必须为正");
    GRUCell *c = xmalloc(sizeof(GRUCell));
    memset(c, 0, sizeof(*c));
    c->input_size = input_size;
    c->hidden_size = hidden_size;
    c->gate_size = 3 * hidden_size;

    int ins = input_size, hs = hidden_size, gs = c->gate_size;
    size_t nw_ih = (size_t)gs * (size_t)ins;
    size_t nw_hh = (size_t)gs * (size_t)hs;

    c->W_ih = dalloc((int)nw_ih);
    c->W_hh = dalloc((int)nw_hh);
    c->b_ih = dalloc(gs);
    c->b_hh = dalloc(gs);

    c->grad_W_ih = dalloc((int)nw_ih);
    c->grad_W_hh = dalloc((int)nw_hh);
    c->grad_b_ih = dalloc(gs);
    c->grad_b_hh = dalloc(gs);

    c->m_W_ih = dalloc((int)nw_ih);  c->v_W_ih = dalloc((int)nw_ih);
    c->m_W_hh = dalloc((int)nw_hh);  c->v_W_hh = dalloc((int)nw_hh);
    c->m_b_ih = dalloc(gs);          c->v_b_ih = dalloc(gs);
    c->m_b_hh = dalloc(gs);          c->v_b_hh = dalloc(gs);

    c->x = dalloc(ins);
    c->h_prev = dalloc(hs);
    c->z = dalloc(hs);
    c->r = dalloc(hs);
    c->n_ih = dalloc(hs);
    c->n_hh = dalloc(hs);
    c->z_a = dalloc(hs);
    c->r_a = dalloc(hs);
    c->n_val = dalloc(hs);
    c->h = dalloc(hs);

    c->tmp_ih = dalloc(hs);
    c->tmp_hh = dalloc(hs);
    c->dz = dalloc(hs);
    c->dr = dalloc(hs);
    c->dn_ih = dalloc(hs);
    c->dn_hh = dalloc(hs);
    c->d_all_gates = dalloc(gs);

    unsigned long long seed = 0x9E3779B97F4A7C15ULL;
    gru_cell_init_params(c, &seed);
    return c;
}

void gru_cell_init_params(GRUCell *c, unsigned long long *rng_state)
{
    double std_ih = sqrt(1.0 / (double)c->input_size);
    double std_hh = sqrt(1.0 / (double)c->hidden_size);
    for (int j = 0; j < c->gate_size; j++) {
        for (int i = 0; i < c->input_size; i++)
            c->W_ih[(size_t)j * c->input_size + i] = rng_gauss(rng_state) * std_ih;
        for (int i = 0; i < c->hidden_size; i++)
            c->W_hh[(size_t)j * c->hidden_size + i] = rng_gauss(rng_state) * std_hh;
        c->b_ih[j] = 0.0;
        c->b_hh[j] = 0.0;
    }
}

void gru_cell_free(GRUCell *c)
{
    if (!c) return;
    free(c->W_ih); free(c->W_hh); free(c->b_ih); free(c->b_hh);
    free(c->grad_W_ih); free(c->grad_W_hh); free(c->grad_b_ih); free(c->grad_b_hh);
    free(c->m_W_ih); free(c->v_W_ih); free(c->m_W_hh); free(c->v_W_hh);
    free(c->m_b_ih); free(c->v_b_ih); free(c->m_b_hh); free(c->v_b_hh);
    free(c->x); free(c->h_prev); free(c->z); free(c->r);
    free(c->n_ih); free(c->n_hh); free(c->z_a); free(c->r_a); free(c->n_val); free(c->h);
    free(c->tmp_ih); free(c->tmp_hh); free(c->dz); free(c->dr);
    free(c->dn_ih); free(c->dn_hh); free(c->d_all_gates);
    memset(c, 0, sizeof(*c));
}

void gru_cell_set_params(GRUCell *c, const double *W_ih, const double *W_hh,
                         const double *b_ih, const double *b_hh)
{
    const char *who = "gru_cell_set_params";
    require_ptr(W_ih, who, "W_ih");
    require_ptr(W_hh, who, "W_hh");
    memcpy(c->W_ih, W_ih, (size_t)c->gate_size * c->input_size * sizeof(double));
    memcpy(c->W_hh, W_hh, (size_t)c->gate_size * c->hidden_size * sizeof(double));
    if (b_ih) memcpy(c->b_ih, b_ih, (size_t)c->gate_size * sizeof(double));
    else      memset(c->b_ih, 0, (size_t)c->gate_size * sizeof(double));
    if (b_hh) memcpy(c->b_hh, b_hh, (size_t)c->gate_size * sizeof(double));
    else      memset(c->b_hh, 0, (size_t)c->gate_size * sizeof(double));
}

void gru_cell_zero_grad(GRUCell *c)
{
    memset(c->grad_W_ih, 0, (size_t)c->gate_size * c->input_size * sizeof(double));
    memset(c->grad_W_hh, 0, (size_t)c->gate_size * c->hidden_size * sizeof(double));
    memset(c->grad_b_ih, 0, (size_t)c->gate_size * sizeof(double));
    memset(c->grad_b_hh, 0, (size_t)c->gate_size * sizeof(double));
}

/* Python 的 _linear(W_slice, b_slice, inp, size)：
 *     out = [b[j] ...]; out[j] += W[j][i] * inp[i]
 * 从偏置起累加，与 sum(map(...)) 不同（顺序敏感，必须照抄） */
static void gru_linear(double *out, const double *W, const double *b,
                       const double *inp, int n_in, int n_out)
{
    for (int j = 0; j < n_out; j++) {
        double s = b[j];
        for (int i = 0; i < n_in; i++) s += W[(size_t)j * n_in + i] * inp[i];
        out[j] = s;
    }
}

const double *gru_cell_forward(GRUCell *c, const double *x, const double *h_prev)
{
    const char *who = "gru_cell_forward";
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(x, who, "x");

    if (x != c->x) memcpy(c->x, x, (size_t)ins * sizeof(double));
    if (h_prev && h_prev != c->h_prev) memcpy(c->h_prev, h_prev, (size_t)hs * sizeof(double));
    else if (!h_prev) memset(c->h_prev, 0, (size_t)hs * sizeof(double));

    /* 三组门权重: [z: 0:hs], [r: hs:2*hs], [n: 2*hs:3*hs] */
    gru_linear(c->tmp_ih, c->W_ih, c->b_ih, c->x, ins, hs);
    gru_linear(c->tmp_hh, c->W_hh, c->b_hh, c->h_prev, hs, hs);
    for (int j = 0; j < hs; j++) c->z[j] = c->tmp_ih[j] + c->tmp_hh[j];

    gru_linear(c->tmp_ih, c->W_ih + (size_t)hs * ins, c->b_ih + hs, c->x, ins, hs);
    gru_linear(c->tmp_hh, c->W_hh + (size_t)hs * hs, c->b_hh + hs, c->h_prev, hs, hs);
    for (int j = 0; j < hs; j++) c->r[j] = c->tmp_ih[j] + c->tmp_hh[j];

    gru_linear(c->n_ih, c->W_ih + (size_t)2 * hs * ins, c->b_ih + 2 * hs, c->x, ins, hs);
    gru_linear(c->n_hh, c->W_hh + (size_t)2 * hs * hs, c->b_hh + 2 * hs, c->h_prev, hs, hs);

    for (int j = 0; j < hs; j++) c->z_a[j] = sigmoid(c->z[j]);
    for (int j = 0; j < hs; j++) c->r_a[j] = sigmoid(c->r[j]);
    for (int j = 0; j < hs; j++) c->n_val[j] = tanh(c->n_ih[j] + c->r_a[j] * c->n_hh[j]);

    for (int j = 0; j < hs; j++)
        c->h[j] = (1 - c->z_a[j]) * c->n_val[j] + c->z_a[j] * c->h_prev[j];
    return c->h;
}

void gru_cell_backward(GRUCell *c, const double *dh, double *dx, double *dh_prev)
{
    const char *who = "gru_cell_backward";
    int ins = c->input_size, hs = c->hidden_size, gs = c->gate_size;
    require_ptr(dh, who, "dh");

    /* h = (1 - z_a) * n_val + z_a * h_prev */
    for (int j = 0; j < hs; j++) {
        double dz_a = dh[j] * (c->h_prev[j] - c->n_val[j]);
        double dn   = dh[j] * (1.0 - c->z_a[j]);
        /* n_val = tanh(n_ih + r_a * n_hh) */
        double dn_pre = dn * (1.0 - c->n_val[j] * c->n_val[j]);
        c->dn_ih[j] = dn_pre;
        c->dn_hh[j] = dn_pre * c->r_a[j];
        double dr_a_from_n = dn_pre * c->n_hh[j];
        /* r_a = sigmoid(r)：梯度只来自 n；z_a = sigmoid(z) */
        c->dr[j] = dr_a_from_n * c->r_a[j] * (1.0 - c->r_a[j]);
        c->dz[j] = dz_a * c->z_a[j] * (1.0 - c->z_a[j]);
        /* dh_prev 的直接路径：h -> z_a * h_prev */
        if (dh_prev) dh_prev[j] = dh[j] * c->z_a[j];
    }

    /* 累积参数梯度：更新门 z 行 [0:hs] */
    for (int j = 0; j < hs; j++) {
        c->grad_b_ih[j] += c->dz[j];
        c->grad_b_hh[j] += c->dz[j];
        for (int i = 0; i < ins; i++) c->grad_W_ih[(size_t)j * ins + i] += c->dz[j] * c->x[i];
        for (int i = 0; i < hs; i++)  c->grad_W_hh[(size_t)j * hs + i] += c->dz[j] * c->h_prev[i];
    }
    /* 重置门 r 行 [hs:2*hs] */
    for (int j = 0; j < hs; j++) {
        int rj = hs + j;
        c->grad_b_ih[rj] += c->dr[j];
        c->grad_b_hh[rj] += c->dr[j];
        for (int i = 0; i < ins; i++) c->grad_W_ih[(size_t)rj * ins + i] += c->dr[j] * c->x[i];
        for (int i = 0; i < hs; i++)  c->grad_W_hh[(size_t)rj * hs + i] += c->dr[j] * c->h_prev[i];
    }
    /* 新门 n 行 [2*hs:3*hs]：ih 走 dn_ih，hh 走 dn_hh */
    for (int j = 0; j < hs; j++) {
        int nj = 2 * hs + j;
        c->grad_b_ih[nj] += c->dn_ih[j];
        c->grad_b_hh[nj] += c->dn_hh[j];
        for (int i = 0; i < ins; i++) c->grad_W_ih[(size_t)nj * ins + i] += c->dn_ih[j] * c->x[i];
        for (int i = 0; i < hs; i++)  c->grad_W_hh[(size_t)nj * hs + i] += c->dn_hh[j] * c->h_prev[i];
    }

    /* dx / dh_prev 通过三个门权重反传。
     * 这两处在 Python 里是 sum(... for j in range(3*hs))，用 Neumaier 补偿求和。
     * 注意 n 门的 ih / hh 分量要分开用（Python 的注释也强调了这一点）。 */
    for (int j = 0; j < hs; j++) {
        c->d_all_gates[j]          = c->dz[j];
        c->d_all_gates[hs + j]     = c->dr[j];
        c->d_all_gates[2 * hs + j] = c->dn_ih[j];
    }
    if (dx) {
        for (int i = 0; i < ins; i++)
            dx[i] = sum_col_neumaier(c->W_ih, ins, i, c->d_all_gates, gs);
    }
    if (dh_prev) {
        for (int j = 0; j < hs; j++) c->d_all_gates[2 * hs + j] = c->dn_hh[j];
        for (int i = 0; i < hs; i++)
            dh_prev[i] = dh_prev[i] + sum_col_neumaier(c->W_hh, hs, i, c->d_all_gates, gs);
    }
}

void gru_cell_get_grads(GRUCell *c, double **grad_W_ih, double **grad_W_hh,
                        double **grad_b_ih, double **grad_b_hh)
{
    if (grad_W_ih) *grad_W_ih = c->grad_W_ih;
    if (grad_W_hh) *grad_W_hh = c->grad_W_hh;
    if (grad_b_ih) *grad_b_ih = c->grad_b_ih;
    if (grad_b_hh) *grad_b_hh = c->grad_b_hh;
}

int gru_cell_param_count(const GRUCell *c)
{
    return c->gate_size * c->input_size + c->gate_size * c->hidden_size
         + 2 * c->gate_size;
}

int gru_cell_cache_size(const GRUCell *c)
{
    /* x, h_prev, z, r, n_ih, n_hh, z_a, r_a, n_val */
    return c->input_size + 8 * c->hidden_size;
}

void gru_cell_cache_snapshot(const GRUCell *c, double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "gru_cell_cache_snapshot", "rec");
    memcpy(rec, c->x, (size_t)ins * sizeof(double));        rec += ins;
    memcpy(rec, c->h_prev, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(rec, c->z, (size_t)hs * sizeof(double));         rec += hs;
    memcpy(rec, c->r, (size_t)hs * sizeof(double));         rec += hs;
    memcpy(rec, c->n_ih, (size_t)hs * sizeof(double));      rec += hs;
    memcpy(rec, c->n_hh, (size_t)hs * sizeof(double));      rec += hs;
    memcpy(rec, c->z_a, (size_t)hs * sizeof(double));       rec += hs;
    memcpy(rec, c->r_a, (size_t)hs * sizeof(double));       rec += hs;
    memcpy(rec, c->n_val, (size_t)hs * sizeof(double));
}

void gru_cell_cache_restore(GRUCell *c, const double *rec)
{
    int ins = c->input_size, hs = c->hidden_size;
    require_ptr(rec, "gru_cell_cache_restore", "rec");
    memcpy(c->x, rec, (size_t)ins * sizeof(double));        rec += ins;
    memcpy(c->h_prev, rec, (size_t)hs * sizeof(double));    rec += hs;
    memcpy(c->z, rec, (size_t)hs * sizeof(double));         rec += hs;
    memcpy(c->r, rec, (size_t)hs * sizeof(double));         rec += hs;
    memcpy(c->n_ih, rec, (size_t)hs * sizeof(double));      rec += hs;
    memcpy(c->n_hh, rec, (size_t)hs * sizeof(double));      rec += hs;
    memcpy(c->z_a, rec, (size_t)hs * sizeof(double));       rec += hs;
    memcpy(c->r_a, rec, (size_t)hs * sizeof(double));       rec += hs;
    memcpy(c->n_val, rec, (size_t)hs * sizeof(double));
}

/* ========================== step cache（多层） ==========================
 * 对应 Python 的 _StepCacheMixin：xxx_forward 每算完一层就 cache_snapshot 一次，
 * BPTT 时 restore_cache 还原，省掉整轮重复前向。
 */
static void cache_alloc(RNNStepCache *sc, int num_layers, int hidden_size, int max_T,
                        int per_layer_extra, int input_size)
{
    sc->max_T = max_T;
    sc->num_layers = num_layers;
    sc->hidden_size = hidden_size;
    sc->cached_T = 0;
    sc->layer_off = xcalloc((size_t)num_layers + 1, sizeof(int));

    int stride = 0;
    for (int l = 0; l < num_layers; l++) {
        int layer_in = (l == 0) ? input_size : hidden_size;
        sc->layer_off[l] = stride;
        stride += layer_in + per_layer_extra * hidden_size;
    }
    sc->layer_off[num_layers] = stride;
    sc->rec_stride = stride;
    sc->rec = xcalloc((size_t)max_T * (size_t)stride, sizeof(double));
}

static void cache_release(RNNStepCache *sc)
{
    free(sc->layer_off);
    free(sc->rec);
    memset(sc, 0, sizeof(*sc));
}

static double *cache_rec(RNNStepCache *sc, int t, int layer)
{
    return sc->rec + (size_t)t * (size_t)sc->rec_stride + (size_t)sc->layer_off[layer];
}

static void cache_check_T(const RNNStepCache *sc, int T, const char *who, const char *reserve_fn)
{
    if (T < 0) rnn_fail(who, "序列长度 T 不能为负");
    if (T > sc->max_T) {
        fprintf(stderr, "%s: 序列长度 T = %d 超过预分配容量 max_T = %d；"
                        "请在训练循环外调用 %s(net, T) 扩容\n",
                who, T, sc->max_T, reserve_fn);
        exit(1);
    }
}

static void cache_check_index(const RNNStepCache *sc, int t, int layer, const char *who)
{
    if (t < 0 || t >= sc->cached_T) {
        fprintf(stderr, "%s: 时间步 t = %d 超出已缓存范围 [0, %d)\n", who, t, sc->cached_T);
        exit(1);
    }
    if (layer < 0 || layer >= sc->num_layers) {
        fprintf(stderr, "%s: 层号 layer = %d 超出范围 [0, %d)\n", who, layer, sc->num_layers);
        exit(1);
    }
}

/* 形状检查的小包装（不匹配时 tensor_check_shape 会报错退出） */
static void check_shape2(const Tensor *t, int a, int b, const char *who)
{
    int shape[2] = { a, b };
    tensor_check_shape(t, 2, shape, who);
}

/* ================================ RNN ================================ */

RNN *rnn_create(int input_size, int hidden_size, int num_layers)
{
    return rnn_create_cap(input_size, hidden_size, num_layers, RNN_DEFAULT_MAX_T);
}

RNN *rnn_create_cap(int input_size, int hidden_size, int num_layers, int max_T)
{
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0)
        rnn_fail("rnn_create", "input_size / hidden_size / num_layers 必须为正");
    if (max_T <= 0) rnn_fail("rnn_create", "max_T 必须为正");

    RNN *net = xmalloc(sizeof(RNN));
    memset(net, 0, sizeof(*net));
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->num_layers = num_layers;

    net->cells = xcalloc((size_t)num_layers, sizeof(RNNCell));
    for (int l = 0; l < num_layers; l++) {
        int layer_in = (l == 0) ? input_size : hidden_size;
        /* create 返回的包装结构体按值搬进数组，内部缓冲的所有权随之转移 */
        RNNCell *cell = rnn_cell_create(layer_in, hidden_size);
        net->cells[l] = *cell;
        free(cell);
    }
    cache_alloc(&net->step_cache, num_layers, hidden_size, max_T, 2, input_size);

    int n = (input_size > hidden_size) ? input_size : hidden_size;
    net->dh_slab = dalloc(max_T * num_layers * hidden_size);
    net->scratch_dx = dalloc(n);
    net->scratch_dh = dalloc(hidden_size);
    return net;
}

void rnn_free(RNN *net)
{
    if (!net) return;
    for (int l = 0; l < net->num_layers; l++) rnn_cell_free(&net->cells[l]);
    free(net->cells);
    cache_release(&net->step_cache);
    free(net->dh_slab);
    free(net->scratch_dx);
    free(net->scratch_dh);
    free(net);
}

int rnn_reserve(RNN *net, int max_T)
{
    if (max_T <= net->step_cache.max_T) return 1;
    /* 训练循环外扩容：直接重建缓冲，旧缓存失效 */
    cache_release(&net->step_cache);
    cache_alloc(&net->step_cache, net->num_layers, net->hidden_size, max_T, 2, net->input_size);
    free(net->dh_slab);
    net->dh_slab = dalloc(max_T * net->num_layers * net->hidden_size);
    return 1;
}

void rnn_zero_grad(RNN *net)
{
    for (int l = 0; l < net->num_layers; l++) rnn_cell_zero_grad(&net->cells[l]);
}

RNNCell *rnn_layer(RNN *net, int layer)
{
    if (layer < 0 || layer >= net->num_layers) {
        fprintf(stderr, "rnn_layer: 层号 layer = %d 超出范围 [0, %d)\n", layer, net->num_layers);
        exit(1);
    }
    return &net->cells[layer];
}

int rnn_cached_steps(const RNN *net) { return net->step_cache.cached_T; }
int rnn_param_count(const RNN *net)
{
    int n = 0;
    for (int l = 0; l < net->num_layers; l++) n += rnn_cell_param_count(&net->cells[l]);
    return n;
}

void rnn_forward(RNN *net, const Tensor *x_seq, Tensor *out_seq, Tensor *h_last)
{
    rnn_forward_h0(net, x_seq, NULL, out_seq, h_last);
}

void rnn_forward_h0(RNN *net, const Tensor *x_seq, const Tensor *h0,
                    Tensor *out_seq, Tensor *h_last)
{
    const char *who = "rnn_forward";
    if (!x_seq) rnn_fail(who, "x_seq 不能为 NULL");
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = (x_seq->ndim >= 1) ? x_seq->shape[0] : 0;
    check_shape2(x_seq, T, ins, who);
    cache_check_T(&net->step_cache, T, who, "rnn_reserve");
    if (h0) check_shape2(h0, L, hs, "rnn_forward: h0");
    if (out_seq) check_shape2(out_seq, T, hs, "rnn_forward: out_seq");
    if (h_last) check_shape2(h_last, L, hs, "rnn_forward: h_last");

    /* 初始状态（Python: h = list(h0)） */
    for (int l = 0; l < L; l++) {
        if (h0) memcpy(net->cells[l].h, h0->data + (size_t)l * hs, (size_t)hs * sizeof(double));
        else    memset(net->cells[l].h, 0, (size_t)hs * sizeof(double));
    }

    net->step_cache.cached_T = T;
    for (int t = 0; t < T; t++) {
        const double *layer_in = x_seq->data + (size_t)t * ins;
        for (int l = 0; l < L; l++) {
            RNNCell *c = &net->cells[l];
            const double *h = rnn_cell_forward(c, layer_in, c->h);
            rnn_cell_cache_snapshot(c, cache_rec(&net->step_cache, t, l));
            layer_in = h;
        }
        if (out_seq) memcpy(out_seq->data + (size_t)t * hs, layer_in, (size_t)hs * sizeof(double));
    }
    if (h_last) {
        for (int l = 0; l < L; l++)
            memcpy(h_last->data + (size_t)l * hs, net->cells[l].h, (size_t)hs * sizeof(double));
    }
}

void rnn_restore_cache(RNN *net, int t, int layer)
{
    cache_check_index(&net->step_cache, t, layer, "rnn_restore_cache");
    rnn_cell_cache_restore(&net->cells[layer], cache_rec(&net->step_cache, t, layer));
}

void rnn_backward(RNN *net, const Tensor *dh_last, Tensor *dx_seq)
{
    rnn_backward_seq(net, NULL, dh_last, dx_seq);
}

/* 统一的反向实现：只依赖 step_cache（不重跑前向）。
 * 时间步 t 从 T-1 递减、层 l 从 L-1 递减，dh 累加进 dh_slab：
 *   处理 (t, l) 时 slab[t][l] 已经收齐了来自 (t+1, l) 与 (t, l+1) 的贡献。
 * 单层情形与 Python 示例的 BPTT 循环逐位一致（dh_cur 就是 slab[t][0]）。 */
void rnn_backward_seq(RNN *net, const Tensor *dout_seq, const Tensor *dh_last, Tensor *dx_seq)
{
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = net->step_cache.cached_T;
    if (dout_seq) check_shape2(dout_seq, T, hs, "rnn_backward: dout_seq");
    if (dh_last)  check_shape2(dh_last, L, hs, "rnn_backward: dh_last");
    if (dx_seq)   check_shape2(dx_seq, T, ins, "rnn_backward: dx_seq");
    if (T == 0) return;

    double *slab = net->dh_slab;
    memset(slab, 0, (size_t)T * (size_t)L * (size_t)hs * sizeof(double));

    /* 最后一层每个时间步的输出梯度 */
    if (dout_seq) {
        for (int t = 0; t < T; t++) {
            double *dst = slab + ((size_t)t * L + (L - 1)) * hs;
            const double *src = dout_seq->data + (size_t)t * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }
    /* 每层末状态的梯度 */
    if (dh_last) {
        for (int l = 0; l < L; l++) {
            double *dst = slab + ((size_t)(T - 1) * L + l) * hs;
            const double *src = dh_last->data + (size_t)l * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }

    for (int t = T - 1; t >= 0; t--) {
        for (int l = L - 1; l >= 0; l--) {
            RNNCell *c = rnn_layer(net, l);
            double *dh_cur = slab + ((size_t)t * L + l) * hs;
            rnn_restore_cache(net, t, l);
            rnn_cell_backward(c, dh_cur, net->scratch_dx, net->scratch_dh);
            if (l > 0) {
                /* 本层的输入梯度 = 上一层的输出梯度（同一时间步） */
                double *up = slab + ((size_t)t * L + (l - 1)) * hs;
                for (int j = 0; j < hs; j++) up[j] += net->scratch_dx[j];
            } else if (dx_seq) {
                memcpy(dx_seq->data + (size_t)t * ins, net->scratch_dx, (size_t)ins * sizeof(double));
            }
            if (t > 0) {
                double *prev = slab + ((size_t)(t - 1) * L + l) * hs;
                for (int j = 0; j < hs; j++) prev[j] += net->scratch_dh[j];
            }
        }
    }
}

/* ================================ LSTM ================================ */

LSTM *lstm_create(int input_size, int hidden_size, int num_layers)
{
    return lstm_create_cap(input_size, hidden_size, num_layers, RNN_DEFAULT_MAX_T);
}

LSTM *lstm_create_cap(int input_size, int hidden_size, int num_layers, int max_T)
{
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0)
        rnn_fail("lstm_create", "input_size / hidden_size / num_layers 必须为正");
    if (max_T <= 0) rnn_fail("lstm_create", "max_T 必须为正");

    LSTM *net = xmalloc(sizeof(LSTM));
    memset(net, 0, sizeof(*net));
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->num_layers = num_layers;

    net->cells = xcalloc((size_t)num_layers, sizeof(LSTMCell));
    for (int l = 0; l < num_layers; l++) {
        int layer_in = (l == 0) ? input_size : hidden_size;
        /* create 返回的包装结构体按值搬进数组，内部缓冲的所有权随之转移 */
        LSTMCell *cell = lstm_cell_create(layer_in, hidden_size);
        net->cells[l] = *cell;
        free(cell);
    }
    cache_alloc(&net->step_cache, num_layers, hidden_size, max_T, 7, input_size);

    int n = (input_size > hidden_size) ? input_size : hidden_size;
    net->dh_slab = dalloc(max_T * num_layers * hidden_size);
    net->dc_slab = dalloc(max_T * num_layers * hidden_size);
    net->scratch_dx = dalloc(n);
    net->scratch_dh = dalloc(hidden_size);
    net->scratch_dc = dalloc(hidden_size);
    return net;
}

void lstm_free(LSTM *net)
{
    if (!net) return;
    for (int l = 0; l < net->num_layers; l++) lstm_cell_free(&net->cells[l]);
    free(net->cells);
    cache_release(&net->step_cache);
    free(net->dh_slab);
    free(net->dc_slab);
    free(net->scratch_dx);
    free(net->scratch_dh);
    free(net->scratch_dc);
    free(net);
}

int lstm_reserve(LSTM *net, int max_T)
{
    if (max_T <= net->step_cache.max_T) return 1;
    cache_release(&net->step_cache);
    cache_alloc(&net->step_cache, net->num_layers, net->hidden_size, max_T, 7, net->input_size);
    free(net->dh_slab);
    free(net->dc_slab);
    net->dh_slab = dalloc(max_T * net->num_layers * net->hidden_size);
    net->dc_slab = dalloc(max_T * net->num_layers * net->hidden_size);
    return 1;
}

void lstm_zero_grad(LSTM *net)
{
    for (int l = 0; l < net->num_layers; l++) lstm_cell_zero_grad(&net->cells[l]);
}

LSTMCell *lstm_layer(LSTM *net, int layer)
{
    if (layer < 0 || layer >= net->num_layers) {
        fprintf(stderr, "lstm_layer: 层号 layer = %d 超出范围 [0, %d)\n", layer, net->num_layers);
        exit(1);
    }
    return &net->cells[layer];
}

int lstm_cached_steps(const LSTM *net) { return net->step_cache.cached_T; }
int lstm_param_count(const LSTM *net)
{
    int n = 0;
    for (int l = 0; l < net->num_layers; l++) n += lstm_cell_param_count(&net->cells[l]);
    return n;
}

void lstm_forward(LSTM *net, const Tensor *x_seq, Tensor *out_seq,
                  Tensor *h_last, Tensor *c_last)
{
    lstm_forward_h0(net, x_seq, NULL, NULL, out_seq, h_last, c_last);
}

void lstm_forward_h0(LSTM *net, const Tensor *x_seq, const Tensor *h0, const Tensor *c0,
                     Tensor *out_seq, Tensor *h_last, Tensor *c_last)
{
    const char *who = "lstm_forward";
    if (!x_seq) rnn_fail(who, "x_seq 不能为 NULL");
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = (x_seq->ndim >= 1) ? x_seq->shape[0] : 0;
    check_shape2(x_seq, T, ins, who);
    cache_check_T(&net->step_cache, T, who, "lstm_reserve");
    if (h0) check_shape2(h0, L, hs, "lstm_forward: h0");
    if (c0) check_shape2(c0, L, hs, "lstm_forward: c0");
    if (out_seq) check_shape2(out_seq, T, hs, "lstm_forward: out_seq");
    if (h_last) check_shape2(h_last, L, hs, "lstm_forward: h_last");
    if (c_last) check_shape2(c_last, L, hs, "lstm_forward: c_last");

    for (int l = 0; l < L; l++) {
        if (h0) memcpy(net->cells[l].h, h0->data + (size_t)l * hs, (size_t)hs * sizeof(double));
        else    memset(net->cells[l].h, 0, (size_t)hs * sizeof(double));
        if (c0) memcpy(net->cells[l].c, c0->data + (size_t)l * hs, (size_t)hs * sizeof(double));
        else    memset(net->cells[l].c, 0, (size_t)hs * sizeof(double));
    }

    net->step_cache.cached_T = T;
    for (int t = 0; t < T; t++) {
        const double *layer_in = x_seq->data + (size_t)t * ins;
        for (int l = 0; l < L; l++) {
            LSTMCell *c = &net->cells[l];
            const double *h = lstm_cell_forward(c, layer_in, c->h, c->c);
            lstm_cell_cache_snapshot(c, cache_rec(&net->step_cache, t, l));
            layer_in = h;
        }
        if (out_seq) memcpy(out_seq->data + (size_t)t * hs, layer_in, (size_t)hs * sizeof(double));
    }
    if (h_last) {
        for (int l = 0; l < L; l++)
            memcpy(h_last->data + (size_t)l * hs, net->cells[l].h, (size_t)hs * sizeof(double));
    }
    if (c_last) {
        for (int l = 0; l < L; l++)
            memcpy(c_last->data + (size_t)l * hs, net->cells[l].c, (size_t)hs * sizeof(double));
    }
}

void lstm_restore_cache(LSTM *net, int t, int layer)
{
    cache_check_index(&net->step_cache, t, layer, "lstm_restore_cache");
    lstm_cell_cache_restore(&net->cells[layer], cache_rec(&net->step_cache, t, layer));
}

void lstm_backward(LSTM *net, const Tensor *dh_last, Tensor *dx_seq)
{
    lstm_backward_full(net, NULL, dh_last, NULL, dx_seq);
}

void lstm_backward_full(LSTM *net, const Tensor *dout_seq, const Tensor *dh_last,
                        const Tensor *dc_last, Tensor *dx_seq)
{
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = net->step_cache.cached_T;
    if (dout_seq) check_shape2(dout_seq, T, hs, "lstm_backward: dout_seq");
    if (dh_last)  check_shape2(dh_last, L, hs, "lstm_backward: dh_last");
    if (dc_last)  check_shape2(dc_last, L, hs, "lstm_backward: dc_last");
    if (dx_seq)   check_shape2(dx_seq, T, ins, "lstm_backward: dx_seq");
    if (T == 0) return;

    double *slab = net->dh_slab;
    double *cslab = net->dc_slab;
    memset(slab, 0, (size_t)T * (size_t)L * (size_t)hs * sizeof(double));
    memset(cslab, 0, (size_t)T * (size_t)L * (size_t)hs * sizeof(double));

    if (dout_seq) {
        for (int t = 0; t < T; t++) {
            double *dst = slab + ((size_t)t * L + (L - 1)) * hs;
            const double *src = dout_seq->data + (size_t)t * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }
    if (dh_last) {
        for (int l = 0; l < L; l++) {
            double *dst = slab + ((size_t)(T - 1) * L + l) * hs;
            const double *src = dh_last->data + (size_t)l * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }
    if (dc_last) {
        for (int l = 0; l < L; l++) {
            double *dst = cslab + ((size_t)(T - 1) * L + l) * hs;
            const double *src = dc_last->data + (size_t)l * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }

    for (int t = T - 1; t >= 0; t--) {
        for (int l = L - 1; l >= 0; l--) {
            LSTMCell *c = lstm_layer(net, l);
            double *dh_cur = slab + ((size_t)t * L + l) * hs;
            double *dc_cur = cslab + ((size_t)t * L + l) * hs;
            lstm_restore_cache(net, t, l);
            lstm_cell_backward(c, dh_cur, dc_cur, net->scratch_dx,
                               net->scratch_dh, net->scratch_dc);
            if (l > 0) {
                double *up = slab + ((size_t)t * L + (l - 1)) * hs;
                for (int j = 0; j < hs; j++) up[j] += net->scratch_dx[j];
            } else if (dx_seq) {
                memcpy(dx_seq->data + (size_t)t * ins, net->scratch_dx, (size_t)ins * sizeof(double));
            }
            if (t > 0) {
                double *prev = slab + ((size_t)(t - 1) * L + l) * hs;
                double *cprev = cslab + ((size_t)(t - 1) * L + l) * hs;
                for (int j = 0; j < hs; j++) prev[j] += net->scratch_dh[j];
                for (int j = 0; j < hs; j++) cprev[j] += net->scratch_dc[j];
            }
        }
    }
}

/* ================================ GRU ================================ */

GRU *gru_create(int input_size, int hidden_size, int num_layers)
{
    return gru_create_cap(input_size, hidden_size, num_layers, RNN_DEFAULT_MAX_T);
}

GRU *gru_create_cap(int input_size, int hidden_size, int num_layers, int max_T)
{
    if (input_size <= 0 || hidden_size <= 0 || num_layers <= 0)
        rnn_fail("gru_create", "input_size / hidden_size / num_layers 必须为正");
    if (max_T <= 0) rnn_fail("gru_create", "max_T 必须为正");

    GRU *net = xmalloc(sizeof(GRU));
    memset(net, 0, sizeof(*net));
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->num_layers = num_layers;

    net->cells = xcalloc((size_t)num_layers, sizeof(GRUCell));
    for (int l = 0; l < num_layers; l++) {
        int layer_in = (l == 0) ? input_size : hidden_size;
        /* create 返回的包装结构体按值搬进数组，内部缓冲的所有权随之转移 */
        GRUCell *cell = gru_cell_create(layer_in, hidden_size);
        net->cells[l] = *cell;
        free(cell);
    }
    cache_alloc(&net->step_cache, num_layers, hidden_size, max_T, 8, input_size);

    int n = (input_size > hidden_size) ? input_size : hidden_size;
    net->dh_slab = dalloc(max_T * num_layers * hidden_size);
    net->scratch_dx = dalloc(n);
    net->scratch_dh = dalloc(hidden_size);
    return net;
}

void gru_free(GRU *net)
{
    if (!net) return;
    for (int l = 0; l < net->num_layers; l++) gru_cell_free(&net->cells[l]);
    free(net->cells);
    cache_release(&net->step_cache);
    free(net->dh_slab);
    free(net->scratch_dx);
    free(net->scratch_dh);
    free(net);
}

int gru_reserve(GRU *net, int max_T)
{
    if (max_T <= net->step_cache.max_T) return 1;
    cache_release(&net->step_cache);
    cache_alloc(&net->step_cache, net->num_layers, net->hidden_size, max_T, 8, net->input_size);
    free(net->dh_slab);
    net->dh_slab = dalloc(max_T * net->num_layers * net->hidden_size);
    return 1;
}

void gru_zero_grad(GRU *net)
{
    for (int l = 0; l < net->num_layers; l++) gru_cell_zero_grad(&net->cells[l]);
}

GRUCell *gru_layer(GRU *net, int layer)
{
    if (layer < 0 || layer >= net->num_layers) {
        fprintf(stderr, "gru_layer: 层号 layer = %d 超出范围 [0, %d)\n", layer, net->num_layers);
        exit(1);
    }
    return &net->cells[layer];
}

int gru_cached_steps(const GRU *net) { return net->step_cache.cached_T; }
int gru_param_count(const GRU *net)
{
    int n = 0;
    for (int l = 0; l < net->num_layers; l++) n += gru_cell_param_count(&net->cells[l]);
    return n;
}

void gru_forward(GRU *net, const Tensor *x_seq, Tensor *out_seq, Tensor *h_last)
{
    gru_forward_h0(net, x_seq, NULL, out_seq, h_last);
}

void gru_forward_h0(GRU *net, const Tensor *x_seq, const Tensor *h0,
                    Tensor *out_seq, Tensor *h_last)
{
    const char *who = "gru_forward";
    if (!x_seq) rnn_fail(who, "x_seq 不能为 NULL");
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = (x_seq->ndim >= 1) ? x_seq->shape[0] : 0;
    check_shape2(x_seq, T, ins, who);
    cache_check_T(&net->step_cache, T, who, "gru_reserve");
    if (h0) check_shape2(h0, L, hs, "gru_forward: h0");
    if (out_seq) check_shape2(out_seq, T, hs, "gru_forward: out_seq");
    if (h_last) check_shape2(h_last, L, hs, "gru_forward: h_last");

    for (int l = 0; l < L; l++) {
        if (h0) memcpy(net->cells[l].h, h0->data + (size_t)l * hs, (size_t)hs * sizeof(double));
        else    memset(net->cells[l].h, 0, (size_t)hs * sizeof(double));
    }

    net->step_cache.cached_T = T;
    for (int t = 0; t < T; t++) {
        const double *layer_in = x_seq->data + (size_t)t * ins;
        for (int l = 0; l < L; l++) {
            GRUCell *c = &net->cells[l];
            const double *h = gru_cell_forward(c, layer_in, c->h);
            gru_cell_cache_snapshot(c, cache_rec(&net->step_cache, t, l));
            layer_in = h;
        }
        if (out_seq) memcpy(out_seq->data + (size_t)t * hs, layer_in, (size_t)hs * sizeof(double));
    }
    if (h_last) {
        for (int l = 0; l < L; l++)
            memcpy(h_last->data + (size_t)l * hs, net->cells[l].h, (size_t)hs * sizeof(double));
    }
}

void gru_restore_cache(GRU *net, int t, int layer)
{
    cache_check_index(&net->step_cache, t, layer, "gru_restore_cache");
    gru_cell_cache_restore(&net->cells[layer], cache_rec(&net->step_cache, t, layer));
}

void gru_backward(GRU *net, const Tensor *dh_last, Tensor *dx_seq)
{
    gru_backward_seq(net, NULL, dh_last, dx_seq);
}

void gru_backward_seq(GRU *net, const Tensor *dout_seq, const Tensor *dh_last, Tensor *dx_seq)
{
    int ins = net->input_size, hs = net->hidden_size, L = net->num_layers;
    int T = net->step_cache.cached_T;
    if (dout_seq) check_shape2(dout_seq, T, hs, "gru_backward: dout_seq");
    if (dh_last)  check_shape2(dh_last, L, hs, "gru_backward: dh_last");
    if (dx_seq)   check_shape2(dx_seq, T, ins, "gru_backward: dx_seq");
    if (T == 0) return;

    double *slab = net->dh_slab;
    memset(slab, 0, (size_t)T * (size_t)L * (size_t)hs * sizeof(double));

    if (dout_seq) {
        for (int t = 0; t < T; t++) {
            double *dst = slab + ((size_t)t * L + (L - 1)) * hs;
            const double *src = dout_seq->data + (size_t)t * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }
    if (dh_last) {
        for (int l = 0; l < L; l++) {
            double *dst = slab + ((size_t)(T - 1) * L + l) * hs;
            const double *src = dh_last->data + (size_t)l * hs;
            for (int j = 0; j < hs; j++) dst[j] += src[j];
        }
    }

    for (int t = T - 1; t >= 0; t--) {
        for (int l = L - 1; l >= 0; l--) {
            GRUCell *c = gru_layer(net, l);
            double *dh_cur = slab + ((size_t)t * L + l) * hs;
            gru_restore_cache(net, t, l);
            gru_cell_backward(c, dh_cur, net->scratch_dx, net->scratch_dh);
            if (l > 0) {
                double *up = slab + ((size_t)t * L + (l - 1)) * hs;
                for (int j = 0; j < hs; j++) up[j] += net->scratch_dx[j];
            } else if (dx_seq) {
                memcpy(dx_seq->data + (size_t)t * ins, net->scratch_dx, (size_t)ins * sizeof(double));
            }
            if (t > 0) {
                double *prev = slab + ((size_t)(t - 1) * L + l) * hs;
                for (int j = 0; j < hs; j++) prev[j] += net->scratch_dh[j];
            }
        }
    }
}

/* ============================== 优化器 ==============================
 * 与 ai.c 的 optimizer_step 同一套运算顺序（Adam 偏差修正每次 step 只算一次），
 * 只是参数被组织成若干“扁平块”。这里不调用 ai.c 的符号，便于单独链接。 */
typedef struct {
    double *W;   /* 参数 */
    double *g;   /* 梯度 */
    double *m;   /* 一阶动量 */
    double *v;   /* 二阶动量 / RMSprop 缓存 */
    int n;       /* 元素个数 */
} RNNParamBlock;

/* 超参默认值与 ai.c 的 optimizer_init 完全一致 */
void rnn_optimizer_init(Optimizer *o, OptKind kind, double lr)
{
    memset(o, 0, sizeof(*o));
    o->kind = kind;
    o->lr = lr;
    o->beta1 = 0.9;
    o->beta2 = 0.999;
    o->eps = 1e-8;
    o->momentum = 0.9;
    o->alpha = 0.99;
    o->weight_decay = 0.0;
    o->step_count = 0;
}

static void optimizer_apply(RNNParamBlock *blk, int nblk, Optimizer *o)
{
    o->step_count++;
    double lr = o->lr;

    double c1 = 0.0, c2 = 0.0, o1 = 0.0, o2 = 0.0;
    if (o->kind == OPT_ADAM) {
        c1 = 1.0 - pow(o->beta1, o->step_count);
        c2 = 1.0 - pow(o->beta2, o->step_count);
        o1 = 1.0 - o->beta1;
        o2 = 1.0 - o->beta2;
    }

    for (int b = 0; b < nblk; b++) {
        double *W = blk[b].W, *g = blk[b].g, *m = blk[b].m, *v = blk[b].v;
        int n = blk[b].n;
        switch (o->kind) {
        case OPT_ADAM: {
            double b1 = o->beta1, b2 = o->beta2, eps = o->eps;
            for (int k = 0; k < n; k++) {
                double gk = g[k];
                double mm = b1 * m[k] + o1 * gk;
                double vv = b2 * v[k] + o2 * gk * gk;
                m[k] = mm;
                v[k] = vv;
                W[k] -= lr * (mm / c1) / (sqrt(vv / c2) + eps);
            }
            break;
        }
        case OPT_SGD: {
            double wd = o->weight_decay;
            for (int k = 0; k < n; k++) {
                double gk = g[k];
                if (wd > 0.0) gk += wd * W[k];
                W[k] -= lr * gk;
            }
            break;
        }
        case OPT_SGD_MOMENTUM: {
            double mu = o->momentum;
            for (int k = 0; k < n; k++) {
                double vv = mu * m[k] + lr * g[k];
                m[k] = vv;
                W[k] -= vv;
            }
            break;
        }
        case OPT_RMSPROP: {
            double al = o->alpha, eps = o->eps, om = 1.0 - al;
            for (int k = 0; k < n; k++) {
                double gk = g[k];
                double s = al * v[k] + om * gk * gk;
                v[k] = s;
                W[k] -= lr * gk / (sqrt(s) + eps);
            }
            break;
        }
        }
    }
}

/* 每个 cell 最多 4 个参数块（W_ih / W_hh / b_ih / b_hh） */
static int rnn_cell_param_blocks(RNNCell *c, RNNParamBlock *blk)
{
    int ins = c->input_size, hs = c->hidden_size;
    blk[0] = (RNNParamBlock){ c->W_ih, c->grad_W_ih, c->m_W_ih, c->v_W_ih, hs * ins };
    blk[1] = (RNNParamBlock){ c->W_hh, c->grad_W_hh, c->m_W_hh, c->v_W_hh, hs * hs };
    blk[2] = (RNNParamBlock){ c->b_ih, c->grad_b_ih, c->m_b_ih, c->v_b_ih, hs };
    blk[3] = (RNNParamBlock){ c->b_hh, c->grad_b_hh, c->m_b_hh, c->v_b_hh, hs };
    return 4;
}

static int lstm_cell_param_blocks(LSTMCell *c, RNNParamBlock *blk)
{
    int ins = c->input_size, hs = c->hidden_size, gs = c->gate_size;
    blk[0] = (RNNParamBlock){ c->W_ih, c->grad_W_ih, c->m_W_ih, c->v_W_ih, gs * ins };
    blk[1] = (RNNParamBlock){ c->W_hh, c->grad_W_hh, c->m_W_hh, c->v_W_hh, gs * hs };
    blk[2] = (RNNParamBlock){ c->b_ih, c->grad_b_ih, c->m_b_ih, c->v_b_ih, gs };
    blk[3] = (RNNParamBlock){ c->b_hh, c->grad_b_hh, c->m_b_hh, c->v_b_hh, gs };
    return 4;
}

static int gru_cell_param_blocks(GRUCell *c, RNNParamBlock *blk)
{
    int ins = c->input_size, hs = c->hidden_size, gs = c->gate_size;
    blk[0] = (RNNParamBlock){ c->W_ih, c->grad_W_ih, c->m_W_ih, c->v_W_ih, gs * ins };
    blk[1] = (RNNParamBlock){ c->W_hh, c->grad_W_hh, c->m_W_hh, c->v_W_hh, gs * hs };
    blk[2] = (RNNParamBlock){ c->b_ih, c->grad_b_ih, c->m_b_ih, c->v_b_ih, gs };
    blk[3] = (RNNParamBlock){ c->b_hh, c->grad_b_hh, c->m_b_hh, c->v_b_hh, gs };
    return 4;
}

void rnn_optimizer_step(RNN *net, Optimizer *o)
{
    /* 偏差修正只依赖 step_count，整个网络只算一次：先收集全部参数块再统一更新 */
    RNNParamBlock blk[4 * 64];
    int n = 0;
    if (net->num_layers > 64)
        rnn_fail("rnn_optimizer_step", "num_layers > 64，请扩大 blk 缓冲");
    for (int l = 0; l < net->num_layers; l++)
        n += rnn_cell_param_blocks(&net->cells[l], blk + n);
    optimizer_apply(blk, n, o);
}

void lstm_optimizer_step(LSTM *net, Optimizer *o)
{
    RNNParamBlock blk[4 * 64];
    int n = 0;
    if (net->num_layers > 64)
        rnn_fail("lstm_optimizer_step", "num_layers > 64，请扩大 blk 缓冲");
    for (int l = 0; l < net->num_layers; l++)
        n += lstm_cell_param_blocks(&net->cells[l], blk + n);
    optimizer_apply(blk, n, o);
}

void gru_optimizer_step(GRU *net, Optimizer *o)
{
    RNNParamBlock blk[4 * 64];
    int n = 0;
    if (net->num_layers > 64)
        rnn_fail("gru_optimizer_step", "num_layers > 64，请扩大 blk 缓冲");
    for (int l = 0; l < net->num_layers; l++)
        n += gru_cell_param_blocks(&net->cells[l], blk + n);
    optimizer_apply(blk, n, o);
}
