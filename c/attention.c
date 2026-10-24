/*
 * attention.c —— Transformer 组件（移植自 AI.py）
 *
 * 本文件的核心原则是「逐行照抄 Python 的运算顺序」，而不是重写成等价的矩阵形式：
 *   - Python 的 _matmul(A, B) 内部是 sum(map(_mul, row, col))，即沿 k 升序做点积；
 *     C 侧用 numeric.h 的 Neumaier 补偿求和，保持同样的下标顺序，做到逐位一致。
 *   - Python 3.12+ 的 sum() 对浮点走 Neumaier 快路径，所以源码里凡是 sum(...) 的位置
 *     （softmax 的分母、grad_b 的列和、LayerNorm 的均值/方差……）在 C 侧都用 Neumaier。
 *   - 权重梯度的形式（例如 grad_W_o = dout^T @ concat）历史上出过「整体转置」的 bug，
 *     这里按当前 Python 版逐字搬运，并用数值梯度回归测试锁住（见 tests/test_attention.c）。
 *
 * 内存：所有缓冲在 create 时按 max_seq_len 一次分配，forward/backward 热路径零 malloc。
 */
#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================ 内存 ============================ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "attention: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void xfree(void *p)
{
    free(p);
}

/* ==================== 矩阵乘 / 点积（对齐 Python 的 _matmul） ====================
 *
 * Python:
 *     def _matmul(A, B):
 *         Bt = list(zip(*B))
 *         return [[sum(map(_mul, row, col)) for col in Bt] for row in A]
 * 对每一对 (row, col) 沿 k 升序做 Neumaier 求和。下面的 helper 保持同样的遍历顺序，
 * 只是允许三个矩阵各自有不同的行跨度（leading dim），这样可以：
 *   - 直接在 (h, T, d_k) 里取某个头的切片参与运算，不必额外拷贝；
 *   - 直接在 (T, d_model) 里取 head 对应的那 d_k 列参与运算。
 * 行跨度只影响「从哪里取数」，不影响每个点积内部的累加顺序。
 */

/* 与 numeric.h 的 dot_neumaier 完全同算法，仅多了两处跨度参数 */
static double dot_strided(const double *a, int sa, const double *b, int sb, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a[(size_t)i * sa] * b[(size_t)i * sb];
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* 生成器式 sum 的跨步版，对应 Python 的 sum(f(t) for t in ...) */
static double sum_strided(const double *a, int sa, int n)
{
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        double x = a[(size_t)i * sa];
        double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return s + c;
}

/* C(m×n) = A(m×k) @ B(k×n)；lda/ldb/ldc 为各自的行跨度 */
static void mm_nn(const double *A, int lda, const double *B, int ldb, double *C, int ldc,
                  int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        const double *arow = A + (size_t)i * lda;
        for (int j = 0; j < n; j++)
            C[(size_t)i * ldc + j] = dot_strided(arow, 1, B + j, ldb, k);
    }
}

/* C(m×n) = A(m×k) @ B(n×k)^T（B 的行是 k 维），对应 _matmul(A, _transpose(B)) */
static void mm_nt(const double *A, int lda, const double *B, int ldb, double *C, int ldc,
                  int m, int k, int n)
{
    for (int i = 0; i < m; i++) {
        const double *arow = A + (size_t)i * lda;
        for (int j = 0; j < n; j++) {
            const double *brow = B + (size_t)j * ldb;
            /* 两边都连续时直接用 numeric.h 的 dot_neumaier（同一算法，等价写法） */
            C[(size_t)i * ldc + j] = (lda == k && ldb == k)
                ? dot_neumaier(arow, brow, k)
                : dot_strided(arow, 1, brow, 1, k);
        }
    }
}

/* C(k×n) = A(m×k)^T @ B(m×n)，对应 _matmul(_transpose(A), B)。
 * acc=1 时累加到 C（对应 Python 的 grads[...] += grad_W）。 */
static void mm_tn(const double *A, int lda, const double *B, int ldb, double *C, int ldc,
                  int m, int k, int n, int acc)
{
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            double v = dot_strided(A + i, lda, B + j, ldb, m);
            if (acc) C[(size_t)i * ldc + j] += v;
            else     C[(size_t)i * ldc + j] = v;
        }
    }
}

/* 一维 softmax（就地），与 Python 的 _softmax_row 同序：先 max、再 exp(x-mx)、再逐元素除 */
static void softmax_row_inplace(double *v, int n)
{
    double mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    for (int i = 0; i < n; i++) v[i] = exp(v[i] - mx);
    /* Python 的 s = sum(exps) 走 CPython 3.12+ 的 Neumaier 快路径 */
    double s = sum_neumaier(v, n);
    for (int i = 0; i < n; i++) v[i] = v[i] / s;
}

/* ============================ 形状检查 ============================ */

/* 校验二维序列张量 (T, d)，返回 T；T 超过预分配上限时报错退出 */
static int seq_check(const Tensor *t, int d, int max_seq, const char *who)
{
    if (t->ndim != 2 || t->shape[1] != d) {
        fprintf(stderr, "%s: expected a 2D tensor (seq_len, %d), got ndim=%d shape=[", who, d, t->ndim);
        for (int i = 0; i < t->ndim; i++) fprintf(stderr, "%s%d", i ? "," : "", t->shape[i]);
        fprintf(stderr, "]\n");
        exit(1);
    }
    int T = t->shape[0];
    if (T > max_seq) {
        fprintf(stderr, "%s: seq_len=%d exceeds max_seq_len=%d set at create time"
                        "（预分配缓冲不够，禁止静默截断）\n", who, T, max_seq);
        exit(1);
    }
    return T;
}

/* 把预分配的裸缓冲包装成 (T, d) 张量视图（不拥有数据，不参与释放） */
static Tensor make_view(double *data, int T, int d)
{
    Tensor t;
    t.ndim = 2;
    t.shape[0] = T;
    t.shape[1] = d;
    t.shape[2] = 1;
    t.shape[3] = 1;
    t.size = T * d;
    t.data = data;
    t.owns_data = 0;
    return t;
}

/* ============================ 私有 LayerNorm ============================ */

void tfln_init(TfLayerNorm *ln, int d, int max_seq, double eps)
{
    memset(ln, 0, sizeof(*ln));
    ln->d = d;
    ln->max_seq = max_seq;
    ln->eps = eps;
    ln->gamma = xmalloc((size_t)d * sizeof(double));
    ln->beta = xmalloc((size_t)d * sizeof(double));
    ln->ggamma = xmalloc((size_t)d * sizeof(double));
    ln->gbeta = xmalloc((size_t)d * sizeof(double));
    ln->tmp = xmalloc((size_t)d * sizeof(double));
    ln->tmp2 = xmalloc((size_t)d * sizeof(double));
    ln->mean = xmalloc((size_t)max_seq * sizeof(double));
    ln->inv_std = xmalloc((size_t)max_seq * sizeof(double));
    ln->x_hat = xmalloc((size_t)max_seq * d * sizeof(double));
    for (int j = 0; j < d; j++) { ln->gamma[j] = 1.0; ln->beta[j] = 0.0; }
    memset(ln->ggamma, 0, (size_t)d * sizeof(double));
    memset(ln->gbeta, 0, (size_t)d * sizeof(double));
    memset(ln->x_hat, 0, (size_t)max_seq * d * sizeof(double));
}

void tfln_free(TfLayerNorm *ln)
{
    if (!ln) return;
    xfree(ln->gamma); xfree(ln->beta); xfree(ln->ggamma); xfree(ln->gbeta);
    xfree(ln->tmp); xfree(ln->tmp2);
    xfree(ln->mean); xfree(ln->inv_std); xfree(ln->x_hat);
    memset(ln, 0, sizeof(*ln));
}

void tfln_zero_grad(TfLayerNorm *ln)
{
    memset(ln->ggamma, 0, (size_t)ln->d * sizeof(double));
    memset(ln->gbeta, 0, (size_t)ln->d * sizeof(double));
}

void tfln_forward_row(TfLayerNorm *ln, int row, const double *x, double *out)
{
    const int d = ln->d;
    double *xh = ln->x_hat + (size_t)row * d;

    /* 照抄 LayerNorm.forward：
     *   mean = sum(x)/d ; var = sum((xi-mean)**2)/d ; inv_std = 1/sqrt(var+eps)
     *   x_hat = (xi-mean)*inv_std ; out = gamma*x_hat + beta */
    double mean = sum_neumaier(x, d) / (double)d;
    for (int j = 0; j < d; j++) {
        double t = x[j] - mean;
        xh[j] = t * t;                 /* 对应 Python 的 (xi - mean) ** 2 */
    }
    double var = sum_neumaier(xh, d) / (double)d;
    double inv_std = 1.0 / sqrt(var + ln->eps);
    ln->mean[row] = mean;
    ln->inv_std[row] = inv_std;
    for (int j = 0; j < d; j++) xh[j] = (x[j] - mean) * inv_std;
    for (int j = 0; j < d; j++) out[j] = ln->gamma[j] * xh[j] + ln->beta[j];
}

void tfln_backward_row(TfLayerNorm *ln, int row, const double *dout,
                       double *grad_gamma, double *grad_beta, double *dx)
{
    const int d = ln->d;
    const double *xh = ln->x_hat + (size_t)row * d;
    double *dx_hat = ln->tmp;
    double *tmp2 = ln->tmp2;

    /* 照抄 LayerNorm.backward 的返回顺序 (grad_gamma, grad_beta, dx) */
    for (int j = 0; j < d; j++) {
        grad_gamma[j] = dout[j] * xh[j];
        grad_beta[j] = dout[j];
        dx_hat[j] = dout[j] * ln->gamma[j];
    }
    double s1 = sum_neumaier(dx_hat, d);
    for (int j = 0; j < d; j++) tmp2[j] = dx_hat[j] * xh[j];
    double s2 = sum_neumaier(tmp2, d);
    for (int j = 0; j < d; j++)
        dx[j] = ln->inv_std[row] * (d * dx_hat[j] - s1 - xh[j] * s2) / d;
}

/* ============================ 私有 Dropout ============================ */

/* splitmix64：只为 dropout mask 提供一个可复现的均匀分布源。
 * 注意：与 Python 的 Mersenne Twister 不同源，p>0 时不可能与 Python 逐位对齐。 */
static double rng_uniform(unsigned long long *s)
{
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);   /* [0,1) */
}

/* 参数检查失败直接终止：C 里没有异常，越界/非法参数会破坏内存 */
static void tfd_die(const char *where, const char *msg)
{
    fprintf(stderr, "error [%s]: %s\n", where, msg);
    exit(1);
}

void tfd_init(TfDropout *dp, double p, int d, int max_seq, unsigned long long seed)
{
    memset(dp, 0, sizeof(*dp));
    /* p 越界会产生除零/全零/放大，与 Python 的 Dropout(p) 校验一致 */
    if (!(p >= 0.0 && p < 1.0))
        tfd_die("tfd_init", "dropout p 必须在 [0,1) 内");
    dp->p = p;
    dp->train = 1;
    dp->d = d;
    dp->max_seq = max_seq;
    dp->rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    dp->mask = xmalloc((size_t)d * max_seq * sizeof(double));
    /* 未 forward 就 backward 时退化为恒等（Python 里 masks 为 None 时会直接跳过 mask 那一步） */
    for (int i = 0; i < d * max_seq; i++) dp->mask[i] = 1.0;
}

void tfd_free(TfDropout *dp)
{
    if (!dp) return;
    xfree(dp->mask);
    memset(dp, 0, sizeof(*dp));
}

void tfd_forward_row(TfDropout *dp, int row, const double *x, double *out)
{
    const int d = dp->d;
    double *mask = dp->mask + (size_t)row * d;
    if (dp->train) {
        double scale = 1.0 / (1.0 - dp->p);
        for (int j = 0; j < d; j++)
            mask[j] = (rng_uniform(&dp->rng) > dp->p) ? scale : 0.0;
        for (int j = 0; j < d; j++) out[j] = x[j] * mask[j];
    } else {
        for (int j = 0; j < d; j++) mask[j] = 1.0;
        if (out != x) memcpy(out, x, (size_t)d * sizeof(double));
    }
}

void tfd_backward_row(const TfDropout *dp, int row, const double *dout, double *dx)
{
    const int d = dp->d;
    const double *mask = dp->mask + (size_t)row * d;
    for (int j = 0; j < d; j++) dx[j] = dout[j] * mask[j];
}

/* ==========================================================================
 *  MultiHeadAttention
 * ========================================================================== */

MultiHeadAttention *mha_create(int d_model, int num_heads, int max_seq_len, double dropout_p)
{
    if (d_model <= 0 || num_heads <= 0 || max_seq_len <= 0) {
        fprintf(stderr, "mha_create: params非法 (d_model=%d, num_heads=%d, max_seq_len=%d)\n",
                d_model, num_heads, max_seq_len);
        exit(1);
    }
    if (d_model % num_heads != 0) {
        fprintf(stderr, "mha_create: d_model(%d) 必须能被 num_heads(%d) 整除\n", d_model, num_heads);
        exit(1);
    }
    MultiHeadAttention *m = xmalloc(sizeof(*m));
    memset(m, 0, sizeof(*m));
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_k = d_model / num_heads;
    m->max_seq_len = max_seq_len;
    m->dropout_p = dropout_p;

    const size_t dd = (size_t)d_model * d_model;
    const size_t d1 = (size_t)d_model;
    const size_t T = (size_t)max_seq_len;
    const size_t hT = (size_t)num_heads * T;
    const size_t hTT = (size_t)num_heads * T * T;

    m->W_q = xmalloc(dd * sizeof(double)); m->W_k = xmalloc(dd * sizeof(double));
    m->W_v = xmalloc(dd * sizeof(double)); m->W_o = xmalloc(dd * sizeof(double));
    m->b_q = xmalloc(d1 * sizeof(double)); m->b_k = xmalloc(d1 * sizeof(double));
    m->b_v = xmalloc(d1 * sizeof(double)); m->b_o = xmalloc(d1 * sizeof(double));
    m->grad_W_q = xmalloc(dd * sizeof(double)); m->grad_W_k = xmalloc(dd * sizeof(double));
    m->grad_W_v = xmalloc(dd * sizeof(double)); m->grad_W_o = xmalloc(dd * sizeof(double));
    m->grad_b_q = xmalloc(d1 * sizeof(double)); m->grad_b_k = xmalloc(d1 * sizeof(double));
    m->grad_b_v = xmalloc(d1 * sizeof(double)); m->grad_b_o = xmalloc(d1 * sizeof(double));

    m->x = xmalloc(T * d1 * sizeof(double));
    m->q_full = xmalloc(T * d1 * sizeof(double));
    m->k_full = xmalloc(T * d1 * sizeof(double));
    m->v_full = xmalloc(T * d1 * sizeof(double));
    m->Q = xmalloc(hT * m->d_k * sizeof(double));
    m->K = xmalloc(hT * m->d_k * sizeof(double));
    m->V = xmalloc(hT * m->d_k * sizeof(double));
    m->attn = xmalloc(hTT * sizeof(double));
    m->head_out = xmalloc(hT * m->d_k * sizeof(double));
    m->concat = xmalloc(T * d1 * sizeof(double));

    m->d_concat = xmalloc(T * d1 * sizeof(double));
    m->d_head = xmalloc(hT * m->d_k * sizeof(double));
    m->d_scores = xmalloc(hTT * sizeof(double));
    m->dQ = xmalloc(hT * m->d_k * sizeof(double));
    m->dK = xmalloc(hT * m->d_k * sizeof(double));
    m->dV = xmalloc(hT * m->d_k * sizeof(double));
    m->dQ_full = xmalloc(T * d1 * sizeof(double));
    m->dK_full = xmalloc(T * d1 * sizeof(double));
    m->dV_full = xmalloc(T * d1 * sizeof(double));
    m->tmp = xmalloc(T * d1 * sizeof(double));

    /* 权重初始化：与 Python 一样用 N(0, sqrt(2/d_model))，但 C 侧 rand 序列不同
     * （跨语言对齐时用 mha_set_params 灌入 Python 的权重） */
    unsigned long long rng = 0x243F6A8885A308D3ULL;
    double std = sqrt(2.0 / (double)d_model);
    double *allW[4] = { m->W_q, m->W_k, m->W_v, m->W_o };
    for (int k = 0; k < 4; k++) {
        for (size_t i = 0; i < dd; i++) {
            /* Box-Muller */
            double u1 = rng_uniform(&rng), u2 = rng_uniform(&rng);
            if (u1 < 1e-300) u1 = 1e-300;
            allW[k][i] = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
        }
    }
    for (int j = 0; j < d_model; j++) {
        m->b_q[j] = m->b_k[j] = m->b_v[j] = m->b_o[j] = 0.0;
    }
    mha_zero_grad(m);
    return m;
}

void mha_free(MultiHeadAttention *m)
{
    if (!m) return;
    xfree(m->W_q); xfree(m->W_k); xfree(m->W_v); xfree(m->W_o);
    xfree(m->b_q); xfree(m->b_k); xfree(m->b_v); xfree(m->b_o);
    xfree(m->grad_W_q); xfree(m->grad_W_k); xfree(m->grad_W_v); xfree(m->grad_W_o);
    xfree(m->grad_b_q); xfree(m->grad_b_k); xfree(m->grad_b_v); xfree(m->grad_b_o);
    xfree(m->x); xfree(m->q_full); xfree(m->k_full); xfree(m->v_full);
    xfree(m->Q); xfree(m->K); xfree(m->V); xfree(m->attn); xfree(m->head_out); xfree(m->concat);
    xfree(m->d_concat); xfree(m->d_head); xfree(m->d_scores);
    xfree(m->dQ); xfree(m->dK); xfree(m->dV);
    xfree(m->dQ_full); xfree(m->dK_full); xfree(m->dV_full); xfree(m->tmp);
    xfree(m);
}

void mha_zero_grad(MultiHeadAttention *m)
{
    const size_t dd = (size_t)m->d_model * m->d_model;
    const size_t d1 = (size_t)m->d_model;
    memset(m->grad_W_q, 0, dd * sizeof(double));
    memset(m->grad_W_k, 0, dd * sizeof(double));
    memset(m->grad_W_v, 0, dd * sizeof(double));
    memset(m->grad_W_o, 0, dd * sizeof(double));
    memset(m->grad_b_q, 0, d1 * sizeof(double));
    memset(m->grad_b_k, 0, d1 * sizeof(double));
    memset(m->grad_b_v, 0, d1 * sizeof(double));
    memset(m->grad_b_o, 0, d1 * sizeof(double));
}

void mha_set_params(MultiHeadAttention *m,
                    const double *W_q, const double *b_q,
                    const double *W_k, const double *b_k,
                    const double *W_v, const double *b_v,
                    const double *W_o, const double *b_o)
{
    const double *src[8] = { W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o };
    for (int i = 0; i < 8; i++) {
        if (!src[i]) {
            fprintf(stderr, "mha_set_params: 第 %d 个参数指针为 NULL (8 个都必须提供)\n", i);
            exit(1);
        }
    }
    const size_t dd = (size_t)m->d_model * m->d_model;
    const size_t d1 = (size_t)m->d_model;
    memcpy(m->W_q, W_q, dd * sizeof(double));
    memcpy(m->W_k, W_k, dd * sizeof(double));
    memcpy(m->W_v, W_v, dd * sizeof(double));
    memcpy(m->W_o, W_o, dd * sizeof(double));
    memcpy(m->b_q, b_q, d1 * sizeof(double));
    memcpy(m->b_k, b_k, d1 * sizeof(double));
    memcpy(m->b_v, b_v, d1 * sizeof(double));
    memcpy(m->b_o, b_o, d1 * sizeof(double));
}

int mha_param_count(const MultiHeadAttention *m)
{
    return 4 * m->d_model * m->d_model + 4 * m->d_model;
}

void mha_forward(MultiHeadAttention *m, const Tensor *x, Tensor *out)
{
    const int d = m->d_model, h = m->num_heads, dk = m->d_k;
    const int T = seq_check(x, d, m->max_seq_len, "mha_forward");
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(out, 2, dims, "mha_forward(out)");
    m->seq_len = T;

    /* Python 存的是引用（self.x = x_seq）；C 侧拷一份，避免调用方复用缓冲后反向算错 */
    memcpy(m->x, x->data, (size_t)T * d * sizeof(double));

    /* q_full = x @ W_q^T —— 等价于 Python 的 _matmul(x_seq, _transpose(self.W_q)) */
    mm_nt(m->x, d, m->W_q, d, m->q_full, d, T, d, d);
    mm_nt(m->x, d, m->W_k, d, m->k_full, d, T, d, d);
    mm_nt(m->x, d, m->W_v, d, m->v_full, d, T, d, d);

    /* 加 bias（照抄 Python 的三行 +=） */
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d; j++) {
            m->q_full[(size_t)t * d + j] += m->b_q[j];
            m->k_full[(size_t)t * d + j] += m->b_k[j];
            m->v_full[(size_t)t * d + j] += m->b_v[j];
        }
    }

    /* 拆多头：(T, d_model) → h × (T, d_k)（对应 self.Q / self.K / self.V） */
    for (int head = 0; head < h; head++) {
        const int off = head * dk;
        for (int t = 0; t < T; t++) {
            const size_t dst = ((size_t)head * T + t) * dk;
            const size_t src = (size_t)t * d + off;
            memcpy(m->Q + dst, m->q_full + src, (size_t)dk * sizeof(double));
            memcpy(m->K + dst, m->k_full + src, (size_t)dk * sizeof(double));
            memcpy(m->V + dst, m->v_full + src, (size_t)dk * sizeof(double));
        }
    }

    const double scale = 1.0 / sqrt((double)dk);
    for (int head = 0; head < h; head++) {
        const int off = head * dk;
        const double *qh = m->Q + (size_t)head * T * dk;
        const double *kh = m->K + (size_t)head * T * dk;
        const double *vh = m->V + (size_t)head * T * dk;
        double *attn = m->attn + (size_t)head * T * T;
        double *ho = m->head_out + (size_t)head * T * dk;

        /* scores = q_h @ k_h^T * scale，随后逐行 softmax（就地） */
        mm_nt(qh, dk, kh, dk, attn, T, T, dk, T);
        for (int i = 0; i < T * T; i++) attn[i] *= scale;
        for (int t = 0; t < T; t++) softmax_row_inplace(attn + (size_t)t * T, T);

        /* out = attn @ v_h */
        mm_nn(attn, T, vh, dk, ho, dk, T, T, dk);

        /* 拼回 d_model */
        for (int t = 0; t < T; t++)
            memcpy(m->concat + (size_t)t * d + off, ho + (size_t)t * dk,
                   (size_t)dk * sizeof(double));
    }

    /* output = concat @ W_o^T + b_o */
    mm_nt(m->concat, d, m->W_o, d, out->data, d, T, d, d);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < d; j++) out->data[(size_t)t * d + j] += m->b_o[j];
}

void mha_backward(MultiHeadAttention *m, const Tensor *dout, Tensor *dx)
{
    const int d = m->d_model, h = m->num_heads, dk = m->d_k;
    const int T = m->seq_len;
    if (T <= 0) {
        fprintf(stderr, "mha_backward: 尚未调用过 forward (seq_len=0)\n");
        exit(1);
    }
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(dout, 2, dims, "mha_backward(dout)");
    tensor_check_shape(dx, 2, dims, "mha_backward(dx)");

    const double scale = 1.0 / sqrt((double)dk);
    const double *doutp = dout->data;

    /* 1. d_concat = dout @ W_o
     *    grad_W_o = dout^T @ concat
     *    （Python 注释：前向 output = concat @ W_o^T，所以 grad_W_o[j][i] = sum_t dout[t][j]*concat[t][i]，
     *      写成 concat^T @ dout 会得到转置 —— 这条曾经是真实 bug，务必保持此形式） */
    mm_nn(doutp, d, m->W_o, d, m->d_concat, d, T, d, d);
    mm_tn(doutp, d, m->concat, d, m->grad_W_o, d, T, d, d, 1);
    /* grad_b_o[j] = sum_t dout[t][j]（生成器 sum → Neumaier，t 升序） */
    for (int j = 0; j < d; j++) m->grad_b_o[j] += sum_strided(doutp + j, d, T);

    /* 2. 拆回各头 */
    for (int head = 0; head < h; head++) {
        const int off = head * dk;
        for (int t = 0; t < T; t++)
            memcpy(m->d_head + ((size_t)head * T + t) * dk,
                   m->d_concat + (size_t)t * d + off, (size_t)dk * sizeof(double));
    }

    /* 3. 各头分别反向 */
    for (int head = 0; head < h; head++) {
        const double *attn = m->attn + (size_t)head * T * T;
        const double *vh = m->V + (size_t)head * T * dk;
        const double *qh = m->Q + (size_t)head * T * dk;
        const double *kh = m->K + (size_t)head * T * dk;
        const double *d_out = m->d_head + (size_t)head * T * dk;
        double *dQ = m->dQ + (size_t)head * T * dk;
        double *dK = m->dK + (size_t)head * T * dk;
        double *dV = m->dV + (size_t)head * T * dk;
        double *ds = m->d_scores + (size_t)head * T * T;

        /* dV = attn^T @ d_out ：attn 是 (T×T)，结果 (T×d_k) */
        mm_tn(attn, T, d_out, dk, dV, dk, T, T, dk, 0);

        /* d_attn = d_out @ V^T —— 先写进 d_scores，随后原地覆盖成 d_scores */
        mm_nt(d_out, dk, vh, dk, ds, T, T, dk, T);

        /* softmax 反向：d_scores[t][j] = attn[t][j] * (d_attn[t][j] - s_t) * scale，
         * 其中 s_t = sum_i attn[t][i]*d_attn[t][i] 只依赖 t，
         * 必须提到 j 循环外面（Python 的注释：原来每个 (t,j) 都重算 → O(seq²) 冗余）。
         * 原地写回是安全的：s_t 在覆盖 row_d 之前就已算好，且每个 j 只读自己的旧值。 */
        for (int t = 0; t < T; t++) {
            const double *row_a = attn + (size_t)t * T;
            double *row_d = ds + (size_t)t * T;
            double s = dot_neumaier(row_a, row_d, T);
            for (int j = 0; j < T; j++)
                row_d[j] = row_a[j] * (row_d[j] - s) * scale;
        }

        /* dQ = d_scores @ K_h ; dK = d_scores^T @ Q_h */
        mm_nn(ds, T, kh, dk, dQ, dk, T, T, dk);
        mm_tn(ds, T, qh, dk, dK, dk, T, T, dk, 0);
    }

    /* 4. 拼接各头的 dQ/dK/dV → (T, d_model) */
    for (int head = 0; head < h; head++) {
        const int off = head * dk;
        for (int t = 0; t < T; t++) {
            const size_t src = ((size_t)head * T + t) * dk;
            const size_t dst = (size_t)t * d + off;
            memcpy(m->dQ_full + dst, m->dQ + src, (size_t)dk * sizeof(double));
            memcpy(m->dK_full + dst, m->dK + src, (size_t)dk * sizeof(double));
            memcpy(m->dV_full + dst, m->dV + src, (size_t)dk * sizeof(double));
        }
    }

    /* 5. 通过投影矩阵反传梯度（累加）
     *    grad_W_q[b][a] = sum_t dQ_full[t][b] * x[t][a] 即 dQ_full^T @ x；
     *    写成 x^T @ dQ_full 是它的转置（方阵下不报错但会静默训歪）—— 这里保持正确形式 */
    mm_tn(m->dQ_full, d, m->x, d, m->grad_W_q, d, T, d, d, 1);
    mm_tn(m->dK_full, d, m->x, d, m->grad_W_k, d, T, d, d, 1);
    mm_tn(m->dV_full, d, m->x, d, m->grad_W_v, d, T, d, d, 1);
    for (int j = 0; j < d; j++) {
        m->grad_b_q[j] += sum_strided(m->dQ_full + j, d, T);
        m->grad_b_k[j] += sum_strided(m->dK_full + j, d, T);
        m->grad_b_v[j] += sum_strided(m->dV_full + j, d, T);
    }

    /* 6. d_x = (dx_q + dx_k) + dx_v —— 加法顺序与 Python 一致 */
    mm_nn(m->dQ_full, d, m->W_q, d, dx->data, d, T, d, d);
    mm_nn(m->dK_full, d, m->W_k, d, m->tmp, d, T, d, d);
    for (int i = 0; i < T * d; i++) dx->data[i] += m->tmp[i];
    mm_nn(m->dV_full, d, m->W_v, d, m->tmp, d, T, d, d);
    for (int i = 0; i < T * d; i++) dx->data[i] += m->tmp[i];
}

/* ==========================================================================
 *  FeedForward
 * ========================================================================== */

FeedForward *ff_create(int d_model, int d_ff, int max_seq_len)
{
    if (d_model <= 0 || d_ff <= 0 || max_seq_len <= 0) {
        fprintf(stderr, "ff_create: params非法 (d_model=%d, d_ff=%d, max_seq_len=%d)\n",
                d_model, d_ff, max_seq_len);
        exit(1);
    }
    FeedForward *f = xmalloc(sizeof(*f));
    memset(f, 0, sizeof(*f));
    f->d_model = d_model;
    f->d_ff = d_ff;
    f->max_seq_len = max_seq_len;

    const size_t w1 = (size_t)d_ff * d_model, w2 = (size_t)d_model * d_ff;
    const size_t T = (size_t)max_seq_len;
    f->W1 = xmalloc(w1 * sizeof(double));
    f->b1 = xmalloc((size_t)d_ff * sizeof(double));
    f->W2 = xmalloc(w2 * sizeof(double));
    f->b2 = xmalloc((size_t)d_model * sizeof(double));
    f->grad_W1 = xmalloc(w1 * sizeof(double));
    f->grad_b1 = xmalloc((size_t)d_ff * sizeof(double));
    f->grad_W2 = xmalloc(w2 * sizeof(double));
    f->grad_b2 = xmalloc((size_t)d_model * sizeof(double));
    f->x = xmalloc(T * d_model * sizeof(double));
    f->h = xmalloc(T * d_ff * sizeof(double));
    f->dh = xmalloc(T * d_ff * sizeof(double));

    /* 与 Python 一致：W1 ~ N(0, sqrt(2/d_model))，W2 ~ N(0, sqrt(2/d_ff)) */
    unsigned long long rng = 0x13198A2E03707344ULL;
    double std1 = sqrt(2.0 / (double)d_model), std2 = sqrt(2.0 / (double)d_ff);
    for (size_t i = 0; i < w1; i++) {
        double u1 = rng_uniform(&rng), u2 = rng_uniform(&rng);
        if (u1 < 1e-300) u1 = 1e-300;
        f->W1[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std1;
    }
    for (size_t i = 0; i < w2; i++) {
        double u1 = rng_uniform(&rng), u2 = rng_uniform(&rng);
        if (u1 < 1e-300) u1 = 1e-300;
        f->W2[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std2;
    }
    for (int j = 0; j < d_ff; j++) f->b1[j] = 0.0;
    for (int j = 0; j < d_model; j++) f->b2[j] = 0.0;
    ff_zero_grad(f);
    return f;
}

void ff_free(FeedForward *f)
{
    if (!f) return;
    xfree(f->W1); xfree(f->b1); xfree(f->W2); xfree(f->b2);
    xfree(f->grad_W1); xfree(f->grad_b1); xfree(f->grad_W2); xfree(f->grad_b2);
    xfree(f->x); xfree(f->h); xfree(f->dh);
    xfree(f);
}

void ff_zero_grad(FeedForward *f)
{
    memset(f->grad_W1, 0, (size_t)f->d_ff * f->d_model * sizeof(double));
    memset(f->grad_b1, 0, (size_t)f->d_ff * sizeof(double));
    memset(f->grad_W2, 0, (size_t)f->d_model * f->d_ff * sizeof(double));
    memset(f->grad_b2, 0, (size_t)f->d_model * sizeof(double));
}

void ff_set_params(FeedForward *f, const double *W1, const double *b1,
                   const double *W2, const double *b2)
{
    if (!W1 || !b1 || !W2 || !b2) {
        fprintf(stderr, "ff_set_params: 4 个参数指针都必须非 NULL\n");
        exit(1);
    }
    memcpy(f->W1, W1, (size_t)f->d_ff * f->d_model * sizeof(double));
    memcpy(f->b1, b1, (size_t)f->d_ff * sizeof(double));
    memcpy(f->W2, W2, (size_t)f->d_model * f->d_ff * sizeof(double));
    memcpy(f->b2, b2, (size_t)f->d_model * sizeof(double));
}

int ff_param_count(const FeedForward *f)
{
    return 2 * f->d_model * f->d_ff + f->d_model + f->d_ff;
}

void ff_forward(FeedForward *f, const Tensor *x, Tensor *out)
{
    const int dm = f->d_model, dff = f->d_ff;
    const int T = seq_check(x, dm, f->max_seq_len, "ff_forward");
    int dims[2];
    dims[0] = T; dims[1] = dm;
    tensor_check_shape(out, 2, dims, "ff_forward(out)");
    f->seq_len = T;

    memcpy(f->x, x->data, (size_t)T * dm * sizeof(double));

    /* h = x @ W1^T，然后 h = max(0, h + b1)（Python 就地覆盖成 ReLU 之后的值） */
    mm_nt(f->x, dm, f->W1, dm, f->h, dff, T, dm, dff);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < dff; j++) {
            double v = f->h[(size_t)t * dff + j] + f->b1[j];
            f->h[(size_t)t * dff + j] = (v > 0.0) ? v : 0.0;   /* max(0, v) */
        }
    }
    /* out = h @ W2^T + b2 */
    mm_nt(f->h, dff, f->W2, dff, out->data, dm, T, dff, dm);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < dm; j++) out->data[(size_t)t * dm + j] += f->b2[j];
}

void ff_backward(FeedForward *f, const Tensor *dout, Tensor *dx)
{
    const int dm = f->d_model, dff = f->d_ff;
    const int T = f->seq_len;
    if (T <= 0) {
        fprintf(stderr, "ff_backward: 尚未调用过 forward (seq_len=0)\n");
        exit(1);
    }
    int dims[2];
    dims[0] = T; dims[1] = dm;
    tensor_check_shape(dout, 2, dims, "ff_backward(dout)");
    tensor_check_shape(dx, 2, dims, "ff_backward(dx)");

    /* grad_W2 = dout^T @ h ；grad_b2[j] = sum_t dout[t][j] */
    mm_tn(dout->data, dm, f->h, dff, f->grad_W2, dff, T, dm, dff, 1);
    for (int j = 0; j < dm; j++) f->grad_b2[j] += sum_strided(dout->data + j, dm, T);

    /* dh = dout @ W2，再过 ReLU 反向（h <= 0 的位置清零，与 Python 的判据一致） */
    mm_nn(dout->data, dm, f->W2, dff, f->dh, dff, T, dm, dff);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < dff; j++)
            if (f->h[(size_t)t * dff + j] <= 0.0) f->dh[(size_t)t * dff + j] = 0.0;

    /* grad_W1 = dh^T @ x ；grad_b1[j] = sum_t dh[t][j] ；dx = dh @ W1 */
    mm_tn(f->dh, dff, f->x, dm, f->grad_W1, dm, T, dff, dm, 1);
    for (int j = 0; j < dff; j++) f->grad_b1[j] += sum_strided(f->dh + j, dff, T);
    mm_nn(f->dh, dff, f->W1, dm, dx->data, dm, T, dff, dm);
}

/* ==========================================================================
 *  TransformerEncoderLayer
 * ========================================================================== */

TransformerEncoderLayer *tel_create(int d_model, int num_heads, int d_ff,
                                    int max_seq_len, double dropout_p,
                                    unsigned long long seed)
{
    if (d_model <= 0 || num_heads <= 0 || d_ff <= 0 || max_seq_len <= 0) {
        fprintf(stderr, "tel_create: params非法\n");
        exit(1);
    }
    if (dropout_p < 0.0 || dropout_p >= 1.0) {
        fprintf(stderr, "tel_create: dropout=%g 必须在 [0,1) 内\n", dropout_p);
        exit(1);
    }
    TransformerEncoderLayer *l = xmalloc(sizeof(*l));
    memset(l, 0, sizeof(*l));
    l->d_model = d_model;
    l->max_seq_len = max_seq_len;
    l->self_attn = mha_create(d_model, num_heads, max_seq_len, dropout_p);
    l->ffn = ff_create(d_model, d_ff, max_seq_len);
    tfln_init(&l->ln1, d_model, max_seq_len, 1e-5);
    tfln_init(&l->ln2, d_model, max_seq_len, 1e-5);
    tfd_init(&l->dropout1, dropout_p, d_model, max_seq_len, seed);
    tfd_init(&l->dropout2, dropout_p, d_model, max_seq_len, seed ^ 0x9E3779B97F4A7C15ULL);

    const size_t TD = (size_t)max_seq_len * d_model;
    l->buf1 = xmalloc(TD * sizeof(double));
    l->buf2 = xmalloc(TD * sizeof(double));
    l->buf3 = xmalloc(TD * sizeof(double));
    l->ln_dgamma = xmalloc((size_t)d_model * sizeof(double));
    l->ln_dbeta = xmalloc((size_t)d_model * sizeof(double));
    return l;
}

void tel_free(TransformerEncoderLayer *l)
{
    if (!l) return;
    mha_free(l->self_attn);
    ff_free(l->ffn);
    tfln_free(&l->ln1);
    tfln_free(&l->ln2);
    tfd_free(&l->dropout1);
    tfd_free(&l->dropout2);
    xfree(l->buf1); xfree(l->buf2); xfree(l->buf3);
    xfree(l->ln_dgamma); xfree(l->ln_dbeta);
    xfree(l);
}

void tel_zero_grad(TransformerEncoderLayer *l)
{
    /* 对应 Python 的 zero_grad：清 LayerNorm 梯度 + 子模块梯度 */
    tfln_zero_grad(&l->ln1);
    tfln_zero_grad(&l->ln2);
    mha_zero_grad(l->self_attn);
    ff_zero_grad(l->ffn);
}

void tel_set_train(TransformerEncoderLayer *l, int train)
{
    l->dropout1.train = train ? 1 : 0;
    l->dropout2.train = train ? 1 : 0;
}

void tel_forward(TransformerEncoderLayer *l, const Tensor *x, Tensor *out)
{
    const int d = l->d_model;
    const int T = seq_check(x, d, l->max_seq_len, "tel_forward");
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(out, 2, dims, "tel_forward(out)");
    l->seq_len = T;

    Tensor b1 = make_view(l->buf1, T, d);
    Tensor b2 = make_view(l->buf2, T, d);
    Tensor b3 = make_view(l->buf3, T, d);

    /* 1) Self-Attention（Python 的 MHA 内部不做 dropout，dropout1 在外面逐 token 做） */
    mha_forward(l->self_attn, x, &b1);                       /* buf1 = attn_out */
    for (int t = 0; t < T; t++)
        tfd_forward_row(&l->dropout1, t, l->buf1 + (size_t)t * d, l->buf1 + (size_t)t * d);
    /* 2) Residual + LN1 */
    for (int i = 0; i < T * d; i++) l->buf1[i] += x->data[i];   /* res1 = x + attn_out */
    for (int t = 0; t < T; t++)
        tfln_forward_row(&l->ln1, t, l->buf1 + (size_t)t * d, l->buf2 + (size_t)t * d);
    /* 3) FFN（Python 里没有在 FFN 内部做 dropout，dropout2 在外面逐 token 做） */
    ff_forward(l->ffn, &b2, &b3);                            /* buf3 = ffn_out */
    for (int t = 0; t < T; t++)
        tfd_forward_row(&l->dropout2, t, l->buf3 + (size_t)t * d, l->buf3 + (size_t)t * d);
    /* 4) Residual + LN2 */
    for (int i = 0; i < T * d; i++) l->buf2[i] += l->buf3[i];  /* res2 = ln1_out + ffn_out */
    for (int t = 0; t < T; t++)
        tfln_forward_row(&l->ln2, t, l->buf2 + (size_t)t * d, out->data + (size_t)t * d);
}

void tel_backward(TransformerEncoderLayer *l, const Tensor *dout, Tensor *dx)
{
    const int d = l->d_model;
    const int T = l->seq_len;
    if (T <= 0) {
        fprintf(stderr, "tel_backward: 尚未调用过 forward (seq_len=0)\n");
        exit(1);
    }
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(dout, 2, dims, "tel_backward(dout)");
    tensor_check_shape(dx, 2, dims, "tel_backward(dx)");

    /* 缓冲的角色分配（前向数据到这里全部已无用，可安全复用）：
     *   buf1 = d_ffn（masked，喂给 FFN 反向）→ 稍后复用为 d_attn
     *   buf2 = d_ln1（FFN 反传结果，再加直通分支 d_res2）
     *   buf3 = d_res2（未加 mask）→ 稍后复用为 d_res1（未加 mask） */
    Tensor b1 = make_view(l->buf1, T, d);
    Tensor b2 = make_view(l->buf2, T, d);

    /* LN2 反向（逐 token）→ buf3 = d_res2
     * 注意：Python 里 d_res2 是「未加 dropout mask」的版本，残差直通分支后面要用的正是它，
     * 所以这里不能像 dropout 那样原地覆盖，必须写到另一块缓冲。 */
    for (int t = 0; t < T; t++) {
        tfln_backward_row(&l->ln2, t, dout->data + (size_t)t * d,
                          l->ln_dgamma, l->ln_dbeta, l->buf3 + (size_t)t * d);
        for (int j = 0; j < d; j++) {   /* 与 Python 一样逐 token 累加 ln2 的 gamma/beta 梯度 */
            l->ln2.ggamma[j] += l->ln_dgamma[j];
            l->ln2.gbeta[j] += l->ln_dbeta[j];
        }
    }
    /* Dropout2 反向：buf3 → buf1 = d_ffn（masked） */
    for (int t = 0; t < T; t++)
        tfd_backward_row(&l->dropout2, t, l->buf3 + (size_t)t * d, l->buf1 + (size_t)t * d);

    /* FFN 反向 → buf2 = d_ln1；随后加上残差直通分支（未 mask 的 d_res2） */
    ff_backward(l->ffn, &b1, &b2);
    for (int i = 0; i < T * d; i++) l->buf2[i] += l->buf3[i];

    /* LN1 反向（逐 token）→ buf3 = d_res1（未加 mask；buf3 里的 d_res2 已经用完了） */
    for (int t = 0; t < T; t++) {
        tfln_backward_row(&l->ln1, t, l->buf2 + (size_t)t * d,
                          l->ln_dgamma, l->ln_dbeta, l->buf3 + (size_t)t * d);
        for (int j = 0; j < d; j++) {
            l->ln1.ggamma[j] += l->ln_dgamma[j];
            l->ln1.gbeta[j] += l->ln_dbeta[j];
        }
    }
    /* Dropout1 反向：buf3 → buf1 = d_attn（masked；b1 在 FFN 反向之后已经空出来） */
    for (int t = 0; t < T; t++)
        tfd_backward_row(&l->dropout1, t, l->buf3 + (size_t)t * d, l->buf1 + (size_t)t * d);

    /* Self-Attn 反向 → dx；最后加残差直通分支（Python: d_x = d_res1; d_x += d_attn_out） */
    mha_backward(l->self_attn, &b1, dx);
    for (int i = 0; i < T * d; i++) dx->data[i] += l->buf3[i];
}

void tel_sgd_step(TransformerEncoderLayer *l, double lr)
{
    /* 对应 Python 的 sgd_step：对 self_attn / ffn / LayerNorm 依次 param -= lr*grad */
    MultiHeadAttention *a = l->self_attn;
    FeedForward *f = l->ffn;
    const int d = l->d_model;
    const size_t dd = (size_t)d * d;

    for (size_t i = 0; i < dd; i++) {
        a->W_q[i] -= lr * a->grad_W_q[i];
        a->W_k[i] -= lr * a->grad_W_k[i];
        a->W_v[i] -= lr * a->grad_W_v[i];
        a->W_o[i] -= lr * a->grad_W_o[i];
    }
    for (int i = 0; i < d; i++) {
        a->b_q[i] -= lr * a->grad_b_q[i];
        a->b_k[i] -= lr * a->grad_b_k[i];
        a->b_v[i] -= lr * a->grad_b_v[i];
        a->b_o[i] -= lr * a->grad_b_o[i];
    }
    for (size_t i = 0; i < (size_t)f->d_ff * f->d_model; i++) f->W1[i] -= lr * f->grad_W1[i];
    for (int i = 0; i < f->d_ff; i++) f->b1[i] -= lr * f->grad_b1[i];
    for (size_t i = 0; i < (size_t)f->d_model * f->d_ff; i++) f->W2[i] -= lr * f->grad_W2[i];
    for (int i = 0; i < f->d_model; i++) f->b2[i] -= lr * f->grad_b2[i];

    for (int i = 0; i < d; i++) {
        l->ln1.gamma[i] -= lr * l->ln1.ggamma[i];
        l->ln1.beta[i] -= lr * l->ln1.gbeta[i];
        l->ln2.gamma[i] -= lr * l->ln2.ggamma[i];
        l->ln2.beta[i] -= lr * l->ln2.gbeta[i];
    }
}

int tel_param_count(const TransformerEncoderLayer *l)
{
    return mha_param_count(l->self_attn) + ff_param_count(l->ffn) + 4 * l->d_model;
}

/* ==========================================================================
 *  TransformerEncoder
 * ========================================================================== */

TransformerEncoder *te_create(int d_model, int num_heads, int d_ff, int num_layers,
                              int max_seq_len, double dropout_p, unsigned long long seed)
{
    if (num_layers <= 0) {
        fprintf(stderr, "te_create: num_layers 必须 >= 1 (空栈请直接用恒等模块)\n");
        exit(1);
    }
    TransformerEncoder *e = xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->num_layers = num_layers;
    e->d_model = d_model;
    e->max_seq_len = max_seq_len;
    e->layers = xmalloc((size_t)num_layers * sizeof(TransformerEncoderLayer *));
    for (int i = 0; i < num_layers; i++)
        e->layers[i] = tel_create(d_model, num_heads, d_ff, max_seq_len, dropout_p,
                                  seed + (unsigned long long)i * 0x9E3779B97F4A7C15ULL);
    int dims[2];
    dims[0] = max_seq_len; dims[1] = d_model;
    e->buf_a = tensor_new(2, dims);
    e->buf_b = tensor_new(2, dims);
    return e;
}

void te_free(TransformerEncoder *e)
{
    if (!e) return;
    for (int i = 0; i < e->num_layers; i++) tel_free(e->layers[i]);
    xfree(e->layers);
    tensor_free(e->buf_a);
    tensor_free(e->buf_b);
    xfree(e);
}

void te_zero_grad(TransformerEncoder *e)
{
    for (int i = 0; i < e->num_layers; i++) tel_zero_grad(e->layers[i]);
}

void te_set_train(TransformerEncoder *e, int train)
{
    for (int i = 0; i < e->num_layers; i++) tel_set_train(e->layers[i], train);
}

/* 把 (max_seq_len, d_model) 的乒乓缓冲按当前 T 取视图 */
static Tensor te_view(const Tensor *buf, int T, int d)
{
    return make_view(buf->data, T, d);
}

void te_forward(TransformerEncoder *e, const Tensor *x, Tensor *out)
{
    const int d = e->d_model;
    const int T = seq_check(x, d, e->max_seq_len, "te_forward");
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(out, 2, dims, "te_forward(out)");
    e->seq_len = T;

    if (e->num_layers == 1) {          /* 单层：直接写 out */
        tel_forward(e->layers[0], x, out);
        return;
    }
    /* 乒乓：注意 cur 必须指向「(T,d) 视图」而不是 (max_seq_len,d) 的整块缓冲，
     * 否则 T < max_seq_len 时下一层会按整块缓冲的形状做检查而报错。
     * 视图对象放在 views[] 里保证生命周期跨越整轮循环。 */
    const Tensor *cur = x;
    Tensor views[2];
    for (int i = 0; i < e->num_layers - 1; i++) {
        Tensor *dst = (i % 2 == 0) ? e->buf_a : e->buf_b;
        Tensor *v = &views[i % 2];
        *v = te_view(dst, T, d);
        tel_forward(e->layers[i], cur, v);
        cur = v;
    }
    tel_forward(e->layers[e->num_layers - 1], cur, out);
}

void te_backward(TransformerEncoder *e, const Tensor *dout, Tensor *dx)
{
    const int d = e->d_model;
    const int T = e->seq_len;
    if (T <= 0) {
        fprintf(stderr, "te_backward: 尚未调用过 forward (seq_len=0)\n");
        exit(1);
    }
    int dims[2];
    dims[0] = T; dims[1] = d;
    tensor_check_shape(dout, 2, dims, "te_backward(dout)");
    tensor_check_shape(dx, 2, dims, "te_backward(dx)");

    /* 各层的输入梯度形状 = 该层输入的形状：
     *   第 i 层（i>=1）的输入是第 i-1 层的输出，也就是前向时写进乒乓缓冲的那块。
     *   第 i 层反传写进 buf[(i-1)%2]（与前向 buf[i%2] 的对应关系一致），不会与输入梯度别名。 */
    Tensor views[2];
    const Tensor *dcur = dout;
    for (int i = e->num_layers - 1; i >= 1; i--) {
        Tensor *dst = ((i - 1) % 2 == 0) ? e->buf_a : e->buf_b;
        Tensor *v = &views[(i - 1) % 2];
        *v = te_view(dst, T, d);
        tel_backward(e->layers[i], dcur, v);
        dcur = v;
    }
    tel_backward(e->layers[0], dcur, dx);
}

void te_sgd_step(TransformerEncoder *e, double lr)
{
    for (int i = 0; i < e->num_layers; i++) tel_sgd_step(e->layers[i], lr);
}

int te_param_count(const TransformerEncoder *e)
{
    int n = 0;
    for (int i = 0; i < e->num_layers; i++) n += tel_param_count(e->layers[i]);
    return n;
}

/* ==========================================================================
 *  PositionalEncoding
 * ========================================================================== */

PositionalEncoding *pe_create(int d_model, int max_len)
{
    if (d_model <= 0 || max_len <= 0) {
        fprintf(stderr, "pe_create: params非法 (d_model=%d, max_len=%d)\n", d_model, max_len);
        exit(1);
    }
    PositionalEncoding *p = xmalloc(sizeof(*p));
    memset(p, 0, sizeof(*p));
    p->d_model = d_model;
    p->max_len = max_len;
    p->pe = xmalloc((size_t)max_len * d_model * sizeof(double));
    memset(p->pe, 0, (size_t)max_len * d_model * sizeof(double));

    /* 照抄 Python：
     *   for pos in range(max_len):
     *       for i in range(0, d_model, 2):
     *           div = exp(i * -log(10000.0) / d_model)
     *           pe[pos][i] = sin(pos * div)
     *           if i + 1 < d_model: pe[pos][i+1] = cos(pos * div)
     * 注意 div 是在内层循环里按 i 重算的（值只依赖 i），d_model 为奇数时最后一列保持 0 */
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            double div = exp((double)i * -log(10000.0) / (double)d_model);
            p->pe[(size_t)pos * d_model + i] = sin((double)pos * div);
            if (i + 1 < d_model)
                p->pe[(size_t)pos * d_model + i + 1] = cos((double)pos * div);
        }
    }
    return p;
}

void pe_free(PositionalEncoding *p)
{
    if (!p) return;
    xfree(p->pe);
    xfree(p);
}

void pe_forward(PositionalEncoding *p, const Tensor *x, Tensor *out)
{
    /* Python 版是 seq_len = min(len(x_seq), len(self.pe))，超长会静默截断；
     * 按本项目约定（形状不匹配必须报错），这里改成显式报错。 */
    const int T = seq_check(x, p->d_model, p->max_len, "pe_forward");
    int dims[2];
    dims[0] = T; dims[1] = p->d_model;
    tensor_check_shape(out, 2, dims, "pe_forward(out)");
    for (int t = 0; t < T; t++)
        for (int j = 0; j < p->d_model; j++)
            out->data[(size_t)t * p->d_model + j] =
                x->data[(size_t)t * p->d_model + j] + p->pe[(size_t)t * p->d_model + j];
}

void pe_backward(const Tensor *dout, Tensor *dx)
{
    tensor_check_shape(dx, dout->ndim, dout->shape, "pe_backward(dx)");
    if (dx->data != dout->data)   /* 允许 dx == dout（原地直通） */
        memcpy(dx->data, dout->data, (size_t)dout->size * sizeof(double));
}

int pe_param_count(const PositionalEncoding *p)
{
    (void)p;
    return 0;
}

/* ==========================================================================
 *  Embedding
 * ========================================================================== */

Embedding *emb_create(int vocab_size, int embed_dim, int max_len)
{
    if (vocab_size <= 0 || embed_dim <= 0 || max_len <= 0) {
        fprintf(stderr, "emb_create: params非法 (vocab=%d, embed_dim=%d, max_len=%d)\n",
                vocab_size, embed_dim, max_len);
        exit(1);
    }
    Embedding *e = xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->vocab_size = vocab_size;
    e->embed_dim = embed_dim;
    e->max_len = max_len;
    const size_t n = (size_t)vocab_size * embed_dim;
    e->weight = xmalloc(n * sizeof(double));
    e->grad_weight = xmalloc(n * sizeof(double));
    e->indices = xmalloc((size_t)max_len * sizeof(int));
    /* 与 Python 一致：N(0, sqrt(2/vocab_size)) */
    unsigned long long rng = 0xA4093822299F31D0ULL;
    double std = sqrt(2.0 / (double)vocab_size);
    for (size_t i = 0; i < n; i++) {
        double u1 = rng_uniform(&rng), u2 = rng_uniform(&rng);
        if (u1 < 1e-300) u1 = 1e-300;
        e->weight[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
    }
    emb_zero_grad(e);
    return e;
}

void emb_free(Embedding *e)
{
    if (!e) return;
    xfree(e->weight);
    xfree(e->grad_weight);
    xfree(e->indices);
    xfree(e);
}

void emb_zero_grad(Embedding *e)
{
    memset(e->grad_weight, 0, (size_t)e->vocab_size * e->embed_dim * sizeof(double));
}

void emb_set_weight(Embedding *e, const double *W)
{
    if (!W) {
        fprintf(stderr, "emb_set_weight: W 为 NULL\n");
        exit(1);
    }
    memcpy(e->weight, W, (size_t)e->vocab_size * e->embed_dim * sizeof(double));
}

void emb_forward(Embedding *e, const int *indices, int n, Tensor *out)
{
    const int E = e->embed_dim;
    if (n < 0 || n > e->max_len) {
        fprintf(stderr, "emb_forward: 序列长度 %d 超出 create 时的 max_len=%d\n", n, e->max_len);
        exit(1);
    }
    int dims[2];
    dims[0] = n; dims[1] = E;
    tensor_check_shape(out, 2, dims, "emb_forward(out)");
    for (int t = 0; t < n; t++) {
        const int idx = indices[t];
        /* Python 允许负索引（会从尾部绕回）；C 侧按越界报错，不做这种隐式回绕 */
        if (idx < 0 || idx >= e->vocab_size) {
            fprintf(stderr, "emb_forward: 索引 %d out of range [0,%d)\n", idx, e->vocab_size);
            exit(1);
        }
        e->indices[t] = idx;       /* 缓存（indices 可能就指向 e->indices，逐元素写是安全的） */
        memcpy(out->data + (size_t)t * E, e->weight + (size_t)idx * E,
               (size_t)E * sizeof(double));
    }
    e->seq_len = n;
}

void emb_backward(Embedding *e, const Tensor *dout, Tensor *dx)
{
    const int E = e->embed_dim;
    const int T = e->seq_len;
    if (T <= 0) {
        fprintf(stderr, "emb_backward: 尚未调用过 forward (seq_len=0)\n");
        exit(1);
    }
    int dims[2];
    dims[0] = T; dims[1] = E;
    tensor_check_shape(dout, 2, dims, "emb_backward(dout)");
    if (dx) {
        /* 输入是索引，没有实数梯度：Python 版返回 None，C 版统一写成 0，
         * 便于 Sequential / ResidualBlock 用同一种契约处理（见 containers.h 的说明） */
        int dshape[1];
        dshape[0] = T;
        tensor_check_shape(dx, 1, dshape, "emb_backward(dx)");
        memset(dx->data, 0, (size_t)T * sizeof(double));
    }
    /* 照抄 Python：for t, idx in enumerate(self.indices): grad_weight[idx][j] += dout[t][j]
     * 同一个 idx 出现多次时按 t 升序累加，顺序必须保持 */
    for (int t = 0; t < T; t++) {
        double *g = e->grad_weight + (size_t)e->indices[t] * E;
        const double *d = dout->data + (size_t)t * E;
        for (int j = 0; j < E; j++) g[j] += d[j];
    }
}

int emb_param_count(const Embedding *e)
{
    return e->vocab_size * e->embed_dim;
}

/* ==========================================================================
 *  Module 包装（tensor.h 的 vtable）
 *
 *  所有权约定：包装器接管底层对象，module_free() 会连同底层一起释放。
 *  也就是说同一个底层对象不要包装两次，也不要既调 module_free 又调 xxx_free。
 * ========================================================================== */

static Module *wrap_module(const ModuleVT *vt, void *self)
{
    Module *m = xmalloc(sizeof(*m));
    m->vt = vt;
    m->self = self;
    return m;
}

/* ---------- MultiHeadAttention ---------- */
static void mha_mod_forward(Module *m, const Tensor *x, Tensor *out)
{
    mha_forward(m->self, x, out);
}
static void mha_mod_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    mha_backward(m->self, dout, dx);
}
static void mha_mod_zero_grad(Module *m) { mha_zero_grad(m->self); }
static void mha_mod_free(Module *m) { mha_free(m->self); m->self = NULL; }
static int  mha_mod_param_count(Module *m) { return mha_param_count(m->self); }
static void mha_mod_to_string(Module *m, char *buf, size_t n)
{
    MultiHeadAttention *a = m->self;
    snprintf(buf, n, "MultiHeadAttention(d_model=%d, heads=%d, d_k=%d, max_seq=%d)",
             a->d_model, a->num_heads, a->d_k, a->max_seq_len);
}
static const ModuleVT MHA_MODULE_VT = {
    "MultiHeadAttention", mha_mod_forward, mha_mod_backward, mha_mod_zero_grad,
    mha_mod_free, mha_mod_param_count, mha_mod_to_string
};
Module *mha_module(MultiHeadAttention *a) { return wrap_module(&MHA_MODULE_VT, a); }

/* ---------- FeedForward ---------- */
static void ff_mod_forward(Module *m, const Tensor *x, Tensor *out) { ff_forward(m->self, x, out); }
static void ff_mod_backward(Module *m, const Tensor *dout, Tensor *dx) { ff_backward(m->self, dout, dx); }
static void ff_mod_zero_grad(Module *m) { ff_zero_grad(m->self); }
static void ff_mod_free(Module *m) { ff_free(m->self); m->self = NULL; }
static int  ff_mod_param_count(Module *m) { return ff_param_count(m->self); }
static void ff_mod_to_string(Module *m, char *buf, size_t n)
{
    FeedForward *f = m->self;
    snprintf(buf, n, "FeedForward(d_model=%d, d_ff=%d, max_seq=%d)",
             f->d_model, f->d_ff, f->max_seq_len);
}
static const ModuleVT FF_MODULE_VT = {
    "FeedForward", ff_mod_forward, ff_mod_backward, ff_mod_zero_grad,
    ff_mod_free, ff_mod_param_count, ff_mod_to_string
};
Module *ff_module(FeedForward *f) { return wrap_module(&FF_MODULE_VT, f); }

/* ---------- TransformerEncoderLayer ---------- */
static void tel_mod_forward(Module *m, const Tensor *x, Tensor *out) { tel_forward(m->self, x, out); }
static void tel_mod_backward(Module *m, const Tensor *dout, Tensor *dx) { tel_backward(m->self, dout, dx); }
static void tel_mod_zero_grad(Module *m) { tel_zero_grad(m->self); }
static void tel_mod_free(Module *m) { tel_free(m->self); m->self = NULL; }
static int  tel_mod_param_count(Module *m) { return tel_param_count(m->self); }
static void tel_mod_to_string(Module *m, char *buf, size_t n)
{
    TransformerEncoderLayer *l = m->self;
    snprintf(buf, n, "TransformerEncoderLayer(d_model=%d, heads=%d, d_ff=%d, max_seq=%d)",
             l->d_model, l->self_attn->num_heads, l->ffn->d_ff, l->max_seq_len);
}
static const ModuleVT TEL_MODULE_VT = {
    "TransformerEncoderLayer", tel_mod_forward, tel_mod_backward, tel_mod_zero_grad,
    tel_mod_free, tel_mod_param_count, tel_mod_to_string
};
Module *tel_module(TransformerEncoderLayer *l) { return wrap_module(&TEL_MODULE_VT, l); }

/* ---------- TransformerEncoder ---------- */
static void te_mod_forward(Module *m, const Tensor *x, Tensor *out) { te_forward(m->self, x, out); }
static void te_mod_backward(Module *m, const Tensor *dout, Tensor *dx) { te_backward(m->self, dout, dx); }
static void te_mod_zero_grad(Module *m) { te_zero_grad(m->self); }
static void te_mod_free(Module *m) { te_free(m->self); m->self = NULL; }
static int  te_mod_param_count(Module *m) { return te_param_count(m->self); }
static void te_mod_to_string(Module *m, char *buf, size_t n)
{
    TransformerEncoder *e = m->self;
    snprintf(buf, n, "TransformerEncoder(layers=%d, d_model=%d, max_seq=%d)",
             e->num_layers, e->d_model, e->max_seq_len);
}
static const ModuleVT TE_MODULE_VT = {
    "TransformerEncoder", te_mod_forward, te_mod_backward, te_mod_zero_grad,
    te_mod_free, te_mod_param_count, te_mod_to_string
};
Module *te_module(TransformerEncoder *e) { return wrap_module(&TE_MODULE_VT, e); }

/* ---------- PositionalEncoding ---------- */
static void pe_mod_forward(Module *m, const Tensor *x, Tensor *out) { pe_forward(m->self, x, out); }
static void pe_mod_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    (void)m;
    pe_backward(dout, dx);
}
static void pe_mod_zero_grad(Module *m) { (void)m; }
static void pe_mod_free(Module *m) { pe_free(m->self); m->self = NULL; }
static int  pe_mod_param_count(Module *m) { return pe_param_count(m->self); }
static void pe_mod_to_string(Module *m, char *buf, size_t n)
{
    PositionalEncoding *p = m->self;
    snprintf(buf, n, "PositionalEncoding(d_model=%d, max_len=%d)", p->d_model, p->max_len);
}
static const ModuleVT PE_MODULE_VT = {
    "PositionalEncoding", pe_mod_forward, pe_mod_backward, pe_mod_zero_grad,
    pe_mod_free, pe_mod_param_count, pe_mod_to_string
};
Module *pe_module(PositionalEncoding *p) { return wrap_module(&PE_MODULE_VT, p); }

/* ---------- Embedding ---------- */
static void emb_mod_forward(Module *m, const Tensor *x, Tensor *out)
{
    Embedding *e = m->self;
    if (x->ndim != 1) {
        fprintf(stderr, "emb_module.forward: 输入必须是 ndim=1 的索引张量, 实得 ndim=%d\n", x->ndim);
        exit(1);
    }
    const int n = x->size;
    if (n > e->max_len) {
        fprintf(stderr, "emb_module.forward: 序列长度 %d 超出 max_len=%d\n", n, e->max_len);
        exit(1);
    }
    for (int t = 0; t < n; t++) {
        const double v = x->data[t];
        const int k = (int)v;
        if ((double)k != v) {
            fprintf(stderr, "emb_module.forward: 索引 %g 不是整数\n", v);
            exit(1);
        }
        e->indices[t] = k;      /* 越界检查交给 emb_forward */
    }
    emb_forward(e, e->indices, n, out);
}
static void emb_mod_backward(Module *m, const Tensor *dout, Tensor *dx)
{
    emb_backward(m->self, dout, dx);
}
static void emb_mod_zero_grad(Module *m) { emb_zero_grad(m->self); }
static void emb_mod_free(Module *m) { emb_free(m->self); m->self = NULL; }
static int  emb_mod_param_count(Module *m) { return emb_param_count(m->self); }
static void emb_mod_to_string(Module *m, char *buf, size_t n)
{
    Embedding *e = m->self;
    snprintf(buf, n, "Embedding(vocab=%d, embed_dim=%d, max_len=%d)",
             e->vocab_size, e->embed_dim, e->max_len);
}
static const ModuleVT EMB_MODULE_VT = {
    "Embedding", emb_mod_forward, emb_mod_backward, emb_mod_zero_grad,
    emb_mod_free, emb_mod_param_count, emb_mod_to_string
};
Module *emb_module(Embedding *e) { return wrap_module(&EMB_MODULE_VT, e); }
