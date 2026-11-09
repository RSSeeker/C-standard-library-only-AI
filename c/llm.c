/*
 * llm.c —— 因果语言模型 CausalLM 的实现（对应 AI.py 的 CausalLM）
 *
 * 运算顺序刻意对齐 Python：
 *   - 词表头 logits = dot_neumaier(h[t], W_out[v]) + b_out[v]
 *     （Python 是 _matmul(h, _transpose(W_out))，每个元素都是 sum(map(mul, ...))）
 *   - 交叉熵的 loss 用除法 /= n，而 d_logits 用乘法 * (1/n)，两者在 Python 里写法不同
 *   - 词表头反向的梯度累加是朴素 +=（Python 里同样是普通循环，不是 Neumaier）
 *   - Adam 与 _adam_apply 同序：先更新 m/v，再 p -= lr*(m/c1)/(sqrt(v/c2)+eps)
 * 注意：Python 的 len(h) 取的是 forward 时的序列长度（T-1），C 侧用 m->seq_len 记录。
 */
#include "llm.h"

#include "numeric.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------ 小工具 ------------------------------ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "llm: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) {
        fprintf(stderr, "llm: out of memory (%zu x %zu bytes)\n", n, sz);
        exit(1);
    }
    return p;
}

/* splitmix64：与 attention.c 里的初始化同源（C 侧自己的确定性随机流） */
static double rng_uniform(unsigned long long *state)
{
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

/* Box-Muller，与 Python 的 random.gauss 分布相同（数值不同源，故不能逐位对齐） */
static double rng_gauss(unsigned long long *state, double std)
{
    double u1 = rng_uniform(state), u2 = rng_uniform(state);
    if (u1 < 1e-300) u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2) * std;
}

/* 整数次幂：long double 反复平方，最后一次舍入到 double。
 *
 * 为什么要自己写：Adam 的偏差修正要算 b1**t / b2**t，Python 侧是 CPython 的 pow()
 * （Windows 上来自 ucrtbase），而 MinGW 默认链接 msvcrt.dll，两边的 pow 从 t=4 起
 * 就会差 1 ULP。实测（0.9**t，t=1..40）：msvcrt 的 pow 有 37 个 t 与 Python 不同，
 * 这里的实现前 31 个 t 与 Python 逐位相同；t 更大时仍可能差 1 ULP（long double
 * 反复平方也不是严格正确舍入），影响在 1e-16 量级，已在 README「已知差异」写明。
 */
static double pow_int_cr(double b, int t)
{
    long double r = 1.0L, base = (long double)b;
    int e = t;
    while (e > 0) {
        if (e & 1) r *= base;
        base *= base;
        e >>= 1;
    }
    return (double)r;
}

/* 把预分配缓冲当成 (T, C) 的视图：只改 shape/size，不动 data（热路径零 malloc） */
static void set_shape2(Tensor *t, int rows, int cols)
{
    t->ndim = 2;
    t->shape[0] = rows;
    t->shape[1] = cols;
    for (int i = 2; i < TENSOR_MAX_DIM; i++) t->shape[i] = 1;
    t->size = rows * cols;
}

/* 一维 softmax（就地），与 Python 的 _softmax_row 同序 */
static void softmax_row_inplace(double *v, int n)
{
    double mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    /* 整行都是 -inf（例如 query 的可见 key 全被掩码掉）时，exp(-inf - -inf)
     * 是 NaN；与 Python 的 _softmax_row 一致，这种情况下返回全 0。 */
    if (mx == -INFINITY) {
        for (int i = 0; i < n; i++) v[i] = 0.0;
        return;
    }
    for (int i = 0; i < n; i++) v[i] = exp(v[i] - mx);
    double s = sum_neumaier(v, n);
    for (int i = 0; i < n; i++) v[i] = v[i] / s;
}

/* ------------------------------ 参数块表 ------------------------------ */

/* 往块表里追加一项；name 以 malloc 复制（llm_free 释放） */
static void add_block(CausalLM *m, int *cap, int *n, const char *fmt, int li,
                      double *param, double *grad, int count)
{
    if (*n == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        m->blocks = realloc(m->blocks, (size_t)*cap * sizeof(*m->blocks));
        if (!m->blocks) {
            fprintf(stderr, "llm: out of memory (blocks)\n");
            exit(1);
        }
    }
    char buf[64];
    if (li >= 0) snprintf(buf, sizeof(buf), fmt, li);
    else         snprintf(buf, sizeof(buf), "%s", fmt);
    char *copy = xmalloc(strlen(buf) + 1);
    memcpy(copy, buf, strlen(buf) + 1);
    m->blocks[*n].name = copy;
    m->blocks[*n].param = param;
    m->blocks[*n].grad = grad;
    m->blocks[*n].n = count;
    (*n)++;
}

static void build_blocks(CausalLM *m)
{
    int cap = 0, n = 0;
    const int d = m->d_model;
    const int dd = d * d;

    add_block(m, &cap, &n, "emb.weight", -1, m->emb->weight, m->emb->grad_weight,
              m->vocab_size * d);
    for (int li = 0; li < m->enc->num_layers; li++) {
        TransformerEncoderLayer *l = m->enc->layers[li];
        MultiHeadAttention *a = l->self_attn;
        /* W_k / W_v 是 (kv_dim, d_model)、b_k / b_v 是 (kv_dim)；普通多头时 kv_dim == d，
         * 块尺寸与旧版完全相同（llm_create 走的就是这条默认路径）。 */
        const int kvd = a->kv_dim;
        const int kvdd = kvd * d;
        add_block(m, &cap, &n, "L%d.attn.W_q", li, a->W_q, a->grad_W_q, dd);
        add_block(m, &cap, &n, "L%d.attn.W_k", li, a->W_k, a->grad_W_k, kvdd);
        add_block(m, &cap, &n, "L%d.attn.W_v", li, a->W_v, a->grad_W_v, kvdd);
        add_block(m, &cap, &n, "L%d.attn.W_o", li, a->W_o, a->grad_W_o, dd);
        add_block(m, &cap, &n, "L%d.attn.b_q", li, a->b_q, a->grad_b_q, d);
        add_block(m, &cap, &n, "L%d.attn.b_k", li, a->b_k, a->grad_b_k, kvd);
        add_block(m, &cap, &n, "L%d.attn.b_v", li, a->b_v, a->grad_b_v, kvd);
        add_block(m, &cap, &n, "L%d.attn.b_o", li, a->b_o, a->grad_b_o, d);

        FeedForward *f = l->ffn;
        add_block(m, &cap, &n, "L%d.ffn.W1", li, f->W1, f->grad_W1, m->d_ff * d);
        add_block(m, &cap, &n, "L%d.ffn.b1", li, f->b1, f->grad_b1, m->d_ff);
        add_block(m, &cap, &n, "L%d.ffn.W2", li, f->W2, f->grad_W2, d * m->d_ff);
        add_block(m, &cap, &n, "L%d.ffn.b2", li, f->b2, f->grad_b2, d);

        add_block(m, &cap, &n, "L%d.ln1.gamma", li, l->ln1.gamma, l->ln1.ggamma, d);
        add_block(m, &cap, &n, "L%d.ln1.beta", li, l->ln1.beta, l->ln1.gbeta, d);
        add_block(m, &cap, &n, "L%d.ln2.gamma", li, l->ln2.gamma, l->ln2.ggamma, d);
        add_block(m, &cap, &n, "L%d.ln2.beta", li, l->ln2.beta, l->ln2.gbeta, d);
    }
    add_block(m, &cap, &n, "out.W", -1, m->W_out, m->grad_W_out,
              m->vocab_size * d);
    add_block(m, &cap, &n, "out.b", -1, m->b_out, m->grad_b_out, m->vocab_size);

    m->n_blocks = n;
    m->adam_m = xcalloc((size_t)n, sizeof(double *));
    m->adam_v = xcalloc((size_t)n, sizeof(double *));
}

/* ------------------------------ 生命周期 ------------------------------ */

CausalLM *llm_create(int vocab_size, int d_model, int num_heads, int d_ff,
                     int num_layers, int max_len, double dropout,
                     unsigned long long seed)
{
    if (vocab_size <= 0 || d_model <= 0 || num_heads <= 0 || d_ff <= 0 ||
        num_layers <= 0 || max_len <= 0) {
        fprintf(stderr, "llm_create: 参数必须是正整数 (vocab=%d, d_model=%d, heads=%d,"
                        " d_ff=%d, layers=%d, max_len=%d)\n",
                vocab_size, d_model, num_heads, d_ff, num_layers, max_len);
        exit(1);
    }
    if (d_model % num_heads != 0) {
        fprintf(stderr, "llm_create: d_model(%d) 必须能被 num_heads(%d) 整除\n",
                d_model, num_heads);
        exit(1);
    }
    if (!(dropout >= 0.0 && dropout < 1.0)) {
        fprintf(stderr, "llm_create: dropout 必须在 [0,1) 内，收到 %g\n", dropout);
        exit(1);
    }

    CausalLM *m = xmalloc(sizeof(*m));
    memset(m, 0, sizeof(*m));
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;
    m->dropout = dropout;
    m->use_abs_pos = 1;   /* 默认走 PositionalEncoding；llm_set_rope(..., 1, ...) 会关掉它 */

    m->emb = emb_create(vocab_size, d_model, max_len);
    m->pe = pe_create(d_model, max_len);
    m->enc = te_create(d_model, num_heads, d_ff, num_layers, max_len, dropout, seed);
    te_set_causal(m->enc, 1);          /* ← 与 Python 的 TransformerEncoder(..., causal=True) 对应 */

    /* 词表头：与 Python 一样用 N(0, sqrt(2/d_model)) 初始化，b_out 全 0 */
    const size_t n_head = (size_t)vocab_size * d_model;
    m->W_out = xmalloc(n_head * sizeof(double));
    m->grad_W_out = xcalloc(n_head, sizeof(double));
    m->b_out = xcalloc((size_t)vocab_size, sizeof(double));
    m->grad_b_out = xcalloc((size_t)vocab_size, sizeof(double));
    double std = sqrt(2.0 / (double)d_model);
    unsigned long long rng = seed ^ 0x5DEECE66DULL;
    for (size_t i = 0; i < n_head; i++) m->W_out[i] = rng_gauss(&rng, std);

    /* 缓冲：全部按 max_len 预分配，前向/反向只改 shape */
    int dims2[2] = { max_len, d_model };
    int dims_logits[2] = { max_len, vocab_size };
    m->emb_buf = tensor_new(2, dims2);
    m->x_buf = tensor_new(2, dims2);
    m->h_buf = tensor_new(2, dims2);
    m->dh_buf = tensor_new(2, dims2);
    m->dx_buf = tensor_new(2, dims2);
    m->logits_buf = tensor_new(2, dims_logits);
    m->dlogits = xcalloc((size_t)max_len * vocab_size, sizeof(double));
    m->probs = xcalloc((size_t)vocab_size, sizeof(double));
    m->tmp_row = xcalloc((size_t)(vocab_size > d_model ? vocab_size : d_model),
                         sizeof(double));
    m->keep = xcalloc((size_t)vocab_size, sizeof(unsigned char));
    m->penalty_counts = xcalloc((size_t)vocab_size, sizeof(int));

    build_blocks(m);
    return m;
}

void llm_free(CausalLM *m)
{
    if (!m) return;
    if (m->blocks) {
        for (int i = 0; i < m->n_blocks; i++) free((void *)m->blocks[i].name);
        free(m->blocks);
    }
    if (m->adam_m) {
        for (int i = 0; i < m->n_blocks; i++) free(m->adam_m[i]);
        free(m->adam_m);
    }
    if (m->adam_v) {
        for (int i = 0; i < m->n_blocks; i++) free(m->adam_v[i]);
        free(m->adam_v);
    }
    emb_free(m->emb);
    pe_free(m->pe);
    te_free(m->enc);
    free(m->W_out);
    free(m->grad_W_out);
    free(m->b_out);
    free(m->grad_b_out);
    tensor_free(m->emb_buf);
    tensor_free(m->x_buf);
    tensor_free(m->h_buf);
    tensor_free(m->dh_buf);
    tensor_free(m->dx_buf);
    tensor_free(m->logits_buf);
    free(m->dlogits);
    free(m->probs);
    free(m->tmp_row);
    free(m->keep);
    free(m->penalty_counts);
    free(m);
}

void llm_set_train(CausalLM *m, int train)
{
    te_set_train(m->enc, train ? 1 : 0);
}

void llm_set_rope(CausalLM *m, int enable, double base)
{
    if (!m) {
        fprintf(stderr, "llm_set_rope: m 为 NULL\n");
        exit(1);
    }
    m->use_abs_pos = enable ? 0 : 1;
    for (int i = 0; i < m->enc->num_layers; i++) {
        mha_set_rope(m->enc->layers[i]->self_attn, enable, base);
    }
}

void llm_zero_grad(CausalLM *m)
{
    emb_zero_grad(m->emb);
    te_zero_grad(m->enc);
    memset(m->grad_W_out, 0, (size_t)m->vocab_size * m->d_model * sizeof(double));
    memset(m->grad_b_out, 0, (size_t)m->vocab_size * sizeof(double));
}

/* ------------------------------ 前向 ------------------------------ */

/* 校验 ids：长度 1..max_len、索引在 [0, vocab_size) */
static void check_ids(const CausalLM *m, const int *ids, int T, const char *who)
{
    if (!ids) {
        fprintf(stderr, "%s: ids 为 NULL\n", who);
        exit(1);
    }
    if (T < 1) {
        fprintf(stderr, "%s: 序列长度必须 >= 1（收到 %d）\n", who, T);
        exit(1);
    }
    if (T > m->max_len) {
        fprintf(stderr, "%s: 序列长度 %d 超过 max_len=%d\n", who, T, m->max_len);
        exit(1);
    }
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) {
            fprintf(stderr, "%s: 第 %d 个索引 %d 越界（词表大小 %d）\n",
                    who, t, ids[t], m->vocab_size);
            exit(1);
        }
    }
}

/* 校验 logits 全部有限：权重里混进 NaN/Inf 时，loss 会被 clamp 成
 * -log(1e-12)=27.63，训练照跑但梯度全是 NaN。宁可在这里打断。 */
static void check_finite_logits(const double *logits, int T, int V,
                                const char *who)
{
    for (int t = 0; t < T; t++) {
        const double *row = logits + (size_t)t * V;
        for (int v = 0; v < V; v++) {
            if (!isfinite(row[v])) {
                fprintf(stderr, "%s: 第 %d 行 logits 含 NaN/Inf（参数可能已损坏）\n",
                        who, t);
                exit(1);
            }
        }
    }
}

void llm_forward(CausalLM *m, const int *ids, int T, Tensor *logits)
{
    check_ids(m, ids, T, "llm_forward");
    int dims[2] = { T, m->vocab_size };
    tensor_check_shape(logits, 2, dims, "llm_forward(logits)");
    m->seq_len = T;

    const int d = m->d_model, V = m->vocab_size;

    set_shape2(m->emb_buf, T, d);
    emb_forward(m->emb, ids, T, m->emb_buf);

    set_shape2(m->x_buf, T, d);
    if (m->use_abs_pos) {
        pe_forward(m->pe, m->emb_buf, m->x_buf);
    } else {
        /* RoPE 模式：位置信息由 q/k 的旋转携带，这里只做原样搬运
         * （pe_forward 在 pe 全零时也是逐元素加 0.0，但 memcpy 更直白） */
        memcpy(m->x_buf->data, m->emb_buf->data, (size_t)T * d * sizeof(double));
    }

    set_shape2(m->h_buf, T, d);
    te_forward(m->enc, m->x_buf, m->h_buf);

    /* logits[t][v] = h[t]·W_out[v] + b_out[v]（Neumaier，与 Python 的 sum(map(mul)) 同序） */
    for (int t = 0; t < T; t++) {
        const double *h = m->h_buf->data + (size_t)t * d;
        for (int v = 0; v < V; v++) {
            logits->data[(size_t)t * V + v] =
                dot_neumaier(h, m->W_out + (size_t)v * d, d) + m->b_out[v];
        }
    }
}

/* ------------------------------ 训练一步 ------------------------------ */

double llm_loss_and_backward(CausalLM *m, const int *ids, int T)
{
    check_ids(m, ids, T, "llm_loss_and_backward");
    if (T < 2) {
        fprintf(stderr, "llm_loss_and_backward: 序列至少要有 2 个 token（输入 + 目标）\n");
        exit(1);
    }
    const int d = m->d_model, V = m->vocab_size;
    const int n = T - 1;

    /* forward 只看前 T-1 个 token（最后一个没有「下一个」）。
     * 缓冲按 max_len 预分配，这里用栈上视图把 shape 改成 (n, V)。 */
    Tensor logits_view = *m->logits_buf;
    set_shape2(&logits_view, n, V);
    llm_forward(m, ids, n, &logits_view);

    check_finite_logits(m->logits_buf->data, n, V, "llm_loss_and_backward");

    double loss = 0.0;
    for (int t = 0; t < n; t++) {
        double *row = m->dlogits + (size_t)t * V;
        memcpy(m->probs, m->logits_buf->data + (size_t)t * V, (size_t)V * sizeof(double));
        softmax_row_inplace(m->probs, V);
        int target = ids[t + 1];
        double p = m->probs[target];
        loss += -log(p > 1e-12 ? p : 1e-12);
        memcpy(row, m->probs, (size_t)V * sizeof(double));
        row[target] -= 1.0;
    }
    /* Python: loss /= n；d_logits 用乘法 inv = 1/n（两者写法不同，这里照抄） */
    loss = loss / n;
    const double inv = 1.0 / n;
    for (size_t i = 0; i < (size_t)n * V; i++) m->dlogits[i] *= inv;

    /* ---- 词表头反向：grad_W_out[v][j] += Σ_t d_logits[t][v] * h[t][j]（朴素 +=） ---- */
    const double *h = m->h_buf->data;
    for (int v = 0; v < V; v++) {
        double *g_row = m->grad_W_out + (size_t)v * d;
        double gb = 0.0;
        for (int t = 0; t < n; t++) {
            double dl = m->dlogits[(size_t)t * V + v];
            gb += dl;
            const double *ht = h + (size_t)t * d;
            for (int j = 0; j < d; j++) g_row[j] += dl * ht[j];
        }
        m->grad_b_out[v] += gb;
    }

    /* ---- d_h[t][j] = Σ_v d_logits[t][v] * W_out[v][j]（朴素 +=，与 Python 同序） ---- */
    double *dh = m->dh_buf->data;
    for (int t = 0; t < n; t++) {
        const double *dl = m->dlogits + (size_t)t * V;
        double *row = dh + (size_t)t * d;
        for (int j = 0; j < d; j++) {
            double s = 0.0;
            for (int v = 0; v < V; v++) s += dl[v] * m->W_out[(size_t)v * d + j];
            row[j] = s;
        }
    }

    /* 编码器反向（kernel 内部按层倒序），再把梯度交给词向量 */
    set_shape2(m->dh_buf, n, d);
    set_shape2(m->dx_buf, n, d);
    te_backward(m->enc, m->dh_buf, m->dx_buf);
    emb_backward(m->emb, m->dx_buf, NULL);
    return loss;
}

void llm_adam_step(CausalLM *m, double lr)
{
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    m->step_count++;
    const int t = m->step_count;
    const double c1 = 1.0 - pow_int_cr(b1, t);
    const double c2 = 1.0 - pow_int_cr(b2, t);

    for (int b = 0; b < m->n_blocks; b++) {
        LLMParamBlock *blk = &m->blocks[b];
        if (!m->adam_m[b]) {
            m->adam_m[b] = xcalloc((size_t)blk->n, sizeof(double));
            m->adam_v[b] = xcalloc((size_t)blk->n, sizeof(double));
        }
        double *p = blk->param, *g = blk->grad, *mm = m->adam_m[b], *vv = m->adam_v[b];
        for (int k = 0; k < blk->n; k++) {
            double gk = g[k];
            double mk = b1 * mm[k] + (1.0 - b1) * gk;
            double vk = b2 * vv[k] + (1.0 - b2) * gk * gk;
            mm[k] = mk;
            vv[k] = vk;
            p[k] -= lr * (mk / c1) / (sqrt(vk / c2) + eps);
        }
    }
}

/* ------------------------------ 评估 / 训练 ------------------------------ */

double llm_evaluate(CausalLM *m, const int *const *seqs, const int *lens, int nseq)
{
    if (nseq < 1) {
        fprintf(stderr, "llm_evaluate: 序列数量必须 >= 1\n");
        exit(1);
    }
    llm_set_train(m, 0);
    double total = 0.0;
    int count = 0;
    const int V = m->vocab_size;
    for (int s = 0; s < nseq; s++) {
        const int *ids = seqs[s];
        int T = lens[s];
        check_ids(m, ids, T, "llm_evaluate");
        if (T < 2) {
            fprintf(stderr, "llm_evaluate: 每条序列至少 2 个 token\n");
            exit(1);
        }
        Tensor logits_view = *m->logits_buf;
        set_shape2(&logits_view, T - 1, V);
        llm_forward(m, ids, T - 1, &logits_view);
        check_finite_logits(m->logits_buf->data, T - 1, V, "llm_evaluate");
        for (int t = 0; t < T - 1; t++) {
            memcpy(m->probs, m->logits_buf->data + (size_t)t * V, (size_t)V * sizeof(double));
            softmax_row_inplace(m->probs, V);
            double p = m->probs[ids[t + 1]];
            total += -log(p > 1e-12 ? p : 1e-12);
            count++;
        }
    }
    return total / count;
}

double llm_perplexity(CausalLM *m, const int *const *seqs, const int *lens, int nseq)
{
    /* 对应 Python 的 perplexity()：loss > 700 直接返回 inf，避免 exp 溢出 */
    const double loss = llm_evaluate(m, seqs, lens, nseq);
    if (loss > 700.0) return INFINITY;
    return exp(loss);
}

double llm_fit(CausalLM *m, const int *const *seqs, const int *lens, int nseq,
               int epochs, double lr, int shuffle, unsigned long long seed,
               double *history)
{
    if (nseq < 1) {
        fprintf(stderr, "llm_fit: 序列数量必须 >= 1\n");
        exit(1);
    }
    if (epochs < 0) {
        fprintf(stderr, "llm_fit: epochs 不能为负（收到 %d）\n", epochs);
        exit(1);
    }
    for (int s = 0; s < nseq; s++) {
        if (lens[s] < 2) {
            fprintf(stderr, "llm_fit: 第 %d 条序列只有 %d 个 token（至少 2 个）\n", s, lens[s]);
            exit(1);
        }
    }
    int *order = xmalloc((size_t)nseq * sizeof(int));
    unsigned long long rng = seed;
    double last = 0.0;
    /* 不重置 step_count：Adam 的偏差修正依赖它。新建模型时本来就是 0（见 llm_create），
     * 而多次调用 fit 应当像 Python 一样接着上一次的步数继续 —— 早先在这里清零，
     * 导致「分多次 fit」与「一次 fit 多个 epoch」结果不同，与 Python 对比立刻暴露。 */
    llm_set_train(m, 1);
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < nseq; i++) order[i] = i;
        if (shuffle) {
            for (int i = nseq - 1; i > 0; i--) {
                int j = (int)(rng_uniform(&rng) * (double)(i + 1));
                if (j > i) j = i;
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
        double total = 0.0;
        for (int k = 0; k < nseq; k++) {
            int i = order[k];
            llm_zero_grad(m);
            total += llm_loss_and_backward(m, seqs[i], lens[i]);
            llm_adam_step(m, lr);
        }
        last = total / (double)nseq;
        if (history) history[ep] = last;
    }
    free(order);
    return last;
}

/* ------------------------------ 生成 ------------------------------ */

int llm_generate(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                 double temperature, int top_k, int *out, unsigned long long seed)
{
    return llm_generate_p(m, prompt, prompt_len, max_new, temperature, top_k, 1.0, out, seed);
}

/* 生成罚分（对应 Python 的 _apply_generation_penalties）：
 *   - 按**完整历史 ids**（不只是当前窗口）统计出现次数；
 *   - repetition_penalty：logit > 0 除以 rp，否则乘以 rp（0 走乘法分支，仍是 0）；
 *   - presence_penalty：每个出现过的 token 减一次；
 *   - frequency_penalty：每个出现过的 token 减 count * frequency_penalty；
 *   - no_repeat_ngram_size >= 2：最近 n-1 个 token 的片段若在历史中出现过，
 *     把后继 token 的 logit 置 -INFINITY。
 * 全部默认时直接返回、不改 row —— 这是旧生成路径逐位不变的关键。 */
void llm_apply_generation_penalties(CausalLM *m, double *row, const int *ids, int len,
                                    double repetition_penalty, double presence_penalty,
                                    double frequency_penalty, int no_repeat_ngram_size)
{
    const int V = m->vocab_size;
    const int has_ngram = no_repeat_ngram_size >= 2;
    if (repetition_penalty == 1.0 && presence_penalty == 0.0 &&
        frequency_penalty == 0.0 && !has_ngram) {
        return;
    }

    int *counts = m->penalty_counts;
    for (int v = 0; v < V; v++) counts[v] = 0;
    for (int i = 0; i < len; i++) {
        const int tok = ids[i];
        if (tok >= 0 && tok < V) counts[tok]++;
    }

    for (int v = 0; v < V; v++) {
        if (counts[v] == 0) continue;
        if (repetition_penalty != 1.0) {
            if (row[v] > 0.0) row[v] /= repetition_penalty;
            else row[v] *= repetition_penalty;
        }
        row[v] -= presence_penalty;
        row[v] -= frequency_penalty * (double)counts[v];
    }

    if (has_ngram) {
        /* Python: prefix = ids[len-(n-1):]；for i in range(len(ids)-(n-1))。 */
        const int k = no_repeat_ngram_size - 1;
        const int npos = len - k;
        if (npos > 0) {
            const int *prefix = ids + len - k;
            for (int i = 0; i < npos; i++) {
                int match = 1;
                for (int j = 0; j < k; j++) {
                    if (ids[i + j] != prefix[j]) { match = 0; break; }
                }
                if (match) {
                    const int nxt = ids[i + k];
                    if (nxt >= 0 && nxt < V) row[nxt] = -INFINITY;
                }
            }
        }
    }
}

/* 把 min_p 截尾应用到一行 logits（就地改写），返回保留的候选数。
 * 与 Python 的 _mask_min_p / _softmax_row 同构：softmax 用普通的顺序求和
 * （不是 Neumaier），先逐元素除以总和，再取 max 作为 peak，
 * 阈值 = min_p * peak，保留 prob >= 阈值的 token。probs 是长度 n 的调用方缓冲。
 * 整行 -inf 时 Python 的 softmax 返回全 0、阈值 0，所有位置都 >= 阈值，
 * 因此不掩码（行本身仍是 -inf），这里返回 n。 */
static int mask_row_min_p(double *row, double *probs, int n, double min_p)
{
    double mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    if (mx == -INFINITY) return n;
    for (int i = 0; i < n; i++) probs[i] = exp(row[i] - mx);
    double s = 0.0;
    for (int i = 0; i < n; i++) s += probs[i];
    double peak = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = probs[i] / s;
        if (probs[i] > peak) peak = probs[i];
    }
    const double thresh = min_p * peak;
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (probs[i] >= thresh) kept++;
        else row[i] = -INFINITY;
    }
    return kept;
}

/* 从一行 logits 采样一个 token：top_k → 温度缩放 → top_p（nucleus）→ min_p →
 * softmax → 逆变换采样。会就地改写 row（调用方传入 m->tmp_row），并消耗一次 rng。
 * 这段逻辑原先内联在 llm_generate_p 里；抽出来是为了让 llm_generate_cached 复用
 * 完全相同的采样顺序（两者逐位一致的另一个前提）。 */
static int sample_row(CausalLM *m, double *row, double temperature, int top_k,
                      double top_p, double min_p, unsigned long long *rng)
{
    const int V = m->vocab_size;

    if (top_k > 0 && top_k < V) {
        /* 只保留最大的 top_k 个，其余置 -INFINITY（等价于 Python 的 sorted + keep） */
        for (int v = 0; v < V; v++) {
            int rank = 0;
            for (int u = 0; u < V; u++) {
                if (row[u] > row[v] || (row[u] == row[v] && u < v)) rank++;
            }
            if (rank >= top_k) row[v] = -INFINITY;
        }
    }
    for (int v = 0; v < V; v++) row[v] = row[v] / temperature;

    if (top_p < 1.0) {
        /* nucleus 采样（Python 的 CausalLM._sample 中 top_p < 1 的分支）：
         * 1) 用同一个温度缩放的 logits 算出原始概率 probs_raw；
         * 2) 按概率降序（并列时小下标在前）累加，取累积刚达到 top_p 的最小集合；
         * 3) 集合外的位置置 -INFINITY，随后照常重新 softmax 再采样。
         * 复用 m->probs 当 probs_raw 缓冲（生成路径不跑反向，不会冲突）。 */
        double *probs_raw = m->probs;
        memcpy(probs_raw, row, (size_t)V * sizeof(double));
        softmax_row_inplace(probs_raw, V);
        for (int v = 0; v < V; v++) m->keep[v] = 0;
        double acc_p = 0.0;
        for (int k = 0; k < V; k++) {
            /* 每轮取「尚未保留、概率最大」的 token；相等时保留下标更小的那个 ——
             * 与 Python 的 sorted(..., reverse=True)（稳定排序）顺序一致。 */
            int best_v = -1;
            double best_p = 0.0;
            for (int v = 0; v < V; v++) {
                if (m->keep[v]) continue;
                if (best_v < 0 || probs_raw[v] > best_p) { best_v = v; best_p = probs_raw[v]; }
            }
            if (best_v < 0) break;   /* 理论上不会发生 */
            m->keep[best_v] = 1;
            acc_p += best_p;
            if (acc_p >= top_p) break;
        }
        for (int v = 0; v < V; v++) {
            if (!m->keep[v]) row[v] = -INFINITY;
        }
    }

    if (min_p > 0.0) {
        /* 对应 Python 的 _mask_min_p（温度缩放后调用），细节见 mask_row_min_p。 */
        (void)mask_row_min_p(row, m->probs, V, min_p);
    }

    softmax_row_inplace(row, V);

    double u = rng_uniform(rng);
    double acc = 0.0;
    int pick = V - 1;
    for (int v = 0; v < V; v++) {
        acc += row[v];
        if (u < acc) { pick = v; break; }
    }
    return pick;
}

int llm_generate_ex(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                    double temperature, int top_k, double top_p, double min_p,
                    double repetition_penalty, double presence_penalty,
                    double frequency_penalty, int no_repeat_ngram_size,
                    int *out, unsigned long long seed)
{
    if (!prompt || prompt_len < 1) {
        fprintf(stderr, "llm_generate: prompt 不能为空\n");
        exit(1);
    }
    if (max_new < 0) {
        fprintf(stderr, "llm_generate: max_new 不能为负（收到 %d）\n", max_new);
        exit(1);
    }
    if (!(temperature > 0.0)) {
        fprintf(stderr, "llm_generate: temperature 必须 > 0（收到 %g）\n", temperature);
        exit(1);
    }
    if (!(top_p > 0.0) || top_p > 1.0) {
        fprintf(stderr, "llm_generate_p: top_p 必须在 (0, 1] 内（收到 %g）\n", top_p);
        exit(1);
    }
    if (!(min_p >= 0.0) || min_p >= 1.0) {
        fprintf(stderr, "llm_generate_ex: min_p 必须在 [0, 1) 内（收到 %g）\n", min_p);
        exit(1);
    }
    if (!(repetition_penalty > 0.0)) {
        fprintf(stderr, "llm_generate_ex: repetition_penalty 必须 > 0（收到 %g）\n",
                repetition_penalty);
        exit(1);
    }
    if (!isfinite(presence_penalty) || !isfinite(frequency_penalty)) {
        fprintf(stderr, "llm_generate_ex: presence_penalty / frequency_penalty"
                        " 必须是有限数（收到 %g / %g）\n",
                presence_penalty, frequency_penalty);
        exit(1);
    }
    if (no_repeat_ngram_size < 0 || no_repeat_ngram_size == 1) {
        fprintf(stderr, "llm_generate_ex: no_repeat_ngram_size 必须是 >= 0 且 != 1 的整数"
                        "（收到 %d）\n", no_repeat_ngram_size);
        exit(1);
    }
    if (top_k < 0) {
        fprintf(stderr, "llm_generate_p: top_k 必须 >= 0（0 表示不过滤，收到 %d）\n",
                top_k);
        exit(1);
    }
    check_ids(m, prompt, prompt_len, "llm_generate");
    if (prompt_len > m->max_len) {
        fprintf(stderr, "llm_generate: prompt 长度 %d 超过 max_len=%d\n",
                prompt_len, m->max_len);
        exit(1);
    }
    const int V = m->vocab_size;
    int *ids = xmalloc((size_t)(prompt_len + max_new) * sizeof(int));
    memcpy(ids, prompt, (size_t)prompt_len * sizeof(int));
    int len = prompt_len;

    unsigned long long rng = seed;
    llm_set_train(m, 0);

    for (int step = 0; step < max_new; step++) {
        int start = len > m->max_len ? len - m->max_len : 0;
        int T = len - start;
        Tensor logits_view = *m->logits_buf;
        set_shape2(&logits_view, T, V);
        llm_forward(m, ids + start, T, &logits_view);

        const double *last = m->logits_buf->data + (size_t)(T - 1) * V;
        double *row = m->tmp_row;
        memcpy(row, last, (size_t)V * sizeof(double));

        /* 与 Python 的 generate 相同：罚分用完整历史 ids（含已滑出窗口的部分），
         * 追加在采样之前；全默认时本调用直接返回，row 一个字节都不改。 */
        llm_apply_generation_penalties(m, row, ids, len, repetition_penalty,
                                       presence_penalty, frequency_penalty,
                                       no_repeat_ngram_size);
        int pick = sample_row(m, row, temperature, top_k, top_p, min_p, &rng);
        ids[len++] = pick;
    }

    memcpy(out, ids, (size_t)len * sizeof(int));
    free(ids);
    return len;
}

int llm_generate_p(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                   double temperature, int top_k, double top_p, int *out,
                   unsigned long long seed)
{
    /* 旧入口：全部新参数取默认值，路径与 llm_generate_ex 的默认分支逐位相同。 */
    return llm_generate_ex(m, prompt, prompt_len, max_new, temperature, top_k, top_p,
                           0.0, 1.0, 0.0, 0.0, 0, out, seed);
}

/* ---------------------- 增量 KV cache 生成 ----------------------
 * 与 llm_generate_p 的采样顺序完全相同，只是把「每步对整段窗口重算」换成
 * 「预填充一次 + 每个新 token 一次单 token 前向」。窗口滑动时位置整体左移，
 * 增量 cache 无法原地续用，于是按新窗口重建（Python _generate_cached 同样如此）。
 * 因此结果（token 序列与每步 logits）与 llm_generate_p 逐位相同。 */

void llm_cache_reset(CausalLM *m)
{
    if (!m) {
        fprintf(stderr, "llm_cache_reset: m 为 NULL\n");
        exit(1);
    }
    te_cache_reset(m->enc);
}

void llm_forward_step(CausalLM *m, int token, int pos, double *logits)
{
    if (!m || !logits) {
        fprintf(stderr, "llm_forward_step: m / logits 为 NULL\n");
        exit(1);
    }
    const int d = m->d_model, V = m->vocab_size;
    if (token < 0 || token >= V) {
        fprintf(stderr, "llm_forward_step: token=%d 越界 [0, %d)\n", token, V);
        exit(1);
    }
    if (pos < 0 || pos >= m->max_len) {
        fprintf(stderr, "llm_forward_step: pos=%d 越界 [0, %d)\n", pos, m->max_len);
        exit(1);
    }
    /* cache 必须正好走到 pos：调用方要先 llm_cache_reset 再按 0,1,2,... 喂 token。
     * 这个断言能立刻抓住「窗口滑动后忘了重建」这类错误。 */
    const int cache_pos = m->enc->layers[0]->self_attn->cache_pos;
    if (cache_pos != pos) {
        fprintf(stderr, "llm_forward_step: cache_pos=%d 与 pos=%d 不一致"
                        "（需要先 llm_cache_reset 并按顺序喂 token）\n", cache_pos, pos);
        exit(1);
    }

    /* x = embedding[token] (+ 绝对位置编码[pos])；RoPE 模式不加绝对 PE，
     * 位置由各层 MHA 的 cache_pos 决定。 */
    double *x = m->x_buf->data;   /* 只用第 0 行（长度 d） */
    const double *row = m->emb->weight + (size_t)token * d;
    if (m->use_abs_pos) {
        const double *pe = m->pe->pe + (size_t)pos * d;
        for (int j = 0; j < d; j++) x[j] = row[j] + pe[j];
    } else {
        memcpy(x, row, (size_t)d * sizeof(double));
    }

    double *h = m->h_buf->data;   /* 第 0 行 */
    te_forward_step(m->enc, x, h);

    for (int v = 0; v < V; v++)
        logits[v] = dot_neumaier(h, m->W_out + (size_t)v * d, d) + m->b_out[v];
}

int llm_generate_cached_ex(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                           double temperature, int top_k, double top_p, double min_p,
                           double repetition_penalty, double presence_penalty,
                           double frequency_penalty, int no_repeat_ngram_size,
                           int *out, unsigned long long seed)
{
    if (!prompt || prompt_len < 1) {
        fprintf(stderr, "llm_generate_cached: prompt 不能为空\n");
        exit(1);
    }
    if (max_new < 0) {
        fprintf(stderr, "llm_generate_cached: max_new 不能为负（收到 %d）\n", max_new);
        exit(1);
    }
    if (!(temperature > 0.0)) {
        fprintf(stderr, "llm_generate_cached: temperature 必须 > 0（收到 %g）\n", temperature);
        exit(1);
    }
    if (!(top_p > 0.0) || top_p > 1.0) {
        fprintf(stderr, "llm_generate_cached: top_p 必须在 (0, 1] 内（收到 %g）\n", top_p);
        exit(1);
    }
    if (top_k < 0) {
        fprintf(stderr, "llm_generate_cached: top_k 必须 >= 0（0 表示不过滤，收到 %d）\n",
                top_k);
        exit(1);
    }
    if (!(min_p >= 0.0) || min_p >= 1.0) {
        fprintf(stderr, "llm_generate_cached_ex: min_p 必须在 [0, 1) 内（收到 %g）\n", min_p);
        exit(1);
    }
    if (!(repetition_penalty > 0.0)) {
        fprintf(stderr, "llm_generate_cached_ex: repetition_penalty 必须 > 0（收到 %g）\n",
                repetition_penalty);
        exit(1);
    }
    if (!isfinite(presence_penalty) || !isfinite(frequency_penalty)) {
        fprintf(stderr, "llm_generate_cached_ex: presence_penalty / frequency_penalty"
                        " 必须是有限数（收到 %g / %g）\n",
                presence_penalty, frequency_penalty);
        exit(1);
    }
    if (no_repeat_ngram_size < 0 || no_repeat_ngram_size == 1) {
        fprintf(stderr, "llm_generate_cached_ex: no_repeat_ngram_size 必须是 >= 0 且 != 1"
                        " 的整数（收到 %d）\n", no_repeat_ngram_size);
        exit(1);
    }
    check_ids(m, prompt, prompt_len, "llm_generate_cached");
    if (prompt_len > m->max_len) {
        fprintf(stderr, "llm_generate_cached: prompt 长度 %d 超过 max_len=%d\n",
                prompt_len, m->max_len);
        exit(1);
    }
    const int V = m->vocab_size;
    int *ids = xmalloc((size_t)(prompt_len + max_new) * sizeof(int));
    memcpy(ids, prompt, (size_t)prompt_len * sizeof(int));
    int len = prompt_len;

    unsigned long long rng = seed;
    llm_set_train(m, 0);

    double *logits = m->logits_buf->data;   /* 第 0 行，长度 V */

    /* 预填充：只喂当前窗口（prompt 长于 max_len 时取最后 max_len 个） */
    int start = prompt_len > m->max_len ? prompt_len - m->max_len : 0;
    const int T0 = prompt_len - start;
    llm_cache_reset(m);
    for (int i = 0; i < T0; i++)
        llm_forward_step(m, ids[start + i], i, logits);
    int pos = T0;

    for (int step = 0; step < max_new; step++) {
        double *row = m->tmp_row;
        memcpy(row, logits, (size_t)V * sizeof(double));
        /* 与整段重算路径完全相同的罚分 + 采样顺序；ids 是完整历史。 */
        llm_apply_generation_penalties(m, row, ids, len, repetition_penalty,
                                       presence_penalty, frequency_penalty,
                                       no_repeat_ngram_size);
        int pick = sample_row(m, row, temperature, top_k, top_p, min_p, &rng);
        ids[len++] = pick;

        int new_start = len > m->max_len ? len - m->max_len : 0;
        if (new_start == start) {
            /* 窗口没滑：增量算新 token，位置就是窗口内下标 */
            llm_forward_step(m, pick, pos, logits);
            pos++;
        } else {
            /* 窗口左移一格：所有位置重编号，旧 cache 作废，按新窗口重建。
             * 与 Python _generate_cached 的 new_window 重建分支一致。 */
            start = new_start;
            const int Tw = m->max_len;   /* 此时 len - start 恒等于 max_len */
            llm_cache_reset(m);
            for (int i = 0; i < Tw; i++)
                llm_forward_step(m, ids[start + i], i, logits);
            pos = Tw;
        }
    }

    memcpy(out, ids, (size_t)len * sizeof(int));
    free(ids);
    return len;
}

int llm_generate_cached(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                        double temperature, int top_k, double top_p, int *out,
                        unsigned long long seed)
{
    /* 旧入口：全部新参数取默认值，与 llm_generate_cached_ex 的默认分支逐位相同。 */
    return llm_generate_cached_ex(m, prompt, prompt_len, max_new, temperature, top_k,
                                  top_p, 0.0, 1.0, 0.0, 0.0, 0, out, seed);
}

/* ------------------------------ 查询 ------------------------------ */

int llm_min_p_candidates(const double *logits, int n, double min_p)
{
    if (!logits || n <= 0) return 0;
    if (!(min_p > 0.0)) return n;   /* 与 Python _mask_min_p 的 min_p <= 0 分支一致 */
    double *row = xmalloc((size_t)n * sizeof(double));
    double *probs = xmalloc((size_t)n * sizeof(double));
    memcpy(row, logits, (size_t)n * sizeof(double));
    const int kept = mask_row_min_p(row, probs, n, min_p);
    free(row);
    free(probs);
    return kept;
}

const double *llm_hidden(const CausalLM *m, int *T_out)
{
    if (!m) return NULL;
    if (T_out) *T_out = m->seq_len;
    return m->h_buf ? m->h_buf->data : NULL;
}

int llm_param_count(const CausalLM *m)
{
    int total = 0;
    for (int i = 0; i < m->n_blocks; i++) total += m->blocks[i].n;
    return total;
}

int llm_param_blocks(CausalLM *m, LLMParamBlock *out, int cap)
{
    if (cap < m->n_blocks) {
        fprintf(stderr, "llm_param_blocks: cap=%d 不够（需要 %d）\n", cap, m->n_blocks);
        exit(1);
    }
    memcpy(out, m->blocks, (size_t)m->n_blocks * sizeof(*out));
    return m->n_blocks;
}
