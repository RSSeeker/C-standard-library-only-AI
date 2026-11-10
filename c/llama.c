/*
 * llama.c —— LLaMA 式因果语言模型（前向推理）的 C 实现
 *
 * 与 Python 版（AI.py 的 LlamaBlock / LlamaModel）逐项对应，全部前向运算走
 * attention.h / layers_norm.h 的现有构件：
 *   MultiHeadAttention（GQA + causal + RoPE）
 *   RMSNorm（逐 token，单行接口）
 *   SwiGLU（本文件实现：h1 = xW1+b1，g = silu(xWg)+bg，h = h1⊙g，out = hW2+b2）
 *   Embedding
 *
 * 本实现只做前向；不实现反向、Adam 与生成循环（README 与头注释已声明边界）。
 */
#include "llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LL_PI 3.14159265358979323846

/* ============================ 小工具 ============================ */

static void *ll_calloc(size_t n, size_t size)
{
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "llama: calloc 失败\n");
        exit(1);
    }
    return p;
}

static void *ll_realloc(void *old, size_t size)
{
    void *p = realloc(old, size);
    if (!p) {
        fprintf(stderr, "llama: realloc 失败\n");
        exit(1);
    }
    return p;
}

static double ll_gauss(unsigned long long *state)
{
    /* splitmix64 生成两个 [0,1) 均匀数，Box-Muller 转正态（与 C 侧其它模块同源思路） */
    unsigned long long z;
    double u1, u2;
    *state += 0x9e3779b97f4a7c15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    u1 = ((double)(z >> 11) + 0.5) / 9007199254740992.0;
    *state += 0x9e3779b97f4a7c15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    u2 = ((double)(z >> 11) + 0.5) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * LL_PI * u2);
}

static double ll_silu(double z)
{
    return z / (1.0 + exp(-z));
}

static int ll_tensor_cap(int *cap, double **buf, int need)
{
    if (need <= *cap) return 0;
    *buf = (double *)ll_realloc(*buf, (size_t)need * sizeof(double));
    memset(*buf + *cap, 0, (size_t)(need - *cap) * sizeof(double));
    *cap = need;
    return 0;
}

/* ============================== SwiGLU ============================== */

SwiGLU *swiglu_create(int d_model, int d_ff)
{
    if (d_model < 1 || d_ff < 1) {
        fprintf(stderr, "swiglu_create: d_model / d_ff 必须为正\n");
        exit(1);
    }
    SwiGLU *s = (SwiGLU *)ll_calloc(1, sizeof(SwiGLU));
    s->d_model = d_model;
    s->d_ff = d_ff;
    s->W1 = (double *)ll_calloc((size_t)d_ff * d_model, sizeof(double));
    s->Wg = (double *)ll_calloc((size_t)d_ff * d_model, sizeof(double));
    s->W2 = (double *)ll_calloc((size_t)d_model * d_ff, sizeof(double));
    s->b1 = (double *)ll_calloc((size_t)d_ff, sizeof(double));
    s->bg = (double *)ll_calloc((size_t)d_ff, sizeof(double));
    s->b2 = (double *)ll_calloc((size_t)d_model, sizeof(double));
    s->h1 = s->g = s->h = NULL;
    s->cap_h1 = s->cap_g = s->cap_h = 0;

    unsigned long long seed = 0x1234abcdU;
    double std1 = sqrt(2.0 / (double)d_model);
    double std2 = sqrt(2.0 / (double)d_ff);
    for (int i = 0; i < d_ff * d_model; i++) {
        s->W1[i] = ll_gauss(&seed) * std1;
        s->Wg[i] = ll_gauss(&seed) * std1;
    }
    for (int i = 0; i < d_model * d_ff; i++) s->W2[i] = ll_gauss(&seed) * std2;
    return s;
}

void swiglu_free(SwiGLU *s)
{
    if (!s) return;
    free(s->W1);
    free(s->b1);
    free(s->Wg);
    free(s->bg);
    free(s->W2);
    free(s->b2);
    free(s->h1);
    free(s->g);
    free(s->h);
    free(s);
}

void swiglu_set_params(SwiGLU *s, const double *W1, const double *b1,
                       const double *Wg, const double *bg,
                       const double *W2, const double *b2)
{
    if (!s || !W1 || !b1 || !Wg || !bg || !W2 || !b2) {
        fprintf(stderr, "swiglu_set_params: 参数不能为 NULL\n");
        exit(1);
    }
    memcpy(s->W1, W1, (size_t)s->d_ff * s->d_model * sizeof(double));
    memcpy(s->Wg, Wg, (size_t)s->d_ff * s->d_model * sizeof(double));
    memcpy(s->W2, W2, (size_t)s->d_model * s->d_ff * sizeof(double));
    memcpy(s->b1, b1, (size_t)s->d_ff * sizeof(double));
    memcpy(s->bg, bg, (size_t)s->d_ff * sizeof(double));
    memcpy(s->b2, b2, (size_t)s->d_model * sizeof(double));
}

int swiglu_param_count(const SwiGLU *s)
{
    if (!s) return 0;
    return 2 * s->d_ff * s->d_model + 2 * s->d_ff + s->d_model * s->d_ff + s->d_model;
}

void swiglu_forward(SwiGLU *s, const double *x, int T, double *out)
{
    if (!s || !x || !out || T <= 0) {
        fprintf(stderr, "swiglu_forward: 参数非法\n");
        exit(1);
    }
    const int d = s->d_model, ff = s->d_ff;
    ll_tensor_cap(&s->cap_h1, &s->h1, T * ff);
    ll_tensor_cap(&s->cap_g, &s->g, T * ff);
    ll_tensor_cap(&s->cap_h, &s->h, T * ff);

    for (int t = 0; t < T; t++) {
        const double *xr = x + (size_t)t * d;
        for (int j = 0; j < ff; j++) {
            double a = s->b1[j];
            double b = 0.0;
            for (int i = 0; i < d; i++) {
                a += xr[i] * s->W1[(size_t)j * d + i];
                b += xr[i] * s->Wg[(size_t)j * d + i];
            }
            s->h1[(size_t)t * ff + j] = a;
            s->g[(size_t)t * ff + j] = ll_silu(b) + s->bg[j];
            s->h[(size_t)t * ff + j] = a * (ll_silu(b) + s->bg[j]);
        }
        for (int k = 0; k < d; k++) {
            double acc = s->b2[k];
            for (int j = 0; j < ff; j++) {
                acc += s->h[(size_t)t * ff + j] * s->W2[(size_t)k * ff + j];
            }
            out[(size_t)t * d + k] = acc;
        }
    }
}

/* ============================== LlamaBlock ============================== */

LlamaBlock *llama_block_create(int d_model, int num_heads, int num_kv_heads,
                               int d_ff, int max_seq_len, double rope_base)
{
    if (d_model < 1 || num_heads < 1 || d_ff < 1 || max_seq_len < 1) {
        fprintf(stderr, "llama_block_create: 维度参数必须为正\n");
        exit(1);
    }
    if (rope_base <= 0.0) {
        fprintf(stderr, "llama_block_create: rope_base 必须 > 0\n");
        exit(1);
    }
    if (num_kv_heads <= 0) num_kv_heads = num_heads;
    if (num_heads % num_kv_heads != 0) {
        fprintf(stderr, "llama_block_create: num_heads 必须能被 num_kv_heads 整除\n");
        exit(1);
    }

    LlamaBlock *b = (LlamaBlock *)ll_calloc(1, sizeof(LlamaBlock));
    b->d_model = d_model;
    b->num_heads = num_heads;
    b->num_kv_heads = num_kv_heads;
    b->d_ff = d_ff;
    b->max_seq_len = max_seq_len;

    if (num_kv_heads == num_heads) {
        b->attn = mha_create(d_model, num_heads, max_seq_len, 0.0);
    } else {
        b->attn = mha_create_gqa(d_model, num_heads, num_kv_heads, max_seq_len, 0.0);
    }
    mha_set_causal(b->attn, 1);
    mha_set_rope(b->attn, 1, rope_base);
    b->ln1 = rmsnorm_create(d_model, 1e-6);
    b->ln2 = rmsnorm_create(d_model, 1e-6);
    b->ffn = swiglu_create(d_model, d_ff);
    b->buf1 = (double *)ll_calloc((size_t)max_seq_len * d_model, sizeof(double));
    b->buf2 = (double *)ll_calloc((size_t)max_seq_len * d_model, sizeof(double));
    b->buf3 = (double *)ll_calloc((size_t)max_seq_len * d_model, sizeof(double));
    return b;
}

void llama_block_free(LlamaBlock *b)
{
    if (!b) return;
    mha_free(b->attn);
    rmsnorm_free(b->ln1);
    rmsnorm_free(b->ln2);
    swiglu_free(b->ffn);
    free(b->buf1);
    free(b->buf2);
    free(b->buf3);
    free(b);
}

int llama_block_param_count(const LlamaBlock *b)
{
    if (!b) return 0;
    return mha_param_count(b->attn) + 2 * b->d_model + swiglu_param_count(b->ffn);
}

MultiHeadAttention *llama_block_attn(LlamaBlock *b) { return b ? b->attn : NULL; }
RMSNorm *llama_block_ln1(LlamaBlock *b) { return b ? b->ln1 : NULL; }
RMSNorm *llama_block_ln2(LlamaBlock *b) { return b ? b->ln2 : NULL; }
SwiGLU *llama_block_ffn(LlamaBlock *b) { return b ? b->ffn : NULL; }

void llama_block_forward(LlamaBlock *b, const double *x, int T, double *out)
{
    if (!b || !x || !out || T <= 0 || T > b->max_seq_len) {
        fprintf(stderr, "llama_block_forward: 参数非法或 T 超过 max_seq_len\n");
        exit(1);
    }
    const int d = b->d_model;

    int dims2[2] = { T, d };
    int dims1[1] = { d };
    Tensor *tx = tensor_new(2, dims2);
    Tensor *th1 = tensor_new(2, dims2);
    Tensor *tattn = tensor_new(2, dims2);
    Tensor *trow_in = tensor_new(1, dims1);
    Tensor *trow_out = tensor_new(1, dims1);
    Tensor *th2 = tensor_new(2, dims2);
    Tensor *tffn = tensor_new(2, dims2);

    memcpy(tx->data, x, (size_t)T * d * sizeof(double));

    /* h1 = ln1(x) */
    for (int t = 0; t < T; t++) {
        memcpy(trow_in->data, tx->data + (size_t)t * d, (size_t)d * sizeof(double));
        rmsnorm_forward(b->ln1, trow_in, trow_out);
        memcpy(b->buf1 + (size_t)t * d, trow_out->data, (size_t)d * sizeof(double));
    }
    memcpy(th1->data, b->buf1, (size_t)T * d * sizeof(double));

    /* attn(h1) */
    mha_forward(b->attn, th1, tattn);

    /* res1 = x + attn_out；h2 = ln2(res1) */
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d; j++) {
            double v = tx->data[(size_t)t * d + j] + tattn->data[(size_t)t * d + j];
            b->buf2[(size_t)t * d + j] = v;
        }
        memcpy(trow_in->data, b->buf2 + (size_t)t * d, (size_t)d * sizeof(double));
        rmsnorm_forward(b->ln2, trow_in, trow_out);
        memcpy(b->buf3 + (size_t)t * d, trow_out->data, (size_t)d * sizeof(double));
    }
    memcpy(th2->data, b->buf3, (size_t)T * d * sizeof(double));

    /* ffn(h2)，输出加上 res1 */
    swiglu_forward(b->ffn, th2->data, T, tffn->data);
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < d; j++) {
            out[(size_t)t * d + j] = b->buf2[(size_t)t * d + j] + tffn->data[(size_t)t * d + j];
        }
    }

    tensor_free(tx);
    tensor_free(th1);
    tensor_free(tattn);
    tensor_free(trow_in);
    tensor_free(trow_out);
    tensor_free(th2);
    tensor_free(tffn);
}

/* ============================== LlamaModel ============================== */

LlamaModel *llama_create(int vocab_size, int d_model, int num_heads, int d_ff,
                         int num_layers, int num_kv_heads, int max_len,
                         double rope_base)
{
    if (vocab_size < 1 || d_model < 1 || num_heads < 1 || d_ff < 1 ||
        num_layers < 1 || max_len < 1) {
        fprintf(stderr, "llama_create: 维度参数必须为正\n");
        exit(1);
    }
    LlamaModel *m = (LlamaModel *)ll_calloc(1, sizeof(LlamaModel));
    m->vocab_size = vocab_size;
    m->d_model = d_model;
    m->num_heads = num_heads;
    m->num_kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
    m->d_ff = d_ff;
    m->num_layers = num_layers;
    m->max_len = max_len;

    m->emb = emb_create(vocab_size, d_model, max_len);
    m->blocks = (LlamaBlock **)ll_calloc((size_t)num_layers, sizeof(LlamaBlock *));
    for (int i = 0; i < num_layers; i++) {
        m->blocks[i] = llama_block_create(d_model, num_heads, num_kv_heads, d_ff, max_len, rope_base);
    }
    m->norm = rmsnorm_create(d_model, 1e-6);
    m->W_out = (double *)ll_calloc((size_t)vocab_size * d_model, sizeof(double));
    m->b_out = (double *)ll_calloc((size_t)vocab_size, sizeof(double));
    unsigned long long seed = 0x9e3779b9U;
    double std = sqrt(2.0 / (double)d_model);
    for (int i = 0; i < vocab_size * d_model; i++) m->W_out[i] = ll_gauss(&seed) * std;

    m->t_x = tensor_new(2, (int[]){ 1, d_model });
    m->t_y = tensor_new(2, (int[]){ 1, d_model });
    m->buf_a = (double *)ll_calloc((size_t)max_len * d_model, sizeof(double));
    m->buf_b = (double *)ll_calloc((size_t)max_len * d_model, sizeof(double));
    m->cap = max_len * d_model;
    return m;
}

void llama_free(LlamaModel *m)
{
    if (!m) return;
    emb_free(m->emb);
    for (int i = 0; i < m->num_layers; i++) llama_block_free(m->blocks[i]);
    free(m->blocks);
    rmsnorm_free(m->norm);
    free(m->W_out);
    free(m->b_out);
    tensor_free(m->t_x);
    tensor_free(m->t_y);
    free(m->buf_a);
    free(m->buf_b);
    free(m);
}

int llama_param_count(const LlamaModel *m)
{
    if (!m) return 0;
    int total = emb_param_count(m->emb);
    for (int i = 0; i < m->num_layers; i++) total += llama_block_param_count(m->blocks[i]);
    total += rmsnorm_param_count((RMSNorm *)m->norm);
    total += m->vocab_size * m->d_model + m->vocab_size;
    return total;
}

Embedding *llama_emb(LlamaModel *m) { return m ? m->emb : NULL; }
LlamaBlock *llama_block(LlamaModel *m, int index)
{
    if (!m || index < 0 || index >= m->num_layers) return NULL;
    return m->blocks[index];
}
RMSNorm *llama_norm(LlamaModel *m) { return m ? m->norm : NULL; }

void llama_set_head(LlamaModel *m, const double *W_out, const double *b_out)
{
    if (!m || !W_out || !b_out) {
        fprintf(stderr, "llama_set_head: 参数不能为 NULL\n");
        exit(1);
    }
    memcpy(m->W_out, W_out, (size_t)m->vocab_size * m->d_model * sizeof(double));
    memcpy(m->b_out, b_out, (size_t)m->vocab_size * sizeof(double));
}

void llama_forward(LlamaModel *m, const int *ids, int T, double *logits)
{
    if (!m || !ids || !logits || T <= 0 || T > m->max_len) {
        fprintf(stderr, "llama_forward: 参数非法或 T 超过 max_len\n");
        exit(1);
    }
    for (int t = 0; t < T; t++) {
        if (ids[t] < 0 || ids[t] >= m->vocab_size) {
            fprintf(stderr, "llama_forward: token id 越界\n");
            exit(1);
        }
    }
    const int d = m->d_model;

    /* Embedding：直接复用 Tensor (T, d_model) */
    int dims2[2] = { T, d };
    Tensor *tx = tensor_new(2, dims2);
    emb_forward(m->emb, ids, T, tx);
    memcpy(m->buf_a, tx->data, (size_t)T * d * sizeof(double));
    tensor_free(tx);

    double *cur = m->buf_a;
    double *nxt = m->buf_b;
    for (int i = 0; i < m->num_layers; i++) {
        llama_block_forward(m->blocks[i], cur, T, nxt);
        double *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    /* 最终 RMSNorm */
    int dims1[1] = { d };
    Tensor *trow_in = tensor_new(1, dims1);
    Tensor *trow_out = tensor_new(1, dims1);
    double *norm_out = (double *)ll_calloc((size_t)T * d, sizeof(double));
    for (int t = 0; t < T; t++) {
        memcpy(trow_in->data, cur + (size_t)t * d, (size_t)d * sizeof(double));
        rmsnorm_forward(m->norm, trow_in, trow_out);
        memcpy(norm_out + (size_t)t * d, trow_out->data, (size_t)d * sizeof(double));
    }
    tensor_free(trow_in);
    tensor_free(trow_out);

    /* 词表头：logits = norm_out @ W_out^T + b_out */
    for (int t = 0; t < T; t++) {
        const double *row = norm_out + (size_t)t * d;
        for (int v = 0; v < m->vocab_size; v++) {
            double acc = m->b_out[v];
            const double *w = m->W_out + (size_t)v * d;
            for (int j = 0; j < d; j++) acc += row[j] * w[j];
            logits[(size_t)t * m->vocab_size + v] = acc;
        }
    }
    free(norm_out);
}
