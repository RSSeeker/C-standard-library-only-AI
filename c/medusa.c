/*
 * medusa.c —— Medusa 多头草稿 / EAGLE 特征草稿与贪心验证解码（C17 实现）
 *
 * 与 Python 版 AI.py 的对应关系见 medusa.h 顶部注释。所有数值运算顺序
 * 与 Python 版逐项一致：softmax（减 max 再 exp/sum）、交叉熵梯度
 * (probs - onehot) / (num_heads * n_rows)，验证时用目标模型整段前向。
 */
#include "medusa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================ 小工具 ============================ */

static unsigned long long sm64_next(unsigned long long *s)
{
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double sm64_uniform(unsigned long long *s)
{
    return ((double)(sm64_next(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double sm64_gauss(unsigned long long *s)
{
    double u1 = sm64_uniform(s);
    double u2 = sm64_uniform(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static int argmax_row(const double *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

static void softmax_row(const double *v, int n, double *out)
{
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(v[i] - m);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static void log_softmax_row(const double *v, int n, double *out)
{
    double m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) m = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(v[i] - m);
        sum += out[i];
    }
    double lse = log(sum) + m;
    for (int i = 0; i < n; i++) out[i] = v[i] - lse;
}

static void *xmalloc0(size_t n)
{
    void *p = calloc(n ? n : 1, 1);
    if (!p) {
        fprintf(stderr, "medusa: 内存分配失败\n");
        exit(1);
    }
    return p;
}

/* ============================ MedusaHeads ============================ */

MedusaHeads *medusa_heads_create(int d_model, int vocab_size, int num_heads,
                                 unsigned long long seed)
{
    if (d_model < 1 || vocab_size < 1 || num_heads < 1) {
        fprintf(stderr, "medusa_heads_create: 维度必须 >= 1\n");
        return NULL;
    }
    MedusaHeads *h = xmalloc0(sizeof *h);
    h->d_model = d_model;
    h->vocab_size = vocab_size;
    h->num_heads = num_heads;
    h->heads = xmalloc0((size_t)num_heads * sizeof(MLP *));
    unsigned long long st = seed;
    for (int k = 0; k < num_heads; k++) {
        int sizes[2] = {d_model, vocab_size};
        Activation acts[1] = {ACT_LINEAR};
        MLP *m = mlp_create(sizes, acts, 2);
        if (!m) {
            fprintf(stderr, "medusa_heads_create: mlp_create 失败\n");
            medusa_heads_free(h);
            return NULL;
        }
        for (int i = 0; i < m->n_layers; i++) {
            Layer *l = &m->layers[i];
            int nw = l->fan_out * l->fan_in;
            double std = sqrt(2.0 / (double)l->fan_in);
            for (int j = 0; j < nw; j++) l->W[j] = sm64_gauss(&st) * std;
            for (int j = 0; j < l->fan_out; j++) l->b[j] = 0.0;
        }
        h->heads[k] = m;
    }
    return h;
}

void medusa_heads_free(MedusaHeads *h)
{
    if (!h) return;
    if (h->heads) {
        for (int k = 0; k < h->num_heads; k++) mlp_free(h->heads[k]);
        free(h->heads);
    }
    free(h);
}

long medusa_heads_param_count(const MedusaHeads *h)
{
    if (!h) return 0;
    long total = 0;
    for (int k = 0; k < h->num_heads; k++)
        for (int i = 0; i < h->heads[k]->n_layers; i++) {
            const Layer *l = &h->heads[k]->layers[i];
            total += (long)l->fan_out * l->fan_in + l->fan_out;
        }
    return total;
}

void medusa_heads_zero_grad(MedusaHeads *h)
{
    if (!h) return;
    for (int k = 0; k < h->num_heads; k++)
        for (int i = 0; i < h->heads[k]->n_layers; i++) {
            Layer *l = &h->heads[k]->layers[i];
            int nw = l->fan_out * l->fan_in;
            for (int j = 0; j < nw; j++) l->gW[j] = 0.0;
            for (int j = 0; j < l->fan_out; j++) l->gb[j] = 0.0;
        }
}

void medusa_heads_sgd_step(MedusaHeads *h, double lr)
{
    if (!h) return;
    for (int k = 0; k < h->num_heads; k++)
        for (int i = 0; i < h->heads[k]->n_layers; i++) {
            Layer *l = &h->heads[k]->layers[i];
            int nw = l->fan_out * l->fan_in;
            for (int j = 0; j < nw; j++) l->W[j] -= lr * l->gW[j];
            for (int j = 0; j < l->fan_out; j++) l->b[j] -= lr * l->gb[j];
        }
}

void medusa_heads_forward(MedusaHeads *h, const double *hidden_rows, int n_rows,
                          double *out)
{
    for (int k = 0; k < h->num_heads; k++) {
        for (int r = 0; r < n_rows; r++)
            mlp_forward(h->heads[k], hidden_rows + (size_t)r * h->d_model,
                        out + ((size_t)k * n_rows + (size_t)r) * h->vocab_size);
    }
}

double medusa_heads_loss_and_backward(MedusaHeads *h, const double *hidden_rows,
                                      int n_rows, const int *targets)
{
    const int V = h->vocab_size;
    const double scale = 1.0 / ((double)h->num_heads * (double)n_rows);
    double total = 0.0;
    double *logits = xmalloc0((size_t)V * sizeof(double));
    double *dlogits = xmalloc0((size_t)V * sizeof(double));
    for (int k = 0; k < h->num_heads; k++) {
        for (int r = 0; r < n_rows; r++) {
            const double *row = hidden_rows + (size_t)r * h->d_model;
            int tok = targets[(size_t)k * n_rows + (size_t)r];
            mlp_forward(h->heads[k], row, logits);
            log_softmax_row(logits, V, dlogits);   /* 先放 log 概率 */
            total -= dlogits[tok] * scale;
            for (int j = 0; j < V; j++) dlogits[j] = exp(dlogits[j]) * scale;
            dlogits[tok] -= scale;
            mlp_backward(h->heads[k], dlogits);
        }
    }
    free(logits);
    free(dlogits);
    return total;
}

void medusa_heads_predict(MedusaHeads *h, const double *hidden_row, double *probs)
{
    const int V = h->vocab_size;
    double *logits = xmalloc0((size_t)V * sizeof(double));
    for (int k = 0; k < h->num_heads; k++) {
        mlp_forward(h->heads[k], hidden_row, logits);
        softmax_row(logits, V, probs + (size_t)k * V);
    }
    free(logits);
}

/* ============================ EagleDrafter ============================ */

EagleDrafter *eagle_drafter_create(int d_model, int vocab_size, int hidden,
                                   unsigned long long seed)
{
    if (d_model < 1 || vocab_size < 1 || hidden < 1) {
        fprintf(stderr, "eagle_drafter_create: 维度必须 >= 1\n");
        return NULL;
    }
    EagleDrafter *d = xmalloc0(sizeof *d);
    d->d_model = d_model;
    d->vocab_size = vocab_size;
    d->hidden = hidden;
    d->Wf = xmalloc0((size_t)hidden * d_model * sizeof(double));
    d->We = xmalloc0((size_t)hidden * d_model * sizeof(double));
    d->bh = xmalloc0((size_t)hidden * sizeof(double));
    d->Wh = xmalloc0((size_t)vocab_size * hidden * sizeof(double));
    d->bo = xmalloc0((size_t)vocab_size * sizeof(double));
    unsigned long long st = seed;
    double std = sqrt(2.0 / (double)d_model);
    for (int j = 0; j < hidden; j++)
        for (int i = 0; i < d_model; i++) {
            d->Wf[(size_t)j * d_model + i] = sm64_gauss(&st) * std;
            d->We[(size_t)j * d_model + i] = sm64_gauss(&st) * std;
        }
    double std2 = sqrt(2.0 / (double)hidden);
    for (int j = 0; j < vocab_size; j++)
        for (int i = 0; i < hidden; i++)
            d->Wh[(size_t)j * hidden + i] = sm64_gauss(&st) * std2;
    return d;
}

void eagle_drafter_free(EagleDrafter *d)
{
    if (!d) return;
    free(d->Wf);
    free(d->We);
    free(d->bh);
    free(d->Wh);
    free(d->bo);
    free(d);
}

long eagle_drafter_param_count(const EagleDrafter *d)
{
    if (!d) return 0;
    return (long)d->hidden * d->d_model * 2 + d->hidden
         + (long)d->vocab_size * d->hidden + d->vocab_size;
}

void eagle_drafter_forward(const EagleDrafter *d, const double *feature,
                           const double *prev_emb, double *logits)
{
    const int H = d->hidden, D = d->d_model, V = d->vocab_size;
    double *hid = xmalloc0((size_t)H * sizeof(double));
    for (int j = 0; j < H; j++) {
        double v = d->bh[j];
        const double *wf = d->Wf + (size_t)j * D;
        const double *we = d->We + (size_t)j * D;
        for (int i = 0; i < D; i++) v += wf[i] * feature[i] + we[i] * prev_emb[i];
        hid[j] = tanh(v);
    }
    for (int j = 0; j < V; j++) {
        double v = d->bo[j];
        const double *wh = d->Wh + (size_t)j * H;
        for (int i = 0; i < H; i++) v += wh[i] * hid[i];
        logits[j] = v;
    }
    free(hid);
}

/* ============================ 解码循环 ============================ */

static Tensor *lm_forward(CausalLM *m, const int *ids, int T)
{
    int dims[2] = {T, m->vocab_size};
    Tensor *lt = tensor_new(2, dims);
    llm_forward(m, ids, T, lt);
    return lt;
}

static int check_prompt(const CausalLM *target, const int *prompt, int prompt_len)
{
    if (!target || !prompt || prompt_len < 1) return -1;
    if (prompt_len > target->max_len) return -1;
    for (int i = 0; i < prompt_len; i++)
        if (prompt[i] < 0 || prompt[i] >= target->vocab_size) return -1;
    return 0;
}

static void fill_stats(SpecStats *stats, int drafted, int accepted, int calls)
{
    if (!stats) return;
    stats->drafted = drafted;
    stats->accepted = accepted;
    stats->target_calls = calls;
    stats->accept_rate = drafted ? (double)accepted / (double)drafted : 0.0;
}

int medusa_decode(CausalLM *target, const int *prompt, int prompt_len, int max_new,
                  int eos_id, MedusaHeads *heads, int *out, int out_cap,
                  SpecStats *stats)
{
    if (!heads || !out || check_prompt(target, prompt, prompt_len) != 0) return -1;
    if (max_new < 1 || heads->vocab_size != target->vocab_size
        || heads->d_model != target->d_model) return -1;
    if (out_cap < prompt_len + max_new) return -1;
    if (target->max_len > 4096) return -1;

    llm_set_train(target, 0);
    const int V = target->vocab_size, d = target->d_model, K = heads->num_heads;
    int ctx[4096];
    memcpy(ctx, prompt, (size_t)prompt_len * sizeof(int));
    memcpy(out, prompt, (size_t)prompt_len * sizeof(int));
    int ctx_len = prompt_len;
    int n_new = 0;
    int drafted = 0, accepted = 0, calls = 0;
    double *head_logits = calloc((size_t)K * V, sizeof(double));
    int drafts[4096];
    int verify[8192];
    if (!head_logits) return -1;

    while (n_new < max_new) {
        int quick = (ctx_len + K + 1 > target->max_len);
        Tensor *lt = lm_forward(target, ctx, ctx_len);
        calls++;
        const double *last = lt->data + (size_t)(ctx_len - 1) * V;
        if (quick) {
            int tok = argmax_row(last, V);
            tensor_free(lt);
            ctx[ctx_len++] = tok;
            out[prompt_len + n_new++] = tok;
            if (eos_id >= 0 && tok == eos_id) break;
            continue;
        }
        int hid_T = 0;
        const double *hid = llm_hidden(target, &hid_T);
        if (!hid || hid_T != ctx_len) { tensor_free(lt); free(head_logits); return -1; }
        tensor_free(lt);
        medusa_heads_forward(heads, hid + (size_t)(ctx_len - 1) * d, 1, head_logits);
        for (int k = 0; k < K; k++) {
            drafts[k] = argmax_row(head_logits + (size_t)k * V, V);
            drafted++;
        }
        memcpy(verify, ctx, (size_t)ctx_len * sizeof(int));
        memcpy(verify + ctx_len, drafts, (size_t)K * sizeof(int));
        Tensor *vt = lm_forward(target, verify, ctx_len + K);
        calls++;
        int acc = 0;
        for (int i = 0; i < K; i++) {
            int pos = ctx_len - 1 + i;
            if (argmax_row(vt->data + (size_t)pos * V, V) == drafts[i]) acc++;
            else break;
        }
        accepted += acc;
        int stop = 0;
        for (int i = 0; i < acc && n_new < max_new; i++) {
            ctx[ctx_len++] = drafts[i];
            out[prompt_len + n_new++] = drafts[i];
            if (eos_id >= 0 && drafts[i] == eos_id) { stop = 1; break; }
        }
        if (stop || n_new >= max_new) { tensor_free(vt); break; }
        int tok = argmax_row(vt->data + (size_t)(ctx_len - 1) * V, V);
        tensor_free(vt);
        ctx[ctx_len++] = tok;
        out[prompt_len + n_new++] = tok;
        if (eos_id >= 0 && tok == eos_id) break;
    }
    fill_stats(stats, drafted, accepted, calls);
    free(head_logits);
    return prompt_len + n_new;
}

int eagle_decode(CausalLM *target, const int *prompt, int prompt_len, int max_new,
                 int gamma, int eos_id, EagleDrafter *drafter, int *out, int out_cap,
                 SpecStats *stats)
{
    if (!drafter || !out || check_prompt(target, prompt, prompt_len) != 0) return -1;
    if (max_new < 1 || gamma < 1 || drafter->vocab_size != target->vocab_size
        || drafter->d_model != target->d_model) return -1;
    if (out_cap < prompt_len + max_new) return -1;
    if (target->max_len > 4096 || gamma > 4095) return -1;

    llm_set_train(target, 0);
    const int V = target->vocab_size, d = target->d_model;
    int ctx[4096];
    memcpy(ctx, prompt, (size_t)prompt_len * sizeof(int));
    memcpy(out, prompt, (size_t)prompt_len * sizeof(int));
    int ctx_len = prompt_len;
    int n_new = 0;
    int drafted = 0, accepted = 0, calls = 0;
    double *dlogits = calloc((size_t)V, sizeof(double));
    int drafts[4096];
    if (!dlogits) return -1;

    while (n_new < max_new) {
        int quick = (ctx_len + gamma + 1 > target->max_len);
        Tensor *lt = lm_forward(target, ctx, ctx_len);
        calls++;
        const double *last = lt->data + (size_t)(ctx_len - 1) * V;
        if (quick) {
            int tok = argmax_row(last, V);
            tensor_free(lt);
            ctx[ctx_len++] = tok;
            out[prompt_len + n_new++] = tok;
            if (eos_id >= 0 && tok == eos_id) break;
            continue;
        }
        int first_tok = argmax_row(last, V);
        int hid_T = 0;
        const double *hid = llm_hidden(target, &hid_T);
        if (!hid || hid_T != ctx_len) { tensor_free(lt); free(dlogits); return -1; }
        tensor_free(lt);
        const double *feature = hid + (size_t)(ctx_len - 1) * d;
        drafts[0] = first_tok;
        drafted++;
        for (int g = 1; g < gamma; g++) {
            const double *prev_emb = target->emb->weight + (size_t)drafts[g - 1] * d;
            eagle_drafter_forward(drafter, feature, prev_emb, dlogits);
            drafts[g] = argmax_row(dlogits, V);
            drafted++;
        }
        memcpy(ctx + ctx_len, drafts, (size_t)gamma * sizeof(int));
        Tensor *vt = lm_forward(target, ctx, ctx_len + gamma);
        calls++;
        int acc = 0;
        for (int i = 0; i < gamma; i++) {
            int pos = ctx_len - 1 + i;
            if (argmax_row(vt->data + (size_t)pos * V, V) == drafts[i]) acc++;
            else break;
        }
        accepted += acc;
        int stop = 0;
        for (int i = 0; i < acc && n_new < max_new; i++) {
            ctx[ctx_len++] = drafts[i];
            out[prompt_len + n_new++] = drafts[i];
            if (eos_id >= 0 && drafts[i] == eos_id) { stop = 1; break; }
        }
        if (stop || n_new >= max_new) { tensor_free(vt); break; }
        int tok = argmax_row(vt->data + (size_t)(ctx_len - 1) * V, V);
        tensor_free(vt);
        ctx[ctx_len++] = tok;
        out[prompt_len + n_new++] = tok;
        if (eos_id >= 0 && tok == eos_id) break;
    }
    fill_stats(stats, drafted, accepted, calls);
    free(dlogits);
    return prompt_len + n_new;
}
