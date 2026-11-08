/*
 * text_tok.c —— 字节级 BPE（Python `BytePairTokenizer` 的 C17 移植）
 *
 * 关键点（与 Python 逐位对齐的地方）：
 *   - 合并规则的 id 分配：第 k 条（从 0 数）合并产生 id 256 + k；
 *   - 训练挑对的次序：(count 降序, a 升序, b 升序)；
 *   - 应用合并：从左往右、不重叠（`_bpe_apply_merge`）；
 *   - 编码：每次挑 **rank 最小** 的相邻对替换，直到没有可替换的对。
 */
#include "text_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

/* ------------------------------------------------------------------ 助手 */

static void *tk_malloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "text_tok: 内存不足（%lu 字节）\n", (unsigned long)n);
        exit(1);
    }
    return p;
}

static void *tk_calloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) {
        fprintf(stderr, "text_tok: 内存不足（%lu x %lu）\n",
                (unsigned long)count, (unsigned long)size);
        exit(1);
    }
    return p;
}

static char *tk_strdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *p = (char *)tk_malloc(n);
    memcpy(p, s, n);
    return p;
}

/* ------------------------------------------------------------------ UTF-8 */

/* 解码一个码点；成功返回字节数（1..4），非法返回 0。 */
static int u8_decode(const unsigned char *s, unsigned int *cp)
{
    unsigned char c = s[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        if ((s[1] & 0xC0) != 0x80) return 0;
        *cp = ((unsigned int)(c & 0x1F) << 6) | (unsigned int)(s[1] & 0x3F);
        if (*cp < 0x80) return 0;                    /* 过长编码 */
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
        *cp = ((unsigned int)(c & 0x0F) << 12)
            | ((unsigned int)(s[1] & 0x3F) << 6)
            | (unsigned int)(s[2] & 0x3F);
        if (*cp < 0x800 || (*cp >= 0xD800 && *cp <= 0xDFFF)) return 0;
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return 0;
        *cp = ((unsigned int)(c & 0x07) << 18)
            | ((unsigned int)(s[1] & 0x3F) << 12)
            | ((unsigned int)(s[2] & 0x3F) << 6)
            | (unsigned int)(s[3] & 0x3F);
        if (*cp < 0x10000 || *cp > 0x10FFFF) return 0;
        return 4;
    }
    return 0;
}

/* 码点分类：1 = 字母数字（unicode 子表），2 = 空白，0 = 其它。
 * 与 Python 的 str.isalnum()/isspace() 在表内一致；表外按「其它」处理。 */
static int cp_class(unsigned int cp)
{
    static const unsigned int alpha_ranges[][2] = {
        {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02AF},
        {0x0370, 0x03FF}, {0x0400, 0x052F}, {0x0531, 0x0588},
        {0x0590, 0x05FF}, {0x0600, 0x06FF}, {0x0900, 0x097F},
        {0x3041, 0x3096}, {0x30A1, 0x30FA}, {0x3400, 0x4DBF},
        {0x4E00, 0x9FFF}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF},
        {0xFF10, 0xFF19}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
        {0xFF66, 0xFFDC}, {0x20000, 0x2FA1F}
    };
    static const unsigned int space_cps[] = {
        0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0x200B
    };
    size_t i;

    if (cp < 0x80) {
        if ((cp >= '0' && cp <= '9') || (cp >= 'a' && cp <= 'z')
            || (cp >= 'A' && cp <= 'Z')) {
            return 1;
        }
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\v'
            || cp == '\f' || cp == '\r') {
            return 2;
        }
        return 0;
    }
    for (i = 0; i < sizeof(space_cps) / sizeof(space_cps[0]); i++) {
        if (cp == space_cps[i]) return 2;
    }
    if (cp >= 0x2000 && cp <= 0x200A) return 2;
    for (i = 0; i < sizeof(alpha_ranges) / sizeof(alpha_ranges[0]); i++) {
        if (cp >= alpha_ranges[i][0] && cp <= alpha_ranges[i][1]) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ 内部结构 */

#define TK_INIT_CAP 16

typedef struct {
    int *v;
    int n;
    int cap;
} IntVec;

static void iv_push(IntVec *a, int x)
{
    if (a->n == a->cap) {
        int cap = a->cap ? a->cap * 2 : TK_INIT_CAP;
        int *nv = (int *)tk_malloc((size_t)cap * sizeof(int));
        if (a->n) memcpy(nv, a->v, (size_t)a->n * sizeof(int));
        free(a->v);
        a->v = nv;
        a->cap = cap;
    }
    a->v[a->n++] = x;
}

static void iv_free(IntVec *a)
{
    free(a->v);
    a->v = NULL;
    a->n = a->cap = 0;
}

/* 词块列表：每个词是一段字节（编码前会被转成 int 数组） */
typedef struct {
    char **items;
    int n;
    int cap;
} StrVec;

static void sv_push(StrVec *a, char *s)
{
    if (a->n == a->cap) {
        int cap = a->cap ? a->cap * 2 : TK_INIT_CAP;
        char **nv = (char **)tk_malloc((size_t)cap * sizeof(char *));
        if (a->n) memcpy(nv, a->items, (size_t)a->n * sizeof(char *));
        free(a->items);
        a->items = nv;
        a->cap = cap;
    }
    a->items[a->n++] = s;
}

static void sv_free(StrVec *a)
{
    int i;
    for (i = 0; i < a->n; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL;
    a->n = a->cap = 0;
}

/* (a, b) -> count 的计数表（训练用） */
typedef struct {
    long long *keys;
    int *counts;
    int cap;
    int size;
} PairCounts;

static unsigned long long pair_key(int a, int b)
{
    return ((unsigned long long)(unsigned int)a << 32) | (unsigned int)b;
}

static void pc_rehash(PairCounts *pc, int cap)
{
    long long *keys = (long long *)tk_calloc((size_t)cap, sizeof(long long));
    int *counts = (int *)tk_calloc((size_t)cap, sizeof(int));
    int i;

    for (i = 0; i < pc->cap; i++) {
        if (pc->counts[i] > 0) {
            unsigned long long k = (unsigned long long)pc->keys[i];
            int idx = (int)((k * 11400714819323198485ULL) >> 40) & (cap - 1);
            while (counts[idx] > 0) idx = (idx + 1) & (cap - 1);
            keys[idx] = (long long)k;
            counts[idx] = pc->counts[i];
        }
    }
    free(pc->keys);
    free(pc->counts);
    pc->keys = keys;
    pc->counts = counts;
    pc->cap = cap;
}

static void pc_add(PairCounts *pc, int a, int b)
{
    unsigned long long k = pair_key(a, b);
    int idx;

    if (pc->cap == 0) pc_rehash(pc, 1024);
    if ((pc->size + 1) * 10 >= pc->cap * 7) pc_rehash(pc, pc->cap * 2);
    idx = (int)((k * 11400714819323198485ULL) >> 40) & (pc->cap - 1);
    while (pc->counts[idx] > 0) {
        if ((unsigned long long)pc->keys[idx] == k) {
            pc->counts[idx] += 1;
            return;
        }
        idx = (idx + 1) & (pc->cap - 1);
    }
    pc->keys[idx] = (long long)k;
    pc->counts[idx] = 1;
    pc->size += 1;
}

static void pc_free(PairCounts *pc)
{
    free(pc->keys);
    free(pc->counts);
    pc->keys = NULL;
    pc->counts = NULL;
    pc->cap = pc->size = 0;
}

/* ------------------------------------------------------------------ 词表重建 */

static void tok_release_tables(BpeTokenizer *t)
{
    int i;
    if (t->bytes_of) {
        for (i = 0; i < t->vocab_size; i++) free(t->bytes_of[i]);
        free(t->bytes_of);
        t->bytes_of = NULL;
    }
    free(t->len_of);
    t->len_of = NULL;
    free(t->keys);
    t->keys = NULL;
    free(t->ranks);
    t->ranks = NULL;
    t->hash_cap = 0;
}

/* 按 merges 重建 id->bytes 与 (a,b)->rank。 */
static void tok_rebuild(BpeTokenizer *t)
{
    int i, cap = 1024;

    tok_release_tables(t);
    t->vocab_size = 256 + t->n_merges;
    t->bytes_of = (unsigned char **)tk_calloc((size_t)t->vocab_size, sizeof(unsigned char *));
    t->len_of = (int *)tk_calloc((size_t)t->vocab_size, sizeof(int));
    for (i = 0; i < 256; i++) {
        t->bytes_of[i] = (unsigned char *)tk_malloc(1);
        t->bytes_of[i][0] = (unsigned char)i;
        t->len_of[i] = 1;
    }
    for (i = 0; i < t->n_merges; i++) {
        int a = t->merges[i].a, b = t->merges[i].b;
        int la = t->len_of[a], lb = t->len_of[b];
        unsigned char *p = (unsigned char *)tk_malloc((size_t)la + (size_t)lb);
        memcpy(p, t->bytes_of[a], (size_t)la);
        memcpy(p + la, t->bytes_of[b], (size_t)lb);
        t->bytes_of[256 + i] = p;
        t->len_of[256 + i] = la + lb;
    }

    while (cap < 2 * (t->n_merges + 1)) cap <<= 1;
    t->hash_cap = cap;
    t->keys = (long long *)tk_calloc((size_t)cap, sizeof(long long));
    t->ranks = (int *)tk_malloc((size_t)cap * sizeof(int));
    for (i = 0; i < cap; i++) t->ranks[i] = -1;
    for (i = 0; i < t->n_merges; i++) {
        unsigned long long k = pair_key(t->merges[i].a, t->merges[i].b);
        int idx = (int)((k * 11400714819323198485ULL) >> 40) & (cap - 1);
        while (t->ranks[idx] >= 0) idx = (idx + 1) & (cap - 1);
        t->keys[idx] = (long long)k;
        t->ranks[idx] = i;
    }
}

/* 查 (a, b) 的合并优先级；没有返回 -1。 */
static int tok_rank(const BpeTokenizer *t, int a, int b)
{
    unsigned long long k;
    int idx;

    if (t->n_merges == 0) return -1;
    k = pair_key(a, b);
    idx = (int)((k * 11400714819323198485ULL) >> 40) & (t->hash_cap - 1);
    for (;;) {
        if (t->ranks[idx] < 0) return -1;
        if ((unsigned long long)t->keys[idx] == k) return t->ranks[idx];
        idx = (idx + 1) & (t->hash_cap - 1);
    }
}

/* ------------------------------------------------------------------ 公开 API */

BpeTokenizer *bpe_create(void)
{
    BpeTokenizer *t = (BpeTokenizer *)tk_calloc(1, sizeof(BpeTokenizer));
    tok_rebuild(t);
    return t;
}

BpeTokenizer *bpe_create_from_merges(const int *pairs, int n_merges)
{
    BpeTokenizer *t;
    int k;

    if (!pairs || n_merges < 0) return NULL;
    for (k = 0; k < n_merges; k++) {
        int a = pairs[2 * k], b = pairs[2 * k + 1];
        int limit = 256 + k;            /* 只能引用此前已定义的 id */
        if (a < 0 || a >= limit || b < 0 || b >= limit) return NULL;
    }
    t = (BpeTokenizer *)tk_calloc(1, sizeof(BpeTokenizer));
    if (n_merges > 0) {
        t->merges = (BpePair *)tk_malloc((size_t)n_merges * sizeof(BpePair));
        memcpy(t->merges, pairs, (size_t)n_merges * 2 * sizeof(int));
        t->n_merges = n_merges;
        t->cap_merges = n_merges;
    }
    tok_rebuild(t);
    return t;
}

void bpe_free(BpeTokenizer *t)
{
    if (!t) return;
    tok_release_tables(t);
    free(t->merges);
    free(t);
}

int bpe_vocab_size(const BpeTokenizer *t) { return t ? t->vocab_size : 0; }

int bpe_merge_count(const BpeTokenizer *t) { return t ? t->n_merges : 0; }

int bpe_merge_at(const BpeTokenizer *t, int k, int *a, int *b)
{
    if (!t || k < 0 || k >= t->n_merges) return 0;
    if (a) *a = t->merges[k].a;
    if (b) *b = t->merges[k].b;
    return 1;
}

int bpe_pretokenize(const char *text, char ***out)
{
    StrVec sv;
    const unsigned char *p = (const unsigned char *)text;
    size_t begin = 0;
    int kind = -1, n = 0;

    if (!text || !out) return -1;
    memset(&sv, 0, sizeof(sv));
    while (*p) {
        unsigned int cp = 0;
        int len = u8_decode(p, &cp);
        int k;

        if (len <= 0) {
            sv_free(&sv);
            return -1;
        }
        k = cp_class(cp);
        if (kind < 0) {
            kind = k;
            begin = (size_t)(p - (const unsigned char *)text);
        } else if (k != kind) {
            size_t end = (size_t)(p - (const unsigned char *)text);
            size_t nbytes = end - begin;
            char *piece = (char *)tk_malloc(nbytes + 1);
            memcpy(piece, text + begin, nbytes);
            piece[nbytes] = '\0';
            sv_push(&sv, piece);
            kind = k;
            begin = end;
        }
        p += len;
    }
    if (kind >= 0) {
        char *piece = tk_strdup(text + begin);
        sv_push(&sv, piece);
    }
    n = sv.n;
    *out = sv.items;                   /* 交给 bpe_pretokenize_free */
    return n;
}

void bpe_pretokenize_free(char **pieces, int n)
{
    int i;
    if (!pieces) return;
    for (i = 0; i < n; i++) free(pieces[i]);
    free(pieces);
}

/* 把词块（UTF-8）转成 int 词块（字节 id）。 */
static IntVec word_to_bytes(const char *piece)
{
    IntVec a;
    const unsigned char *p = (const unsigned char *)piece;
    memset(&a, 0, sizeof(a));
    while (*p) {
        iv_push(&a, (int)*p);
        p++;
    }
    return a;
}

/* 应用一条合并（从左往右、不重叠），原地写回 a。 */
static void apply_merge_inplace(IntVec *a, int left, int right, int new_id)
{
    IntVec out;
    int i = 0;

    memset(&out, 0, sizeof(out));
    while (i < a->n) {
        if (i + 1 < a->n && a->v[i] == left && a->v[i + 1] == right) {
            iv_push(&out, new_id);
            i += 2;
        } else {
            iv_push(&out, a->v[i]);
            i += 1;
        }
    }
    free(a->v);
    *a = out;
}

int bpe_train(BpeTokenizer *t, const char *text, int vocab_size, int min_frequency)
{
    char **pieces = NULL;
    int n_pieces, i, step, target, learned = 0;
    IntVec *words;

    if (!t || !text) return -1;
    if (vocab_size < 256) return -1;
    if (min_frequency < 1) return -1;
    n_pieces = bpe_pretokenize(text, &pieces);
    if (n_pieces < 0) return -1;

    words = (IntVec *)tk_calloc((size_t)(n_pieces > 0 ? n_pieces : 1), sizeof(IntVec));
    for (i = 0; i < n_pieces; i++) words[i] = word_to_bytes(pieces[i]);
    bpe_pretokenize_free(pieces, n_pieces);

    /* 重建为空词表（Python 的 train 也是从 self.merges = [] 开始） */
    t->n_merges = 0;
    tok_rebuild(t);

    target = vocab_size - 256;
    for (step = 0; step < target; step++) {
        PairCounts pc;
        int best_a = -1, best_b = -1, best_count = -1;
        int new_id;

        memset(&pc, 0, sizeof(pc));
        for (i = 0; i < n_pieces; i++) {
            int k;
            for (k = 0; k + 1 < words[i].n; k++) {
                pc_add(&pc, words[i].v[k], words[i].v[k + 1]);
            }
        }
        if (pc.size == 0) {
            pc_free(&pc);
            break;
        }
        /* 挑最大 count；并列时取更小的 a，再取更小的 b。 */
        for (i = 0; i < pc.cap; i++) {
            int a, b, c;
            if (pc.counts[i] <= 0) continue;
            a = (int)((unsigned long long)pc.keys[i] >> 32);
            b = (int)((unsigned long long)pc.keys[i] & 0xFFFFFFFFULL);
            c = pc.counts[i];
            if (c > best_count
                || (c == best_count && (best_a < 0 || a < best_a))
                || (c == best_count && a == best_a && b < best_b)) {
                best_count = c;
                best_a = a;
                best_b = b;
            }
        }
        pc_free(&pc);
        if (best_a < 0 || best_count < min_frequency) break;

        if (t->n_merges == t->cap_merges) {
            int cap = t->cap_merges ? t->cap_merges * 2 : 16;
            BpePair *np = (BpePair *)tk_malloc((size_t)cap * sizeof(BpePair));
            if (t->n_merges) memcpy(np, t->merges, (size_t)t->n_merges * sizeof(BpePair));
            free(t->merges);
            t->merges = np;
            t->cap_merges = cap;
        }
        t->merges[t->n_merges].a = best_a;
        t->merges[t->n_merges].b = best_b;
        t->n_merges += 1;
        tok_rebuild(t);
        new_id = 255 + t->n_merges;    /* 第 k 条（1 起）-> 256 + k - 1 */
        learned += 1;
        for (i = 0; i < n_pieces; i++) {
            apply_merge_inplace(&words[i], best_a, best_b, new_id);
        }
    }

    for (i = 0; i < n_pieces; i++) iv_free(&words[i]);
    free(words);
    return learned;
}

int bpe_encode(const BpeTokenizer *t, const char *text, int *ids, int cap)
{
    char **pieces = NULL;
    int n_pieces, i, written = 0;

    if (!t || !text || !ids || cap < 0) return -1;
    n_pieces = bpe_pretokenize(text, &pieces);
    if (n_pieces < 0) return -2;
    for (i = 0; i < n_pieces; i++) {
        IntVec toks = word_to_bytes(pieces[i]);
        for (;;) {
            int best_rank = -1, best_i = -1, k;
            for (k = 0; k + 1 < toks.n; k++) {
                int r = tok_rank(t, toks.v[k], toks.v[k + 1]);
                if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                    best_rank = r;
                    best_i = k;
                }
            }
            if (best_rank < 0) break;
            toks.v[best_i] = 256 + best_rank;
            memmove(toks.v + best_i + 1, toks.v + best_i + 2,
                    (size_t)(toks.n - best_i - 2) * sizeof(int));
            toks.n -= 1;
        }
        if (written + toks.n > cap) {
            iv_free(&toks);
            bpe_pretokenize_free(pieces, n_pieces);
            return -1;
        }
        memcpy(ids + written, toks.v, (size_t)toks.n * sizeof(int));
        written += toks.n;
        iv_free(&toks);
    }
    bpe_pretokenize_free(pieces, n_pieces);
    return written;
}

int bpe_decode(const BpeTokenizer *t, const int *ids, int n, char *out, int cap)
{
    int i, total = 0;

    if (!t || !out || cap <= 0) return -3;
    out[0] = '\0';
    for (i = 0; i < n; i++) {
        int id = ids[i];
        int len;
        if (id < 0 || id >= t->vocab_size) return -1;
        len = t->len_of[id];
        if (total + len + 1 > cap) return -3;
        memcpy(out + total, t->bytes_of[id], (size_t)len);
        total += len;
    }
    out[total] = '\0';
    /* 校验拼出来的字节是合法 UTF-8（与 Python 的 decode 一样不静默替换） */
    {
        const unsigned char *p = (const unsigned char *)out;
        const unsigned char *end = p + total;
        while (p < end) {
            unsigned int cp = 0;
            int len = u8_decode(p, &cp);
            if (len <= 0 || p + len > end) return -2;
            p += len;
        }
    }
    return total;
}

int bpe_save(const BpeTokenizer *t, const char *path)
{
    FILE *f;
    int i;

    if (!t || !path) return -1;
    f = fopen(path, "w");
    if (!f) return -1;
    fputs("{\"format\": \"ai_bpe_v1\", \"merges\": [", f);
    for (i = 0; i < t->n_merges; i++) {
        fprintf(f, "%s[%d, %d]", i ? ", " : "", t->merges[i].a, t->merges[i].b);
    }
    fputs("]}", f);
    fclose(f);
    return 0;
}

BpeTokenizer *bpe_load(const char *path, char *err, size_t errlen)
{
    JsonValue *root;
    const JsonValue *merges;
    BpeTokenizer *t;
    int n, i;

    if (!path) return NULL;
    root = json_parse_file(path, err, errlen);
    if (!root) return NULL;
    if (root->kind != JSON_OBJECT
        || !(merges = json_get(root, "merges")) || merges->kind != JSON_ARRAY) {
        if (err && errlen) snprintf(err, errlen, "text_tok: 存档缺少 merges 数组");
        json_free(root);
        return NULL;
    }
    n = json_array_len(merges);
    for (i = 0; i < n; i++) {
        const JsonValue *pair = json_at(merges, i);
        int a, b, limit = 256 + i;
        if (!pair || pair->kind != JSON_ARRAY || json_array_len(pair) != 2) {
            if (err && errlen) snprintf(err, errlen, "text_tok: 第 %d 条规则不是二元组", i);
            json_free(root);
            return NULL;
        }
        a = (int)json_num(json_at(pair, 0), -1.0);
        b = (int)json_num(json_at(pair, 1), -1.0);
        if (a < 0 || a >= limit || b < 0 || b >= limit) {
            if (err && errlen) snprintf(err, errlen, "text_tok: 第 %d 条规则引用了未定义的 id", i);
            json_free(root);
            return NULL;
        }
    }
    t = (BpeTokenizer *)tk_calloc(1, sizeof(BpeTokenizer));
    if (n > 0) {
        t->merges = (BpePair *)tk_malloc((size_t)n * sizeof(BpePair));
        t->cap_merges = n;
        t->n_merges = n;
        for (i = 0; i < n; i++) {
            const JsonValue *pair = json_at(merges, i);
            t->merges[i].a = (int)json_num(json_at(pair, 0), 0.0);
            t->merges[i].b = (int)json_num(json_at(pair, 1), 0.0);
        }
    }
    json_free(root);
    tok_rebuild(t);
    return t;
}
