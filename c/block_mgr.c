/*
 * block_mgr.c —— PagedAttention 块分配器（PagedBlockManager 的 C 版本）
 */
#include "block_mgr.h"

#include <stdlib.h>
#include <string.h>

struct AiBlockManager {
    int num_blocks;
    int block_size;
    int swap_capacity; /* <0 表示 None */
    int *free_list;
    int n_free;
    int *ref;       /* num_blocks，0 表示空闲 */
    int cap;        /* 请求容量 */
    int **tables;   /* tables[rid] 或 NULL */
    int *table_lens;
    int **swapped;
    int *swapped_lens;
    int seq;
    int cow_count;
};

static int grow(AiBlockManager *m, int need)
{
    int new_cap;
    int **nt, **ns;
    int *nl, *nsl;
    if (need <= m->cap) return 0;
    new_cap = m->cap ? m->cap : 4;
    while (new_cap < need) new_cap *= 2;
    nt = (int **)realloc(m->tables, sizeof(int *) * (size_t)new_cap);
    if (!nt) return -1;
    m->tables = nt;
    ns = (int **)realloc(m->swapped, sizeof(int *) * (size_t)new_cap);
    if (!ns) return -1;
    m->swapped = ns;
    nl = (int *)realloc(m->table_lens, sizeof(int) * (size_t)new_cap);
    if (!nl) return -1;
    m->table_lens = nl;
    nsl = (int *)realloc(m->swapped_lens, sizeof(int) * (size_t)new_cap);
    if (!nsl) return -1;
    m->swapped_lens = nsl;
    for (int i = m->cap; i < new_cap; i++) {
        m->tables[i] = NULL;
        m->swapped[i] = NULL;
        m->table_lens[i] = 0;
        m->swapped_lens[i] = 0;
    }
    m->cap = new_cap;
    return 0;
}

static int *alloc_table(AiBlockManager *m, int len)
{
    int *t = (int *)malloc(sizeof(int) * (size_t)(len > 0 ? len : 1));
    (void)m;
    return t;
}

AiBlockManager *ai_block_mgr_create(int num_blocks, int block_size,
                                    int swap_capacity)
{
    AiBlockManager *m;
    if (num_blocks < 1 || block_size < 1) return NULL;
    if (swap_capacity < -1) return NULL;
    m = (AiBlockManager *)calloc(1, sizeof(AiBlockManager));
    if (!m) return NULL;
    m->num_blocks = num_blocks;
    m->block_size = block_size;
    m->swap_capacity = swap_capacity < 0 ? -1 : swap_capacity;
    m->free_list = (int *)malloc(sizeof(int) * (size_t)num_blocks);
    m->ref = (int *)calloc((size_t)num_blocks, sizeof(int));
    if (!m->free_list || !m->ref) {
        ai_block_mgr_free(m);
        return NULL;
    }
    for (int i = 0; i < num_blocks; i++) m->free_list[i] = i;
    m->n_free = num_blocks;
    return m;
}

void ai_block_mgr_free(AiBlockManager *m)
{
    if (!m) return;
    if (m->tables) {
        for (int i = 0; i < m->cap; i++) free(m->tables[i]);
    }
    if (m->swapped) {
        for (int i = 0; i < m->cap; i++) free(m->swapped[i]);
    }
    free(m->free_list);
    free(m->ref);
    free(m->tables);
    free(m->table_lens);
    free(m->swapped);
    free(m->swapped_lens);
    free(m);
}

static void sort_free(AiBlockManager *m)
{
    for (int i = 1; i < m->n_free; i++) {
        int v = m->free_list[i];
        int j = i - 1;
        while (j >= 0 && m->free_list[j] > v) {
            m->free_list[j + 1] = m->free_list[j];
            j--;
        }
        m->free_list[j + 1] = v;
    }
}

static int valid_rid(AiBlockManager *m, int rid)
{
    return rid >= 0 && rid < m->cap && rid < m->seq;
}

static int is_swapped(AiBlockManager *m, int rid)
{
    return valid_rid(m, rid) && m->swapped[rid] != NULL;
}

static int known(AiBlockManager *m, int rid)
{
    return valid_rid(m, rid) && m->tables[rid] != NULL;
}

int ai_block_mgr_new_request(AiBlockManager *m)
{
    int rid;
    if (!m) return -1;
    rid = m->seq++;
    if (grow(m, m->seq) != 0) return -1;
    m->tables[rid] = alloc_table(m, 0);
    if (!m->tables[rid]) return -1;
    m->table_lens[rid] = 0;
    m->swapped[rid] = NULL;
    m->swapped_lens[rid] = 0;
    return rid;
}

static int append_block(AiBlockManager *m, int rid, int *out, int *out_len)
{
    int *t;
    if (!m->free_list || m->n_free < 1) return -1;
    {
        int bid = m->free_list[0];
        for (int i = 1; i < m->n_free; i++) m->free_list[i - 1] = m->free_list[i];
        m->n_free--;
        m->ref[bid] = 1;
        t = (int *)realloc(m->tables[rid],
                           sizeof(int) * (size_t)(m->table_lens[rid] + 1));
        if (!t) {
            m->free_list[m->n_free++] = bid;
            sort_free(m);
            return -1;
        }
        m->tables[rid] = t;
        t[m->table_lens[rid]++] = bid;
    }
    if (out) {
        memcpy(out, m->tables[rid], sizeof(int) * (size_t)m->table_lens[rid]);
        *out_len = m->table_lens[rid];
    }
    return 0;
}

int ai_block_mgr_allocate(AiBlockManager *m, int rid, int count, int *out,
                          int *out_len)
{
    if (!m || count < 0) return -1;
    if (!known(m, rid)) return -1;
    if (count > m->n_free) return -1;
    for (int i = 0; i < count; i++) {
        if (append_block(m, rid, NULL, NULL) != 0) return -1;
    }
    if (out) {
        memcpy(out, m->tables[rid], sizeof(int) * (size_t)m->table_lens[rid]);
        *out_len = m->table_lens[rid];
    }
    return 0;
}

int ai_block_mgr_ensure(AiBlockManager *m, int rid, int tokens, int *out,
                        int *out_len)
{
    int need, have;
    if (!m || tokens < 0 || !known(m, rid)) return -1;
    need = (tokens + m->block_size - 1) / m->block_size;
    have = m->table_lens[rid];
    if (need > have) {
        if (ai_block_mgr_allocate(m, rid, need - have, out, out_len) != 0)
            return -1;
    } else if (out) {
        memcpy(out, m->tables[rid], sizeof(int) * (size_t)have);
        *out_len = have;
    }
    return 0;
}

static void release_block(AiBlockManager *m, int bid, int *released,
                          int *n_released)
{
    m->ref[bid]--;
    if (m->ref[bid] <= 0) {
        m->ref[bid] = 0;
        m->free_list[m->n_free++] = bid;
        if (released) released[(*n_released)++] = bid;
    }
}

int ai_block_mgr_free_request(AiBlockManager *m, int rid)
{
    if (!m) return -1;
    if (!valid_rid(m, rid)) return -1;
    if (is_swapped(m, rid)) {
        free(m->swapped[rid]);
        m->swapped[rid] = NULL;
        m->swapped_lens[rid] = 0;
        return 0;
    }
    if (!known(m, rid)) return -1;
    for (int i = 0; i < m->table_lens[rid]; i++)
        release_block(m, m->tables[rid][i], NULL, NULL);
    free(m->tables[rid]);
    m->tables[rid] = NULL;
    m->table_lens[rid] = 0;
    sort_free(m);
    return 0;
}

int ai_block_mgr_fork(AiBlockManager *m, int src, int dst)
{
    if (!m) return -1;
    if (!known(m, src)) return -1;
    if (dst < 0) {
        dst = m->seq++;
        if (grow(m, m->seq) != 0) return -1;
    } else {
        if (!valid_rid(m, dst) && dst >= m->cap) {
            if (grow(m, dst + 1) != 0) return -1;
        }
        if (dst < m->cap && (m->tables[dst] != NULL || m->swapped[dst] != NULL))
            return -1;
        if (dst >= m->seq) m->seq = dst + 1;
    }
    m->tables[dst] = alloc_table(m, m->table_lens[src]);
    if (!m->tables[dst]) return -1;
    m->table_lens[dst] = m->table_lens[src];
    memcpy(m->tables[dst], m->tables[src],
           sizeof(int) * (size_t)m->table_lens[src]);
    m->swapped[dst] = NULL;
    m->swapped_lens[dst] = 0;
    for (int i = 0; i < m->table_lens[dst]; i++) m->ref[m->tables[dst][i]]++;
    return dst;
}

int ai_block_mgr_cow(AiBlockManager *m, int rid, int index)
{
    int old, new_bid;
    if (!m || index < 0) return -1;
    if (!known(m, rid)) return -1;
    if (index >= m->table_lens[rid]) return -1;
    old = m->tables[rid][index];
    if (m->ref[old] == 1) return old;
    if (m->n_free < 1) return -1;
    new_bid = m->free_list[0];
    for (int i = 1; i < m->n_free; i++) m->free_list[i - 1] = m->free_list[i];
    m->n_free--;
    m->ref[new_bid] = 1;
    m->ref[old]--;
    m->tables[rid][index] = new_bid;
    m->cow_count++;
    return new_bid;
}

int ai_block_mgr_swap_out(AiBlockManager *m, int rid, int *released,
                          int *n_released)
{
    if (!m || !n_released) return -1;
    if (is_swapped(m, rid)) return -1;
    if (!known(m, rid)) return -1;
    if (m->swap_capacity >= 0) {
        int n_swapped = 0;
        for (int i = 0; i < m->cap; i++)
            if (m->swapped[i]) n_swapped++;
        if (n_swapped >= m->swap_capacity) return -1;
    }
    *n_released = 0;
    for (int i = 0; i < m->table_lens[rid]; i++)
        release_block(m, m->tables[rid][i], released, n_released);
    m->swapped[rid] = m->tables[rid];
    m->swapped_lens[rid] = m->table_lens[rid];
    m->tables[rid] = NULL;
    m->table_lens[rid] = 0;
    sort_free(m);
    return 0;
}

int ai_block_mgr_swap_in(AiBlockManager *m, int rid, int *old_bids,
                         int *new_bids, int *n_map)
{
    int *table, len, n_miss = 0, k = 0;
    int *out;
    if (!m || !n_map) return -1;
    if (!is_swapped(m, rid)) return -1;
    table = m->swapped[rid];
    len = m->swapped_lens[rid];
    for (int i = 0; i < len; i++) {
        int bid = table[i];
        if (bid < 0 || bid >= m->num_blocks) return -1;
        if (m->ref[bid] == 0) n_miss++;
    }
    if (n_miss > m->n_free) return -1;
    out = alloc_table(m, len);
    if (!out) return -1;
    for (int i = 0; i < len; i++) {
        int bid = table[i];
        old_bids[k] = bid;
        if (m->ref[bid] > 0) {
            m->ref[bid]++;
            new_bids[k] = bid;
        } else {
            int nb = m->free_list[0];
            for (int j = 1; j < m->n_free; j++)
                m->free_list[j - 1] = m->free_list[j];
            m->n_free--;
            m->ref[nb] = 1;
            new_bids[k] = nb;
        }
        out[k] = new_bids[k];
        k++;
    }
    m->tables[rid] = out;
    m->table_lens[rid] = k;
    free(m->swapped[rid]);
    m->swapped[rid] = NULL;
    m->swapped_lens[rid] = 0;
    *n_map = k;
    return 0;
}

int ai_block_mgr_stats(AiBlockManager *m, AiBlockStats *out)
{
    int used = 0, shared = 0, swapped_blocks = 0, requests = 0;
    if (!m || !out) return -1;
    for (int i = 0; i < m->num_blocks; i++) {
        if (m->ref[i] > 0) used++;
        if (m->ref[i] > 1) shared++;
    }
    for (int i = 0; i < m->cap; i++) {
        if (m->tables[i]) requests++;
        if (m->swapped[i]) {
            requests++;
            swapped_blocks += m->swapped_lens[i];
        }
    }
    out->used_blocks = used;
    out->free_blocks = m->n_free;
    out->shared_blocks = shared;
    out->swapped_blocks = swapped_blocks;
    out->requests = requests;
    out->copy_on_write = m->cow_count;
    return 0;
}

int ai_block_mgr_tables(AiBlockManager *m, int *out, int *out_len)
{
    int n = 0;
    if (!m || !out || !out_len) return -1;
    for (int i = 0; i < m->cap; i++) {
        if (!m->tables[i]) continue;
        out[n++] = i;
        out[n++] = m->table_lens[i];
        for (int j = 0; j < m->table_lens[i]; j++) out[n++] = m->tables[i][j];
    }
    *out_len = n;
    return 0;
}

int ai_block_mgr_table(AiBlockManager *m, int rid, int *out, int *out_len)
{
    if (!m || !out || !out_len || !known(m, rid)) return -1;
    memcpy(out, m->tables[rid], sizeof(int) * (size_t)m->table_lens[rid]);
    *out_len = m->table_lens[rid];
    return 0;
}
