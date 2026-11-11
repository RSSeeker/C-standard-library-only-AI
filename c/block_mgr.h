/*
 * block_mgr.h —— PagedAttention 块分配器（PagedBlockManager 的 C 版本）
 *
 * 纯整数簿记：分配 / 扩块 / 释放、fork 共享、写时复制（COW）、
 * swap_out/swap_in 与统计。不涉及真实显存。
 */
#ifndef AI_BLOCK_MGR_H
#define AI_BLOCK_MGR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AiBlockManager AiBlockManager;

/* swap_capacity < 0 表示不限制。返回 NULL 表示参数非法。 */
AiBlockManager *ai_block_mgr_create(int num_blocks, int block_size,
                                    int swap_capacity);
void ai_block_mgr_free(AiBlockManager *m);

/* 创建空块表请求，返回非负 rid；失败返回 -1。 */
int ai_block_mgr_new_request(AiBlockManager *m);

/* 追加 count 个物理块；out 写入块表（至少 num_blocks 个 int 空间）。 */
int ai_block_mgr_allocate(AiBlockManager *m, int rid, int count, int *out,
                          int *out_len);
/* 确保块表可容纳 tokens 个 token。 */
int ai_block_mgr_ensure(AiBlockManager *m, int rid, int tokens, int *out,
                        int *out_len);
/* 归还请求的全部块（共享块只减引用计数）。 */
int ai_block_mgr_free_request(AiBlockManager *m, int rid);
/* dst < 0 自动分配新 rid；成功返回 dst。 */
int ai_block_mgr_fork(AiBlockManager *m, int src, int dst);
/* 写前复制第 index 块；返回新块号（独占时返回原块号），失败 -1。 */
int ai_block_mgr_cow(AiBlockManager *m, int rid, int index);
/* 换出：释放物理块，released 写入真正回收的块号。 */
int ai_block_mgr_swap_out(AiBlockManager *m, int rid, int *released,
                          int *n_released);
/* 换入：重新分配，old/new 成对写入（长度至少 num_blocks）。 */
int ai_block_mgr_swap_in(AiBlockManager *m, int rid, int *old_bids,
                         int *new_bids, int *n_map);

/* 统计；tables 为按 rid 升序的扁平表（块号展平），长度写入 n_tables。 */
typedef struct {
    int used_blocks;
    int free_blocks;
    int shared_blocks;
    int swapped_blocks;
    int requests;
    int copy_on_write;
} AiBlockStats;

int ai_block_mgr_stats(AiBlockManager *m, AiBlockStats *out);
int ai_block_mgr_tables(AiBlockManager *m, int *out, int *out_len);

/* 单请求块表查询。 */
int ai_block_mgr_table(AiBlockManager *m, int rid, int *out, int *out_len);

#ifdef __cplusplus
}
#endif

#endif /* AI_BLOCK_MGR_H */
