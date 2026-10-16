/*
 * check_noalloc.c —— 证明"训练热路径零分配"
 *
 * 用链接器的 --wrap 把 malloc / calloc / realloc 换成计数版：
 *   gcc -std=c17 -O2 -Wall -Wextra -Wl,--wrap=malloc -Wl,--wrap=calloc \
 *       -Wl,--wrap=realloc -o check_noalloc check_noalloc.c ai.c -lm
 *
 * 这个检查与机器负载无关（数的是调用次数，不是时间），因此在 CPU 被别的
 * 进程占满时依然可信。
 */
#include "ai.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);

static long g_allocs = 0;
static long g_bytes = 0;

void *__wrap_malloc(size_t n)          { g_allocs++; g_bytes += (long)n; return __real_malloc(n); }
void *__wrap_calloc(size_t c, size_t s){ g_allocs++; g_bytes += (long)(c * s); return __real_calloc(c, s); }
void *__wrap_realloc(void *p, size_t n){ g_allocs++; g_bytes += (long)n; return __real_realloc(p, n); }

static void reset(void) { g_allocs = 0; g_bytes = 0; }

int main(void)
{
    const int sizes[] = { 64, 256, 128, 8 };
    const Activation acts[] = { ACT_LEAKY_RELU, ACT_LEAKY_RELU, ACT_LINEAR };
    enum { N = 200, IN = 64, OUT = 8 };

    static double X[N * IN], Y[N * OUT];
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < IN; k++) X[i * IN + k] = (double)((i * 7 + k * 13) % 100) / 50.0 - 1.0;
        for (int k = 0; k < OUT; k++) {
            double s = 0.0;
            for (int t = 0; t < 8; t++) s += X[i * IN + (k * 8 + t) % IN];
            Y[i * OUT + k] = s / 8.0;
        }
    }

    MLP *net = mlp_create(sizes, acts, 4);
    Trainer t;
    trainer_init(&t, net, LOSS_MSE, OPT_ADAM, 0.001);
    Dataset train = { X, Y, NULL, N, IN, OUT };

    FitConfig cfg = { 0 };
    cfg.epochs = 3;
    cfg.batch_size = 32;
    cfg.shuffle = 1;
    cfg.seed = 5;

    /* 预热：首次 fit 会懒分配内部缓冲（索引 / pred / grad），不计入统计 */
    trainer_fit(&t, &train, NULL, &cfg, NULL, NULL);

    reset();
    int done = trainer_fit(&t, &train, NULL, &cfg, NULL, NULL);
    long allocs_after_warmup = g_allocs;
    long bytes = g_bytes;

    printf("training %d epochs x %d samples x MLP[64,256,128,8]\n", done, N);
    printf("  allocations during fit (after warmup): %ld  (%ld bytes)\n", allocs_after_warmup, bytes);
    printf("  -> %s\n", allocs_after_warmup == 0
           ? "zero-allocation hot path: OK"
           : "still allocating in the hot path");

    trainer_release(&t);
    mlp_free(net);
    return allocs_after_warmup == 0 ? 0 : 1;
}
