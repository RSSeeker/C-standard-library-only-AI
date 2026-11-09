/*
 * parallel.c —— 分布式训练数值模拟实现
 */
#include "parallel.h"

#include <math.h>
#include <string.h>

static int par_check_finite(const double *v, int n)
{
    if (!v || n < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

int par_split_ranges(int n, int world_size, int *starts, int *ends)
{
    if (n < 0 || world_size < 1 || !starts || !ends) return -1;
    int base = n / world_size;
    int rem = n % world_size;
    int start = 0;
    for (int rank = 0; rank < world_size; rank++) {
        int size = base + (rank < rem ? 1 : 0);
        starts[rank] = start;
        ends[rank] = start + size;
        start += size;
    }
    return 0;
}

int par_fsdp_shard_flat(const double *flat, int n, int world_size,
                        double *out, int *sizes)
{
    if (!par_check_finite(flat, n) || world_size < 1 || !out || !sizes) return -1;
    int starts[4096];
    int ends[4096];
    if (world_size > 4096) return -1;
    if (par_split_ranges(n, world_size, starts, ends) != 0) return -1;
    for (int rank = 0; rank < world_size; rank++) {
        int len = ends[rank] - starts[rank];
        sizes[rank] = len;
        memcpy(out + starts[rank], flat + starts[rank],
               (size_t)len * sizeof(double));
    }
    return 0;
}

int par_fsdp_all_gather(const double *shard_values, const int *starts,
                        const int *ends, int n_shards, double *out)
{
    if (!shard_values || !starts || !ends || !out || n_shards < 1) return -1;
    int expect = 0;
    for (int r = 0; r < n_shards; r++) {
        if (starts[r] != expect || ends[r] < starts[r]) return -1;
        int len = ends[r] - starts[r];
        memcpy(out + starts[r], shard_values + starts[r],
               (size_t)len * sizeof(double));
        expect = ends[r];
    }
    return 0;
}

int par_zero_step(const double *params, const double *grads, int n, double lr,
                  int world_size, double *out_params,
                  int *upd_starts, int *upd_ends)
{
    if (!par_check_finite(params, n) || !par_check_finite(grads, n)) return -1;
    if (!(lr >= 0.0) || !isfinite(lr) || world_size < 1) return -1;
    if (!out_params || !upd_starts || !upd_ends) return -1;
    int starts[4096], ends[4096];
    if (world_size > 4096) return -1;
    if (par_split_ranges(n, world_size, starts, ends) != 0) return -1;
    memcpy(out_params, params, (size_t)n * sizeof(double));
    for (int rank = 0; rank < world_size; rank++) {
        int s = starts[rank], e = ends[rank];
        for (int i = s; i < e; i++) out_params[i] = params[i] - lr * grads[i];
        upd_starts[rank] = s;
        upd_ends[rank] = e;
    }
    return 0;
}

int par_fsdp_reduce_scatter(const double *rank_grads, int world_size, int n,
                            double *out)
{
    if (!rank_grads || !out || world_size < 1 || n < 1) return -1;
    for (int i = 0; i < world_size * n; i++) {
        if (!isfinite(rank_grads[i])) return -1;
    }
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int r = 0; r < world_size; r++) acc += rank_grads[(size_t)r * n + i];
        out[i] = acc / world_size;
    }
    return 0;
}

int par_tensor_parallel(const double *W, int rows, int cols,
                        const double *bias, int has_bias,
                        const double *x, int world_size, int mode,
                        double *full_out, double *shard_out, int *shard_sizes)
{
    if (!W || rows < 1 || cols < 1 || world_size < 1 || !x) return -1;
    if (!full_out || !shard_out || !shard_sizes) return -1;
    if (has_bias && !bias) return -1;
    if (mode != 0 && mode != 1) return -1;
    int starts[4096], ends[4096];
    if (world_size > 4096) return -1;

    if (mode == 0) {
        if (par_split_ranges(rows, world_size, starts, ends) != 0) return -1;
        for (int rank = 0; rank < world_size; rank++) {
            int s = starts[rank], e = ends[rank];
            shard_sizes[rank] = e - s;
            int pos = s;
            for (int r = s; r < e; r++) {
                double acc = 0.0;
                for (int c = 0; c < cols; c++) {
                    acc += W[(size_t)r * cols + c] * x[c];
                }
                if (has_bias) acc += bias[r];
                full_out[r] = acc;
                shard_out[pos++] = acc;
            }
        }
        return 0;
    }

    /* mode == 1: split input columns, partial sums, then add bias once. */
    if (par_split_ranges(cols, world_size, starts, ends) != 0) return -1;
    double partial[4096];
    if (rows > 4096) return -1;
    for (int r = 0; r < rows; r++) full_out[r] = 0.0;
    for (int rank = 0; rank < world_size; rank++) {
        int s = starts[rank], e = ends[rank];
        for (int r = 0; r < rows; r++) partial[r] = 0.0;
        for (int c = s; c < e; c++) {
            double xc = x[c];
            for (int r = 0; r < rows; r++) partial[r] += W[(size_t)r * cols + c] * xc;
        }
        memcpy(shard_out + (size_t)rank * rows, partial,
               (size_t)rows * sizeof(double));
        shard_sizes[rank] = rows;
        for (int r = 0; r < rows; r++) full_out[r] += partial[r];
    }
    if (has_bias) {
        for (int r = 0; r < rows; r++) full_out[r] += bias[r];
    }
    return 0;
}

int par_sequence_parallel_layer_norm(const double *x_seq, int n_tokens, int d,
                                     double eps, double *out)
{
    if (!x_seq || !out || n_tokens < 1 || d < 1) return -1;
    if (!(eps > 0.0) || !isfinite(eps)) return -1;
    for (int t = 0; t < n_tokens; t++) {
        const double *row = x_seq + (size_t)t * d;
        double sum = 0.0;
        for (int k = 0; k < d; k++) sum += row[k];
        double mu = sum / d;
        double var = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = row[k] - mu;
            var += diff * diff;
        }
        var /= d;
        double denom = sqrt(var + eps);
        for (int k = 0; k < d; k++) out[(size_t)t * d + k] = (row[k] - mu) / denom;
    }
    return 0;
}
