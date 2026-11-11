/*
 * rl_utils.c —— 强化学习/μP 工具集的 C 实现
 *
 * 与 Python 版逐项对齐：GRPO 组相对优势、Best-of-N、拒绝采样、
 * 整数样本自洽性投票，以及 μP 的 target_std / layer_kinds / lr /
 * MuPAdam 单步更新。
 */
#include "rl_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ai_group_relative_advantages(const double *rewards, int n, int group_size,
                                  double eps, int normalize, double *out)
{
    int size, ngroups, g, i;
    if (!rewards || !out || n <= 0) return;
    if (group_size <= 0) {
        size = n;
        ngroups = 1;
    } else {
        size = group_size;
        ngroups = n / size;
        if (ngroups <= 0) return;
    }
    for (g = 0; g < ngroups; g++) {
        const int lo = g * size;
        const int hi = lo + size;
        double mean = 0.0;
        for (i = lo; i < hi; i++) mean += rewards[i];
        mean /= (double)size;
        if (normalize) {
            double var = 0.0;
            double std;
            for (i = lo; i < hi; i++) {
                const double c = rewards[i] - mean;
                var += c * c;
            }
            var /= (double)size;
            std = var > 0.0 ? sqrt(var) : 0.0;
            for (i = lo; i < hi; i++) {
                const double c = rewards[i] - mean;
                out[i] = c / (std + eps);
            }
        } else {
            for (i = lo; i < hi; i++) out[i] = rewards[i] - mean;
        }
    }
}

int ai_best_of_n(const double *rewards, int n, double *out_reward)
{
    int best = 0;
    int i;
    if (!rewards || n <= 0) return -1;
    for (i = 1; i < n; i++) {
        if (rewards[i] > rewards[best]) best = i;
    }
    if (out_reward) *out_reward = rewards[best];
    return best;
}

int ai_rejection_filter_threshold(const double *rewards, int n, double threshold,
                                  int *kept)
{
    int count = 0;
    int i;
    if (!rewards || !kept || n <= 0) return 0;
    for (i = 0; i < n; i++) {
        if (rewards[i] >= threshold) kept[count++] = i;
    }
    return count;
}

/* 选择排序：按 (-reward, index) 升序；小 n 足够，且结果确定。 */
static void sort_indices_by_loss(const double *rewards, int *idx, int n)
{
    int i, j;
    for (i = 0; i < n; i++) {
        int best = i;
        for (j = i + 1; j < n; j++) {
            const int a = idx[j];
            const int b = idx[best];
            if (rewards[a] > rewards[b] ||
                (rewards[a] == rewards[b] && a < b)) {
                best = j;
            }
        }
        if (best != i) {
            const int t = idx[i];
            idx[i] = idx[best];
            idx[best] = t;
        }
    }
}

int ai_rejection_filter_top_ratio(const double *rewards, int n, double ratio,
                                  int *kept)
{
    int *idx;
    int k, i, count;
    if (!rewards || !kept || n <= 0) return 0;
    k = (int)nearbyint((double)n * ratio);
    if (k < 1) k = 1;
    if (k > n) k = n;
    idx = (int *)malloc(sizeof(int) * (size_t)n);
    if (!idx) return 0;
    for (i = 0; i < n; i++) idx[i] = i;
    sort_indices_by_loss(rewards, idx, n);
    count = k;
    for (i = 0; i < count; i++) kept[i] = idx[i];
    free(idx);
    return count;
}

int ai_self_consistency_int(const int *samples, int n, double *out_conf,
                            int *out_unique, int *keys, int *counts)
{
    int nkeys = 0;
    int best, best_count = -1;
    int i, j;
    if (!samples || n <= 0) return -1;
    for (i = 0; i < n; i++) {
        int found = -1;
        for (j = 0; j < nkeys; j++) {
            if (keys[j] == samples[i]) {
                found = j;
                break;
            }
        }
        if (found < 0) {
            keys[nkeys] = samples[i];
            counts[nkeys] = 1;
            nkeys++;
        } else {
            counts[found]++;
        }
    }
    best = 0;
    for (j = 0; j < nkeys; j++) {
        if (counts[j] > best_count) {
            best_count = counts[j];
            best = keys[j];
        }
    }
    if (out_conf) *out_conf = (double)best_count / (double)n;
    if (out_unique) *out_unique = nkeys;
    return best;
}

double ai_mup_target_std(const char *kind, double fan_in, double base_width,
                         double width, double scale)
{
    double width_mult = 1.0;
    if (!kind || fan_in <= 0.0 || base_width <= 0.0 || scale <= 0.0) {
        return -1.0;
    }
    if (width > 0.0) width_mult = width / base_width;
    if (strcmp(kind, "hidden") == 0) {
        return scale * sqrt(2.0 / fan_in) / sqrt(width_mult);
    }
    if (strcmp(kind, "readout") == 0) {
        return scale * sqrt(1.0 / fan_in) * sqrt(width_mult);
    }
    if (strcmp(kind, "input") == 0) {
        return scale * sqrt(1.0 / fan_in);
    }
    return -1.0;
}

int ai_mup_layer_kinds(int num_layers, char *kinds)
{
    int i;
    if (!kinds || num_layers < 1) return 0;
    if (num_layers == 1) {
        kinds[0] = 'r';
        return 1;
    }
    if (num_layers == 2) {
        kinds[0] = 'i';
        kinds[1] = 'r';
        return 2;
    }
    kinds[0] = 'i';
    for (i = 1; i < num_layers - 1; i++) kinds[i] = 'h';
    kinds[num_layers - 1] = 'r';
    return num_layers;
}

void ai_mup_lr(const double *fan_ins, int n, double base_lr, double width,
               double base_width, double *out_lrs)
{
    char kinds[64];
    double width_mult;
    int nl, i;
    if (!fan_ins || !out_lrs || n <= 0 || base_width <= 0.0) return;
    if (n > 64) n = 64;
    if (width <= 0.0) {
        width = fan_ins[0];
        for (i = 1; i < n; i++) {
            if (fan_ins[i] > width) width = fan_ins[i];
        }
    }
    width_mult = width / base_width;
    nl = ai_mup_layer_kinds(n, kinds);
    for (i = 0; i < nl; i++) {
        if (kinds[i] == 'h') out_lrs[i] = base_lr / width_mult;
        else if (kinds[i] == 'r') out_lrs[i] = base_lr * width_mult;
        else out_lrs[i] = base_lr;
    }
}

void ai_mupadam_step(double *params, const double *grads, double *m, double *v,
                     int count, double lr, double beta1, double beta2,
                     double eps, int step_count)
{
    double c1, c2, o1, o2;
    int i;
    if (!params || !grads || !m || !v || count <= 0) return;
    if (step_count < 1) step_count = 1;
    c1 = 1.0 - pow(beta1, (double)step_count);
    c2 = 1.0 - pow(beta2, (double)step_count);
    o1 = 1.0 - beta1;
    o2 = 1.0 - beta2;
    for (i = 0; i < count; i++) {
        const double g = grads[i];
        const double mi = beta1 * m[i] + o1 * g;
        const double vi = beta2 * v[i] + o2 * g * g;
        m[i] = mi;
        v[i] = vi;
        params[i] -= lr * (mi / c1) / (sqrt(vi / c2) + eps);
    }
}
