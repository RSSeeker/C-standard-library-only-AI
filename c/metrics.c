/*
 * metrics.c —— 分类 / 回归指标
 *
 * 与 AI.py 的关系：
 *   - 分类指标全部建立在整数混淆矩阵上，结果精确；
 *   - 回归指标里凡是 AI.py 用 sum(...) 的地方（mae 的 |e| 和、rmse 的 e² 和、
 *     r2 的均值），C 侧都用 Neumaier 补偿求和，并保持同样的遍历顺序；
 *     而 AI.py 用朴素 += 累加的地方（mape 的 total、r2 的 ss_tot/ss_res）这里也
 *     用朴素 +=，不能互换——换了就会差最后几位。
 */
#include "metrics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numeric.h"

static void die(const char *who, const char *msg)
{
    fprintf(stderr, "metrics: %s: %s\n", who, msg);
    exit(1);
}

static void check_len(const char *who, int n)
{
    if (n <= 0) die(who, "长度必须 > 0");
}

static void check_ptr(const char *who, const void *p, const char *name)
{
    if (!p) die(who, name);
}

/* 标签必须是非负整数（Python 侧 _metric_label 拒绝负数与非整数） */
static void check_labels(const char *who, const int *y, int n, const char *name)
{
    for (int i = 0; i < n; i++) {
        if (y[i] < 0) die(who, name);
    }
}

int *confusion_matrix(const int *y_true, const int *y_pred, int n,
                      int num_classes, int *out_n)
{
    const char *who = "confusion_matrix";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_ptr(who, out_n, "out_n 为空");
    check_len(who, n);
    check_labels(who, y_true, n, "y_true 里有负数标签");
    check_labels(who, y_pred, n, "y_pred 里有负数标签");

    int mx = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] > mx) mx = y_true[i];
        if (y_pred[i] > mx) mx = y_pred[i];
    }
    int k = mx + 1;
    if (num_classes > 0) {
        if (num_classes < k) die(who, "num_classes 小于实际类别数");
        k = num_classes;
    }

    int *mat = (int *)calloc((size_t)k * (size_t)k, sizeof(int));
    if (!mat) die(who, "内存不足");
    for (int i = 0; i < n; i++) mat[(size_t)y_true[i] * (size_t)k + (size_t)y_pred[i]] += 1;
    *out_n = k;
    return mat;
}

double accuracy_score(const int *y_true, const int *y_pred, int n)
{
    const char *who = "accuracy_score";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_len(who, n);

    int hit = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) hit += 1;
    }
    return (double)hit / (double)n;
}

/*
 * per-class 明细。混淆矩阵按需构造：传进来的 mat 为 NULL 时自己建一个。
 * n 由调用方通过 matrix_n 给出。
 */
static void per_class_from_matrix(const char *who, const int *mat, int k,
                                  double *precision_out, double *recall_out,
                                  double *f1_out, int *support_out)
{
    for (int c = 0; c < k; c++) {
        int tp = mat[(size_t)c * (size_t)k + (size_t)c];
        long long support = 0, predicted = 0;
        for (int j = 0; j < k; j++) support += mat[(size_t)c * (size_t)k + (size_t)j];
        for (int r = 0; r < k; r++) predicted += mat[(size_t)r * (size_t)k + (size_t)c];

        double precision = predicted ? (double)tp / (double)predicted : 0.0;
        double recall = support ? (double)tp / (double)support : 0.0;
        double f1 = (precision + recall) ? (2.0 * precision * recall / (precision + recall)) : 0.0;
        if (precision_out) precision_out[c] = precision;
        if (recall_out) recall_out[c] = recall;
        if (f1_out) f1_out[c] = f1;
        if (support_out) support_out[c] = (int)support;
        (void)who;
    }
}

void precision_recall_f1_per_class(const int *y_true, const int *y_pred, int n,
                                   int num_classes, double *precision_out,
                                   double *recall_out, double *f1_out,
                                   int *support_out)
{
    const char *who = "precision_recall_f1";
    int k = 0;
    int *mat = confusion_matrix(y_true, y_pred, n, num_classes, &k);
    per_class_from_matrix(who, mat, k, precision_out, recall_out, f1_out, support_out);
    free(mat);
}

void precision_recall_f1(const int *y_true, const int *y_pred, int n,
                         int num_classes, MetricAverage average, double *out3)
{
    const char *who = "precision_recall_f1";
    check_ptr(who, out3, "out3 为空");
    if (average != METRIC_AVG_MACRO && average != METRIC_AVG_MICRO &&
        average != METRIC_AVG_WEIGHTED && average != METRIC_AVG_NONE) {
        die(who, "average 非法");
    }

    int k = 0;
    int *mat = confusion_matrix(y_true, y_pred, n, num_classes, &k);
    double *precision = (double *)calloc((size_t)k, sizeof(double));
    double *recall = (double *)calloc((size_t)k, sizeof(double));
    double *f1 = (double *)calloc((size_t)k, sizeof(double));
    int *support = (int *)calloc((size_t)k, sizeof(int));
    if (!precision || !recall || !f1 || !support) die(who, "内存不足");

    per_class_from_matrix(who, mat, k, precision, recall, f1, support);

    /* 全局计数（Python 的 total_tp/fp/fn） */
    long long total_tp = 0, total_fp = 0, total_fn = 0;
    for (int c = 0; c < k; c++) {
        int tp = mat[(size_t)c * (size_t)k + (size_t)c];
        long long support_c = 0, predicted = 0;
        for (int j = 0; j < k; j++) support_c += mat[(size_t)c * (size_t)k + (size_t)j];
        for (int r = 0; r < k; r++) predicted += mat[(size_t)r * (size_t)k + (size_t)c];
        total_tp += tp;
        total_fp += predicted - tp;
        total_fn += support_c - tp;
    }

    double p = 0.0, r = 0.0, f = 0.0;
    if (average == METRIC_AVG_MICRO) {
        const long long denom_p = total_tp + total_fp;
        const long long denom_r = total_tp + total_fn;
        p = denom_p ? (double)total_tp / (double)denom_p : 0.0;
        r = denom_r ? (double)total_tp / (double)denom_r : 0.0;
        f = (p + r) ? (2.0 * p * r / (p + r)) : 0.0;
    } else if (average == METRIC_AVG_WEIGHTED) {
        long long total_support = 0;
        for (int c = 0; c < k; c++) total_support += support[c];
        if (total_support) {
            double *support_d = (double *)calloc((size_t)k, sizeof(double));
            if (!support_d) die(who, "内存不足");
            for (int c = 0; c < k; c++) support_d[c] = (double)support[c];
            /* Python: sum(row[...] * row["support"] for row in per_class) / total_support */
            p = dot_neumaier(precision, support_d, k) / (double)total_support;
            r = dot_neumaier(recall, support_d, k) / (double)total_support;
            f = dot_neumaier(f1, support_d, k) / (double)total_support;
            free(support_d);
        }
    } else {
        /* macro / none：只用「支持数或 F1 非零」的类 */
        double *used_p = (double *)calloc((size_t)k, sizeof(double));
        double *used_r = (double *)calloc((size_t)k, sizeof(double));
        double *used_f = (double *)calloc((size_t)k, sizeof(double));
        if (!used_p || !used_r || !used_f) die(who, "内存不足");
        int used = 0;
        for (int c = 0; c < k; c++) {
            if (support[c] != 0 || f1[c] != 0.0) {
                used_p[used] = precision[c];
                used_r[used] = recall[c];
                used_f[used] = f1[c];
                used += 1;
            }
        }
        if (used) {
            p = sum_neumaier(used_p, used) / (double)used;
            r = sum_neumaier(used_r, used) / (double)used;
            f = sum_neumaier(used_f, used) / (double)used;
        }
        free(used_p);
        free(used_r);
        free(used_f);
    }

    out3[0] = p;
    out3[1] = r;
    out3[2] = f;

    free(precision);
    free(recall);
    free(f1);
    free(support);
    free(mat);
}

double matthews_corrcoef(const int *y_true, const int *y_pred, int n,
                         int num_classes)
{
    const char *who = "matthews_corrcoef";
    int k = 0;
    int *mat = confusion_matrix(y_true, y_pred, n, num_classes, &k);

    long long total = 0, correct = 0;
    for (int c = 0; c < k; c++) {
        correct += mat[(size_t)c * (size_t)k + (size_t)c];
        for (int j = 0; j < k; j++) total += mat[(size_t)c * (size_t)k + (size_t)j];
    }
    if (total == 0) die(who, "没有样本");

    long long *row_sums = (long long *)calloc((size_t)k, sizeof(long long));
    long long *col_sums = (long long *)calloc((size_t)k, sizeof(long long));
    if (!row_sums || !col_sums) die(who, "内存不足");
    for (int c = 0; c < k; c++) {
        for (int j = 0; j < k; j++) row_sums[c] += mat[(size_t)c * (size_t)k + (size_t)j];
        for (int r = 0; r < k; r++) col_sums[c] += mat[(size_t)r * (size_t)k + (size_t)c];
    }

    long long cross = 0, sum_col2 = 0, sum_row2 = 0;
    for (int c = 0; c < k; c++) {
        cross += row_sums[c] * col_sums[c];
        sum_col2 += col_sums[c] * col_sums[c];
        sum_row2 += row_sums[c] * row_sums[c];
    }
    const double cov_ytyp = (double)(correct * total) - (double)cross;
    const double cov_ypyp = (double)(total * total) - (double)sum_col2;
    const double cov_ytyt = (double)(total * total) - (double)sum_row2;
    const double denom = sqrt(cov_ypyp * cov_ytyt);

    free(row_sums);
    free(col_sums);
    free(mat);

    if (denom == 0.0) return 0.0;
    return cov_ytyp / denom;
}

double top_k_accuracy(const int *y_true, const double *scores, int n,
                      int num_classes, int k)
{
    const char *who = "top_k_accuracy";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, scores, "scores 为空");
    check_len(who, n);
    if (k < 1) die(who, "k 必须是 >= 1 的整数");
    if (num_classes < 1) die(who, "num_classes 必须 >= 1");

    int hit = 0;
    for (int i = 0; i < n; i++) {
        const int t = y_true[i];
        if (t < 0 || t >= num_classes) die(who, "y_true 越界（超出得分长度）");
        const double *row = scores + (size_t)i * (size_t)num_classes;
        /* 比 t 更优（分数更高，或同分但类别下标更小）的类别个数就是 t 的名次 */
        int better = 0;
        for (int c = 0; c < num_classes; c++) {
            if (c == t) continue;
            if (row[c] > row[t] || (row[c] == row[t] && c < t)) better += 1;
        }
        if (better < k) hit += 1;
    }
    return (double)hit / (double)n;
}

typedef struct {
    double score;
    int idx;
} RankItem;

static int cmp_rank(const void *pa, const void *pb)
{
    const RankItem *a = (const RankItem *)pa;
    const RankItem *b = (const RankItem *)pb;
    if (a->score < b->score) return -1;
    if (a->score > b->score) return 1;
    return (a->idx < b->idx) ? -1 : (a->idx > b->idx);
}

double roc_auc_score(const int *labels, const double *scores, int n)
{
    const char *who = "roc_auc_score";
    check_ptr(who, labels, "labels 为空");
    check_ptr(who, scores, "scores 为空");
    check_len(who, n);

    long long n_pos = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != 0 && labels[i] != 1) die(who, "labels 只能是 0/1");
        n_pos += labels[i];
    }
    const long long n_neg = (long long)n - n_pos;
    if (n_pos == 0 || n_neg == 0) die(who, "需要同时包含正例和负例");

    RankItem *items = (RankItem *)calloc((size_t)n, sizeof(RankItem));
    double *ranks = (double *)calloc((size_t)n, sizeof(double));
    if (!items || !ranks) die(who, "内存不足");
    for (int i = 0; i < n; i++) {
        items[i].score = scores[i];
        items[i].idx = i;
    }
    qsort(items, (size_t)n, sizeof(RankItem), cmp_rank);

    /* 并列值取平均秩（秩从 1 开始），与 Python 的分组写法等价 */
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && items[j + 1].score == items[i].score) j += 1;
        const double avg = (double)(i + j) / 2.0 + 1.0;
        for (int t = i; t <= j; t++) ranks[items[t].idx] = avg;
        i = j + 1;
    }

    /* Python: sum(r for r, t in zip(ranks, labels) if t == 1) —— 按样本下标升序、Neumaier */
    double *pos_ranks = (double *)calloc((size_t)n_pos, sizeof(double));
    if (!pos_ranks) die(who, "内存不足");
    int m = 0;
    for (int t = 0; t < n; t++) {
        if (labels[t] == 1) pos_ranks[m++] = ranks[t];
    }
    const double rank_sum = sum_neumaier(pos_ranks, m);

    free(pos_ranks);
    free(ranks);
    free(items);

    return (rank_sum - (double)n_pos * (double)(n_pos + 1) / 2.0) /
           ((double)n_pos * (double)n_neg);
}

double mean_absolute_error(const double *y_true, const double *y_pred, int n)
{
    const char *who = "mean_absolute_error";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_len(who, n);

    /* Python: sum(abs(t - p) for ...) / len —— Neumaier */
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        const double x = fabs(y_true[i] - y_pred[i]);
        const double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return (s + c) / (double)n;
}

double root_mean_squared_error(const double *y_true, const double *y_pred, int n)
{
    const char *who = "root_mean_squared_error";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_len(who, n);

    /* Python: sqrt(sum((t - p) ** 2 for ...) / len) —— Neumaier */
    double s = 0.0, c = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = y_true[i] - y_pred[i];
        const double x = d * d;
        const double t = s + x;
        if (fabs(s) >= fabs(x)) c += (s - t) + x;
        else                    c += (x - t) + s;
        s = t;
    }
    return sqrt((s + c) / (double)n);
}

double mean_absolute_percentage_error(const double *y_true, const double *y_pred,
                                     int n, double eps)
{
    const char *who = "mean_absolute_percentage_error";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_len(who, n);
    if (!(eps > 0.0)) die(who, "eps 必须是正数");

    /* Python 这里是朴素 +=，不能换成 Neumaier */
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        const double at = fabs(y_true[i]);
        const double denom = (at > eps) ? at : eps;
        total += fabs(y_true[i] - y_pred[i]) / denom;
    }
    return 100.0 * total / (double)n;
}

double r2_score(const double *y_true, const double *y_pred, int n)
{
    const char *who = "r2_score";
    check_ptr(who, y_true, "y_true 为空");
    check_ptr(who, y_pred, "y_pred 为空");
    check_len(who, n);

    const double mean = sum_neumaier(y_true, n) / (double)n;   /* Python: sum(yt) */
    double ss_tot = 0.0, ss_res = 0.0;                        /* Python: 朴素 += */
    for (int i = 0; i < n; i++) {
        const double d = y_true[i] - mean;
        ss_tot += d * d;
        const double r = y_true[i] - y_pred[i];
        ss_res += r * r;
    }
    if (!(ss_tot < INFINITY && ss_res < INFINITY)) {
        die(who, "平方和溢出（目标值量级过大），请先把 y 归一化");
    }
    if (ss_tot == 0.0) return (ss_res == 0.0) ? 1.0 : 0.0;
    return 1.0 - ss_res / ss_tot;
}
