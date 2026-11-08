/*
 * metrics.h —— 分类 / 回归指标（与 AI.py 的同名函数逐位一致）
 *
 * 分类指标吃整数标签与（row-major 的）得分矩阵；回归指标吃 double 数组。
 * 与 AI.py 的一致性由 c/tools/metrics_reference.py + c/tests/compare_metrics.c 保证：
 * 参考值从 AI.py 现场导出，C 侧逐位比对（容差 0）。
 */
#ifndef AI_METRICS_H
#define AI_METRICS_H

/* precision_recall_f1 的 average 口径，取值与 AI.py 的字符串参数一一对应 */
typedef enum {
    METRIC_AVG_MACRO = 0,   /* 各类等权，只用「支持数或 F1 非零」的类（Python 的默认） */
    METRIC_AVG_MICRO = 1,   /* 全局 TP/FP/FN 计数 */
    METRIC_AVG_WEIGHTED = 2,/* 按各类支持数加权 */
    METRIC_AVG_NONE = 3     /* 只返回 per-class（这里的聚合值同 macro） */
} MetricAverage;

/*
 * 混淆矩阵：matrix[t * n + p] = 真实 t 被预测成 p 的样本数。
 * num_classes <= 0 时取 max(标签)+1；否则必须 >= max+1。
 * 成功返回 malloc 出来的 n*n 数组（调用方 free），并通过 out_n 回传 n。
 * 参数非法时打印错误并 exit(1)（与 losses.c 同一口径）。
 */
int *confusion_matrix(const int *y_true, const int *y_pred, int n,
                      int num_classes, int *out_n);

/* 逐样本比较的准确率 */
double accuracy_score(const int *y_true, const int *y_pred, int n);

/*
 * 精确率 / 召回率 / F1 聚合值。out3 依次写 {precision, recall, f1}。
 * 与 Python 的 precision_recall_f1(y_true, y_pred, average) 完全同口径：
 * 空类（支持数与 F1 都为 0）不参与 macro 平均，weighted 用支持数加权。
 */
void precision_recall_f1(const int *y_true, const int *y_pred, int n,
                         int num_classes, MetricAverage average, double *out3);

/*
 * per-class 明细：四个数组各写 num_classes 项（precision/recall/f1/support）。
 * 任一输出指针为 NULL 表示不需要该项。
 */
void precision_recall_f1_per_class(const int *y_true, const int *y_pred, int n,
                                   int num_classes, double *precision_out,
                                   double *recall_out, double *f1_out,
                                   int *support_out);

/* 多分类 MCC（二分类时即标准 MCC） */
double matthews_corrcoef(const int *y_true, const int *y_pred, int n,
                         int num_classes);

/*
 * Top-k 准确率。scores 是 row-major 的 n*num_classes 得分（越大越可能）。
 * 并列时按类别下标升序（与 Python 的 sorted(key=(-score, class)) 一致）。
 */
double top_k_accuracy(const int *y_true, const double *scores, int n,
                      int num_classes, int k);

/*
 * 二分类 ROC-AUC（秩和 / Mann-Whitney U，并列取平均秩）。
 * labels 只能含 0/1，且必须同时出现正负例。
 */
double roc_auc_score(const int *labels, const double *scores, int n);

/* 回归指标：数组长度 n，与 AI.py 的对应函数同公式、同累加顺序 */
double mean_absolute_error(const double *y_true, const double *y_pred, int n);
double root_mean_squared_error(const double *y_true, const double *y_pred, int n);
double mean_absolute_percentage_error(const double *y_true, const double *y_pred,
                                     int n, double eps);
double r2_score(const double *y_true, const double *y_pred, int n);

#endif /* AI_METRICS_H */
