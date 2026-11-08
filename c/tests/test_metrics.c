/*
 * test_metrics.c —— 指标模块自测（不依赖 Python）
 *
 * 覆盖：混淆矩阵、准确率、P/R/F1 四种口径、MCC、top-k、ROC-AUC、
 *       回归四项，以及各指标之间的数学恒等式（micro-F1 == 准确率、
 *       k=类别数时 top-k == 1、标签翻转后 AUC 镜像等）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../metrics.h"

static int g_checks;
static int g_fail;

static void ok(int cond, const char *what)
{
    g_checks++;
    if (cond) return;
    g_fail++;
    printf("  [FAIL] %s\n", what);
}

static void near(double got, double want, double tol, const char *what)
{
    g_checks++;
    if (fabs(got - want) <= tol) return;
    g_fail++;
    printf("  [FAIL] %s：C=%.17g want=%.17g 差=%.3e\n", what, got, want, fabs(got - want));
}

/* ---------- [1] 混淆矩阵 ---------- */
static void test_confusion(void)
{
    printf("[1] 混淆矩阵\n");
    const int yt[5] = {0, 1, 2, 2, 0};
    const int yp[5] = {0, 2, 1, 2, 0};
    int k = 0;
    int *m = confusion_matrix(yt, yp, 5, 0, &k);
    ok(k == 3, "自动类别数 = max+1");
    ok(m[0 * 3 + 0] == 2 && m[1 * 3 + 2] == 1 && m[2 * 3 + 1] == 1 &&
       m[2 * 3 + 2] == 1 && m[1 * 3 + 1] == 0, "矩阵内容");
    /* 每行之和 = 该类真实样本数 */
    ok(m[0 * 3 + 0] + m[0 * 3 + 1] + m[0 * 3 + 2] == 2, "第 0 行合计 2");
    free(m);

    /* num_classes 放大后，多出来的行列全 0 */
    m = confusion_matrix(yt, yp, 5, 5, &k);
    ok(k == 5, "指定 num_classes=5");
    int zero_tail = 1;
    for (int r = 3; r < 5; r++)
        for (int c = 0; c < 5; c++)
            if (m[r * 5 + c]) zero_tail = 0;
    ok(zero_tail, "多出来的行全 0");
    free(m);

    /* 只有一个类别也给出 2x2（max+1=2） */
    const int t1[3] = {1, 1, 1};
    const int p1[3] = {1, 1, 1};
    m = confusion_matrix(t1, p1, 3, 0, &k);
    ok(k == 2 && m[1 * 2 + 1] == 3, "单类别 → 2x2");
    free(m);
}

/* ---------- [2] 准确率 ---------- */
static void test_accuracy(void)
{
    printf("[2] 准确率\n");
    const int yt[5] = {0, 1, 2, 2, 0};
    const int yp[5] = {0, 2, 1, 2, 0};
    near(accuracy_score(yt, yp, 5), 0.6, 0.0, "3/5");
    const int same[3] = {1, 1, 1};
    near(accuracy_score(same, same, 3), 1.0, 0.0, "全对");
    const int other[3] = {0, 0, 0};
    const int zero[3] = {1, 1, 1};
    near(accuracy_score(other, zero, 3), 0.0, 0.0, "全错");
}

/* ---------- [3] precision / recall / f1 ---------- */
static void test_prf(void)
{
    printf("[3] precision / recall / f1\n");
    /* 二分类：TP=3、FP=1、FN=1 → P=R=F1=0.75
       混淆矩阵 [[1,1],[0,2]]：class0 的 P=1/1、class1 的 P=2/3 */
    const int yt[4] = {1, 1, 0, 0};
    const int yp[4] = {1, 1, 0, 1};
    double out[3];
    precision_recall_f1(yt, yp, 4, 2, METRIC_AVG_MICRO, out);
    near(out[0], 0.75, 1e-15, "micro precision");
    near(out[1], 0.75, 1e-15, "micro recall");
    near(out[2], 0.75, 1e-15, "micro f1");

    /* 单标签多分类：micro-F1 恒等于准确率 */
    const int mt[5] = {0, 1, 2, 2, 0};
    const int mp[5] = {0, 2, 1, 2, 0};
    precision_recall_f1(mt, mp, 5, 3, METRIC_AVG_MICRO, out);
    near(out[2], accuracy_score(mt, mp, 5), 1e-15, "micro-F1 == 准确率");

    /* macro 只看出现过的类；macro 与 weighted 在不平衡时不同 */
    const int ut[10] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
    const int up[10] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 2};
    double mac[3], wei[3];
    precision_recall_f1(ut, up, 10, 3, METRIC_AVG_MACRO, mac);
    precision_recall_f1(ut, up, 10, 3, METRIC_AVG_WEIGHTED, wei);
    /* class0: TP=7、predicted=7 → P=1.0、R=7/8
       class1: TP=1、predicted=2 → P=0.5、R=1.0
       class2: TP=1、predicted=1 → P=1.0、R=1.0 */
    near(mac[0], (1.0 + 0.5 + 1.0) / 3.0, 1e-15, "macro precision");
    near(wei[0], (1.0 * 8 + 0.5 * 1 + 1.0 * 1) / 10.0, 1e-15, "weighted precision");
    ok(mac[0] != wei[0], "macro 与 weighted 确实不同");

    /* 放大 num_classes 时，多出来的空类不参与 macro 平均 */
    double padded[3];
    precision_recall_f1(ut, up, 10, 6, METRIC_AVG_MACRO, padded);
    near(padded[0], mac[0], 1e-15, "空类不进 macro");
    double padded_w[3];
    precision_recall_f1(ut, up, 10, 6, METRIC_AVG_WEIGHTED, padded_w);
    near(padded_w[0], wei[0], 1e-15, "空类支持数 0，不影响 weighted");

    /* per-class 明细 */
    double pp[3], rr[3], ff[3];
    int sup[3];
    precision_recall_f1_per_class(ut, up, 10, 3, pp, rr, ff, sup);
    near(pp[0], 1.0, 1e-15, "per_class[0].precision");
    near(rr[1], 1.0, 1e-15, "per_class[1].recall");
    ok(sup[0] == 8 && sup[1] == 1 && sup[2] == 1, "per_class support");
}

/* ---------- [4] MCC ---------- */
static void test_mcc(void)
{
    printf("[4] MCC\n");
    const int t[4] = {0, 0, 1, 1};
    const int p[4] = {0, 0, 1, 1};
    near(matthews_corrcoef(t, p, 4, 0), 1.0, 0.0, "完全正确 → 1");
    const int inv[4] = {1, 1, 0, 0};
    near(matthews_corrcoef(t, inv, 4, 0), -1.0, 0.0, "完全相反 → -1");
    /* 二分类闭式：混淆矩阵 [[1,1],[1,1]] → TP=TN=FP=FN=1 → MCC=0 */
    const int t2[4] = {0, 0, 1, 1};
    const int p2[4] = {0, 1, 1, 0};
    near(matthews_corrcoef(t2, p2, 4, 0), 0.0, 1e-15, "四项各 1 → 0");
    /* 常数预测器：分母为 0，返回 0（Python 同口径） */
    const int c[4] = {0, 0, 0, 0};
    near(matthews_corrcoef(t2, c, 4, 0), 0.0, 0.0, "常数预测 → 0");
}

/* ---------- [5] top-k ---------- */
static void test_topk(void)
{
    printf("[5] top-k\n");
    const int t[3] = {0, 1, 2};
    const double s[9] = {
        0.5, 0.3, 0.2,
        0.1, 0.6, 0.3,
        0.2, 0.3, 0.5,
    };
    near(top_k_accuracy(t, s, 3, 3, 1), 1.0, 0.0, "k=1 全对");
    near(top_k_accuracy(t, s, 3, 3, 3), 1.0, 0.0, "k=类别数 → 恒 1");

    /* 全部预测错、k=1 时 0.0 */
    const int t2[3] = {2, 0, 1};
    near(top_k_accuracy(t2, s, 3, 3, 1), 0.0, 0.0, "k=1 全错");
    /* 样本 0 的真实类 2 得分最低（排第 3），k=2 时只有样本 2 命中 → 1/3 */
    near(top_k_accuracy(t2, s, 3, 3, 2), 1.0 / 3.0, 1e-15, "k=2 命中 1/3");
    near(top_k_accuracy(t2, s, 3, 3, 3), 1.0, 0.0, "k=3 恒 1");

    /* 并列：下标小的优先（Python 的 sorted(key=(-score, class))） */
    const double tie[4] = {0.4, 0.4, 0.2, 0.0};
    const int tt[1] = {1};
    near(top_k_accuracy(tt, tie, 1, 3, 1), 0.0, 0.0, "并列时下标 0 排前");
    const int tt0[1] = {0};
    near(top_k_accuracy(tt0, tie, 1, 3, 1), 1.0, 0.0, "并列时下标 0 命中");
}

/* ---------- [6] ROC-AUC ---------- */
static void test_roc(void)
{
    printf("[6] ROC-AUC\n");
    const int lab[4] = {0, 0, 1, 1};
    const double sep[4] = {0.1, 0.4, 0.35, 0.8};
    /* 正例 0.35/0.8 对负例 0.1/0.4：正确对 3 个、错误对 1 个 → 3/4 */
    near(roc_auc_score(lab, sep, 4), 0.75, 1e-15, "3/4 对");

    const double perfect[4] = {0.0, 0.25, 0.75, 1.0};
    near(roc_auc_score(lab, perfect, 4), 1.0, 0.0, "完全可分 → 1");
    const double rev[4] = {1.0, 0.75, 0.25, 0.0};
    near(roc_auc_score(lab, rev, 4), 0.0, 0.0, "完全相反 → 0");
    const double tie[4] = {0.5, 0.5, 0.5, 0.5};
    near(roc_auc_score(lab, tie, 4), 0.5, 1e-15, "全并列 → 0.5");

    /* 标签翻转 → AUC 镜像成 1-AUC */
    const int flipped[4] = {1, 1, 0, 0};
    near(roc_auc_score(flipped, sep, 4), 0.25, 1e-15, "翻转后 1-0.75");
}

/* ---------- [7] 回归 ---------- */
static void test_regression(void)
{
    printf("[7] 回归指标\n");
    const double t[3] = {1.0, 2.0, 3.0};
    const double p[3] = {1.5, 2.0, 2.0};
    near(mean_absolute_error(t, p, 3), (0.5 + 0.0 + 1.0) / 3.0, 1e-16, "MAE");
    near(root_mean_squared_error(t, p, 3), sqrt((0.25 + 0.0 + 1.0) / 3.0), 1e-16, "RMSE");
    near(mean_absolute_percentage_error(t, p, 3, 1e-12),
         100.0 * (0.5 / 1.0 + 0.0 / 2.0 + 1.0 / 3.0) / 3.0, 1e-13, "MAPE");

    /* 真值为 0 时用 eps 兜底 */
    const double zt[1] = {0.0};
    const double zp[1] = {0.5};
    const double got = mean_absolute_percentage_error(zt, zp, 1, 0.25);
    near(got, 100.0 * 0.5 / 0.25, 1e-13, "MAPE 的 eps 兜底");

    /* R²：完美 1.0；常数目标命中 1.0、否则 0.0 */
    near(r2_score(t, t, 3), 1.0, 0.0, "R² 完美");
    const double ct[3] = {2.0, 2.0, 2.0};
    near(r2_score(ct, ct, 3), 1.0, 0.0, "常数目标命中 → 1");
    const double cp[3] = {2.0, 3.0, 2.0};
    near(r2_score(ct, cp, 3), 0.0, 0.0, "常数目标未命中 → 0");
    /* 用常数均值预测时 ss_res == ss_tot → R² = 0 */
    const double mean_pred[3] = {2.0, 2.0, 2.0};
    near(r2_score(t, mean_pred, 3), 0.0, 1e-16, "均值预测 → 0");

    /* MAE/RMSE 与手写朴素循环一致（同顺序） */
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < 3; i++) {
        s1 += fabs(t[i] - p[i]);
        const double d = t[i] - p[i];
        s2 += d * d;
    }
    near(mean_absolute_error(t, p, 3), s1 / 3.0, 1e-16, "MAE == 朴素和");
    near(root_mean_squared_error(t, p, 3), sqrt(s2 / 3.0), 1e-16, "RMSE == 朴素和");
}

/* ---------- [8] 交叉恒等式 ---------- */
static void test_cross(void)
{
    printf("[8] 交叉恒等式\n");
    /* 同一组数据：accuracy == micro-F1 == 对角线之和 / n */
    const int t[6] = {0, 1, 1, 2, 2, 2};
    const int p[6] = {0, 1, 2, 2, 1, 2};
    double out[3];
    precision_recall_f1(t, p, 6, 3, METRIC_AVG_MICRO, out);
    const double a = accuracy_score(t, p, 6);
    near(out[2], a, 1e-15, "micro-F1 == accuracy");
    int k = 0;
    int *m = confusion_matrix(t, p, 6, 3, &k);
    int diag = 0;
    for (int i = 0; i < k; i++) diag += m[i * k + i];
    near(a, (double)diag / 6.0, 1e-15, "accuracy == 对角线和/n");
    free(m);

    /* top-k 在 k=1 时等于「argmax 后的准确率」 */
    const int t2[3] = {0, 1, 2};
    const double s[9] = {0.6, 0.3, 0.1, 0.2, 0.5, 0.3, 0.1, 0.2, 0.7};
    int arg[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        int best = 0;
        for (int c = 1; c < 3; c++)
            if (s[i * 3 + c] > s[i * 3 + best]) best = c;
        arg[i] = best;
    }
    near(top_k_accuracy(t2, s, 3, 3, 1), accuracy_score(t2, arg, 3), 1e-15,
         "top-1 == argmax 准确率");
}

int main(void)
{
    test_confusion();
    test_accuracy();
    test_prf();
    test_mcc();
    test_topk();
    test_roc();
    test_regression();
    test_cross();

    printf("\n[汇总] %d 项检查，%d 项失败\n", g_checks, g_fail);
    if (g_fail == 0) printf("[OK] metrics 自测全部通过\n");
    return g_fail == 0 ? 0 : 1;
}
