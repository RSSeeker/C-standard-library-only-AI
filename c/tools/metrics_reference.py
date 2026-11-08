"""从 AI.py 导出指标参考值 -> c/metrics_reference.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(C_DIR)
PY_REPO = os.path.join(os.path.dirname(REPO), "Python-standard-library-only-AI-main")
sys.path.insert(0, PY_REPO)

import AI  # noqa: E402

CASES = []


def add(name, func, tol=0.0, **kw):
    CASES.append(dict(name=name, func=func, tol=tol, **kw))


def conf(name, y_true, y_pred, num_classes=None):
    mat = AI.confusion_matrix(y_true, y_pred, num_classes)
    add(name, "confusion_matrix", y_true=y_true, y_pred=y_pred,
        num_classes=num_classes, n=len(mat), matrix=mat)


def acc(name, y_true, y_pred):
    add(name, "accuracy_score", y_true=y_true, y_pred=y_pred,
        value=AI.accuracy_score(y_true, y_pred))


def prf(name, y_true, y_pred, average, num_classes=None):
    r = AI.precision_recall_f1(y_true, y_pred, average, num_classes)
    per = r["per_class"]
    add(name, "precision_recall_f1", y_true=y_true, y_pred=y_pred,
        num_classes=num_classes, average=average,
        precision=r["precision"], recall=r["recall"], f1=r["f1"],
        per_precision=[row["precision"] for row in per],
        per_recall=[row["recall"] for row in per],
        per_f1=[row["f1"] for row in per],
        support=[row["support"] for row in per], n=len(per))


def mcc(name, y_true, y_pred, num_classes=None):
    add(name, "matthews_corrcoef", y_true=y_true, y_pred=y_pred,
        num_classes=num_classes,
        value=AI.matthews_corrcoef(y_true, y_pred, num_classes),
        n=len(AI.confusion_matrix(y_true, y_pred, num_classes)))


def topk(name, y_true, scores, k):
    add(name, "top_k_accuracy", y_true=y_true, scores=scores, k=k,
        value=AI.top_k_accuracy(y_true, scores, k), num_classes=len(scores[0]))


def roc(name, labels, scores):
    add(name, "roc_auc_score", labels=labels, scores=scores,
        value=AI.roc_auc_score(labels, scores))


def reg(name, func, y_true, y_pred, label=None, **kw):
    fn = getattr(AI, func)
    add(name, func, y_true=y_true, y_pred=y_pred, value=fn(y_true, y_pred, **kw),
        **kw)


# --- 混淆矩阵 / 准确率 ---
conf("cm_basic", [0, 1, 2, 2, 0], [0, 2, 1, 2, 0])
conf("cm_padded", [0, 1], [0, 1], 4)
conf("cm_single_class", [1, 1, 1], [1, 1, 1])
acc("acc_basic", [0, 1, 2, 2, 0], [0, 2, 1, 2, 0])
acc("acc_perfect", [1, 1, 0], [1, 1, 0])
acc("acc_all_wrong", [0, 0, 1], [1, 1, 0])

# --- precision / recall / f1 ---
P_TRUE = [0, 1, 2, 2, 0, 1, 0]
P_PRED = [0, 2, 1, 2, 0, 0, 2]
for avg in ("macro", "micro", "weighted", "none"):
    prf("prf_" + avg, P_TRUE, P_PRED, avg)
prf("prf_unbalanced_macro", [0] * 8 + [1, 2], [0] * 7 + [1, 1, 2], "macro")
prf("prf_unbalanced_weighted", [0] * 8 + [1, 2], [0] * 7 + [1, 1, 2], "weighted")
prf("prf_absent_class", [0, 1], [0, 1], "macro", 4)

# --- MCC ---
mcc("mcc_basic", P_TRUE, P_PRED)
mcc("mcc_binary", [0, 0, 1, 1], [0, 1, 1, 1])
mcc("mcc_padded", [0, 1], [1, 1], 3)

# --- top-k ---
topk("topk_k1", [0, 1, 2],
     [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]], 1)
topk("topk_k2", [0, 1, 2],
     [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]], 2)
topk("topk_k3", [0, 1, 2],
     [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]], 3)
topk("topk_ties", [0, 1],
     [[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], 1)

# --- ROC-AUC ---
roc("roc_basic", [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
roc("roc_ties", [0, 1, 1, 0, 1], [0.5, 0.5, 0.5, 0.5, 0.5])
roc("roc_separable", [0, 0, 1, 1], [0.0, 0.25, 0.75, 1.0])
roc("roc_perfect_ties", [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])

# --- 回归 ---
REG_T = [1.5, -2.25, 0.0, 3.75, 10.0, -0.5]
REG_P = [1.25, -2.0, 0.5, 3.0, 9.25, 0.0]
reg("mae_basic", "mean_absolute_error", REG_T, REG_P)
reg("rmse_basic", "root_mean_squared_error", REG_T, REG_P)
reg("mape_basic", "mean_absolute_percentage_error", REG_T, REG_P)
reg("mape_zero_true", "mean_absolute_percentage_error",
    [0.0, 1.0, -2.0], [0.5, 1.5, -1.0])
reg("r2_basic", "r2_score", REG_T, REG_P)
reg("r2_perfect", "r2_score", REG_T, list(REG_T))
reg("r2_constant_target_hit", "r2_score", [2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
reg("r2_constant_target_miss", "r2_score", [2.0, 2.0, 2.0], [2.0, 3.0, 2.0])

out = {
    "meta": {
        "source": "AI.py",
        "generated_by": "c/tools/metrics_reference.py",
        "cases": len(CASES),
        "note": "分类指标为整数/精确值；回归指标的累加顺序与 AI.py 一致，容差 0",
    },
    "cases": CASES,
}

path = os.path.join(C_DIR, "metrics_reference.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=False)
    f.write("\n")
print("wrote", path, "cases:", len(CASES))
