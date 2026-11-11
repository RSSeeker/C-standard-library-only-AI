# -*- coding: utf-8 -*-
"""从 AI.py 导出幻觉检测 / NLI 三分类参考值 -> c/hallu_reference.json

覆盖 cosine_similarity / sentence_similarity_matrix / nli_consistency_score /
hallucination_detect / hallucination_report。
"""
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


def add(**kw):
    CASES.append(kw)


add(kind="cosine", name="cos_parallel", a=[1.0, 0.0, 0.0], b=[2.0, 0.0, 0.0],
    out=AI.cosine_similarity([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]), tol=0.0)
add(kind="cosine", name="cos_orthogonal", a=[1.0, 0.0], b=[0.0, 1.0],
    out=AI.cosine_similarity([1.0, 0.0], [0.0, 1.0]), tol=0.0)
add(kind="cosine", name="cos_zero", a=[0.0, 0.0], b=[1.0, 2.0],
    out=AI.cosine_similarity([0.0, 0.0], [1.0, 2.0]), tol=0.0)
add(kind="cosine", name="cos_generic", a=[0.3, -1.2, 2.0, 0.5],
    b=[-0.7, 0.4, 1.1, -2.0],
    out=AI.cosine_similarity([0.3, -1.2, 2.0, 0.5], [-0.7, 0.4, 1.1, -2.0]),
    tol=0.0)

PRE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
HYP = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.8, 0.0]]
mat = AI.sentence_similarity_matrix(PRE, HYP)
add(kind="matrix", name="matrix_mixed",
    premise=[v for row in PRE for v in row], n_premise=len(PRE),
    hypothesis=[v for row in HYP for v in row], n_hypothesis=len(HYP), dim=3,
    out=[v for row in mat for v in row], tol=0.0)

nli = AI.nli_consistency_score(PRE, HYP)
add(kind="nli", name="nli_mixed",
    premise=[v for row in PRE for v in row], n_premise=len(PRE),
    hypothesis=[v for row in HYP for v in row], n_hypothesis=len(HYP), dim=3,
    entail_threshold=0.75, contradiction_threshold=0.35,
    entail_ratio=nli["entail_ratio"],
    contradiction_ratio=nli["contradiction_ratio"],
    neutral_ratio=nli["neutral_ratio"], score=nli["score"],
    labels=[{"entail": 0, "contradict": 1, "neutral": 2}[x] for x in nli["labels"]],
    tol=0.0)
nli2 = AI.nli_consistency_score(PRE, HYP, entail_threshold=0.2,
                                contradiction_threshold=0.05)
add(kind="nli", name="nli_low_thresholds",
    premise=[v for row in PRE for v in row], n_premise=len(PRE),
    hypothesis=[v for row in HYP for v in row], n_hypothesis=len(HYP), dim=3,
    entail_threshold=0.2, contradiction_threshold=0.05,
    entail_ratio=nli2["entail_ratio"],
    contradiction_ratio=nli2["contradiction_ratio"],
    neutral_ratio=nli2["neutral_ratio"], score=nli2["score"],
    labels=[{"entail": 0, "contradict": 1, "neutral": 2}[x] for x in nli2["labels"]],
    tol=0.0)

ANS_V = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]
SRC_V = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
det = AI.hallucination_detect(["a", "b", "c", "d", "e"], ANS_V,
                              ["s0", "s1"], SRC_V, threshold=0.5)
add(kind="detect", name="detect_partial",
    answer=[v for row in ANS_V for v in row], n_answer=5,
    source=[v for row in SRC_V for v in row], n_source=2, dim=3, threshold=0.5,
    flagged=det["flagged"],
    spans=[v for span in det["spans"] for v in span],
    n_spans=len(det["spans"]),
    hallucination_ratio=det["hallucination_ratio"],
    similarities=det["similarities"], tol=0.0)

det2 = AI.hallucination_detect(["a", "b"], [[1.0, 0.0], [0.0, 1.0]],
                               ["s0"], [[1.0, 0.0]], threshold=0.0)
add(kind="detect", name="detect_supported",
    answer=[1.0, 0.0, 0.0, 1.0], n_answer=2,
    source=[1.0, 0.0], n_source=1, dim=2, threshold=0.0,
    flagged=det2["flagged"], spans=[],
    n_spans=len(det2["spans"]),
    hallucination_ratio=det2["hallucination_ratio"],
    similarities=det2["similarities"], tol=0.0)

ANS_PARTIAL = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
               [0.8, 0.6, 0.0]]
ANS_UNSUP = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
for name, tokens, av, sv, thr in (
    ("report_supported", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]],
     [[1.0, 0.0]], 0.0),
    ("report_partial", ["a", "b", "c", "d"], ANS_PARTIAL, SRC_V, 0.5),
    ("report_unsupported", ["a", "b", "c"], ANS_UNSUP, SRC_V, 0.99),
):
    src_tokens = ["s%d" % i for i in range(len(sv))]
    rep = AI.hallucination_report(tokens, av, src_tokens, sv, threshold=thr)
    add(kind="report", name=name,
        answer=[v for row in av for v in row], n_answer=len(av),
        source=[v for row in sv for v in row], n_source=len(sv),
        dim=len(av[0]), threshold=thr,
        unsupported=[["a", "b", "c", "d", "e"][i] for i in rep["flagged"]],
        n_unsupported=len(rep["flagged"]),
        verdict={"supported": 0, "partially_supported": 1,
                 "unsupported": 2}[rep["verdict"]], tol=0.0)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "hallu_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
