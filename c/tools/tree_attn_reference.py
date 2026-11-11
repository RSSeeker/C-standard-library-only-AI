# -*- coding: utf-8 -*-
"""从 AI.py 导出树状注意力辅助函数参考值 -> c/tree_attn_reference.json

覆盖 build_draft_tree / tree_attention_bias / tree_attention_mask，
供 c/tests/compare_tree_attn.c 逐位对拍。
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


SEQS = (
    ("tree_basic", [[1, 2, 3], [1, 2, 4], [5, 6]]),
    ("tree_single", [[7]]),
    ("tree_dup", [[1, 2], [1, 2]]),
    ("tree_deep", [[1, 2, 3, 4], [1, 2, 5], [9]]),
    ("tree_roots", [[10, 11], [12, 13], [10, 14]]),
)
for name, seqs in SEQS:
    tree = AI.build_draft_tree(seqs)
    flat = [t for s in seqs for t in s]
    lens = [len(s) for s in seqs]
    bias = AI.tree_attention_bias(tree["parents"], -1e30)
    mask = AI.tree_attention_mask(tree["parents"])
    add(kind="tree", name=name, seqs=seqs, flat=flat, seq_lens=lens,
        parents=tree["parents"], tokens=tree["tokens"], leaves=tree["leaves"],
        paths=tree["paths"], n_nodes=len(tree["parents"]),
        n_leaves=len(tree["leaves"]), neg_inf=-1e30,
        bias=[v for row in bias for v in row],
        mask=[1 if v else 0 for row in mask for v in row], tol=0.0)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "tree_attn_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
