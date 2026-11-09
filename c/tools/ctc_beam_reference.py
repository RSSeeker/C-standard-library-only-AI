# -*- coding: utf-8 -*-
"""从 AI.py 导出 CTC 前缀 beam search 参考值 -> c/ctc_beam_reference.json

覆盖 ctc_prefix_beam_search（直接给 log_probs）与 ctc_beam_decode（给 logits），
包含不同 blank / beam_width / LM 融合 / 长度惩罚 / -inf 表项等情形，供
c/tests/compare_ctc_beam.c 对拍 tokens（精确）与分数（容差 1e-12）。
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


def lm_prefer2(tokens):
    return 5.0 if list(tokens) == [2] else 0.0


def lm_length_bias(tokens):
    return -0.5 * len(tokens) + 0.25


def lm_constant(tokens):
    return 1.25


LM_NAMES = {"prefer2": lm_prefer2, "length_bias": lm_length_bias,
            "constant": lm_constant}

CASES = []


def add(name, func, result, log_probs=None, logits=None, blank=0,
        beam_width=8, lm=None, lm_weight=0.5, length_penalty=0.0):
    CASES.append(dict(
        name=name, func=func, tol=1e-12,
        log_probs=log_probs, logits=logits, blank=blank,
        beam_width=beam_width, lm=lm, lm_weight=lm_weight,
        length_penalty=length_penalty,
        tokens=result["tokens"], score=result["score"],
        ctc_score=result["ctc_score"], lm_score=result["lm_score"]))


def to_log_probs(logits):
    return [AI._asr_log_softmax([float(v) for v in row]) for row in logits]


def prefix(name, logits, blank=0, beam_width=8, lm=None, lm_weight=0.5,
           length_penalty=0.0):
    log_probs = to_log_probs(logits)
    kwargs = {} if lm is None else {"lm": LM_NAMES[lm]}
    result = AI.ctc_prefix_beam_search(
        log_probs, blank=blank, beam_width=beam_width,
        lm_weight=lm_weight, length_penalty=length_penalty, **kwargs)
    add(name, "prefix", result, log_probs=log_probs, blank=blank,
        beam_width=beam_width, lm=lm, lm_weight=lm_weight,
        length_penalty=length_penalty)


def decode(name, logits, blank=0, beam_width=8, lm=None, lm_weight=0.5,
           length_penalty=0.0):
    kwargs = {} if lm is None else {"lm": LM_NAMES[lm]}
    result = AI.ctc_beam_decode(
        logits, blank=blank, beam_width=beam_width,
        lm_weight=lm_weight, length_penalty=length_penalty, **kwargs)
    add(name, "decode", result, logits=logits, blank=blank,
        beam_width=beam_width, lm=lm, lm_weight=lm_weight,
        length_penalty=length_penalty)


# 4 帧、3 类（blank=0）的小例子
base = [
    [2.0, 0.5, -0.5],
    [0.2, 1.8, 0.3],
    [0.1, 0.4, 1.9],
    [1.5, 0.2, 0.1],
]
prefix("prefix_simple", base)
prefix("prefix_blank1", base, blank=1, beam_width=4)
prefix("prefix_beam1", base, beam_width=1)
prefix("prefix_lm_prefer2", base, beam_width=4, lm="prefer2", lm_weight=1.0)
prefix("prefix_lm_length", base, beam_width=4, lm="length_bias",
       lm_weight=0.5, length_penalty=0.3)
prefix("prefix_lm_constant", base, beam_width=4, lm="constant",
       lm_weight=2.0, length_penalty=-0.2)

longer = [
    [1.0, 0.1, 0.2, -1.0],
    [0.2, 1.2, 0.0, 0.3],
    [0.1, 0.2, 1.4, 0.0],
    [0.3, 0.0, 0.1, 1.5],
    [0.0, 0.9, 0.2, 0.1],
    [0.4, 0.1, 1.1, 0.0],
]
prefix("prefix_longer", longer, beam_width=6)
prefix("prefix_longer_blank3", longer, blank=3, beam_width=8)
prefix("prefix_short_t1", [[1.0, -0.5, 0.3]], beam_width=3)
prefix("prefix_neg_inf", [
    [0.0, -1.0, 0.2],
    [1.0, 0.1, 0.3],
    [-0.2, 0.0, 1.1],
], beam_width=5)
prefix("prefix_shape_k4", [
    [0.2, 0.1, 0.0, 1.0],
    [1.1, 0.2, 0.0, 0.1],
    [0.0, 1.2, 0.1, 0.2],
], beam_width=6)

# raw logits 版本（内部做 log_softmax）
decode("decode_simple", base)
decode("decode_lm", base, beam_width=4, lm="prefer2", lm_weight=0.8)
decode("decode_beam2", longer, beam_width=2)
decode("decode_penalty", base, beam_width=4, length_penalty=0.5, lm_weight=0.0)
decode("decode_blank2", longer, blank=2, beam_width=5,
       lm="length_bias", lm_weight=1.0, length_penalty=-0.1)

OUT = os.path.join(C_DIR, "ctc_beam_reference.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES}, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT}: {len(CASES)} cases")
