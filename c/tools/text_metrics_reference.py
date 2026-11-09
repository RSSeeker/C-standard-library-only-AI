# -*- coding: utf-8 -*-
"""从 AI.py 导出文本指标参考值 -> c/text_metrics_reference.json

覆盖 meteor_score / meteor_score_full / chrf_score / word_error_rate /
character_error_rate，供 c/tests/compare_text_metrics.c 逐位（容差 0）对拍。
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


def add(name, func, value, tol=0.0, **kw):
    CASES.append(dict(name=name, func=func, tol=tol, value=value, **kw))


# ---------- meteor_score（简化版，token 数组） ----------
def meteor(name, pred, ref, alpha=0.9, beta=3.0, gamma=0.5):
    add(name, "meteor", AI.meteor_score(pred, ref, alpha, beta, gamma),
        pred=list(pred), ref=list(ref), alpha=alpha, beta=beta, gamma=gamma)


meteor("meteor_identical", ["the", "cat", "sat", "on", "the", "mat"],
       ["the", "cat", "sat", "on", "the", "mat"])
meteor("meteor_one_add", ["the", "cat"], ["the", "cat", "sat"])
meteor("meteor_reordered", ["a", "b", "c"], ["c", "b", "a"])
meteor("meteor_case_diff", ["The", "Cat"], ["the", "cat"])
meteor("meteor_subset", ["a", "b", "c"], ["a", "x", "b", "c"])
meteor("meteor_no_match", ["x", "y"], ["a", "b"])
meteor("meteor_empty_pred", [], ["a"])

# ---------- meteor_score_full（词干 / 同义词） ----------
def meteor_full(name, pred, ref, alpha=0.9, beta=3.0, gamma=0.5,
                synonyms=None, use_stem=True):
    add(name, "meteor_full",
        AI.meteor_score_full(pred, ref, alpha, beta, gamma,
                             synonyms=synonyms, use_stem=use_stem),
        pred=list(pred), ref=list(ref), alpha=alpha, beta=beta, gamma=gamma,
        synonyms=(None if synonyms is None else [list(g) for g in synonyms]),
        use_stem=1 if use_stem else 0)


meteor_full("meteor_stem_on", ["running", "dogs"], ["runs", "dog"], use_stem=True)
meteor_full("meteor_stem_off", ["running", "dogs"], ["runs", "dog"], use_stem=False)
meteor_full("meteor_synonyms", ["car"], ["automobile"],
            synonyms=[["car", "automobile", "vehicle"]])
meteor_full("meteor_synonyms_none", ["car"], ["automobile"])
meteor_full("meteor_mixed_synonyms", ["a", "b"], ["a", "automobile", "b"],
            synonyms=[["car", "automobile"]])
meteor_full("meteor_full_chinese", ["你", "好", "世", "界"], ["你", "好", "世", "界"])
meteor_full("meteor_full_order", ["c", "b", "a"], ["a", "b", "c"])
meteor_full("meteor_full_alpha", ["the", "cat"], ["the", "cat"], alpha=0.5)

# ---------- chrf_score（字符串） ----------
def chrf(name, pred, ref, max_n=6, beta=2.0):
    add(name, "chrf", AI.chrf_score(pred, ref, max_n, beta),
        pred=pred, ref=ref, max_n=max_n, beta=beta)


chrf("chrf_identical", "the cat sat on the mat", "the cat sat on the mat")
chrf("chrf_case", "The Cat", "the cat")
chrf("chrf_ws", "a  b\tc", "a b c")
chrf("chrf_chinese", "你好世界", "你好世界")
chrf("chrf_chinese_partial", "你好世界", "你好")
chrf("chrf_short_ref", "ab", "abcdef", max_n=3)
chrf("chrf_max2", "the cat sat", "the cat sat", max_n=2)
chrf("chrf_beta1", "hello world", "hello there", beta=1.0)

# ---------- chrf_score（token 数组） ----------
def chrf_tok(name, pred, ref, max_n=6, beta=2.0):
    add(name, "chrf_tokens", AI.chrf_score(pred, ref, max_n, beta),
        pred=list(pred), ref=list(ref), max_n=max_n, beta=beta)


chrf_tok("chrf_tok_case", ["Hello", "World"], ["hello", "world"])
chrf_tok("chrf_tok_partial", ["a", "b", "c"], ["a", "b"])
chrf_tok("chrf_tok_empty", [], ["a"])

# ---------- word_error_rate ----------
def wer(name, ref, hyp):
    add(name, "wer", AI.word_error_rate(ref, hyp), ref=ref, hyp=hyp)


def wer_tok(name, ref, hyp):
    add(name, "wer_tokens", AI.word_error_rate(ref, hyp),
        ref=list(ref), hyp=list(hyp))


wer("wer_identical", "the cat sat", "the cat sat")
wer("wer_sub", "a b c", "a x c")
wer("wer_ins", "a b", "a b c")
wer("wer_del", "a b c", "a")
wer("wer_all", "a b c", "x y z")
wer("wer_multi_space", "a   b\tc", "a b c")
wer("wer_chinese", "你好 世界", "你好 世界")
wer_tok("wer_tok_case", ["The", "Cat"], ["the", "cat"])
wer_tok("wer_tok_sub", ["a", "b", "c"], ["a", "x", "c"])

# ---------- character_error_rate ----------
def cer(name, ref, hyp):
    add(name, "cer", AI.character_error_rate(ref, hyp), ref=ref, hyp=hyp)


def cer_tok(name, ref, hyp):
    add(name, "cer_tokens", AI.character_error_rate(ref, hyp),
        ref=list(ref), hyp=list(hyp))


cer("cer_identical", "你好世界", "你好世界")
cer("cer_sub", "你好世界", "你好世")
cer("cer_ins", "你好", "你们好")
cer("cer_empty_hyp", "abc", "")
cer("cer_english", "the cat", "the kat")
cer_tok("cer_tok_sub", ["a", "b", "c"], ["a", "x"])

OUT = os.path.join(C_DIR, "text_metrics_reference.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES}, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT}: {len(CASES)} cases")
