# -*- coding: utf-8 -*-
"""从 AI.py 导出 FIM / T5 span corruption 参考值 -> c/fim_reference.json

覆盖：
  fim_transform / fim_target_mask / fim_batch
  span_corruption_mask / span_corruption_ids

随机 span 用 Python `random.Random(seed)` 抽样；C 侧内置同款 MT19937，
因此 span 起点/长度可以逐位对拍（见 c/tests/compare_fim.c）。
Python 的 seed=None 走 urandom、不可复现，C 侧要求显式 seed，
所以这里所有 span 用例都写明确的非负 seed。

错误用例统一写成 {"kind": "error", "op": ..., "args": {...}}；
导出时先验证 Python 侧确实会 raise，避免记录「假错误」。
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


def add_transform(name, ids, vocab_size, mode="psm", prefix_ratio=0.5,
                  middle_ratio=0.5):
    ex = AI.fim_transform(ids, vocab_size, mode=mode,
                          prefix_ratio=prefix_ratio,
                          middle_ratio=middle_ratio)
    add(kind="fim_transform", name=name, ids=list(ids), vocab_size=vocab_size,
        mode=mode, prefix_ratio=prefix_ratio, middle_ratio=middle_ratio,
        input_ids=ex["input_ids"], prefix=ex["prefix"], middle=ex["middle"],
        suffix=ex["suffix"], middle_start=ex["middle_start"])


def add_target_mask(name, ids, vocab_size, mode="psm", prefix_ratio=0.5,
                    middle_ratio=0.5):
    ex = AI.fim_transform(ids, vocab_size, mode=mode,
                          prefix_ratio=prefix_ratio,
                          middle_ratio=middle_ratio)
    mask = AI.fim_target_mask(ex)
    add(kind="target_mask", name=name, ids=list(ids), vocab_size=vocab_size,
        mode=mode, prefix_ratio=prefix_ratio, middle_ratio=middle_ratio,
        mask=[1 if v else 0 for v in mask])


def add_batch(name, seqs, vocab_size, mode="psm", prefix_ratio=0.5,
              middle_ratio=0.5):
    flat = [t for s in seqs for t in s]
    lens = [len(s) for s in seqs]
    out = AI.fim_batch(seqs, vocab_size, mode=mode, prefix_ratio=prefix_ratio,
                       middle_ratio=middle_ratio)
    add(kind="batch", name=name, flat=flat, seq_lens=lens,
        vocab_size=vocab_size, mode=mode, prefix_ratio=prefix_ratio,
        middle_ratio=middle_ratio,
        examples=[{
            "input_ids": e["input_ids"], "prefix": e["prefix"],
            "middle": e["middle"], "suffix": e["suffix"],
            "middle_start": e["middle_start"],
        } for e in out])


def add_span_mask(name, seq_len, mask_ratio, mean_span, seed):
    spans = AI.span_corruption_mask(seq_len, mask_ratio=mask_ratio,
                                    mean_span=mean_span, seed=seed)
    add(kind="span_mask", name=name, seq_len=seq_len, mask_ratio=mask_ratio,
        mean_span=mean_span, seed=seed,
        spans=[[a, b] for a, b in spans])


def add_span_ids(name, ids, vocab_size, mask_ratio, mean_span, sentinel_id,
                 seed):
    out = AI.span_corruption_ids(ids, vocab_size, mask_ratio=mask_ratio,
                                 mean_span=mean_span, sentinel_id=sentinel_id,
                                 seed=seed)
    add(kind="span_ids", name=name, ids=list(ids), vocab_size=vocab_size,
        mask_ratio=mask_ratio, mean_span=mean_span, sentinel_id=sentinel_id,
        seed=seed, input_ids=out["input_ids"], target_ids=out["target_ids"],
        spans=[[a, b] for a, b in out["spans"]],
        sentinel=out["sentinel_id"])


def add_error(name, op, args, call):
    """记录一个错误用例；先在 Python 侧确认它确实会 raise。"""
    try:
        call()
    except (ValueError, TypeError) as exc:
        pass
    else:
        raise SystemExit("错误用例 %s 在 Python 侧没有 raise" % name)
    add(kind="error", name=name, op=op, args=args)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "fim_reference.json")

    # ---------------- fim_transform：主用例 ----------------
    add_transform("psm_basic", [5, 6, 7, 8, 9, 10], 16)
    add_transform("spm_basic", [5, 6, 7, 8, 9, 10], 16, mode="spm")
    add_transform("psm_min3", [1, 2, 3], 5)
    add_transform("spm_min3", [1, 2, 3], 5, mode="spm")
    add_transform("psm_n4", [3, 4, 5, 6], 8)
    add_transform("psm_vocab2", [0, 1, 0, 1, 1], 2)
    add_transform("psm_prefix_hi", [1, 2, 3, 4, 5], 8, prefix_ratio=0.9)
    add_transform("psm_prefix_lo", [1, 2, 3, 4, 5, 6, 7], 8,
                  prefix_ratio=0.1)
    add_transform("psm_middle_hi", [2, 3, 4, 5, 6, 7], 9, middle_ratio=0.9)
    add_transform("psm_middle_lo", [2, 3, 4, 5, 6, 7], 9, middle_ratio=0.1)
    add_transform("psm_quarter", [10, 11, 12, 13, 14, 15, 16, 17], 24,
                  prefix_ratio=0.25, middle_ratio=0.25)
    add_transform("spm_quarter", [10, 11, 12, 13, 14, 15, 16, 17], 24,
                  mode="spm", prefix_ratio=0.25, middle_ratio=0.75)
    add_transform("psm_long", list(range(20)), 32, prefix_ratio=0.6,
                  middle_ratio=0.4)
    # 3 块钱序列：p = round(3*0.99)=3 -> clamp1；m 同理
    add_transform("psm_clamp", [7, 8, 9], 10, prefix_ratio=0.99,
                  middle_ratio=0.99)

    # ---------------- fim_target_mask ----------------
    add_target_mask("mask_psm_basic", [5, 6, 7, 8, 9, 10], 16)
    add_target_mask("mask_spm_basic", [5, 6, 7, 8, 9, 10], 16, mode="spm")
    add_target_mask("mask_min3", [1, 2, 3], 5)
    add_target_mask("mask_quarter", [10, 11, 12, 13, 14, 15, 16, 17], 24,
                    prefix_ratio=0.25, middle_ratio=0.25)

    # ---------------- fim_batch ----------------
    add_batch("batch_3seq", [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14, 15]], 20)
    add_batch("batch_spm", [[1, 2, 3, 4], [5, 6, 7, 8]], 12, mode="spm",
              prefix_ratio=0.25, middle_ratio=0.5)
    add_batch("batch_single", [[2, 3, 4]], 6)

    # ---------------- span_corruption_mask ----------------
    SPAN_MASKS = (
        ("span_s1", 1, 0.15, 3.0, 0),
        ("span_s2", 2, 0.15, 3.0, 1),
        ("span_s5", 5, 0.15, 3.0, 7),
        ("span_s13", 13, 0.15, 3.0, 42),
        ("span_s16_r30", 16, 0.3, 3.0, 12345),
        ("span_s16_r50", 16, 0.5, 3.0, 20241106),
        ("span_s31_mean1", 31, 0.15, 1.0, 7),
        ("span_s31_mean2", 31, 0.15, 2.0, 7),
        ("span_s31_mean5", 31, 0.15, 5.5, 7),
        ("span_s64_full", 64, 1.0, 3.0, 42),
        ("span_s64_r15", 64, 0.15, 3.0, 0),
        ("span_s100", 100, 0.15, 3.0, 1099511627776),
        ("span_seed0", 16, 0.25, 2.0, 0),
        ("span_seed1", 16, 0.25, 2.0, 1),
        ("span_seed7", 16, 0.25, 2.0, 7),
        ("span_seed42", 16, 0.25, 2.0, 42),
        ("span_bigseed", 24, 0.2, 2.5, 4294967297),
        ("span_longseq", 200, 0.15, 3.0, 777),
    )
    for name, seq_len, ratio, mean_span, seed in SPAN_MASKS:
        add_span_mask(name, seq_len, ratio, mean_span, seed)

    # ---------------- span_corruption_ids ----------------
    add_span_ids("span_ids_default_sentinel", list(range(10, 30)), 32,
                 0.15, 3.0, None, 7)
    add_span_ids("span_ids_explicit_sentinel", list(range(10, 30)), 40,
                 0.15, 3.0, 39, 7)
    add_span_ids("span_ids_dense", list(range(1, 13)), 16, 0.5, 2.0, 15, 42)
    add_span_ids("span_ids_short", [3, 4, 5], 8, 0.3, 1.0, None, 0)
    add_span_ids("span_ids_full", [1, 2, 3, 4, 5], 8, 1.0, 3.0, None, 12345)
    add_span_ids("span_ids_split", [9, 8, 7, 6, 5, 4, 3, 2, 1], 16, 0.3,
                 2.5, 3, 20241106)

    # ---------------- 错误用例 ----------------
    add_error("err_transform_empty", "fim_transform",
              {"ids": [], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([], 8))
    add_error("err_transform_vocab1", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 1, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 1))
    add_error("err_transform_id_oor", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 3, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 3))
    add_error("err_transform_id_neg", "fim_transform",
              {"ids": [-1, 0, 1], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([-1, 0, 1], 8))
    add_error("err_transform_too_short", "fim_transform",
              {"ids": [1, 2], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2], 8))
    add_error("err_transform_mode", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 7,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 8, mode="bogus"))
    add_error("err_transform_prefix0", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.0, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 8, prefix_ratio=0.0))
    add_error("err_transform_prefix1", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 1.0, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 8, prefix_ratio=1.0))
    add_error("err_transform_prefix15", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 1.5, "middle_ratio": 0.5},
              lambda: AI.fim_transform([1, 2, 3], 8, prefix_ratio=1.5))
    add_error("err_transform_middle0", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.0},
              lambda: AI.fim_transform([1, 2, 3], 8, middle_ratio=0.0))
    add_error("err_transform_middle_hi", "fim_transform",
              {"ids": [1, 2, 3], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 2.0},
              lambda: AI.fim_transform([1, 2, 3], 8, middle_ratio=2.0))
    add_error("err_target_empty", "fim_target_mask",
              {"input_ids": [], "middle_start": 0},
              lambda: AI.fim_target_mask({"input_ids": [], "middle_start": 0}))
    add_error("err_target_start_hi", "fim_target_mask",
              {"input_ids": [1, 2, 3], "middle_start": 3},
              lambda: AI.fim_target_mask({"input_ids": [1, 2, 3],
                                          "middle_start": 3}))
    add_error("err_target_start_neg", "fim_target_mask",
              {"input_ids": [1, 2, 3], "middle_start": -1},
              lambda: AI.fim_target_mask({"input_ids": [1, 2, 3],
                                          "middle_start": -1}))
    add_error("err_batch_empty", "fim_batch",
              {"flat": [], "seq_lens": [], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_batch([], 8))
    add_error("err_batch_short_seq", "fim_batch",
              {"flat": [1, 2], "seq_lens": [2], "vocab_size": 8, "mode": 0,
               "prefix_ratio": 0.5, "middle_ratio": 0.5},
              lambda: AI.fim_batch([[1, 2]], 8))
    add_error("err_mask_seq0", "span_corruption_mask",
              {"seq_len": 0, "mask_ratio": 0.15, "mean_span": 3.0, "seed": 0},
              lambda: AI.span_corruption_mask(0, seed=0))
    add_error("err_mask_ratio0", "span_corruption_mask",
              {"seq_len": 8, "mask_ratio": 0.0, "mean_span": 3.0, "seed": 0},
              lambda: AI.span_corruption_mask(8, mask_ratio=0.0, seed=0))
    add_error("err_mask_ratio_hi", "span_corruption_mask",
              {"seq_len": 8, "mask_ratio": 1.5, "mean_span": 3.0, "seed": 0},
              lambda: AI.span_corruption_mask(8, mask_ratio=1.5, seed=0))
    add_error("err_mask_mean_lo", "span_corruption_mask",
              {"seq_len": 8, "mask_ratio": 0.15, "mean_span": 0.5, "seed": 0},
              lambda: AI.span_corruption_mask(8, mean_span=0.5, seed=0))
    add_error("err_span_ids_vocab1", "span_corruption_ids",
              {"ids": [1, 2, 3], "vocab_size": 1, "mask_ratio": 0.15,
               "mean_span": 3.0, "sentinel_id": None, "seed": 0},
              lambda: AI.span_corruption_ids([1, 2, 3], 1, seed=0))
    add_error("err_span_ids_sentinel_hi", "span_corruption_ids",
              {"ids": [1, 2, 3], "vocab_size": 8, "mask_ratio": 0.15,
               "mean_span": 3.0, "sentinel_id": 8, "seed": 0},
              lambda: AI.span_corruption_ids([1, 2, 3], 8, sentinel_id=8,
                                             seed=0))
    add_error("err_span_ids_sentinel_neg", "span_corruption_ids",
              {"ids": [1, 2, 3], "vocab_size": 8, "mask_ratio": 0.15,
               "mean_span": 3.0, "sentinel_id": -5, "seed": 0},
              lambda: AI.span_corruption_ids([1, 2, 3], 8, sentinel_id=-5,
                                             seed=0))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    n_err = sum(1 for c in CASES if c["kind"] == "error")
    print("wrote %s (%d cases, %d error cases)" % (out_path, len(CASES), n_err))


if __name__ == "__main__":
    main()
