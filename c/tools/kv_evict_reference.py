# -*- coding: utf-8 -*-
"""从 AI.py 导出 KV cache 淘汰策略参考值 -> c/kv_evict_reference.json

覆盖 streaming_kv_keep / h2o_kv_keep / snapkv_kv_keep / kv_cache_evict
（policy=streaming/h2o/snapkv），供 c/tests/compare_kv_evict.c 逐位对拍。
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


# ---------------- streaming_kv_keep ----------------
for name, seq_len, budget, sink in (
    ("stream_5_3_1", 5, 3, 1),
    ("stream_8_4_2", 8, 4, 2),
    ("stream_6_6_0", 6, 6, 0),
    ("stream_4_2_1", 4, 2, 1),
    ("stream_10_5_2", 10, 5, 2),
):
    add(kind="streaming", name=name, seq_len=seq_len, budget=budget, sink=sink,
        keep=AI.streaming_kv_keep(seq_len, budget, sink=sink))

# ---------------- h2o_kv_keep ----------------
S2x5 = [[0.1, 0.0, 0.3, 0.2, 0.0], [0.0, 0.4, 0.0, 0.1, 0.5]]
S3x7 = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2],
        [0.0, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0]]
for name, scores, budget, sink, recent in (
    ("h2o_2x5_b3_s1_r0", S2x5, 3, 1, 0),
    ("h2o_2x5_b3_s1_r1", S2x5, 3, 1, 1),
    ("h2o_2x5_b4_s1_r0", S2x5, 4, 1, 0),
    ("h2o_2x5_b2_s0_r0", S2x5, 2, 0, 0),
    ("h2o_3x7_b4_s1_r2", S3x7, 4, 1, 2),
    ("h2o_3x7_b3_s0_r0_ties", S3x7, 3, 0, 0),
):
    add(kind="h2o", name=name, scores=scores, nq=len(scores),
        seq_len=len(scores[0]), budget=budget, sink=sink, recent=recent,
        keep=AI.h2o_kv_keep(scores, budget, sink=sink, recent=recent))

# ---------------- snapkv_kv_keep ----------------
S8x6 = [[0.01 * (r + 1) + 0.001 * j for j in range(6)] for r in range(8)]
for name, scores, budget, sink, obs in (
    ("snapkv_8x6_b3_s1_o2", S8x6, 3, 1, 2),
    ("snapkv_8x6_b4_s1_o1", S8x6, 4, 1, 1),
    ("snapkv_8x6_b3_s1_auto", S8x6, 3, 1, None),
    ("snapkv_8x6_b6_s2_auto", S8x6, 6, 2, None),
):
    add(kind="snapkv", name=name, scores=scores, nq=len(scores),
        seq_len=len(scores[0]), budget=budget, sink=sink,
        obs=-1 if obs is None else obs,
        keep=AI.snapkv_kv_keep(scores, budget, sink=sink, obs=obs))

# ---------------- kv_cache_evict ----------------
S8x8 = [[0.01 * (r + 1) + 0.001 * j for j in range(8)] for r in range(8)]
K5 = [[1.0], [2.0], [3.0], [4.0], [5.0]]
V5 = [[10.0], [20.0], [30.0], [40.0], [50.0]]
K5b = [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5]]
V5b = [[-1.0, -0.5], [-2.0, -1.0], [-3.0, -1.5], [-4.0, -2.0], [-5.0, -2.5]]
for name, keys, values, budget, policy, sink, recent, obs, scores in (
    ("evict_stream_5_b2", K5, V5, 2, "streaming", 1, 0, None, None),
    ("evict_stream_5_b5", K5, V5, 5, "streaming", 1, 0, None, None),
    ("evict_stream_5_b2_dim2", K5b, V5b, 2, "streaming", 1, 0, None, None),
    ("evict_h2o_2x5_b3", K5b, V5b, 3, "h2o", 1, 0, None, S2x5),
    ("evict_h2o_2x5_b3_recent", K5b, V5b, 3, "h2o", 1, 1, None, S2x5),
    ("evict_snapkv_8_b3", [[float(i)] for i in range(8)],
     [[100.0 * i] for i in range(8)], 3, "snapkv", 1, 0, 2, S8x8),
):
    out = AI.kv_cache_evict(keys, values, budget, scores=scores, policy=policy,
                            sink=sink, recent=recent, obs=obs)
    add(kind="evict", name=name, keys=keys, values=values, seq_len=len(keys),
        dim=len(keys[0]), budget=budget, policy=policy, sink=sink,
        recent=recent, obs=-1 if obs is None else obs,
        scores=scores if scores is not None else [],
        nq=len(scores) if scores else 0,
        kept_indices=out["kept_indices"], evicted=out["evicted"],
        new_keys=out["keys"], new_values=out["values"])


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "kv_evict_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
