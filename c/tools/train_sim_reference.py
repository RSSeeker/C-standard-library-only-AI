#!/usr/bin/env python3
"""生成「分布式训练模拟 + 低精度舍入」批次的 C 对拍参考。

覆盖 parallel.h（FSDP/ZeRO/张量并行/序列并行 LayerNorm）与
quant.h（fp16/bf16/fp8 舍入、对称伪量化）。

用法: python tools/train_sim_reference.py train_sim_reference.json
"""
import json
import os
import sys

import _py_path  # noqa: F401  (把 Python 主仓库加入 sys.path)

import AI  # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "train_sim_reference.json"

    flat = [0.1 * i - 0.25 for i in range(10)]
    shards = AI.fsdp_shard_flat(flat, 3)
    gathered = AI.fsdp_all_gather(shards)

    params = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    grads = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    new_params, updates = AI.zero_optimizer_step(params, grads, 0.1, 3)

    rank_grads = [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]]
    rs = AI.fsdp_reduce_scatter(rank_grads, 2)

    W = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    bias = [0.5, -0.5, 1.0]
    x = [0.25, -0.75]
    col_full, col_shards = AI.tensor_parallel_linear_forward(W, bias, x, 3, "column")
    row_full, row_shards = AI.tensor_parallel_linear_forward(W, None, x, 2, "row")

    x_seq = [[0.1, 0.2, 0.3, 0.4], [-1.0, 0.0, 1.0, 2.0]]
    sp_out = AI.sequence_parallel_layer_norm(x_seq, eps=1e-5)

    round_vals = [0.0, 0.1, -0.1, 1.0 / 3.0, 123.456, 65504.0, 65000.0,
                  1e-8, -1e-8, 448.0, 500.0, 0.5, -0.5, 2.5, 3.5]
    fp16 = [AI.fp16_round(v) for v in round_vals]
    bf16 = [AI.bf16_round(v) for v in round_vals]
    fp8_e4m3 = [AI.fp8_round(v, "e4m3") for v in round_vals]
    fp8_e5m2 = [AI.fp8_round(v, "e5m2") for v in round_vals]

    fq_vals = [1.0, -2.0, 0.5, -0.25, 3.0]
    fq_auto = AI.fake_quantize(fq_vals, bits=8)
    fq_scaled = AI.fake_quantize(fq_vals, bits=4, scale=0.1)

    payload = {
        "meta": {"source": "train_sim_reference.py", "tolerance": 1e-12},
        "fsdp": {"flat": flat, "world": 3,
                 "shards": [[s, e, vals] for s, e, vals in shards],
                 "gather": gathered},
        "zero": {"params": params, "grads": grads, "lr": 0.1, "world": 3,
                 "out": new_params, "updates": [[s, e] for s, e in updates]},
        "reduce_scatter": {"rank_grads": rank_grads, "world": 2, "n": 4,
                           "out": [v for _, _, vals in rs for v in vals]},
        "tp": {"W": [v for row in W for v in row], "rows": 3, "cols": 2,
               "bias": bias, "has_bias": 1, "x": x, "world": 3,
               "col_full": col_full, "col_sizes": [1, 1, 1],
               "row_full": row_full, "row_world": 2, "row_sizes": [3, 3],
               "row_shards": [v for shard in row_shards for v in shard]},
        "sp_ln": {"x": [v for row in x_seq for v in row], "tokens": 2, "d": 4,
                  "eps": 1e-5, "out": [v for row in sp_out for v in row]},
        "round": {"values": round_vals, "fp16": fp16, "bf16": bf16,
                  "fp8_e4m3": fp8_e4m3, "fp8_e5m2": fp8_e5m2},
        "fq": {"auto": {"values": fq_vals, "bits": 8, "out": fq_auto},
               "scaled": {"values": fq_vals, "bits": 4, "scale": 0.1,
                          "out": fq_scaled}},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
