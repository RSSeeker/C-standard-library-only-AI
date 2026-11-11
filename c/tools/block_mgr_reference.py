# -*- coding: utf-8 -*-
"""从 AI.py 导出 PagedBlockManager 参考操作序列 -> c/block_mgr_reference.json

每个用例是一串操作（含参数与期望输出）；compare 程序按同样顺序执行 C 版本
并逐步比较。
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


def run(num_blocks, block_size, swap_capacity, ops):
    m = AI.PagedBlockManager(num_blocks, block_size, swap_capacity)
    steps = []
    for op in ops:
        kind = op[0]
        if kind == "new_request":
            steps.append({"op": kind, "rid": m.new_request()})
        elif kind == "allocate":
            _, rid, count = op
            steps.append({"op": kind, "rid": rid, "count": count,
                          "table": m.allocate(rid, count)})
        elif kind == "ensure":
            _, rid, tokens = op
            steps.append({"op": kind, "rid": rid, "tokens": tokens,
                          "table": m.ensure(rid, tokens)})
        elif kind == "free":
            _, rid = op
            steps.append({"op": kind, "rid": rid, "released": m.free(rid)})
        elif kind == "fork":
            _, src, dst = op
            got = m.fork(src, None if dst < 0 else dst)
            steps.append({"op": kind, "src": src, "dst_arg": dst, "dst": got})
        elif kind == "cow":
            _, rid, index = op
            steps.append({"op": kind, "rid": rid, "index": index,
                          "block": m.cow(rid, index)})
        elif kind == "swap_out":
            _, rid = op
            steps.append({"op": kind, "rid": rid, "released": m.swap_out(rid)})
        elif kind == "swap_in":
            _, rid = op
            mp = m.swap_in(rid)
            keys = list(mp.keys())
            steps.append({"op": kind, "rid": rid, "old": keys,
                          "new": [mp[k] for k in keys]})
        elif kind == "stats":
            st = m.stats()
            steps.append({"op": kind, "stats": {k: st[k] for k in (
                "used_blocks", "free_blocks", "shared_blocks", "swapped_blocks",
                "requests", "copy_on_write")}})
        elif kind == "tables":
            flat = []
            for rid, tbl in sorted(m._tables.items()):
                flat.append(rid)
                flat.append(len(tbl))
                flat.extend(tbl)
            steps.append({"op": kind, "flat": flat})
        else:
            raise ValueError(kind)
    return steps


OPS1 = [
    ("new_request",),
    ("ensure", 0, 3),
    ("allocate", 0, 2),
    ("fork", 0, -1),
    ("cow", 1, 0),
    ("cow", 1, 1),
    ("stats",),
    ("free", 0),
    ("stats",),
    ("swap_out", 1),
    ("stats",),
    ("swap_in", 1),
    ("stats",),
    ("tables",),
    ("free", 1),
    ("stats",),
]

OPS2 = [
    ("new_request",),
    ("ensure", 0, 4),
    ("stats",),
    ("swap_out", 0),
    ("stats",),
    ("swap_in", 0),
    ("stats",),
    ("free", 0),
    ("stats",),
]


def main():
    CASES.append({"name": "basic", "num_blocks": 6, "block_size": 2,
                  "swap_capacity": -1, "ops": run(6, 2, None, OPS1)})
    CASES.append({"name": "swap_capacity", "num_blocks": 4, "block_size": 2,
                  "swap_capacity": 1, "ops": run(4, 2, 1, OPS2)})
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        C_DIR, "block_mgr_reference.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": CASES}, f, indent=2, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(CASES)))


if __name__ == "__main__":
    main()
