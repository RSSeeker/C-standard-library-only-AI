# -*- coding: utf-8 -*-
"""导出 BytePairTokenizer 的黄金值，供 C 侧逐位对拍。

语料覆盖面：ASCII 单词 + 标点 + 多空格、换行/制表、CJK、希腊/西里尔字母、
emoji。C 侧 pretokenize 只用一张 Unicode 子表，这里挑的字符都在表内，
或两边都判为「其它符号」（emoji、全角标点）。
"""
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.environ.get("AI_PY_DIR") or os.path.join(
    _HERE, "..", "..", "..", "Python-standard-library-only-AI-main")
sys.path.insert(0, os.path.abspath(_PY))

import AI  # noqa: E402

CASES = [
    ("ascii_words", "the cat sat on the mat. the cat sat.", 300, 2),
    ("punct_spaces", "hello,  world!  foo-bar_baz 42  ok", 290, 2),
    ("cjk", "你好世界，你好世界。再见", 280, 1),
    ("greek_cyrillic", "alpha βήτα γάμμα привет мир", 300, 2),
    ("mixed_emoji", "hi 🙂 there, 你好 x2", 290, 2),
    ("whitespace", "a\tb\nc  d\ne", 265, 1),
    ("single_char", "aaaa", 258, 2),
    ("no_merge", "xyz", 256, 2),
]


def case_dict(name, text, vocab_size, min_frequency):
    tok = AI.BytePairTokenizer()
    learned = tok.train(text, vocab_size=vocab_size, min_frequency=min_frequency)
    merges = [list(m) for m in learned.merges]
    pieces = AI._bpe_pretokenize(text)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # 用学到的规则重建一个新实例，确认「存档 → 加载」等价
    rebuilt = AI.BytePairTokenizer(merges)
    ids2 = rebuilt.encode(text)
    return {
        "name": name,
        "text": text,
        "vocab_size_arg": vocab_size,
        "min_frequency": min_frequency,
        "vocab_size": tok.vocab_size,
        "merges": merges,
        "pieces": pieces,
        "ids": ids,
        "decoded": decoded,
        "ids_rebuilt": ids2,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "text_tok_reference.json"
    cases = [case_dict(*c) for c in CASES]
    payload = {
        "meta": {
            "source": "AI.BytePairTokenizer",
            "note": "C 侧 pretokenize 用 Unicode 子表；本文件挑的字符都在子表内或两边同为『其它』",
        },
        "cases": cases,
    }
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(cases)))
    for c in cases:
        print("  %-16s vocab=%-4d merges=%-4d ids=%-3d pieces=%d"
              % (c["name"], c["vocab_size"], len(c["merges"]), len(c["ids"]),
                 len(c["pieces"])))


if __name__ == "__main__":
    main()
