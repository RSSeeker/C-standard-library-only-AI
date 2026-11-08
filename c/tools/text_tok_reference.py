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


# ---------------------------------------------------------------- WordPiece

WP_TRAIN = [
    ("wp_ascii", "playing played player play plays", 40, 1, "play"),
    ("wp_cjk", "\u4f60\u597d\u4e16\u754c\uff0c\u4f60\u597d\u4e16\u754c\u3002\u518d\u89c1", 30, 1, "\u4f60\u597d"),
    ("wp_minfreq", "the cat sat on the mat", 40, 100, "the"),
    ("wp_mixed", "hi \U0001f642 there, \u4f60\u597d x2", 34, 2, "hi"),
]

WP_FIXED = [
    ("wp_fixed",
     ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "##b", " ", "!", "ab"],
     ["ab a!", "a b", "  ", "zz", "aa"]),
]


def wp_train_case(name, text, vocab_size, min_frequency, second):
    tok = AI.WordPieceTokenizer()
    tok.train(text, vocab_size=vocab_size, min_frequency=min_frequency)
    ids = tok.encode(text)
    small = tok.encode(text, max_chars_per_word=2)
    pair = tok.encode_pair(text, second)
    return {
        "name": name,
        "mode": "train",
        "text": text,
        "vocab_size_arg": vocab_size,
        "min_frequency": min_frequency,
        "tokens": list(tok.tokens),
        "ids": ids,
        "decoded": tok.decode(ids),
        "ids_small": small,
        "decoded_small": tok.decode(small),
        "second": second,
        "pair": pair,
        "pair_decoded": tok.decode(pair),
        "unk_id": tok.unk_id,
        "cls_id": tok.cls_id,
        "sep_id": tok.sep_id,
    }


def wp_fixed_case(name, tokens, probes):
    tok = AI.WordPieceTokenizer(tokens)
    encodes = []
    for text in probes:
        ids = tok.encode(text)
        encodes.append({"text": text, "ids": ids, "decoded": tok.decode(ids)})
    ids_seq = [5, 6, 6, 7, 6, 8, 9]
    return {
        "name": name,
        "mode": "fixed",
        "tokens": list(tokens),
        "encodes": encodes,
        "decode_ids": ids_seq,
        "decode_text": tok.decode(ids_seq),
        "unk_id": tok.unk_id,
    }


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "text_tok_reference.json"
    cases = [case_dict(*c) for c in CASES]
    wp_cases = [wp_train_case(*c) for c in WP_TRAIN]
    wp_cases += [wp_fixed_case(*c) for c in WP_FIXED]
    payload = {
        "wordpiece": wp_cases,
        "meta": {
            "source": "AI.BytePairTokenizer",
            "note": "C 侧 pretokenize 用 Unicode 子表；本文件挑的字符都在子表内或两边同为『其它』",
        },
        "cases": cases,
    }
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote %s (%d cases)" % (out_path, len(cases) + len(wp_cases)))
    for c in cases:
        print("  %-16s vocab=%-4d merges=%-4d ids=%-3d pieces=%d"
              % (c["name"], c["vocab_size"], len(c["merges"]), len(c["ids"]),
                 len(c["pieces"])))
    print("  wordpiece: %d cases" % len(wp_cases))
    for c in wp_cases:
        if c["mode"] == "train":
            print("  %-16s vocab=%-4d ids=%-3d small=%-3d" %
                  (c["name"], len(c["tokens"]), len(c["ids"]), len(c["ids_small"])))
        else:
            print("  %-16s vocab=%-4d probes=%d" %
                  (c["name"], len(c["tokens"]), len(c["encodes"])))


if __name__ == "__main__":
    main()
