# -*- coding: utf-8 -*-
"""导出 CausalLM 的参考数值，供 C 实现（tests/compare_llm.c）逐位比对。

用法：python c/tools/llm_reference.py [输出路径]
默认写到 c/llm_reference.json

不写死任何权重：先用固定种子在 Python 侧建模型，把它的初始参数一起导出，
C 侧再灌进去 —— 两边从完全相同的起点出发，任何差异都只能来自实现本身。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_here = os.path.dirname(os.path.abspath(__file__))
AI_PY_DIR = os.environ.get("AI_PY_DIR") or os.path.abspath(
    os.path.join(_here, "..", "..", "..", "Python-standard-library-only-AI-main"))
if not os.path.isfile(os.path.join(AI_PY_DIR, "AI.py")):
    sys.exit(f"找不到 Python 版 AI.py，请设置环境变量 AI_PY_DIR（当前: {AI_PY_DIR}）")
sys.path.insert(0, AI_PY_DIR)

import random  # noqa: E402

import AI  # noqa: E402


def flat(x):
    """把嵌套 list 按行主序摊平。"""
    if isinstance(x, list):
        out = []
        for row in x:
            out.extend(flat(row))
        return out
    return [float(x)]


def blocks(model):
    return [(name, param, grad) for name, param, grad in model._named_params()]


def snapshot(model):
    return {name: flat(p) for name, p, _g in blocks(model)}


def set_params(model, values):
    for name, param, _g in blocks(model):
        vals = values[name]
        if isinstance(param[0], list):
            k = 0
            for i in range(len(param)):
                for j in range(len(param[i])):
                    param[i][j] = vals[k]
                    k += 1
        else:
            for i in range(len(param)):
                param[i] = vals[i]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.abspath(os.path.join(_here, "..")), "llm_reference.json")

    cfg = dict(vocab_size=11, d_model=8, num_heads=2, d_ff=16,
               num_layers=1, max_len=8, dropout=0.0)

    random.seed(20240101)
    model = AI.CausalLM(**cfg)
    model.eval()
    init = snapshot(model)

    out = {"config": cfg, "init": init}

    # 1) 纯前向的 logits（只有乘加，应当逐位相同）
    prompt = [2, 3, 4]
    out["forward_ids"] = prompt
    out["forward_logits"] = [flat(row) for row in model.forward(prompt)]

    # 2) 一次前向+反向的 loss 与梯度
    seq = [1, 4, 2, 5, 3]
    model.zero_grad()
    out["loss_ids"] = seq
    out["loss"] = model._loss_and_backward(seq)
    out["grads"] = {name: flat(g) for name, _p, g in blocks(model)}

    # 3) 一步 Adam 之后的参数
    lr1 = 0.01
    model.adam_step(lr1)
    out["adam_lr"] = lr1
    out["after_adam"] = snapshot(model)

    # 4) 从初始权重出发训练 3 个 epoch（shuffle=False，顺序确定）
    fit_seqs = [[1, 4, 2, 5, 3], [2, 3, 4, 5, 6]]
    fit_epochs, fit_lr = 3, 0.01
    model2 = AI.CausalLM(**cfg)
    set_params(model2, init)
    out["fit_seqs"] = fit_seqs
    out["fit_epochs"] = fit_epochs
    out["fit_lr"] = fit_lr
    history = []
    per_epoch = []
    for _ in range(fit_epochs):
        history.extend(model2.fit(fit_seqs, epochs=1, lr=fit_lr, shuffle=False))
        per_epoch.append(snapshot(model2))
    out["fit_history"] = history
    out["after_fit"] = snapshot(model2)
    out["per_epoch_fit"] = per_epoch
    out["eval_loss"] = model2.evaluate(fit_seqs)
    out["perplexity"] = model2.perplexity(fit_seqs)

    # 5) 贪心生成（top_k=1 时与随机数无关，两边必然一致）
    model3 = AI.CausalLM(**cfg)
    set_params(model3, init)
    gen_prompt, gen_new = [2, 3], 4
    out["gen_prompt"] = gen_prompt
    out["gen_new"] = gen_new
    out["gen_tokens"] = model3.generate(gen_prompt, max_new_tokens=gen_new,
                                        top_k=1)

    # 6) top_p 的确定性对拍：p_max >= 1/V >> 1e-9，nucleus 只含 argmax，
    #    因此采样结果与 RNG 无关，必须与 top_k=1 的贪心序列相同。
    #    （任意 top_p 下的随机 token 无法跨语言逐位比较：C 用 splitmix64，
    #      Python 用 random.MT19937，消耗的随机数不同。）
    out["gen_top_p"] = 1e-9
    out["gen_tokens_top_p"] = model3.generate(gen_prompt, max_new_tokens=gen_new,
                                              top_k=0, top_p=1e-9)

    # 7) 生成罚分（repetition / presence / frequency / no_repeat_ngram）的确定性对拍。
    #    top_k=1 时采样退化成 argmax，与 RNG 无关；每条用例同时导出 use_cache=True
    #    与 False 两种结果，C 侧分别与 llm_generate_cached_ex / llm_generate_ex 比较。
    penalty_cases = [
        dict(name="rp_gt1", prompt=[2, 3], max_new=4,
             kwargs=dict(repetition_penalty=2.0)),
        dict(name="rp_lt1", prompt=[2, 3], max_new=8,
             kwargs=dict(repetition_penalty=0.5)),
        dict(name="presence", prompt=[2, 3], max_new=4,
             kwargs=dict(presence_penalty=2.0)),
        dict(name="frequency", prompt=[2, 3], max_new=4,
             kwargs=dict(frequency_penalty=2.0)),
        # prompt 里已经有 "1 1"，ngram=2 时第一步就必须禁止 token 1。
        dict(name="ngram2", prompt=[1, 1], max_new=5,
             kwargs=dict(no_repeat_ngram_size=2)),
        dict(name="combined", prompt=[2, 3, 2], max_new=6,
             kwargs=dict(repetition_penalty=2.0, presence_penalty=2.0,
                         frequency_penalty=2.0, no_repeat_ngram_size=3,
                         min_p=0.05)),
    ]
    out["gen_penalty_cases"] = []
    for case in penalty_cases:
        rec = {key: case[key] for key in ("name", "prompt", "max_new")}
        rec["kwargs"] = case["kwargs"]
        rec["tokens_full"] = model3.generate(
            case["prompt"], max_new_tokens=case["max_new"], top_k=1,
            use_cache=False, **case["kwargs"])
        rec["tokens_cached"] = model3.generate(
            case["prompt"], max_new_tokens=case["max_new"], top_k=1,
            use_cache=True, **case["kwargs"])
        if rec["tokens_full"] != rec["tokens_cached"]:
            sys.exit(f"Python cache/full 不一致: {case['name']}")
        out["gen_penalty_cases"].append(rec)

    # 8) min_p 截尾后的候选集合大小。直接用上面已导出的最后一行 logits 计算，
    #    这样 C 侧读到的是同一批 double，计数差异只可能来自算法实现。
    probe_logits = out["forward_logits"][-1]
    probe_values = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    probe_counts = []
    for mp in probe_values:
        masked = AI._mask_min_p(probe_logits, mp)
        probe_counts.append(sum(1 for v in masked if v != float("-inf")))
    out["min_p_probe_logits"] = probe_logits
    out["min_p_probe_values"] = probe_values
    out["min_p_probe_counts"] = probe_counts

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
