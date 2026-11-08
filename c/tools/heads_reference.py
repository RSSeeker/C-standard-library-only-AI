# -*- coding: utf-8 -*-
"""导出 heads（TokenClassificationHead / MultiLabelHead / MoE）的参考数据。

用法：
    python tools/heads_reference.py heads_reference.json

参数全部由固定种子生成并写进 JSON，C 侧用 set_params 覆写后逐项比对，
因此不依赖两边的随机数发生器。
"""
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AI_DIR = os.environ.get("AI_PY_DIR") or os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "Python-standard-library-only-AI-main"))
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

import AI  # noqa: E402


def rnd(seed, n, scale=1.0):
    g = random.Random(seed)
    return [g.gauss(0.0, scale) for _ in range(n)]


def mat(seed, rows, cols, scale=1.0):
    g = random.Random(seed)
    return [[g.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def flat(m):
    return [v for row in m for v in row]


def case_token(name, d_model, num_labels, T, mask, seed):
    head = AI.TokenClassificationHead(d_model, num_labels)
    W = mat(seed, num_labels, d_model, 0.7)
    b = rnd(seed + 1, num_labels, 0.3)
    head.W = [list(row) for row in W]
    head.b = list(b)
    head.zero_grad()
    x = mat(seed + 2, T, d_model, 0.9)
    targets = [(seed + i * 7) % num_labels for i in range(T)]
    logits = head.forward(x)
    loss, dlogits = head.loss_grad(logits, targets, mask)
    dx = head.backward(dlogits)
    pred = head.predict(logits, mask)
    return {
        "name": name, "kind": "token",
        "d_model": d_model, "num_labels": num_labels, "T": T,
        "mask": mask, "targets": targets,
        "w": flat(W), "b": list(b), "x": flat(x),
        "logits": flat(logits), "loss": loss, "dlogits": flat(dlogits),
        "dx": flat(dx), "grad_w": flat(head.grad_W), "grad_b": list(head.grad_b),
        "pred": pred, "param_count": head.param_count(),
    }


def case_multilabel(name, fan_in, num_labels, seed):
    head = AI.MultiLabelHead(fan_in, num_labels)
    W = mat(seed, num_labels, fan_in, 0.8)
    b = rnd(seed + 1, num_labels, 0.25)
    head.W = [list(row) for row in W]
    head.b = list(b)
    head.zero_grad()
    x = rnd(seed + 2, fan_in, 1.1)
    probs = head.forward(x)
    targets = [(seed + i) % 2 for i in range(num_labels)]
    loss, dlogits = head.loss_grad(probs, targets)
    dx = head.backward(dlogits)
    pred = head.predict(probs, 0.5)
    return {
        "name": name, "kind": "multilabel",
        "fan_in": fan_in, "num_labels": num_labels,
        "targets": targets,
        "w": flat(W), "b": list(b), "x": list(x),
        "probs": list(probs), "loss": loss, "dlogits": list(dlogits),
        "dx": list(dx), "grad_w": flat(head.grad_W), "grad_b": list(head.grad_b),
        "pred": pred, "param_count": head.param_count(),
    }


def case_moe(name, d_model, d_ff, n_experts, top_k, T, activation, seed):
    moe = AI.MoE(d_model, d_ff, n_experts, top_k, activation)
    expert_params = []
    for e in range(n_experts):
        s = seed + 10 * e
        W1 = mat(s, d_ff, d_model, 0.6)
        b1 = rnd(s + 1, d_ff, 0.2)
        W2 = mat(s + 2, d_model, d_ff, 0.5)
        b2 = rnd(s + 3, d_model, 0.2)
        moe.experts[e].W1 = [list(r) for r in W1]
        moe.experts[e].b1 = list(b1)
        moe.experts[e].W2 = [list(r) for r in W2]
        moe.experts[e].b2 = list(b2)
        expert_params.append({"w1": flat(W1), "b1": list(b1),
                              "w2": flat(W2), "b2": list(b2)})
    W_gate = mat(seed + 100, n_experts, d_model, 0.9)
    b_gate = rnd(seed + 101, n_experts, 0.2)
    moe.W_gate = [list(r) for r in W_gate]
    moe.b_gate = list(b_gate)
    moe.zero_grad()
    x = mat(seed + 200, T, d_model, 1.0)
    dout = mat(seed + 300, T, d_model, 0.7)
    out = moe.forward(x)
    dx = moe.backward(dout)
    routes = [[[e, w] for (e, w) in r] for r in moe.routes]
    return {
        "name": name, "kind": "moe",
        "d_model": d_model, "d_ff": d_ff, "n_experts": n_experts,
        "top_k": top_k, "T": T, "activation": activation,
        "x": flat(x), "dout": flat(dout),
        "w_gate": flat(W_gate), "b_gate": list(b_gate),
        "experts": expert_params,
        "out": flat(out), "dx": flat(dx),
        "grad_w_gate": flat(moe.grad_W_gate), "grad_b_gate": list(moe.grad_b_gate),
        "expert_grads": [{"w1": flat(moe.experts[e].grads["W1"]),
                          "b1": list(moe.experts[e].grads["b1"]),
                          "w2": flat(moe.experts[e].grads["W2"]),
                          "b2": list(moe.experts[e].grads["b2"])}
                         for e in range(n_experts)],
        "routes": routes, "param_count": moe.param_count(),
        "tol": 1e-12,
    }


def main(argv):
    out = argv[1] if len(argv) > 1 else "heads_reference.json"
    cases = []
    cases.append(case_token("token_masked", 3, 3, 4, [1, 1, 0, 1], 11))
    cases.append(case_token("token_full", 4, 2, 3, None, 23))
    cases.append(case_multilabel("multilabel_basic", 3, 4, 31))
    cases.append(case_moe("moe_gelu_k2", 3, 5, 3, 2, 4, "gelu", 41))
    cases.append(case_moe("moe_relu_k1", 4, 6, 2, 1, 3, "relu", 57))
    cases.append(case_moe("moe_linear_kall", 2, 4, 2, 2, 3, "linear", 71))
    data = {"meta": {"source": "AI.py", "tool": "tools/heads_reference.py",
                     "cases": len(cases)}, "cases": cases}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote %s (%d cases)" % (out, len(cases)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
