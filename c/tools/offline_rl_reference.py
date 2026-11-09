"""C 侧离线 RL（IQL / CQL）对拍参考：从 AI.py 生成确定性用例。

产物 c/offline_rl_reference.json，供 tests/compare_offline_rl.c 使用：
  - 初始权重（q / v / policy；目标网络由 C 侧从在线网络复制）
  - 一批确定性转移（states / actions / rewards / next_states / dones）
  - 3 次 update 的损失序列
  - 更新后的 q logits / v / policy probs（两个探测状态）
  - param_count

用法：python tools/offline_rl_reference.py offline_rl_reference.json
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_REPO = r"D:\vscode\Python-standard-library-only-AI-main"
if PY_REPO not in sys.path:
    sys.path.insert(0, PY_REPO)

import AI  # noqa: E402


def dump_mlp(model):
    """MLP -> [{"w": [[...]], "b": [...]}]，与 C 的层顺序一致。"""
    layers = []
    for layer in model.layers:
        layers.append({
            "w": [[float(v) for v in row] for row in layer.weights],
            "b": [float(v) for v in layer.biases],
        })
    return layers


def make_iql_case():
    states = [
        [0.10, -0.20, 0.30, 0.40],
        [0.50, 0.60, -0.70, 0.80],
        [-0.90, 0.15, 0.25, -0.35],
        [0.05, 0.45, -0.55, 0.65],
    ]
    next_states = [
        [0.11, -0.19, 0.31, 0.39],
        [0.51, 0.61, -0.71, 0.81],
        [-0.89, 0.16, 0.26, -0.34],
        [0.06, 0.46, -0.54, 0.66],
    ]
    batch = {
        "states": states,
        "actions": [0, 1, 2, 1],
        "rewards": [1.0, -0.5, 0.25, 2.0],
        "next_states": next_states,
        "dones": [0, 1, 0, 0],
    }
    agent = AI.IQLAgent(4, 3, hidden=(8,), actor_lr=0.01, critic_lr=0.005,
                        tau=0.7, beta=2.5, gamma=0.9, target_tau=0.2, seed=11)
    init = {
        "q": dump_mlp(agent.q.model),
        "v": dump_mlp(agent.v.model),
        "policy": dump_mlp(agent.policy.model),
    }
    step_losses = []
    for _ in range(3):
        losses = agent.update(batch["states"], batch["actions"], batch["rewards"],
                              batch["next_states"], batch["dones"])
        step_losses.append({k: float(v) for k, v in losses.items()})
    final = {"q": [], "v": [], "probs": []}
    for s in (states[0], states[2]):
        final["q"].append([float(v) for v in agent.q.logits(s)])
        final["v"].append(float(agent.v.value(s)))
        final["probs"].append([float(v) for v in agent.policy.probs(s)])
    return {
        "name": "iql",
        "kind": "iql",
        "state_dim": 4,
        "num_actions": 3,
        "hidden": [8],
        "actor_lr": 0.01,
        "critic_lr": 0.005,
        "tau": 0.7,
        "beta": 2.5,
        "gamma": 0.9,
        "target_tau": 0.2,
        "alpha": 0.0,
        "batch": batch,
        "steps": 3,
        "init": init,
        "step_losses": step_losses,
        "final": final,
        "param_count": int(agent.param_count()),
        "tolerance": 1e-12,
    }


def make_cql_case():
    states = [
        [0.20, 0.10, -0.40],
        [-0.30, 0.50, 0.60],
        [0.70, -0.80, 0.05],
        [-0.15, -0.25, 0.35],
        [0.45, 0.55, -0.65],
    ]
    next_states = [
        [0.21, 0.11, -0.39],
        [-0.29, 0.51, 0.61],
        [0.71, -0.79, 0.06],
        [-0.14, -0.24, 0.36],
        [0.46, 0.56, -0.64],
    ]
    batch = {
        "states": states,
        "actions": [2, 0, 1, 0, 2],
        "rewards": [0.5, 1.5, -1.0, 0.0, 0.75],
        "next_states": next_states,
        "dones": [0, 0, 1, 0, 1],
    }
    agent = AI.CQLAgent(3, 3, hidden=(8,), actor_lr=0.02, critic_lr=0.01,
                        alpha=0.5, beta=2.0, gamma=0.95, target_tau=0.3, seed=12)
    init = {
        "q": dump_mlp(agent.q.model),
        "policy": dump_mlp(agent.policy.model),
    }
    step_losses = []
    for _ in range(3):
        losses = agent.update(batch["states"], batch["actions"], batch["rewards"],
                              batch["next_states"], batch["dones"])
        step_losses.append({k: float(v) for k, v in losses.items()})
    final = {"q": [], "probs": []}
    for s in (states[0], states[3]):
        final["q"].append([float(v) for v in agent.q.logits(s)])
        final["probs"].append([float(v) for v in agent.policy.probs(s)])
    return {
        "name": "cql",
        "kind": "cql",
        "state_dim": 3,
        "num_actions": 3,
        "hidden": [8],
        "actor_lr": 0.02,
        "critic_lr": 0.01,
        "alpha": 0.5,
        "beta": 2.0,
        "gamma": 0.95,
        "target_tau": 0.3,
        "tau": 0.0,
        "batch": batch,
        "steps": 3,
        "init": init,
        "step_losses": step_losses,
        "final": final,
        "param_count": int(agent.param_count()),
        "tolerance": 1e-12,
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "offline_rl_reference.json"
    cases = [make_iql_case(), make_cql_case()]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
