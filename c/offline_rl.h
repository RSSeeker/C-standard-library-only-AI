/*
 * offline_rl.h —— 离线强化学习：IQL / CQL（离散动作，C17，零依赖）
 *
 * 与 AI.py 的 IQLAgent / CQLAgent 对齐：
 *   - IQL：expectile 回归 V；Q 的 TD 目标用 V_target(s')（不做 max）；
 *          策略用 AWR（优势 = Q(s,a) - V(s)）做加权行为克隆。
 *   - CQL：TD 损失 + 保守项 alpha*(logsumexp Q - Q(s,a))；策略 AWR 用同一优势。
 *   网络：Q 与策略是多层感知机（输出线性），V 是标量回归（输出线性）；
 *         Adam 优化器；目标网络按 tau 软更新。
 *
 * 数据约定：states/next_states 为 n×state_dim 行主序；actions 为 int[n]；
 * rewards 为 double[n]；dones 为 int[n]（非 0 表示终止）。
 */
#ifndef AI_OFFLINE_RL_H
#define AI_OFFLINE_RL_H

#include "ai.h"

/* ---------- IQL ---------- */
typedef struct {
    int state_dim;
    int num_actions;
    double tau;        /* expectile 系数 (0,1) */
    double beta;       /* AWR 温度 */
    double gamma;
    double target_tau; /* 目标网络软更新系数 */

    MLP *q;
    MLP *q_target;
    MLP *v;
    MLP *v_target;
    MLP *policy;

    Optimizer q_opt;
    Optimizer v_opt;
    Optimizer policy_opt;
} IqlAgent;

/* 创建智能体；hidden 可为 NULL（n_hidden=0 表示单隐层网络被省略——但按 Python
 * 语义至少需要 hidden=(32,)，C 侧允许 0 个隐层以便测试小网络）。
 * 返回 NULL 表示参数非法（维度 <=0、学习率 <=0、tau/target_tau 越界、hidden 元素 <=0）。 */
IqlAgent *iql_create(int state_dim, int num_actions, const int *hidden,
                     int n_hidden, double actor_lr, double critic_lr,
                     double tau, double beta, double gamma, double target_tau);
void iql_free(IqlAgent *a);

/* 对一批转移做一次更新；losses[0..2] 写入 q_loss / v_loss / policy_loss。
 * 返回 0 成功，-1 表示参数非法（数据为空、动作越界、NaN 等）。 */
int iql_update(IqlAgent *a, const double *states, const int *actions,
               const double *rewards, const double *next_states,
               const int *dones, int n, double losses[3]);

/* 推理辅助（便于对拍/调试）：q_logits 长度 num_actions；probs 同；v 返回标量。 */
void iql_q(const IqlAgent *a, const double *state, double *q_logits);
double iql_v(const IqlAgent *a, const double *state);
void iql_probs(const IqlAgent *a, const double *state, double *probs);
int iql_act(const IqlAgent *a, const double *state);
int iql_param_count(const IqlAgent *a);

/* 直接访问网络权重（便于载入/导出参考权重对拍）：返回对应 MLP 指针。 */
MLP *iql_net_q(IqlAgent *a);
MLP *iql_net_v(IqlAgent *a);
MLP *iql_net_policy(IqlAgent *a);

/* ---------- CQL ---------- */
typedef struct {
    int state_dim;
    int num_actions;
    double alpha;
    double beta;
    double gamma;
    double target_tau;

    MLP *q;
    MLP *q_target;
    MLP *policy;

    Optimizer q_opt;
    Optimizer policy_opt;
} CqlAgent;

CqlAgent *cql_create(int state_dim, int num_actions, const int *hidden,
                     int n_hidden, double actor_lr, double critic_lr,
                     double alpha, double beta, double gamma, double target_tau);
void cql_free(CqlAgent *a);

/* 一次更新；losses[0..1] 写入 q_loss / policy_loss。 */
int cql_update(CqlAgent *a, const double *states, const int *actions,
               const double *rewards, const double *next_states,
               const int *dones, int n, double losses[2]);

void cql_q(const CqlAgent *a, const double *state, double *q_logits);
void cql_probs(const CqlAgent *a, const double *state, double *probs);
int cql_act(const CqlAgent *a, const double *state);
int cql_param_count(const CqlAgent *a);

MLP *cql_net_q(CqlAgent *a);
MLP *cql_net_policy(CqlAgent *a);

#endif /* AI_OFFLINE_RL_H */
