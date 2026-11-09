/*
 * compare_offline_rl.c —— C 侧 IQL / CQL 与 AI.py 的数值对拍
 *
 * 用法：./tests/compare_offline_rl [offline_rl_reference.json]
 *
 * 参考用例（tools/offline_rl_reference.py 生成）提供：
 *   - 初始权重（q / v / policy；目标网络由 C 从在线网络复制）
 *   - 确定性转移批次；3 次 update 的损失；更新后的 q / v / probs；param_count
 * 比较：损失、输出、参数量（浮点按用例 tolerance，默认 1e-12）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ai.h"
#include "../json.h"
#include "../offline_rl.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_offline_rl] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_offline_rl] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/offline_rl_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static int jint(const JsonValue *v) { return json_int(v, 0); }
static double jnum(const JsonValue *v) { return json_num(v, 0.0); }

static double *darr(const JsonValue *v, int n)
{
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    return out;
}

static int *iarr(const JsonValue *v, int n)
{
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
    return out;
}

static void check_num(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    {
        const double d = fabs(got - want);
        if (d > g_worst) g_worst = d;
        if (d <= tol) return;
        printf("  %-28s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
        g_fail++;
    }
}

static void check_int(const char *what, int got, int want)
{
    g_checks++;
    if (got == want) g_exact++;
    else {
        printf("  %-28s C=%d py=%d [不等]\n", what, got, want);
        g_fail++;
    }
}

/* 把 JSON 的层权重载入 MLP；形状不符直接报错。 */
static void load_mlp(MLP *m, const JsonValue *layers, const char *who)
{
    const int n = json_array_len(layers);
    if (n != m->n_layers) {
        fprintf(stderr, "[%s] 层数不一致：C=%d JSON=%d\n", who, m->n_layers, n);
        exit(2);
    }
    for (int i = 0; i < n; i++) {
        const JsonValue *lv = json_at(layers, i);
        const JsonValue *wv = need(lv, "w", who);
        const JsonValue *bv = need(lv, "b", who);
        Layer *l = &m->layers[i];
        if (json_array_len(wv) != l->fan_out
            || json_array_len(json_at(wv, 0)) != l->fan_in
            || json_array_len(bv) != l->fan_out) {
            fprintf(stderr, "[%s] 第 %d 层形状不一致\n", who, i);
            exit(2);
        }
        for (int r = 0; r < l->fan_out; r++) {
            const JsonValue *row = json_at(wv, r);
            for (int c = 0; c < l->fan_in; c++)
                l->W[(size_t)r * l->fan_in + c] = jnum(json_at(row, c));
        }
        for (int r = 0; r < l->fan_out; r++)
            l->b[r] = jnum(json_at(bv, r));
    }
}

static const int *hidden_from(const JsonValue *case_v, int *n_hidden)
{
    const JsonValue *hv = json_get(case_v, "hidden");
    const int n = hv ? json_array_len(hv) : 0;
    int *h = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!h) die("out of memory");
    for (int i = 0; i < n; i++) h[i] = jint(json_at(hv, i));
    *n_hidden = n;
    return h;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "offline_rl_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_offline_rl] 读不了 %s：%s\n"
                        "请先运行 python tools/offline_rl_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_offline_rl] %s：%d 个用例\n", path, n_cases);

    for (int ci = 0; ci < n_cases; ci++) {
        const JsonValue *cs = json_at(cases, ci);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *kind = json_str(json_get(cs, "kind"), "?");
        const int sd = jint(need(cs, "state_dim", name));
        const int na = jint(need(cs, "num_actions", name));
        const double tol = jnum(json_get(cs, "tolerance"));
        int n_hidden = 0;
        const int *hidden = hidden_from(cs, &n_hidden);
        const JsonValue *batch = need(cs, "batch", name);
        const int n = json_array_len(need(batch, "actions", name));
        const JsonValue *states_v = need(batch, "states", name);
        const JsonValue *rewards_v = need(batch, "rewards", name);
        const JsonValue *next_v = need(batch, "next_states", name);
        const JsonValue *dones_v = need(batch, "dones", name);

        /* 数组扁平化：states / next_states 为 n×sd 行主序 */
        double *states = (double *)malloc(sizeof(double) * (size_t)(n * sd));
        double *next_states = (double *)malloc(sizeof(double) * (size_t)(n * sd));
        double *rewards = darr(rewards_v, n);
        int *actions = iarr(need(batch, "actions", name), n);
        int *dones = iarr(dones_v, n);
        if (!states || !next_states) die("out of memory");
        for (int i = 0; i < n; i++) {
            const JsonValue *row = json_at(states_v, i);
            const JsonValue *nrow = json_at(next_v, i);
            for (int j = 0; j < sd; j++) {
                states[(size_t)i * sd + j] = jnum(json_at(row, j));
                next_states[(size_t)i * sd + j] = jnum(json_at(nrow, j));
            }
        }

        printf("\n用例 %-6s kind=%s n=%d sd=%d na=%d\n", name, kind, n, sd, na);

        if (strcmp(kind, "iql") == 0) {
            IqlAgent *agent = iql_create(
                sd, na, hidden, n_hidden,
                jnum(need(cs, "actor_lr", name)),
                jnum(need(cs, "critic_lr", name)),
                jnum(need(cs, "tau", name)),
                jnum(need(cs, "beta", name)),
                jnum(need(cs, "gamma", name)),
                jnum(need(cs, "target_tau", name)));
            if (!agent) die("iql_create 失败");
            const JsonValue *init = need(cs, "init", name);
            load_mlp(agent->q, need(init, "q", name), "iql/q");
            load_mlp(agent->v, need(init, "v", name), "iql/v");
            load_mlp(agent->policy, need(init, "policy", name), "iql/policy");
            /* 目标网络 <- 在线网络 */
            for (int k = 0; k < agent->q->n_layers; k++) {
                Layer *tl = &agent->q_target->layers[k];
                const Layer *sl = &agent->q->layers[k];
                memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
                memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
            }
            for (int k = 0; k < agent->v->n_layers; k++) {
                Layer *tl = &agent->v_target->layers[k];
                const Layer *sl = &agent->v->layers[k];
                memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
                memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
            }

            check_int("param_count", iql_param_count(agent),
                      jint(need(cs, "param_count", name)));

            const JsonValue *losses_v = need(cs, "step_losses", name);
            const int steps = json_array_len(losses_v);
            for (int step = 0; step < steps; step++) {
                double losses[3] = {0.0, 0.0, 0.0};
                if (iql_update(agent, states, actions, rewards, next_states,
                               dones, n, losses) != 0) {
                    printf("  iql_update 返回错误\n");
                    g_fail++;
                    break;
                }
                const JsonValue *want = json_at(losses_v, step);
                check_num("q_loss", losses[0], jnum(need(want, "q_loss", name)), tol);
                check_num("v_loss", losses[1], jnum(need(want, "v_loss", name)), tol);
                check_num("policy_loss", losses[2],
                          jnum(need(want, "policy_loss", name)), tol);
            }

            /* 最终输出：q / v / probs × 2 个探测状态 = 批次前两个/第四个 */
            const JsonValue *final = need(cs, "final", name);
            const JsonValue *fq = need(final, "q", name);
            const JsonValue *fv = need(final, "v", name);
            const JsonValue *fp = need(final, "probs", name);
            const int probes[2] = {0, 2};
            for (int pi = 0; pi < 2; pi++) {
                const int idx = probes[pi];
                const double *s = states + (size_t)idx * sd;
                double *qbuf = (double *)malloc(sizeof(double) * (size_t)na);
                double *pbuf = (double *)malloc(sizeof(double) * (size_t)na);
                if (!qbuf || !pbuf) die("out of memory");
                iql_q(agent, s, qbuf);
                const JsonValue *wq = json_at(fq, pi);
                for (int j = 0; j < na; j++)
                    check_num("q", qbuf[j], jnum(json_at(wq, j)), tol);
                check_num("v", iql_v(agent, s), jnum(json_at(fv, pi)), tol);
                iql_probs(agent, s, pbuf);
                const JsonValue *wp = json_at(fp, pi);
                for (int j = 0; j < na; j++)
                    check_num("probs", pbuf[j], jnum(json_at(wp, j)), tol);
                free(qbuf);
                free(pbuf);
            }
            iql_free(agent);
        } else if (strcmp(kind, "cql") == 0) {
            CqlAgent *agent = cql_create(
                sd, na, hidden, n_hidden,
                jnum(need(cs, "actor_lr", name)),
                jnum(need(cs, "critic_lr", name)),
                jnum(need(cs, "alpha", name)),
                jnum(need(cs, "beta", name)),
                jnum(need(cs, "gamma", name)),
                jnum(need(cs, "target_tau", name)));
            if (!agent) die("cql_create 失败");
            const JsonValue *init = need(cs, "init", name);
            load_mlp(agent->q, need(init, "q", name), "cql/q");
            load_mlp(agent->policy, need(init, "policy", name), "cql/policy");
            for (int k = 0; k < agent->q->n_layers; k++) {
                Layer *tl = &agent->q_target->layers[k];
                const Layer *sl = &agent->q->layers[k];
                memcpy(tl->W, sl->W, (size_t)(tl->fan_out * tl->fan_in) * sizeof(double));
                memcpy(tl->b, sl->b, (size_t)tl->fan_out * sizeof(double));
            }

            check_int("param_count", cql_param_count(agent),
                      jint(need(cs, "param_count", name)));

            const JsonValue *losses_v = need(cs, "step_losses", name);
            const int steps = json_array_len(losses_v);
            for (int step = 0; step < steps; step++) {
                double losses[2] = {0.0, 0.0};
                if (cql_update(agent, states, actions, rewards, next_states,
                               dones, n, losses) != 0) {
                    printf("  cql_update 返回错误\n");
                    g_fail++;
                    break;
                }
                const JsonValue *want = json_at(losses_v, step);
                check_num("q_loss", losses[0], jnum(need(want, "q_loss", name)), tol);
                check_num("policy_loss", losses[1],
                          jnum(need(want, "policy_loss", name)), tol);
            }

            const JsonValue *final = need(cs, "final", name);
            const JsonValue *fq = need(final, "q", name);
            const JsonValue *fp = need(final, "probs", name);
            const int probes[2] = {0, 3};
            for (int pi = 0; pi < 2; pi++) {
                const int idx = probes[pi];
                const double *s = states + (size_t)idx * sd;
                double *qbuf = (double *)malloc(sizeof(double) * (size_t)na);
                double *pbuf = (double *)malloc(sizeof(double) * (size_t)na);
                if (!qbuf || !pbuf) die("out of memory");
                cql_q(agent, s, qbuf);
                const JsonValue *wq = json_at(fq, pi);
                for (int j = 0; j < na; j++)
                    check_num("q", qbuf[j], jnum(json_at(wq, j)), tol);
                cql_probs(agent, s, pbuf);
                const JsonValue *wp = json_at(fp, pi);
                for (int j = 0; j < na; j++)
                    check_num("probs", pbuf[j], jnum(json_at(wp, j)), tol);
                free(qbuf);
                free(pbuf);
            }
            cql_free(agent);
        } else {
            printf("  未知 kind\n");
            g_fail++;
        }

        free((void *)hidden);
        free(states);
        free(next_states);
        free(rewards);
        free(actions);
        free(dones);
    }

    printf("\n[compare_offline_rl] 共 %d 项检查：逐位一致 %d，失败 %d，"
           "最大绝对差 %.3e\n", g_checks, g_exact, g_fail, g_worst);
    json_free(root);
    return (g_fail > 0) ? 1 : 0;
}
