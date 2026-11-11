/*
 * rl_utils.h —— 强化学习/μP 工具集（对应 Python 版 AI.py）
 *
 * 覆盖：
 *   - group_relative_advantages（GRPO 组相对优势）
 *   - best_of_n、rejection_sampling_filter（RFT / 拒绝采样）
 *   - self_consistency（整数样本的自洽性投票）
 *   - mup_target_std / mup_lr / MuPAdam 的一步更新
 *
 * 纯计算、无全局状态；数组默认行主序扁平 double。
 */
#ifndef AI_RL_UTILS_H
#define AI_RL_UTILS_H

/* 组相对优势：out 长度 n；group_size<=0 表示所有奖励同属一组；
 * normalize!=0 时每个组做 (r-mean)/(std+eps)（std 用除 size 的总体方差）。 */
void ai_group_relative_advantages(const double *rewards, int n, int group_size,
                                  double eps, int normalize, double *out);

/* Best-of-N：返回奖励最大的下标（并列取最小）；*out_reward 写入该奖励。 */
int ai_best_of_n(const double *rewards, int n, double *out_reward);

/* 阈值过滤：把 reward >= threshold 的下标按原顺序写入 kept（容量 >= n），返回个数。 */
int ai_rejection_filter_threshold(const double *rewards, int n, double threshold,
                                  int *kept);

/* Top-ratio 过滤：保留 max(1, round(n*ratio)) 个，按 (-reward, index) 升序，
 * 下标写入 kept（容量 >= n），返回个数。 */
int ai_rejection_filter_top_ratio(const double *rewards, int n, double ratio,
                                  int *kept);

/* 自洽性（整数样本）：返回票数最高样本（并列取首次出现者）；
 * *out_conf = best_count/n，*out_unique 唯一键数；
 * keys/counts 用首次出现顺序写入，容量 >= n。 */
int ai_self_consistency_int(const int *samples, int n, double *out_conf,
                            int *out_unique, int *keys, int *counts);

/* ---------- μP ---------- */

/* kind = "input" / "hidden" / "readout"；width<=0 表示未显式给出。 */
double ai_mup_target_std(const char *kind, double fan_in, double base_width,
                         double width, double scale);

/* 把各层类型写入 kinds（'i'/'h'/'r'），返回层数。 */
int ai_mup_layer_kinds(int num_layers, char *kinds);

/* 各层学习率：fan_ins 长度 n；width<=0 时取 fan_ins 的最大值。 */
void ai_mup_lr(const double *fan_ins, int n, double base_lr, double width,
               double base_width, double *out_lrs);

/* MuPAdam 的单参数组一步更新：params/grads/m/v 长度均为 count。 */
void ai_mupadam_step(double *params, const double *grads, double *m, double *v,
                     int count, double lr, double beta1, double beta2,
                     double eps, int step_count);

#endif /* AI_RL_UTILS_H */
