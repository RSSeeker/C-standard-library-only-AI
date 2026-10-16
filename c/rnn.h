/*
 * rnn.h —— RNN / LSTM / GRU 家族（含 BPTT 用的 step cache 机制）
 *
 * 移植自 AI.py 的 RNN 部分：
 *     _StepCacheMixin  -> RNNStepCache + xxx_cell_cache_snapshot / xxx_restore_cache
 *     RNNCell  / RNN   -> RNNCell  / RNN
 *     LSTMCell / LSTM  -> LSTMCell / LSTM
 *     GRUCell  / GRU   -> GRUCell  / GRU
 *
 * 命名：类型前缀小写就是函数前缀
 *     rnn_cell_*   单步 RNNCell        rnn_*   多层 RNN（带 step_cache）
 *     lstm_cell_*  单步 LSTMCell       lstm_*  多层 LSTM
 *     gru_cell_*   单步 GRUCell        gru_*   多层 GRU
 *
 * ============================== 两套接口 ==============================
 * 1) 单步 cell（对应 Python 的 RNNCell.forward / .backward）
 *
 *      const double *h = rnn_cell_forward(c, x, h_prev);
 *      rnn_cell_backward(c, dh, dx, dh_prev);        // 参数梯度原地累加
 *
 *    - x: (input_size,)   h_prev: (hidden_size,)，h_prev 传 NULL 表示全 0（同 Python）
 *    - forward 把中间量写进 cell 自己的缓存，紧接着就能 backward，不必重跑前向
 *    - backward 把 dx / dh_prev 写进调用方缓冲（传 NULL 表示不需要）；
 *      参数梯度累加到 c->grad_*（何时清零由调用方决定，见下面的契约）
 *    - LSTMCell 多一个 c_prev / dc_next / dc_prev：
 *        const double *h = lstm_cell_forward(c, x, h_prev, c_prev);
 *        lstm_cell_backward(c, dh, dc_next, dx, dh_prev, dc_prev);
 *    - GRUCell 与 RNNCell 同形
 *
 * 2) 多层包装（对应 Python 的 RNN / LSTM / GRU）
 *
 *      rnn_forward(net, x_seq, out_seq, h_last);   // 同时把每步每层缓存写进 net->step_cache
 *      rnn_backward(net, dh_last, dx_seq);         // 直接复用 step_cache，不重跑前向
 *
 *    形状：
 *      x_seq   (T, input_size)
 *      out_seq (T, hidden_size)              最后一层每个时间步的 hidden（可为 NULL）
 *      h_last  (num_layers, hidden_size)     每层最后一个时间步的 hidden（可为 NULL）
 *      LSTM 另有 c_last (num_layers, hidden_size)
 *      dh_last (num_layers, hidden_size)     末状态梯度（可为 NULL = 全 0）
 *      dx_seq  (T, input_size)               输入梯度（可为 NULL）
 *
 *    与 Python BPTT 循环一一对应的“逐步”接口：
 *      rnn_restore_cache(net, t, layer);     // == cell.restore_cache(step_cache[t][layer])
 *      rnn_cell_backward(rnn_layer(net, layer), dh, dx, dh_prev);
 *    这样调用方可以自己写 BPTT（每个时间步往 net->step_cache 的对应快照还原缓存），
 *    也可以直接用 rnn_backward()。
 *
 * ==================== get_grads() 契约（重要，别再踩） ====================
 * Python 版曾经的真实 bug：get_grads() 返回的就是 cell 内部的梯度缓冲区本身，
 * 调用方拿到之后又“手工把这些梯度加到 cell 自己的缓冲上”，等于把缓冲区加到它自己身上，
 * 梯度按 2 的幂被放大（lr 也因此被迫调大来掩盖）。
 *
 * C 版沿用同一契约，务必注意：
 *   - 每次 backward 都是“原地累加”进 grad_*（既不覆盖也不清零）；
 *   - xxx_cell_get_grads() 或直接读 c->grad_* 拿到的就是这些内部缓冲本身；
 *     一次 zero_grad + 逐步 backward 之后，它们已经包含全部时间步的贡献，
 *     直接交给优化器即可，**不要再往回加一遍**；
 *   - 唯一合法的清零时机是调用方的 zero_grad（一条序列 / 一个 batch 开始时）。
 *     自测里用“一次 zero_grad + 逐步 backward == 各步隔离梯度之和”验证了这一点。
 *
 * ============================ 数值顺序约定 ============================
 * 为了与 Python 逐位一致，前向 / 反向照抄了 Python 的显式索引循环：
 *   - 预激活一律从偏置开始累加：s = b_ih[j] + b_hh[j]，再按 i 升序累加 W_ih 行、W_hh 行；
 *     GRU 的两段分量各自从偏置开始累加（Python 的 _linear），最后才相加
 *     （z = z_ih + z_hh），不能合成一个和；
 *   - backward 的 dx / dh_prev 也是显式索引循环（for i: s = 0; for j: s += W[j][i]*dz[j]）；
 *     只有 GRU 在 Python 里写的是 sum(... for j in ...)，对应 Neumaier 补偿求和
 *     （CPython 3.12+ 的 sum()，见 numeric.h）；
 *   - 梯度累加用朴素 +=；
 *   - 激活：RNN 用 tanh；LSTM 的门用 sigmoid（1/(1+exp(-x))）、cell gate 用 tanh；
 *     GRU 的 update/reset 用 sigmoid、new gate 用 tanh；
 *   - RNNCell 的 tanh 导数写成 1.0 - pow(tanh(z), 2.0)
 *     （Python 是 math.tanh(zj) ** 2，走 libm 的 pow，不是 z*z）。
 *
 * 门顺序（以 AI.py 源码为准）：
 *     LSTMCell: W_ih / W_hh / b_* 的行顺序是 [i, f, g, o]，每段 hidden_size 行
 *     GRUCell : 行顺序是 [z, r, n]，每段 hidden_size 行
 *
 * ========================== 容量与零分配 ==========================
 * forward / backward / optimizer_step 全程不分配内存：
 *   - xxx_create() 按 RNN_DEFAULT_MAX_T 步预分配全部中间缓冲；
 *   - 序列更长时先调用 xxx_reserve(net, max_T) 扩容（只能在训练循环外调用，
 *     调用后 step_cache 失效，需要重新 forward）；
 *   - forward 时 T 仍超过容量 -> 直接报错退出（绝不静默截断）。
 *
 * ========================== 与 Python 的差异 ==========================
 *   - Python 的 self.x / self.h_prev 存的是引用，这里存的是拷贝（训练语义等价，
 *     并且避免调用方复用缓冲时缓存被偷偷改掉）；
 *   - 权重初始化：Python 用 random.gauss，跨语言无法逐位复现。xxx_cell_init_params()
 *     用 xorshift + Box-Muller 生成同 std 的高斯（C 侧独立使用足够）；
 *     需要跨语言对齐时用 xxx_cell_set_params() 把 Python 的权重灌进来。
 *   - 多层 RNN 的 BPTT 在 Python 里没有现成实现（示例只做单层 BPTT），这里按
 *     “时间步逆序、层逆序”实现；单层情形与 Python 示例的 BPTT 循环逐位一致。
 */
#ifndef AI_RNN_H
#define AI_RNN_H

#include "tensor.h"
#include "ai.h"    /* Optimizer / OptKind：复用 ai.h 的优化器超参定义（不依赖 ai.c 的符号） */

/* xxx_create 默认预分配的最大时间步数 */
#define RNN_DEFAULT_MAX_T 128

/* ============================== RNNCell ============================== */
typedef struct {
    int input_size;
    int hidden_size;

    double *W_ih;    /* hidden × input，行主序 */
    double *W_hh;    /* hidden × hidden，行主序 */
    double *b_ih;    /* hidden */
    double *b_hh;    /* hidden */

    double *grad_W_ih;   /* 梯度（原地累加，见头文件契约） */
    double *grad_W_hh;
    double *grad_b_ih;
    double *grad_b_hh;

    double *m_W_ih;  double *v_W_ih;   /* 优化器状态（一阶 / 二阶动量） */
    double *m_W_hh;  double *v_W_hh;
    double *m_b_ih;  double *v_b_ih;
    double *m_b_hh;  double *v_b_hh;

    /* 前向缓存，对应 Python 的 _CACHE_FIELDS = ("x", "h_prev", "z") */
    double *x;        /* input */
    double *h_prev;   /* hidden */
    double *z;        /* hidden，激活前 */
    double *h;        /* hidden，本步输出（forward 的返回值） */

    double *scratch;  /* 反向临时缓冲（hidden），预分配，热路径零分配 */
} RNNCell;

RNNCell *rnn_cell_create(int input_size, int hidden_size);
void     rnn_cell_free(RNNCell *c);
/* 用给定随机种子做 Xavier 初始化（std = sqrt(1/in)、sqrt(1/hs)），与 Python 的
 * 分布一致但序列不同；跨语言对齐请改用 rnn_cell_set_params() */
void     rnn_cell_init_params(RNNCell *c, unsigned long long *rng_state);
/* 按 Python 的展平顺序灌权重：W_ih 为 (hidden, input) 行主序，W_hh 为 (hidden, hidden) */
void     rnn_cell_set_params(RNNCell *c, const double *W_ih, const double *W_hh,
                             const double *b_ih, const double *b_hh);
void     rnn_cell_zero_grad(RNNCell *c);

/* 单步前向：返回内部 h（长度 hidden_size）。h_prev 可为 NULL（= 全 0） */
const double *rnn_cell_forward(RNNCell *c, const double *x, const double *h_prev);
/* 单步反向：参数梯度原地累加；dx / dh_prev 可为 NULL */
void rnn_cell_backward(RNNCell *c, const double *dh, double *dx, double *dh_prev);

/* 取内部梯度缓冲（同 Python get_grads()：返回的就是内部缓冲本身，不要再手工累加） */
void rnn_cell_get_grads(RNNCell *c, double **grad_W_ih, double **grad_W_hh,
                        double **grad_b_ih, double **grad_b_hh);
int  rnn_cell_param_count(const RNNCell *c);

/* 缓存快照：rec 长度必须是 rnn_cell_cache_size()，字段顺序同 Python 的 _CACHE_FIELDS */
int  rnn_cell_cache_size(const RNNCell *c);
void rnn_cell_cache_snapshot(const RNNCell *c, double *rec);
void rnn_cell_cache_restore(RNNCell *c, const double *rec);

/* ============================== LSTMCell ============================== */
typedef struct {
    int input_size;
    int hidden_size;
    int gate_size;   /* 4 * hidden_size */

    double *W_ih;    /* (4*hidden) × input，行顺序 [i, f, g, o] */
    double *W_hh;    /* (4*hidden) × hidden */
    double *b_ih;    /* 4*hidden */
    double *b_hh;    /* 4*hidden */

    double *grad_W_ih;   /* 梯度（原地累加） */
    double *grad_W_hh;
    double *grad_b_ih;
    double *grad_b_hh;

    double *m_W_ih;  double *v_W_ih;
    double *m_W_hh;  double *v_W_hh;
    double *m_b_ih;  double *v_b_ih;
    double *m_b_hh;  double *v_b_hh;

    /* 前向缓存，对应 Python 的 _CACHE_FIELDS = ("x","h_prev","c_prev","i","f","g","o","c") */
    double *x;
    double *h_prev;
    double *c_prev;
    double *i_gate;   /* Python 的 self.i */
    double *f_gate;   /* self.f */
    double *g_gate;   /* self.g（cell gate） */
    double *o_gate;   /* self.o */
    double *c;        /* self.c（cell state） */
    double *h;        /* 本步输出（forward 的返回值） */

    double *gates;    /* 4*hidden 预激活（Python 的局部变量 gates） */
    double *d_gates;  /* 4*hidden 反向临时 */
    double *dc;       /* hidden 反向临时 */
} LSTMCell;

LSTMCell *lstm_cell_create(int input_size, int hidden_size);
void      lstm_cell_free(LSTMCell *c);
void      lstm_cell_init_params(LSTMCell *c, unsigned long long *rng_state);
/* W_ih 为 (4*hidden, input)，行顺序 [i, f, g, o]（与 AI.py 一致） */
void      lstm_cell_set_params(LSTMCell *c, const double *W_ih, const double *W_hh,
                               const double *b_ih, const double *b_hh);
void      lstm_cell_zero_grad(LSTMCell *c);

/* h_prev / c_prev 可为 NULL（= 全 0）；返回内部 h，cell state 在 c->c */
const double *lstm_cell_forward(LSTMCell *c, const double *x,
                                const double *h_prev, const double *c_prev);
/* dc_next 可为 NULL（= 全 0）；dx / dh_prev / dc_prev 可为 NULL；梯度原地累加 */
void lstm_cell_backward(LSTMCell *c, const double *dh, const double *dc_next,
                        double *dx, double *dh_prev, double *dc_prev);

void lstm_cell_get_grads(LSTMCell *c, double **grad_W_ih, double **grad_W_hh,
                         double **grad_b_ih, double **grad_b_hh);
int  lstm_cell_param_count(const LSTMCell *c);

int  lstm_cell_cache_size(const LSTMCell *c);
void lstm_cell_cache_snapshot(const LSTMCell *c, double *rec);
void lstm_cell_cache_restore(LSTMCell *c, const double *rec);

/* ============================== GRUCell ============================== */
typedef struct {
    int input_size;
    int hidden_size;
    int gate_size;   /* 3 * hidden_size */

    double *W_ih;    /* (3*hidden) × input，行顺序 [z, r, n] */
    double *W_hh;    /* (3*hidden) × hidden */
    double *b_ih;    /* 3*hidden */
    double *b_hh;    /* 3*hidden */

    double *grad_W_ih;   /* 梯度（原地累加） */
    double *grad_W_hh;
    double *grad_b_ih;
    double *grad_b_hh;

    double *m_W_ih;  double *v_W_ih;
    double *m_W_hh;  double *v_W_hh;
    double *m_b_ih;  double *v_b_ih;
    double *m_b_hh;  double *v_b_hh;

    /* 前向缓存，对应 Python 的
     * _CACHE_FIELDS = ("x","h_prev","z","r","n_ih","n_hh","z_a","r_a","n_val") */
    double *x;
    double *h_prev;
    double *z;        /* update gate 预激活 */
    double *r;        /* reset gate 预激活 */
    double *n_ih;     /* new gate 的 ih 分量（Python 单独存，反向要用） */
    double *n_hh;     /* new gate 的 hh 分量 */
    double *z_a;      /* sigmoid(z) */
    double *r_a;      /* sigmoid(r) */
    double *n_val;    /* tanh(n_ih + r_a * n_hh) */
    double *h;        /* 本步输出 */

    double *tmp_ih;       /* hidden，_linear 分段累加用（保证 z = z_ih + z_hh 的顺序） */
    double *tmp_hh;       /* hidden */
    double *dz;           /* hidden 反向临时 */
    double *dr;           /* hidden */
    double *dn_ih;        /* hidden */
    double *dn_hh;        /* hidden */
    double *d_all_gates;  /* 3*hidden，dx / dh_prev 的补偿求和输入 */
} GRUCell;

GRUCell *gru_cell_create(int input_size, int hidden_size);
void     gru_cell_free(GRUCell *c);
void     gru_cell_init_params(GRUCell *c, unsigned long long *rng_state);
/* W_ih 为 (3*hidden, input)，行顺序 [z, r, n]（与 AI.py 一致） */
void     gru_cell_set_params(GRUCell *c, const double *W_ih, const double *W_hh,
                             const double *b_ih, const double *b_hh);
void     gru_cell_zero_grad(GRUCell *c);

const double *gru_cell_forward(GRUCell *c, const double *x, const double *h_prev);
void gru_cell_backward(GRUCell *c, const double *dh, double *dx, double *dh_prev);

void gru_cell_get_grads(GRUCell *c, double **grad_W_ih, double **grad_W_hh,
                        double **grad_b_ih, double **grad_b_hh);
int  gru_cell_param_count(const GRUCell *c);

int  gru_cell_cache_size(const GRUCell *c);
void gru_cell_cache_snapshot(const GRUCell *c, double *rec);
void gru_cell_cache_restore(GRUCell *c, const double *rec);

/* ========================== 多层 step cache ==========================
 * 对应 Python 的 _StepCacheMixin + RNN/LSTM/GRU 的 self.step_cache：
 *   rec 是一整块预分配内存，布局 [t][layer][各字段]，第 l 层的字段布局就是
 *   xxx_cell_cache_size/xxx_cell_cache_snapshot 用的那个记录；
 *   layer_off[l] 是第 l 层在“一个时间步”内的偏移，rec_stride = layer_off[num_layers]。
 *   cached_T 是最近一次 forward 写入的步数，backward 依赖它。
 */
typedef struct rnn_step_cache {
    int max_T;         /* 预分配的最大时间步数 */
    int num_layers;
    int hidden_size;
    int cached_T;      /* 最近一次 forward 实际缓存的步数 */
    int rec_stride;    /* 每个时间步占用的 double 数 */
    int *layer_off;    /* num_layers + 1 个前缀和 */
    double *rec;       /* max_T × rec_stride */
} RNNStepCache;

/* ================================ RNN ================================ */
typedef struct {
    int input_size;
    int hidden_size;
    int num_layers;

    RNNCell *cells;        /* num_layers 个；第 l 层输入宽度 = (l == 0 ? input_size : hidden_size) */
    RNNStepCache step_cache;

    /* BPTT 的 dh 累加区 [t][layer][hidden] 与单步临时缓冲，全部预分配 */
    double *dh_slab;
    double *scratch_dx;    /* max(input_size, hidden_size) */
    double *scratch_dh;    /* hidden_size */
} RNN;

RNN *rnn_create(int input_size, int hidden_size, int num_layers);
RNN *rnn_create_cap(int input_size, int hidden_size, int num_layers, int max_T);
void rnn_free(RNN *net);
/* 扩容到 max_T 步（训练循环外调用）；成功返回 1；调用后 step_cache 失效 */
int  rnn_reserve(RNN *net, int max_T);
void rnn_zero_grad(RNN *net);

/* x_seq (T, input_size) -> out_seq (T, hidden_size) / h_last (num_layers, hidden_size)；
 * out_seq / h_last 可为 NULL。h0 为 (num_layers, hidden_size)，可为 NULL（= 全 0） */
void rnn_forward(RNN *net, const Tensor *x_seq, Tensor *out_seq, Tensor *h_last);
void rnn_forward_h0(RNN *net, const Tensor *x_seq, const Tensor *h0,
                    Tensor *out_seq, Tensor *h_last);

/* 反向：dh_last 为末状态梯度 (num_layers, hidden_size)，可为 NULL；
 * dx_seq (T, input_size) 可为 NULL。只依赖 step_cache，不重跑前向 */
void rnn_backward(RNN *net, const Tensor *dh_last, Tensor *dx_seq);
/* 更一般的形式：dout_seq (T, hidden_size) 是最后一层每个时间步输出的梯度，可为 NULL */
void rnn_backward_seq(RNN *net, const Tensor *dout_seq, const Tensor *dh_last, Tensor *dx_seq);

/* 把 (t, layer) 的前向缓存快照还原进对应 cell（== Python 的 restore_cache） */
void rnn_restore_cache(RNN *net, int t, int layer);
int  rnn_cached_steps(const RNN *net);       /* 最近一次 forward 缓存的步数 T */
RNNCell *rnn_layer(RNN *net, int layer);
int  rnn_param_count(const RNN *net);
/* 优化器：超参用 ai.h 的 Optimizer，逐元素顺序与 ai.c 的 optimizer_step 一致；
 * 不依赖 ai.c 的符号（rnn.c 自包含，可与 tensor.c 单独链接） */
void rnn_optimizer_init(Optimizer *o, OptKind kind, double lr);
void rnn_optimizer_step(RNN *net, Optimizer *o);

/* ================================ LSTM ================================ */
typedef struct {
    int input_size;
    int hidden_size;
    int num_layers;

    LSTMCell *cells;
    RNNStepCache step_cache;

    double *dh_slab;       /* [t][layer][hidden] */
    double *dc_slab;       /* [t][layer][hidden]，cell state 的梯度 */
    double *scratch_dx;
    double *scratch_dh;
    double *scratch_dc;
} LSTM;

LSTM *lstm_create(int input_size, int hidden_size, int num_layers);
LSTM *lstm_create_cap(int input_size, int hidden_size, int num_layers, int max_T);
void  lstm_free(LSTM *net);
int   lstm_reserve(LSTM *net, int max_T);
void  lstm_zero_grad(LSTM *net);

/* x_seq (T, input_size) -> out_seq (T, hidden_size) / h_last / c_last (num_layers, hidden_size)；
 * out_seq / h_last / c_last 可为 NULL */
void lstm_forward(LSTM *net, const Tensor *x_seq, Tensor *out_seq,
                  Tensor *h_last, Tensor *c_last);
void lstm_forward_h0(LSTM *net, const Tensor *x_seq, const Tensor *h0, const Tensor *c0,
                     Tensor *out_seq, Tensor *h_last, Tensor *c_last);

/* dh_last / dc_last 均为 (num_layers, hidden_size)，可为 NULL */
void lstm_backward(LSTM *net, const Tensor *dh_last, Tensor *dx_seq);
void lstm_backward_full(LSTM *net, const Tensor *dout_seq, const Tensor *dh_last,
                        const Tensor *dc_last, Tensor *dx_seq);

void lstm_restore_cache(LSTM *net, int t, int layer);
int  lstm_cached_steps(const LSTM *net);
LSTMCell *lstm_layer(LSTM *net, int layer);
int  lstm_param_count(const LSTM *net);
void lstm_optimizer_step(LSTM *net, Optimizer *o);

/* ================================ GRU ================================ */
typedef struct {
    int input_size;
    int hidden_size;
    int num_layers;

    GRUCell *cells;
    RNNStepCache step_cache;

    double *dh_slab;
    double *scratch_dx;
    double *scratch_dh;
} GRU;

GRU *gru_create(int input_size, int hidden_size, int num_layers);
GRU *gru_create_cap(int input_size, int hidden_size, int num_layers, int max_T);
void gru_free(GRU *net);
int  gru_reserve(GRU *net, int max_T);
void gru_zero_grad(GRU *net);

void gru_forward(GRU *net, const Tensor *x_seq, Tensor *out_seq, Tensor *h_last);
void gru_forward_h0(GRU *net, const Tensor *x_seq, const Tensor *h0,
                    Tensor *out_seq, Tensor *h_last);
void gru_backward(GRU *net, const Tensor *dh_last, Tensor *dx_seq);
void gru_backward_seq(GRU *net, const Tensor *dout_seq, const Tensor *dh_last, Tensor *dx_seq);

void gru_restore_cache(GRU *net, int t, int layer);
int  gru_cached_steps(const GRU *net);
GRUCell *gru_layer(GRU *net, int layer);
int  gru_param_count(const GRU *net);
void gru_optimizer_step(GRU *net, Optimizer *o);

#endif /* AI_RNN_H */
