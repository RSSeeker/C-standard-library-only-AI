/*
 * ai.h —— Python 版神经网络工具箱的 C 移植（核心数值路径）
 *
 * 与 Python 版 AI.py 的对应关系：
 *   Layer / MLP        -> Layer / MLP
 *   mse_loss           -> mse_loss / mse_grad
 *   cross_entropy_loss -> ce_loss / ce_grad（softmax+CE 联合梯度约定）
 *   Adam / SGD / ...   -> adam_step / sgd_step / ...
 *   Model.fit          -> trainer_fit
 *   Model.predict/score-> mlp_forward / trainer_score
 *   save_model/load_model -> mlp_save / mlp_load（文本格式，非 JSON）
 *
 * 设计约定（与 Python 版保持一致，便于数值对齐）：
 *   1. 权重 W 为行主序 fan_out × fan_in，b 长度 fan_out；
      前向 z_j = Σ_i W[j*fan_in + i] * x[i] + b[j]（与 Layer.forward 同序）
 *   2. softmax 输出层的反向使用 softmax+CE 联合梯度：
      Layer.backward 里 act_deriv 恒为 1，CE 直接返回 pred - one_hot。
      因此 softmax 只能配 CE（Python 版对此有硬校验，这里由调用方保证）
 *   3. 训练按样本前向+反向、梯度累加到 batch 缓冲、除以 batch 后再更新，
      与 Model.fit 的累加顺序一致
 *   4. 全部用 double，无任何第三方依赖（仅 libc + libm）
 */
#ifndef AI_H
#define AI_H

#include <stddef.h>

/* ---------- 激活函数 ---------- */
typedef enum {
    ACT_LEAKY_RELU = 0,
    ACT_RELU       = 1,
    ACT_SOFTMAX    = 2,
    ACT_LINEAR     = 3
} Activation;

/* ---------- 损失函数 ---------- */
typedef enum {
    LOSS_MSE = 0,
    LOSS_CROSS_ENTROPY = 1
} LossKind;

/* ---------- 优化器 ---------- */
typedef enum {
    OPT_ADAM = 0,
    OPT_SGD = 1,
    OPT_SGD_MOMENTUM = 2,
    OPT_RMSPROP = 3
} OptKind;

/* ---------- 全连接层 ---------- */
typedef struct {
    int fan_in;
    int fan_out;
    Activation act;

    double *W;    /* fan_out × fan_in, 行主序 */
    double *b;    /* fan_out */

    double *gW;   /* 梯度（累积） */
    double *gb;

    double *mW;   /* 一阶动量 / 动量缓存 */
    double *vW;   /* 二阶动量 / RMSprop 缓存 */
    double *mb;
    double *vb;

    double *x;    /* 前向缓存：输入 */
    double *z;    /* 前向缓存：激活前 */
    double *a;    /* 前向缓存：激活后 */

    double *d_act; /* 反向临时缓冲（预分配，避免热路径 malloc） */
    double *tmpw;
} Layer;

/* ---------- 多层感知机 ---------- */
typedef struct {
    int n_layers;
    Layer *layers;
    /* 反向传播的 delta 乒乓缓冲：预分配后训练过程完全不分配内存 */
    double *scratch_a;
    double *scratch_b;
    int scratch_dim;
} MLP;

/* ---------- 优化器超参 ---------- */
typedef struct {
    OptKind kind;
    double lr;
    double beta1;    /* Adam */
    double beta2;    /* Adam */
    double eps;      /* Adam / RMSprop */
    double momentum; /* SGD+Momentum */
    double alpha;    /* RMSprop */
    double weight_decay;
    int step_count;
} Optimizer;

/* ---------- 训练器（对应 Python 的 Model） ---------- */
typedef struct {
    MLP *net;
    LossKind loss;
    Optimizer opt;
    /* 以下为内部缓冲：首次用到时懒分配，之后训练热路径零分配。
     * 用 { net, loss, {0} } 这样初始化即可，多余的字段会被置零 */
    double *pred_buf;
    double *grad_buf;
    int buf_dim;
    int *idx_buf;      /* 打乱/切分用的索引缓冲 */
    int idx_dim;
} Trainer;

/* 初始化训练器（把结构体清零并设置好网络/损失/优化器），
 * 比用花括号列表初始化更稳——将来给 Trainer 加字段不会漏初始化 */
void trainer_init(Trainer *t, MLP *net, LossKind loss, OptKind kind, double lr);

/* 释放 trainer 内部缓冲（不释放 net） */
void trainer_release(Trainer *t);

/* ================= MLP ================= */

/* sizes 长度 n_sizes（含输入与输出维度），acts 长度 n_sizes-1 */
MLP  *mlp_create(const int *sizes, const Activation *acts, int n_sizes);
void  mlp_free(MLP *m);
void  mlp_zero_grad(MLP *m);
void  mlp_forward(MLP *m, const double *x, double *out);
/* 把 dout 往回传，梯度累积到各层 gW/gb（调用前需 mlp_zero_grad 或在 batch 内累积） */
void  mlp_backward(MLP *m, const double *dout);

/* ================= 损失 ================= */
double mse_loss(const double *pred, const double *target, int n);
void   mse_grad(const double *pred, const double *target, int n, double *grad);
/* 返回 loss，并把 softmax+CE 的联合梯度写入 grad（长度 n） */
double ce_loss_grad(const double *pred, int n, int target_idx, double *grad);

/* ================= 优化器 ================= */
void optimizer_init(Optimizer *o, OptKind kind, double lr);
void optimizer_step(MLP *m, Optimizer *o);

/* ================= 训练 / 推理 ================= */

/* 单样本前向 + 反向 + 梯度累加，返回该样本的损失 */
double trainer_accumulate(Trainer *t, const double *x, const double *y, int y_idx);
/* 把 batch 缓冲的平均梯度应用到参数上 */
void   trainer_apply_batch(Trainer *t, int batch_size);

/* 数据集：x 为 n×in_dim 行主序；y 为 n×out_dim 行主序；
 * 分类任务可用 y_idx（类别索引），传 NULL 表示用 y */
typedef struct {
    const double *x;
    const double *y;
    const int    *y_idx;
    int n;
    int in_dim;
    int out_dim;
} Dataset;

/* 对应 Model.fit：返回实际训练轮数；history_train/history_val 可为 NULL */
typedef struct {
    int epochs;
    int batch_size;     /* <=0 表示全批量 */
    int patience;       /* 0 表示不做 early stopping */
    double val_split;
    int shuffle;
    int verbose;
    int seed;
    double grad_clip;   /* >0 时按全局范数裁剪平均梯度，注意与 Python fit(grad_clip=) 同义 */
    struct LRScheduler *lr_scheduler;  /* 可为 NULL；每轮结束（验证/早停之后）调用其
                                        * step(epoch)，与 Python Model.fit 的时机一致。
                                        * 类型定义见 sched.h（此处只做前向声明） */
    /* 每轮结束后的回调（可为 NULL）：返回非 0 表示请求提前停止训练。
     * GUI 用它实现 Stop 按钮，也用来把外部的学习率同步进优化器。
     * epoch 从 1 开始；无验证集时 val_loss 为 -1。 */
    int (*on_epoch)(void *ctx, int epoch, double train_loss, double val_loss);
    void *on_epoch_ctx;
    /* 1 = 用训练损失做早停判据（默认 0：有验证集时用验证损失）。
     * 验证集太小或分布不同时，用户可能想直接盯训练损失。 */
    int early_stop_on_train;
} FitConfig;

int trainer_fit(Trainer *t, const Dataset *train,
                const Dataset *val,          /* 可为 NULL，用 val_split 时自动切分 */
                const FitConfig *cfg,
                double *history_train, double *history_val);

/* 前向预测多样本：out 为 n×out_dim 行主序 */
void trainer_predict(Trainer *t, const Dataset *d, double *out);
/* 分类准确率（用 y_idx） */
double trainer_accuracy(Trainer *t, const Dataset *d);
/* 回归 R²：多输出时对各输出维度取平均（与 Python 版一致） */
double trainer_r2(Trainer *t, const Dataset *d);

/* ================= 持久化（文本格式） ================= */
/* 格式：n_layers / sizes / acts / W / b，详见 ai.c */
int mlp_save(const MLP *m, const char *path);
MLP *mlp_load(const char *path);

/* ================= 工具 ================= */
double act_apply(Activation act, const double *z, double *out, int n);
void   act_deriv(Activation act, const double *z, const double *a, double *out, int n);
void   he_init(Layer *l, unsigned long long *rng_state);

#endif /* AI_H */
