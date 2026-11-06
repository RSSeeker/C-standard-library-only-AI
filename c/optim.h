/*
 * optim.h —— 解耦权重衰减优化器（AdamW / Lion / Adafactor）
 *
 * 对应 AI.py：
 *   AdamW      (AI.py:16954)  -> AdamW
 *   Lion       (AI.py:17099)  -> Lion
 *   Adafactor  (AI.py:17149)  -> Adafactor
 *
 * 与 ai.h 里 Optimizer 的分工：
 *   Adam / SGD / SGD-Momentum / RMSprop 直接吃 MLP，动量放在 Layer 的 mW/vW/mb/vb 里；
 *   这三个优化器要逐层额外的状态（Adafactor 还有行/列统计量），塞不进 Layer，
 *   所以改成"参数视图"接口：调用方把每一层摊成 OptLayer，优化器自己持有状态。
 *   从 ai.h 的 MLP 构造（W 为行主序 fan_out×fan_in，与 Python 的 weights 嵌套 list
 *   逐元素顺序一致，所以不需要任何重排）：
 *
 *     int n = 0;
 *     for (int k = 0; k < m->n_layers; k++) {
 *         Layer *L = &m->layers[k];
 *         ls[n++] = (OptLayer){ L->W, L->b, L->gW, L->gb, L->fan_out, L->fan_in };
 *     }
 *     AdamW *opt = adamw_create(ls, n, 1e-3, 0.9, 0.999, 1e-8, 0.01, &err);
 *
 * 数值约定（与 Python 逐位对齐的关键，踩过的坑记在这里）：
 *   1. 就地更新，运算顺序与括号照抄 Python，不做任何代数化简。AdamW 的
 *      `B[j] *= decay; B[j] -= lr * (m/c1) / (sqrt(v/c2) + eps)` 必须分两步写：
 *      合成一个表达式就少一次舍入，最后一位会变。
 *   2. 只有 Python 用内置 sum() 的地方才走 numeric.h 的 Neumaier 求和
 *      （本文件里只有 Adafactor 的 mean_R = sum(R)/fo 属于这种情况）；
 *      其余全是朴素 += 与 sqrt/pow，与 Python 的逐元素运算一一对应。
 *   3. 偏差修正 c1/c2 用 libm pow（Python 的 `beta ** t` 落地就是 pow）。
 *      同一个数在本机 CPython（UCRT）与 MinGW gcc（msvcrt.dll）上 pow 可能差 1 ULP，
 *      所以对拍用例里既有"二进制精确超参"的严格逐位用例，也有标注 tol=1e-12 的
 *      真实超参用例，详见 tools/adamw_reference.py 的 note。
 *
 * 错误约定：create 失败返回 NULL，并在 err 非 NULL 时写入静态错误串（string literal，
 * 调用方不要 free，也不要长期持有）；step 成功返回 0，参数不合法返回 -1。
 */
#ifndef AI_OPTIM_H
#define AI_OPTIM_H

/* 一层参数 + 对应梯度。w 为行主序 fan_out×fan_in，b 长度 fan_out。
 * 梯度指针由调用方维护：优化器每步现读，所以训练时原地覆盖梯度缓冲即可。 */
typedef struct {
    double *w;
    double *b;
    const double *gw;
    const double *gb;
    int fan_out;
    int fan_in;
} OptLayer;

/* ============================== AdamW ==============================
 * Python: decay = 1 - lr*wd；c1 = 1 - b1^t；c2 = 1 - b2^t
 *         m = b1*m + (1-b1)*g；v = b2*v + (1-b2)*g^2
 *         w *= decay；w -= lr * (m/c1) / (sqrt(v/c2) + eps)
 * 注意顺序：先乘衰减再减更新量（与 Adam 的 L2 不同，wd 不进梯度、不被 sqrt(v) 缩放）。
 * 状态：每层 m_w/v_w（fan_out×fan_in）、m_b/v_b（fan_out），全零初始。
 */
typedef struct AdamW AdamW;

/* lr/beta1/beta2/eps/weight_decay 与 Python 同名参数同序；layers 会按值拷一份
 * （只拷形状与指针，不拷参数数据）。weight_decay < 0 报错（与 Python 一致），
 * beta 不做范围校验（Python 也没有校验）。 */
AdamW *adamw_create(const OptLayer *layers, int n_layers,
                    double lr, double beta1, double beta2, double eps,
                    double weight_decay, const char **err);
/* 走一步：step_count += 1，然后用 layers[i].gw/gb 里的当前梯度更新 w/b。
 * 梯度缓冲必须非 NULL 且形状与 create 时一致。 */
int  adamw_step(AdamW *self, const char **err);
void adamw_free(AdamW *self);

int    adamw_step_count(const AdamW *self);
int    adamw_num_layers(const AdamW *self);
double adamw_lr(const AdamW *self);
void   adamw_set_lr(AdamW *self, double lr);   /* 调度器就改这个，下一步生效 */
double adamw_weight_decay(const AdamW *self);

/* 状态只读访问（对拍/调试用）：k 为层下标，越界返回 NULL。
 * m_w/v_w 长度 fan_out*fan_in（行主序），m_b/v_b 长度 fan_out。 */
const double *adamw_m_w(const AdamW *self, int k);
const double *adamw_v_w(const AdamW *self, int k);
const double *adamw_m_b(const AdamW *self, int k);
const double *adamw_v_b(const AdamW *self, int k);

#endif /* AI_OPTIM_H */
