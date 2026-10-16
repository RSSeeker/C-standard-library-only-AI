/*
 * c_engine.h —— 把 c/ 下的纯 C 神经网络引擎接进 Win32 GUI 的桥接层
 *
 * 设计要点：
 *   1. 本头文件**不包含任何 C 引擎的头文件**，只用 STL 类型，避免 c/ai.h 里的
 *      Layer / MLP / Optimizer 等名字与 AI.hpp 里的同名类型冲突。
 *      C 头文件只在 c_engine.cpp 里包含（那是独立的编译单元）。
 *   2. 接口面向"界面要做的事"：训练一个模型、跑自检、跑基准，拿回可显示的文本与曲线。
 *   3. 训练在调用线程里同步执行；界面侧如需不卡死应当放到工作线程（见 main.cpp 的用法）。
 */
#ifndef C_ENGINE_H
#define C_ENGINE_H

#include <string>
#include <vector>

namespace ceng {

/* 训练结果（界面直接拿去显示） */
struct TrainResult {
    bool        ok = false;
    int         epochs_run = 0;
    double      train_loss = 0.0;
    double      val_loss = 0.0;      // 无验证集时为 -1
    double      score = 0.0;         // 分类=准确率，回归=R²
    double      seconds = 0.0;
    std::string log;                 // 多行文本，可直接追加到输出框
    std::vector<double> train_curve; // 每轮训练损失
    std::vector<double> val_curve;   // 每轮验证损失（可空）
};

struct TrainConfig {
    bool                     classification = false;
    std::vector<int>         hidden;              // 隐藏层宽度，如 {16, 8}
    std::string              optimizer = "adam";  // adam/sgd/momentum/rmsprop
    double                   lr = 0.01;
    int                      epochs = 500;
    int                      batch = 16;
    double                   val_split = 0.1;
    int                      patience = 50;
    double                   grad_clip = 0.0;     // >0 时按范数裁剪
    double                   lr_decay = 0.0;      // >0 时用 StepLR（每 100 轮 ×0.5）
    int                      seed = 0;
};

/* 用纯 C 引擎训练一个 MLP。
 *   x      : n×in_dim
 *   y      : n×out_dim（回归目标；分类传空）
 *   y_idx  : n（分类类别索引；回归传空）
 * 返回的结构里 log 是可以直接显示的文本。 */
TrainResult train_mlp(const std::vector<std::vector<double>>& x,
                      const std::vector<std::vector<double>>& y,
                      const std::vector<int>& y_idx,
                      const TrainConfig& cfg);

/* 用纯 C 引擎做一次有限差分梯度自检，返回可显示的报告文本 */
std::string self_test();

/* 跑一次性能基准（MLP[64,256,128,8]、1000 样本），返回可显示的报告文本 */
std::string benchmark(int epochs = 5, int samples = 1000);

/* 引擎能力清单（模块、验证结论），返回可显示文本 */
std::string info();

} // namespace ceng

#endif /* C_ENGINE_H */
