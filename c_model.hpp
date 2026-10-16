/*
 * c_model.hpp —— 用纯 C 引擎实现的 C++ 兼容层
 *
 * 背景：界面（main.cpp）原本用的是 AI.hpp 那套 C++ 引擎。本文件提供一套**同名同签名**
 * 的最小 API（Model / Layer / MLP / make_regressor / make_classifier / quick_train /
 * StepLR / Conv2d / MaxPool2d / Flatten / Vec），内部全部转调 c/ 下的纯 C 引擎。
 * 于是 main.cpp 只要把 `#include "AI.hpp"` 换成 `#include "c_model.hpp"`、
 * `using namespace ai;` 换成 `using namespace ceng;` 就能切换引擎。
 *
 * 与旧引擎的差异（有意为之）：
 *   1. 模型存档沿用旧引擎的自定义文本格式（CONFIG: ... + 层 + 权重 + 偏置），
 *      所以之前存下来的模型文件仍然能加载。
 *   2. 回归的 R² 对多输出**按维度取平均**（与修好后的 Python 版一致；旧 C++ 引擎只算第 0 维）。
 *   3. 训练取消通过 std::atomic<bool>* stop_flag（每轮检查一次），行为与旧引擎一致。
 */
#ifndef C_MODEL_HPP
#define C_MODEL_HPP

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/* 补齐 AI.hpp 原本会带进来的标准库头：main.cpp 依赖它们（std::cout / std::random_device /
 * std::setprecision 等），换成这个兼容层后必须继续可用。 */
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ceng {

using Vec   = std::vector<double>;
using Vec2D = std::vector<Vec>;
using Vec3D = std::vector<Vec2D>;

/* ---------------- 随机数辅助（对应旧引擎 AI.hpp 的 seed / randu） ---------------- */
void   seed(int s);
double randu();

/* ---------------- 层（只读视图，供界面读取结构 / 存档使用） ---------------- */
struct Layer {
    enum class Activation { LEAKY_RELU, RELU, SOFTMAX, LINEAR };

    int fan_in = 0;
    int fan_out = 0;
    Activation act_type = Activation::LEAKY_RELU;
    std::string act_name = "leaky_relu";
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

struct Net {
    std::vector<Layer> layers;
    bool empty() const { return layers.empty(); }
};

/* ---------------- 优化器（界面只用 lr 与名字） ---------------- */
struct Optimizer {
    double lr = 0.01;
    std::string name = "adam";
};

/* ---------------- 学习率调度 ---------------- */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual void step(int epoch) = 0;
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer* opt, int step_size = 30, double gamma = 0.1);
    void step(int epoch) override;
private:
    Optimizer* opt_;
    int   step_size_;
    double gamma_;
    double initial_lr_;
};

/* ---------------- 前向数据 ---------------- */
struct Dataset;   // 占位：新引擎用 vector 形式，仅保留签名兼容

/* ---------------- Model：界面训练/预测/存档的全部入口 ---------------- */
class Model {
public:
    Model();
    ~Model();
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /* 配置字段（名字与旧引擎一致，界面直接读写） */
    std::string task = "regression";     // "regression" / "classification"
    std::string loss_name = "mse";
    std::string opt_name = "adam";
    double lr = 0.01;
    int    last_epochs = 0;
    int    last_batch = 0;
    double dropout_rate = 0.0;
    bool   early_stop_on_train = false;   // true = 用训练损失做早停判据
    double l2_lambda = 0.0;
    double test_split = 0.0;
    std::atomic<bool>* stop_flag = nullptr;
    std::shared_ptr<Optimizer> optimizer;
    /* 训练线程写入 history、界面线程读取 history 时用它加锁（名字与旧引擎一致） */
    mutable std::mutex history_mutex;
    Net network;
    std::vector<std::pair<int, double>> train_history;
    std::vector<std::pair<int, double>> val_history;

    /* 训练（对应旧引擎的 3 个 fit 重载） */
    void fit(const std::vector<Vec>& x, const std::vector<Vec>& y,
             int epochs = 100, int batch_size = 32, double val_split = 0.1,
             const Dataset* val_data = nullptr, int patience = 50,
             bool verbose = true, bool shuffle = true,
             LRScheduler* lr_scheduler = nullptr, double grad_clip = 0.0);

    void fit(const std::vector<Vec>& x, const std::vector<int>& y,
             int epochs = 100, int batch_size = 32, double val_split = 0.1,
             int patience = 50, bool verbose = true,
             LRScheduler* lr_scheduler = nullptr, double grad_clip = 0.0);

    /* 推理 */
    Vec predict_one(const Vec& x);
    int predict_class(const Vec& x);

    /* 评估：回归 = R²（多输出取各维平均），分类 = 准确率 */
    double score(const std::vector<Vec>& x, const std::vector<Vec>& y);
    double score(const std::vector<Vec>& x, const std::vector<int>& y);

    void summary();

    /* ---- 模型瘦身与统计（权重在 C 引擎内部，必须由本层操作） ----
     * prune_magnitude: 把 |w| < threshold 的权重置 0，返回被剪掉的个数
     * quantize_weights: 把权重按 bits 位定点量化（全局 min/max 均匀分档）
     * weight_count / zero_count: 参数总数与其中的 0 个数（稀疏度） */
    int    weight_count() const;
    int    zero_count() const;
    int    prune_magnitude(double threshold);
    /* this = wSelf*this + wOther*other（结构必须一致）。用于把 K 折模型平均成一个 */
    bool   merge_weights(const Model& other, double wSelf, double wOther);
    /* 重置权重初始化用的随机状态。做 K 折权重平均时必须让各折从同一初值出发，
     * 否则独立训练出来的网络隐藏单元排列不同，直接平均会得到无意义的模型。 */
    static void reset_init_seed(unsigned long long s = 0x9E3779B97F4A7C15ULL);
    double quantize_weights(int bits);   // 返回量化步长，<=0 表示未做

    /* 与 Python 版 AI.py 的 save_model / load_model 的 JSON 格式互通
     * （底层用 c/model_io.c，两边格式已验证逐位一致）。
     * 导出后可直接被 Python 版加载；导入时按输出层激活推断 task。 */
    bool save_python_json(const std::string& filepath);
    static Model load_python_json(const std::string& filepath);

    /* 存档：沿用旧引擎的文本格式 */
    void save(const std::string& filepath, int epochs = -1, int batch = -1);
    static Model load(const std::string& filepath,
                      const std::string& loss_fn = "mse",
                      const std::string& optimizer_name = "adam",
                      double lr_ = 0.01,
                      int* out_epochs = nullptr,
                      int* out_batch = nullptr);

    /* 内部实现（C 引擎的 MLP/Trainer），不对外暴露 */
    struct Impl;
    Impl* impl = nullptr;

    void sync_network_view();   // 把 C 侧权重同步到 network.layers（存档/界面用）
    void rebuild_from(const int* sizes, const int* acts, int n_sizes);
    int  epochs_run() const;
    int  batch_used() const;
};

/* ---------------- 构造辅助（对应旧引擎的自由函数） ---------------- */
Model make_regressor(int input_dim, int output_dim, std::vector<int> hidden,
                     std::string act = "leaky_relu", double lr = 0.01,
                     std::string opt = "adam");
Model make_classifier(int input_dim, int num_classes, std::vector<int> hidden,
                      std::string act = "leaky_relu", double lr = 0.01,
                      std::string opt = "adam");
Model quick_train(const std::vector<Vec>& x, const std::vector<Vec>& y,
                  const std::string& task, std::vector<int> hidden,
                  int epochs = 200, int batch = 32);
Model quick_train(const std::vector<Vec>& x, const std::vector<int>& y,
                  std::vector<int> hidden, int epochs = 200, int batch = 32);

/* ---------------- 卷积相关（CNN 示例用，包在 c/layers_conv.c 上） ---------------- */
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);
    ~Conv2d();
    Conv2d(const Conv2d&) = delete;
    Conv2d& operator=(const Conv2d&) = delete;
    Vec3D forward(const Vec3D& x);
private:
    void* impl_;
};

class MaxPool2d {
public:
    explicit MaxPool2d(int kernel_size, int stride = -1);
    ~MaxPool2d();
    MaxPool2d(const MaxPool2d&) = delete;
    MaxPool2d& operator=(const MaxPool2d&) = delete;
    Vec3D forward(const Vec3D& x);
private:
    void* impl_;
};

class Flatten {
public:
    Flatten();
    ~Flatten();
    Vec forward(const Vec3D& x);
private:
    void* impl_;
};

/* 一个可独立使用的前向 MLP（CNN 示例里当分类头用） */
class MLP {
public:
    MLP(std::vector<int> sizes, std::vector<std::string> activations);
    ~MLP();
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    Vec forward(const Vec& x);
    void* impl_ = nullptr;
};

} // namespace ceng

#endif /* C_MODEL_HPP */
