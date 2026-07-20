/*
================================================================================
  全栈神经网络工具箱 — C++ 版（纯 STL 实现，零第三方依赖）
  - Layer / MLP（全连接层 + 激活函数）
  - Conv2d / MaxPool2d（卷积与池化）
  - RNN / LSTM / GRU（循环神经网络，含完整 BPTT）
  - Transformer（MultiHeadAttention / FeedForward / Encoder / PositionalEncoding）
  - Dropout / BatchNorm1d / LayerNorm（正则化与归一化）
  - Embedding / ResidualBlock / Sequential（基础组件）
  - Adam / SGD / SGDMomentum / RMSprop（优化器）
  - MSE / CrossEntropy 损失函数
  - 梯度裁剪 / 学习率调度器
  - 训练管线 + Early Stopping + DataLoader
  - 模型保存 / 加载（JSON）
  - 高层 sklearn 风格 API（Model / make_classifier / make_regressor / quick_train）
================================================================================
*/
#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <cfloat>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <set>

namespace ai {

using Vec = std::vector<double>;
using Vec2D = std::vector<std::vector<double>>;
using Vec3D = std::vector<Vec2D>;
using Vec4D = std::vector<Vec3D>;
using GradPair = std::pair<Vec2D, Vec>;
using Dataset = std::vector<std::pair<Vec, Vec>>;
using ClsDataset = std::vector<std::pair<Vec, int>>;

// ============================================================================
//  随机数工具
// ============================================================================
inline std::mt19937& rng() {
    static std::mt19937 gen(42);
    return gen;
}
inline void seed(int s) { rng().seed(s); }

inline double randn(double stddev = 1.0) {
    static std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng()) * stddev;
}
inline double randu() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng());
}

// ============================================================================
//  Layer — 全连接层 + 激活函数
// ============================================================================
class Layer {
public:
    int fan_in, fan_out;
    Vec2D weights;
    Vec biases;
    Vec2D m_w, v_w;  // Adam 状态
    Vec m_b, v_b;
    Vec input, z, a;  // 前向缓存

    enum class Activation { LEAKY_RELU, RELU, SOFTMAX, LINEAR };
    Activation act_type;
    bool is_output;

    Layer(int fan_in, int fan_out, Activation act = Activation::LEAKY_RELU,
          bool is_output = false);

    Vec forward(const Vec& x);
    std::tuple<Vec2D, Vec, Vec> backward(const Vec& delta);

private:
    // 激活函数
    static double leaky_relu(double z)  { return z > 0 ? z : 0.01 * z; }
    static double leaky_relu_deriv(double z) { return z > 0 ? 1.0 : 0.01; }
    static double relu(double z)        { return z > 0 ? z : 0.0; }
    static double relu_deriv(double z)  { return z > 0 ? 1.0 : 0.0; }
    Vec _softmax(const Vec& z);
    Vec _softmax_deriv(const Vec& z);
    Vec _linear(const Vec& z)           { return z; }
};

// ============================================================================
//  MLP — 多层感知机
// ============================================================================
class MLP {
public:
    std::vector<Layer> layers;

    MLP() = default;
    MLP(const std::vector<int>& sizes,
        const std::vector<Layer::Activation>& activations);
    MLP(const std::vector<int>& sizes,
        const std::vector<std::string>& activations);

    Vec forward(Vec x);
    std::string to_string() const;
    int param_count() const;
};

// ============================================================================
//  Conv2d — 二维卷积
// ============================================================================
class Conv2d {
public:
    int in_channels, out_channels, kernel_size, stride, pad;
    int kh() const { return kernel_size; }
    int kw() const { return kernel_size; }

    Vec4D weights;   // [oc][ic][kh][kw]
    Vec biases;      // [oc]
    Vec4D grad_w, grad_w_acc;
    Vec grad_b, grad_b_acc;
    Vec3D input_cache;

    Conv2d(int in_c, int out_c, int kernel_sz, int stride = 1, int pad = 0);

    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
    void zero_grad();
};

// ============================================================================
//  MaxPool2d — 二维最大池化
// ============================================================================
class MaxPool2d {
public:
    int kernel_size;
    int pool_h() const { return kernel_size; }
    int pool_w() const { return kernel_size; }
    Vec3D mask;
    Vec3D input_shape;

    MaxPool2d(int kernel_sz);
    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
};

// ============================================================================
//  Flatten — 展平
// ============================================================================
class Flatten {
public:
    std::vector<int> input_shape;
    Vec forward(const Vec3D& x);
    Vec3D backward(const Vec& dout);
};

// ============================================================================
//  Dropout — 随机失活
// ============================================================================
class Dropout {
public:
    double p;
    bool training;
    Vec mask;

    Dropout(double p_ = 0.5);
    Vec forward(const Vec& x);
    Vec backward(const Vec& dout);
};

// ============================================================================
//  BatchNorm1d
// ============================================================================
class BatchNorm1d {
public:
    int dim;
    double eps, momentum;
    Vec gamma, beta;
    Vec grad_gamma, grad_beta;
    Vec running_mean, running_var;
    Vec x_hat, x_centered;
    double inv_std;
    bool training;

    BatchNorm1d(int dim_, double eps_ = 1e-5, double momentum_ = 0.1);
    Vec forward(const Vec& x);
    std::tuple<Vec, Vec, Vec> backward(const Vec& dout);
    void zero_grad();
};

// ============================================================================
//  LayerNorm
// ============================================================================
class LayerNorm {
public:
    int dim;
    double eps;
    Vec gamma, beta;
    Vec x_centered;
    double inv_std;
    Vec grad_gamma, grad_beta;

    LayerNorm(int dim_, double eps_ = 1e-5);
    Vec forward(const Vec& x);
    std::tuple<Vec, Vec, Vec> backward(const Vec& dout);
    void zero_grad();
};

// ============================================================================
//  RNN Cell
// ============================================================================
class RNNCell {
public:
    int input_size, hidden_size;
    Vec2D W_ih, W_hh;
    Vec b_ih, b_hh;
    // 梯度
    Vec2D grad_W_ih, grad_W_hh;
    Vec grad_b_ih, grad_b_hh;
    // 缓存
    Vec x_t, h_prev, h_t, tanh_z;
    Vec grad_x, grad_h_prev;

    RNNCell(int in_sz, int hid_sz);
    Vec forward(const Vec& x, const Vec& h_prev_);
    std::pair<Vec, Vec> backward(const Vec& dh);
    void zero_grad();
};

// ============================================================================
//  LSTM Cell
// ============================================================================
class LSTMCell {
public:
    int input_size, hidden_size;
    Vec2D W_ih, W_hh;
    Vec b_ih, b_hh;
    // 梯度
    Vec2D grad_W_ih, grad_W_hh;
    Vec grad_b_ih, grad_b_hh;
    // 缓存
    Vec x_t, h_prev, c_prev;
    Vec i_gate, f_gate, g_gate, o_gate;  // 原始值
    Vec i_sig, f_sig, g_tanh, o_sig;      // 激活后
    Vec c_t, h_t;

    LSTMCell(int in_sz, int hid_sz);
    std::pair<Vec, Vec> forward(const Vec& x, const Vec& h_prev_, const Vec& c_prev_);
    std::tuple<Vec, Vec, Vec> backward(const Vec& dh, const Vec& dc_next);
    void zero_grad();
};

// ============================================================================
//  GRU Cell
// ============================================================================
class GRUCell {
public:
    int input_size, hidden_size;
    Vec2D W_ih, W_hh;
    Vec b_ih, b_hh;
    // 梯度
    Vec2D grad_W_ih, grad_W_hh;
    Vec grad_b_ih, grad_b_hh;
    // 缓存
    Vec x_t, h_prev;
    Vec z_gate, r_gate, n_gate;
    Vec z_sig, r_sig, n_tanh;
    Vec h_t;
    Vec grad_x, grad_h_prev;

    GRUCell(int in_sz, int hid_sz);
    Vec forward(const Vec& x, const Vec& h_prev_);
    std::pair<Vec, Vec> backward(const Vec& dh);
    void zero_grad();
};

// ============================================================================
//  RNN / LSTM / GRU 容器
// ============================================================================
class RNN {
public:
    int input_size, hidden_size, num_layers;
    std::vector<RNNCell> cells;
    RNN(int in_sz, int hid_sz, int n_layers = 1);
    std::pair<std::vector<Vec>, std::vector<Vec>> forward(const std::vector<Vec>& x_seq);
    std::vector<Vec> backward(const std::vector<Vec>& dh_seq);
};

class LSTM {
public:
    int input_size, hidden_size, num_layers;
    std::vector<LSTMCell> cells;
    LSTM(int in_sz, int hid_sz, int n_layers = 1);
    std::pair<std::vector<Vec>, std::pair<std::vector<Vec>, std::vector<Vec>>>
        forward(const std::vector<Vec>& x_seq);
    void backward(const std::vector<Vec>& dh_seq);
};

class GRU {
public:
    int input_size, hidden_size, num_layers;
    std::vector<GRUCell> cells;
    GRU(int in_sz, int hid_sz, int n_layers = 1);
    std::pair<std::vector<Vec>, std::vector<Vec>> forward(const std::vector<Vec>& x_seq);
    std::vector<Vec> backward(const std::vector<Vec>& dh_seq);
};

// ============================================================================
//  Transformer 组件
// ============================================================================
class MultiHeadAttention {
public:
    int d_model, num_heads, d_k;
    Vec2D W_q, W_k, W_v, W_o;
    Vec b_q, b_k, b_v, b_o;
    Vec2D grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Vec grad_b_q, grad_b_k, grad_b_v, grad_b_o;
    // 缓存
    Vec3D q_mat, k_mat, v_mat;
    Vec3D attn_weights;
    Vec3D input_cache;

    MultiHeadAttention(int d_model_, int num_heads_);
    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
    void zero_grad();
};

class FeedForward {
public:
    int d_model, d_ff;
    Vec2D W1, W2;
    Vec b1, b2;
    Vec2D grad_W1, grad_W2;
    Vec grad_b1, grad_b2;
    Vec3D input_cache, cache_after_w1, cache_hidden;

    FeedForward(int d_model_, int d_ff_);
    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
    void zero_grad();
};

class TransformerEncoderLayer {
public:
    MultiHeadAttention attn;
    FeedForward ff;
    LayerNorm ln1, ln2;
    int d_model;

    TransformerEncoderLayer(int d_model_, int num_heads, int d_ff);
    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
    void zero_grad();
};

class TransformerEncoder {
public:
    int d_model;
    std::vector<TransformerEncoderLayer> layers;
    TransformerEncoder(int d_model_, int num_heads, int d_ff, int num_layers = 1);
    Vec3D forward(const Vec3D& x);
    Vec3D backward(const Vec3D& dout);
    void zero_grad();
};

class PositionalEncoding {
public:
    int d_model;
    Vec2D pe;
    PositionalEncoding(int d_model_, int max_len = 5000);
    Vec3D forward(const Vec3D& x);
};

// ============================================================================
//  Embedding
// ============================================================================
class Embedding {
public:
    int vocab_size, embed_dim;
    Vec2D weight;
    Vec2D grad_w;
    std::vector<int> indices_cache;
    Embedding(int vocab_sz, int embed_dim_);
    Vec2D forward(const std::vector<int>& indices);
    Vec2D backward(const Vec2D& dout);
    void zero_grad();
};

// ============================================================================
//  ResidualBlock
// ============================================================================
class ResidualBlock {
public:
    MLP main_path;
    LayerNorm ln;
    // 梯度缓存 — backward() 调用后将 MLP 梯度存入此处
    std::vector<GradPair> mlp_grads;
    ResidualBlock(int dim, int hidden_dim = 0);
    Vec forward(const Vec& x);
    Vec backward(const Vec& dout);
    void zero_grad();
};

// ============================================================================
//  Sequential
// ============================================================================
class SequentialOp {
public:
    enum OpType { LINEAR, RELU, LEAKY_RELU, SOFTMAX, DROPOUT, LAYER_NORM };
    OpType type;
    int dim;
    double p;
    // Linear
    Vec2D W, grad_W;
    Vec b, grad_b;
    Vec input_cache, z_cache;
    // LN
    Vec gamma, beta;
    Vec grad_gamma, grad_beta;
    Vec ln_mean, ln_var, ln_x_hat;
    // Dropout
    Vec drop_mask;

    SequentialOp(OpType t, int d = 0, double p_ = 0.0);
    Vec forward(const Vec& x);
    Vec backward(const Vec& dout);
    void zero_grad();
};

// ============================================================================
//  Optimizers
// ============================================================================
class Optimizer {
public:
    MLP* model;
    double lr;
    Optimizer(MLP* m, double lr_);
    virtual void step(const std::vector<GradPair>& grads) = 0;
    virtual ~Optimizer() = default;
};

class Adam : public Optimizer {
public:
    double beta1, beta2, eps;
    int t;
    std::vector<Vec2D> m_w, v_w;
    std::vector<Vec> m_b, v_b;

    Adam(MLP* m, double lr_ = 0.01, double b1 = 0.9, double b2 = 0.999,
         double eps_ = 1e-8);
    void step(const std::vector<GradPair>& grads) override;
};

class SGD : public Optimizer {
public:
    SGD(MLP* m, double lr_ = 0.01);
    void step(const std::vector<GradPair>& grads) override;
};

class SGDMomentum : public Optimizer {
public:
    double momentum;
    std::vector<Vec2D> v_w;
    std::vector<Vec> v_b;
    SGDMomentum(MLP* m, double lr_ = 0.01, double momentum_ = 0.9);
    void step(const std::vector<GradPair>& grads) override;
};

class RMSprop : public Optimizer {
public:
    double beta, eps;
    std::vector<Vec2D> cache_w;
    std::vector<Vec> cache_b;
    RMSprop(MLP* m, double lr_ = 0.01, double beta_ = 0.9, double eps_ = 1e-8);
    void step(const std::vector<GradPair>& grads) override;
};

// ============================================================================
//  Loss Functions
// ============================================================================
std::pair<double, Vec> mse_loss(const Vec& pred, const Vec& target);
std::pair<double, Vec> cross_entropy_loss(const Vec& pred, int target_idx);
std::pair<double, Vec> cross_entropy_loss_vec(const Vec& pred, const Vec& target_vec);

// ============================================================================
//  Gradient Clipping
// ============================================================================
double grad_norm(const std::vector<GradPair>& grads);
void clip_grad_by_norm(std::vector<GradPair>& grads, double max_norm);
void clip_grad_by_value(std::vector<GradPair>& grads, double clip_val);

// ============================================================================
//  Learning Rate Schedulers
// ============================================================================
class LRScheduler {
public:
    Optimizer* optimizer;
    LRScheduler(Optimizer* opt);
    virtual void step(int epoch) = 0;
    virtual ~LRScheduler() = default;
};

class StepLR : public LRScheduler {
public:
    int step_size;
    double gamma;
    StepLR(Optimizer* opt, int step_sz, double gamma_);
    void step(int epoch) override;
};

class ExponentialLR : public LRScheduler {
public:
    double gamma;
    ExponentialLR(Optimizer* opt, double gamma_);
    void step(int epoch) override;
};

class CosineAnnealingLR : public LRScheduler {
public:
    int T_max;
    double eta_min;
    double initial_lr;
    CosineAnnealingLR(Optimizer* opt, int T_max_, double eta_min_ = 0.0);
    void step(int epoch) override;
};

// ============================================================================
//  Data Utilities
// ============================================================================
struct SplitResult {
    std::vector<Dataset> splits;  // 0=train, 1=val, 2=test
};

SplitResult split_data(const Dataset& data, double val_ratio = 0.2,
                       double test_ratio = 0.1, int seed = 42);

class DataLoader {
public:
    Dataset data;
    int batch_size;
    DataLoader(const Dataset& d, int bs, bool shuffle_ = true);
    bool has_next();
    Dataset next_batch();
    void reset();
    int size() const;
private:
    bool do_shuffle;
    int idx;
    std::vector<int> indices;
};

// ============================================================================
//  Text Processing
// ============================================================================
class TextProcessor {
public:
    std::vector<char> chars;
    std::unordered_map<char, int> char_to_idx;
    std::unordered_map<int, char> idx_to_char;
    int vocab_size;

    TextProcessor(const std::string& text);
    Vec encode_one_hot(char c);
    std::vector<int> encode_indices(const std::string& text);
    std::string decode_indices(const std::vector<int>& indices);
    Dataset prepare_data(const std::string& text, int window_size);
    ClsDataset prepare_index_data(const std::string& text, int window_size);
};

// ============================================================================
//  High-Level API (sklearn-style)
// ============================================================================
std::vector<GradPair> mlp_backward(MLP& model, const Vec& loss_grad);

Vec softmax(const Vec& logits);
Vec softmax_with_temp(const Vec& logits, double temperature = 1.0);
int sample_index(const Vec& probs);
std::string generate_text(MLP& network, TextProcessor& tp,
                          const std::string& start_str,
                          int length = 15, int window_size = 3, double temp = 1.0);

class Model {
public:
    MLP network;
    std::string task;         // "regression" or "classification"
    std::string loss_name;    // "mse", "cross_entropy"
    std::string opt_name;     // "adam", "sgd", etc.
    std::unique_ptr<Optimizer> optimizer;
    double lr;
    double dropout_rate;
    double l2_lambda;
    double test_split;
    bool trained;

    // History: [(epoch, train_loss), ...]
    std::vector<std::pair<int, double>> train_history;
    std::vector<std::pair<int, double>> val_history;
    std::vector<std::pair<int, double>> test_history;

    Model() : dropout_rate(0.0), l2_lambda(0.0), test_split(0.0) {}

    // loss_fn: "mse" | "cross_entropy" | "ce"
    // optimizer_name: "adam" | "sgd" | "sgd_momentum" | "rmsprop"
    Model(MLP net, const std::string& loss_fn = "mse",
          const std::string& optimizer_name = "adam", double lr_ = 0.01,
          const std::string& task_ = "auto");

    Vec forward(const Vec& x);
    Vec predict_one(const Vec& x);
    std::vector<Vec> predict(const std::vector<Vec>& x);
    int predict_class(const Vec& x);
    std::vector<int> predict_classes(const std::vector<Vec>& x);

    double evaluate(const Dataset& data);
    double evaluate(const std::vector<Vec>& x, const std::vector<Vec>& y);
    double score(const std::vector<Vec>& x, const std::vector<Vec>& y);   // 回归 R²
    double score(const std::vector<Vec>& x, const std::vector<int>& y);   // 分类 accuracy

    void fit(const Dataset& data, int epochs = 100, int batch_size = 32,
             double val_split = 0.1, const Dataset* val_data = nullptr,
             int patience = 50, bool verbose = true, bool shuffle = true,
             LRScheduler* lr_scheduler = nullptr, double grad_clip = 0.0);

    void fit(const std::vector<Vec>& x, const std::vector<Vec>& y, int epochs = 100,
             int batch_size = 32, double val_split = 0.1,
             const Dataset* val_data = nullptr,
             int patience = 50, bool verbose = true, bool shuffle = true,
             LRScheduler* lr_scheduler = nullptr, double grad_clip = 0.0);

    void fit(const std::vector<Vec>& x, const std::vector<int>& y,
             int epochs = 100, int batch_size = 32, double val_split = 0.1,
             int patience = 50, bool verbose = true,
             LRScheduler* lr_scheduler = nullptr, double grad_clip = 0.0);

    void save(const std::string& filepath, int epochs = -1, int batch = -1);
    static Model load(const std::string& filepath,
                      const std::string& loss_fn = "mse",
                      const std::string& optimizer_name = "adam",
                      double lr_ = 0.01,
                      int* out_epochs = nullptr,
                      int* out_batch = nullptr);
    void summary();
    std::string to_string() const;
};

// 工厂函数
Model make_classifier(int input_dim, int num_classes,
                      const std::vector<int>& hidden_layers = {32},
                      const std::string& act = "leaky_relu",
                      double lr = 0.01, const std::string& optimizer_name = "adam");

Model make_regressor(int input_dim, int output_dim = 1,
                     const std::vector<int>& hidden_layers = {32},
                     const std::string& act = "leaky_relu",
                     double lr = 0.01, const std::string& optimizer_name = "adam");

Model quick_train(const std::vector<Vec>& x, const std::vector<Vec>& y,
                  const std::string& task = "regression",
                  const std::vector<int>& hidden_layers = {32},
                  int epochs = 200, int batch_size = 32, double lr = 0.01,
                  double val_split = 0.1, int patience = 30,
                  const std::string& optimizer_name = "adam",
                  bool verbose = true);

Model quick_train(const std::vector<Vec>& x, const std::vector<int>& y,
                  const std::vector<int>& hidden_layers = {32},
                  int epochs = 200, int batch_size = 32, double lr = 0.01,
                  double val_split = 0.1, int patience = 30,
                  const std::string& optimizer_name = "adam",
                  bool verbose = true);

} // namespace ai
