#include "AI.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ai {

// ============================================================================
//  Layer
// ============================================================================
Layer::Layer(int fan_in_, int fan_out_, Activation act_, bool is_output_)
    : fan_in(fan_in_), fan_out(fan_out_), act_type(act_), is_output(is_output_)
{
    double std_dev = std::sqrt(2.0 / fan_in);
    weights.resize(fan_out, Vec(fan_in));
    for (auto& row : weights)
        for (auto& w : row) w = randn(std_dev);
    biases.assign(fan_out, 0.0);

    // Adam 状态
    m_w.assign(fan_out, Vec(fan_in, 0.0));
    v_w.assign(fan_out, Vec(fan_in, 0.0));
    m_b.assign(fan_out, 0.0);
    v_b.assign(fan_out, 0.0);
}

Vec Layer::_softmax(const Vec& z) {
    Vec out(z.size());
    double mx = *std::max_element(z.begin(), z.end());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        out[i] = std::exp(z[i] - mx);
        sum += out[i];
    }
    for (auto& v : out) v /= sum;
    return out;
}

Vec Layer::_softmax_deriv(const Vec& z) {
    // 对交叉熵来说，联合梯度 = softmax(z) - one_hot，不需要单独使用
    (void)z;
    return {};
}

Vec Layer::forward(const Vec& x) {
    input = x;
    z.resize(fan_out);
    for (int j = 0; j < fan_out; ++j) {
        z[j] = biases[j];
        for (int i = 0; i < fan_in; ++i)
            z[j] += weights[j][i] * x[i];
    }

    // 激活函数
    switch (act_type) {
    case Activation::LEAKY_RELU:
        a.resize(fan_out);
        for (int j = 0; j < fan_out; ++j)
            a[j] = leaky_relu(z[j]);
        break;
    case Activation::RELU:
        a.resize(fan_out);
        for (int j = 0; j < fan_out; ++j)
            a[j] = relu(z[j]);
        break;
    case Activation::SOFTMAX:
        a = _softmax(z);
        break;
    case Activation::LINEAR:
        a = z;
        break;
    }
    return a;
}

std::tuple<Vec2D, Vec, Vec> Layer::backward(const Vec& delta) {
    // 激活函数反向
    Vec d_z(fan_out);
    switch (act_type) {
    case Activation::LEAKY_RELU:
        for (int j = 0; j < fan_out; ++j)
            d_z[j] = delta[j] * leaky_relu_deriv(z[j]);
        break;
    case Activation::RELU:
        for (int j = 0; j < fan_out; ++j)
            d_z[j] = delta[j] * relu_deriv(z[j]);
        break;
    case Activation::SOFTMAX:
    case Activation::LINEAR:
        d_z = delta;
        break;
    }

    // 权重 / 偏置梯度
    Vec2D grad_w(fan_out, Vec(fan_in, 0.0));
    Vec grad_b(fan_out);
    Vec d_input(fan_in, 0.0);

    for (int j = 0; j < fan_out; ++j) {
        grad_b[j] = d_z[j];
        for (int i = 0; i < fan_in; ++i) {
            grad_w[j][i] = d_z[j] * input[i];
            d_input[i] += d_z[j] * weights[j][i];
        }
    }

    return {grad_w, grad_b, d_input};
}

// ============================================================================
//  MLP
// ============================================================================
MLP::MLP(const std::vector<int>& sizes,
         const std::vector<Layer::Activation>& activations) {
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        bool is_out = (i == sizes.size() - 2);
        layers.emplace_back(sizes[i], sizes[i + 1], activations[i], is_out);
    }
}

MLP::MLP(const std::vector<int>& sizes,
         const std::vector<std::string>& activations) {
    std::vector<Layer::Activation> acts;
    for (const auto& a : activations) {
        if (a == "leaky_relu")  acts.push_back(Layer::Activation::LEAKY_RELU);
        else if (a == "relu")    acts.push_back(Layer::Activation::RELU);
        else if (a == "softmax") acts.push_back(Layer::Activation::SOFTMAX);
        else if (a == "linear")  acts.push_back(Layer::Activation::LINEAR);
        else throw std::runtime_error("Unknown activation: " + a);
    }
    *this = MLP(sizes, acts);
}

Vec MLP::forward(Vec x) {
    for (auto& layer : layers)
        x = layer.forward(x);
    return x;
}

std::string MLP::to_string() const {
    std::ostringstream oss;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (i > 0) oss << " → ";
        oss << layers[i].fan_in << "→" << layers[i].fan_out;
    }
    return oss.str();
}

int MLP::param_count() const {
    int total = 0;
    for (auto& l : layers)
        total += l.fan_in * l.fan_out + l.fan_out;
    return total;
}

// ============================================================================
//  Conv2d
// ============================================================================
Conv2d::Conv2d(int in_c, int out_c, int kernel_sz, int stride_, int pad_)
    : in_channels(in_c), out_channels(out_c), kernel_size(kernel_sz),
      stride(stride_), pad(pad_)
{
    weights.resize(out_channels,
        Vec3D(in_channels, Vec2D(kernel_sz, Vec(kernel_sz))));
    double std_dev = std::sqrt(2.0 / (in_channels * kernel_sz * kernel_sz));
    for (int oc = 0; oc < out_channels; ++oc)
        for (int ic = 0; ic < in_channels; ++ic)
            for (int h = 0; h < kernel_sz; ++h)
                for (int w = 0; w < kernel_sz; ++w)
                    weights[oc][ic][h][w] = randn(std_dev);
    biases.assign(out_channels, 0.0);
    grad_w = weights;   // 同形状
    grad_b.assign(out_channels, 0.0);
    grad_w_acc = weights;
    grad_b_acc = biases;
}

Vec3D Conv2d::forward(const Vec3D& x) {
    input_cache = x;
    int H = (int)x[0].size();
    int W = (int)x[0][0].size();
    int H_out = (H + 2 * pad - kernel_size) / stride + 1;
    int W_out = (W + 2 * pad - kernel_size) / stride + 1;

    Vec3D out(out_channels, Vec2D(H_out, Vec(W_out, 0.0)));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                double val = biases[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride + kh - pad;
                            int iw = w * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                val += weights[oc][ic][kh][kw] * x[ic][ih][iw];
                        }
                    }
                }
                out[oc][h][w] = val;
            }
        }
    }
    return out;
}

Vec3D Conv2d::backward(const Vec3D& dout) {
    int H = (int)input_cache[0].size();
    int W = (int)input_cache[0][0].size();
    int H_out = (int)dout[0].size();
    int W_out = (int)dout[0][0].size();

    Vec3D dx(in_channels, Vec2D(H, Vec(W, 0.0)));

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                double d = dout[oc][h][w];
                grad_b_acc[oc] += d;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = h * stride + kh - pad;
                            int iw = w * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                grad_w_acc[oc][ic][kh][kw] += d * input_cache[ic][ih][iw];
                                dx[ic][ih][iw] += d * weights[oc][ic][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
    // 拷贝到单样本梯度
    grad_w = grad_w_acc;
    grad_b = grad_b_acc;
    return dx;
}

void Conv2d::zero_grad() {
    for (auto& a : grad_w_acc)
        for (auto& b : a)
            for (auto& c : b) std::fill(c.begin(), c.end(), 0.0);
    std::fill(grad_b_acc.begin(), grad_b_acc.end(), 0.0);
    // 同步清零对外暴露的梯度
    grad_w = grad_w_acc;
    grad_b = grad_b_acc;
}

// ============================================================================
//  MaxPool2d
// ============================================================================
MaxPool2d::MaxPool2d(int kernel_sz) : kernel_size(kernel_sz) {}

Vec3D MaxPool2d::forward(const Vec3D& x) {
    input_shape = x;
    int C = (int)x.size();
    int H = (int)x[0].size();
    int W = (int)x[0][0].size();
    int H_out = H / pool_h();
    int W_out = W / pool_w();

    Vec3D out(C, Vec2D(H_out, Vec(W_out, 0.0)));
    mask = Vec3D(C, Vec2D(H, Vec(W, 0.0)));

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                double mx = -DBL_MAX;
                int max_h = 0, max_w = 0;
                for (int ph = 0; ph < pool_h(); ++ph) {
                    for (int pw = 0; pw < pool_w(); ++pw) {
                        int ih = h * pool_h() + ph;
                        int iw = w * pool_w() + pw;
                        if (x[c][ih][iw] > mx) {
                            mx = x[c][ih][iw];
                            max_h = ih; max_w = iw;
                        }
                    }
                }
                out[c][h][w] = mx;
                mask[c][max_h][max_w] = 1.0;
            }
        }
    }
    return out;
}

Vec3D MaxPool2d::backward(const Vec3D& dout) {
    int C = (int)input_shape.size();
    int H = (int)input_shape[0].size();
    int W = (int)input_shape[0][0].size();
    Vec3D dx(C, Vec2D(H, Vec(W, 0.0)));

    int H_out = (int)dout[0].size();
    int W_out = (int)dout[0][0].size();
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w)
                for (int ph = 0; ph < pool_h(); ++ph)
                    for (int pw = 0; pw < pool_w(); ++pw) {
                        int ih = h * pool_h() + ph;
                        int iw = w * pool_w() + pw;
                        dx[c][ih][iw] += dout[c][h][w] * mask[c][ih][iw];
                    }
    return dx;
}

// ============================================================================
//  Flatten
// ============================================================================
Vec Flatten::forward(const Vec3D& x) {
    input_shape.clear();
    input_shape.push_back((int)x.size());
    input_shape.push_back((int)x[0].size());
    input_shape.push_back((int)x[0][0].size());
    Vec out;
    for (auto& c : x)
        for (auto& h : c)
            for (auto v : h) out.push_back(v);
    return out;
}

Vec3D Flatten::backward(const Vec& dout) {
    int idx = 0;
    Vec3D out(input_shape[0], Vec2D(input_shape[1], Vec(input_shape[2])));
    for (int c = 0; c < input_shape[0]; ++c)
        for (int h = 0; h < input_shape[1]; ++h)
            for (int w = 0; w < input_shape[2]; ++w)
                out[c][h][w] = dout[idx++];
    return out;
}

// ============================================================================
//  Dropout
// ============================================================================
Dropout::Dropout(double p_) : p(std::min(p_, 0.9)), training(true) {}

Vec Dropout::forward(const Vec& x) {
    if (!training) return x;
    double scale = 1.0 / (1.0 - p);
    mask.resize(x.size());
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        mask[i] = (randu() > p) ? scale : 0.0;
        out[i] = x[i] * mask[i];
    }
    return out;
}

Vec Dropout::backward(const Vec& dout) {
    if (!training) return dout;
    Vec dx(dout.size());
    for (size_t i = 0; i < dout.size(); ++i)
        dx[i] = dout[i] * mask[i];
    return dx;
}

// ============================================================================
//  BatchNorm1d
// ============================================================================
BatchNorm1d::BatchNorm1d(int dim_, double eps_, double momentum_)
    : dim(dim_), eps(eps_), momentum(momentum_)
{
    gamma.assign(dim, 1.0);
    beta.assign(dim, 0.0);
    grad_gamma.assign(dim, 0.0);
    grad_beta.assign(dim, 0.0);
    running_mean.assign(dim, 0.0);
    running_var.assign(dim, 1.0);
    training = true;
}

Vec BatchNorm1d::forward(const Vec& x) {
    if (training) {
        // 计算均值
        double mean = std::accumulate(x.begin(), x.end(), 0.0) / dim;
        double var = 0.0;
        for (auto v : x) var += (v - mean) * (v - mean);
        var = var / dim + eps;
        inv_std = 1.0 / std::sqrt(var);

        // 更新 running stats
        for (int i = 0; i < dim; ++i) {
            running_mean[i] = momentum * running_mean[i] + (1 - momentum) * mean;
            running_var[i] = momentum * running_var[i] + (1 - momentum) * var;
        }

        x_centered.resize(dim);
        x_hat.resize(dim);
        Vec out(dim);
        for (int i = 0; i < dim; ++i) {
            x_centered[i] = x[i] - mean;
            x_hat[i] = x_centered[i] * inv_std;
            out[i] = gamma[i] * x_hat[i] + beta[i];
        }
        return out;
    } else {
        Vec out(dim);
        for (int i = 0; i < dim; ++i) {
            double x_hat_i = (x[i] - running_mean[i]) / std::sqrt(running_var[i] + eps);
            out[i] = gamma[i] * x_hat_i + beta[i];
        }
        return out;
    }
}

std::tuple<Vec, Vec, Vec> BatchNorm1d::backward(const Vec& dout) {
    // 完整链式求导（通过 mean 和 var 的间接依赖）
    // d_gamma_i = dout_i * x_hat_i
    // d_beta_i  = dout_i
    // dx_i = gamma_i * inv_std * (dout_i - mean(dout) - x_hat_i * mean(dout * x_hat))
    Vec d_gamma(dim), d_beta(dim), dx(dim);

    Vec d_xhat(dim);
    double mean_dout = 0.0;
    double mean_dout_xhat = 0.0;
    for (int i = 0; i < dim; ++i) {
        d_gamma[i] = dout[i] * x_hat[i];
        d_beta[i] = dout[i];
        grad_gamma[i] += d_gamma[i];   // 累积到成员变量
        grad_beta[i] += d_beta[i];

        d_xhat[i] = dout[i] * gamma[i];
        mean_dout += dout[i];
        mean_dout_xhat += dout[i] * gamma[i] * x_hat[i];
    }
    mean_dout /= dim;
    mean_dout_xhat /= dim;

    for (int i = 0; i < dim; ++i) {
        dx[i] = gamma[i] * inv_std * (dout[i] - mean_dout - x_hat[i] * mean_dout_xhat);
    }

    return {d_gamma, d_beta, dx};
}

void BatchNorm1d::zero_grad() {
    grad_gamma.assign(dim, 0.0);
    grad_beta.assign(dim, 0.0);
}

// ============================================================================
//  LayerNorm
// ============================================================================
LayerNorm::LayerNorm(int dim_, double eps_) : dim(dim_), eps(eps_) {
    gamma.assign(dim, 1.0);
    beta.assign(dim, 0.0);
    grad_gamma.assign(dim, 0.0);
    grad_beta.assign(dim, 0.0);
}

Vec LayerNorm::forward(const Vec& x) {
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / dim;
    double var = 0.0;
    for (auto v : x) var += (v - mean) * (v - mean);
    var = var / dim;
    inv_std = 1.0 / std::sqrt(var + eps);

    x_centered = x;
    Vec out(dim);
    for (int i = 0; i < dim; ++i) {
        x_centered[i] = x[i] - mean;
        out[i] = gamma[i] * x_centered[i] * inv_std + beta[i];
    }
    return out;
}

std::tuple<Vec, Vec, Vec> LayerNorm::backward(const Vec& dout) {
    // 完整链式求导（通过 mean 和 var 的间接依赖）
    // dx = (inv_std / dim) * (dim * d_xhat - sum(d_xhat) - x_hat * sum(d_xhat * x_hat))
    // 其中 d_xhat_i = dout_i * gamma_i
    Vec d_gamma(dim), d_beta(dim), dx(dim);

    Vec d_xhat(dim);
    double sum_dxhat = 0.0;
    double sum_dxhat_xhat = 0.0;
    for (int i = 0; i < dim; ++i) {
        d_gamma[i] = dout[i] * x_centered[i] * inv_std;
        d_beta[i] = dout[i];
        // 累积到内部梯度缓冲区，供优化器使用
        grad_gamma[i] += d_gamma[i];
        grad_beta[i] += d_beta[i];

        d_xhat[i] = dout[i] * gamma[i];
        double xh = x_centered[i] * inv_std;
        sum_dxhat += d_xhat[i];
        sum_dxhat_xhat += d_xhat[i] * xh;
    }

    for (int i = 0; i < dim; ++i) {
        double xh = x_centered[i] * inv_std;
        dx[i] = (inv_std / dim) * (dim * d_xhat[i] - sum_dxhat - xh * sum_dxhat_xhat);
    }

    return {d_gamma, d_beta, dx};
}

void LayerNorm::zero_grad() {
    std::fill(grad_gamma.begin(), grad_gamma.end(), 0.0);
    std::fill(grad_beta.begin(), grad_beta.end(), 0.0);
}

// ============================================================================
//  RNNCell
// ============================================================================
RNNCell::RNNCell(int in_sz, int hid_sz) : input_size(in_sz), hidden_size(hid_sz) {
    double std_dev = std::sqrt(2.0 / (in_sz + hid_sz));
    W_ih.assign(hid_sz, Vec(in_sz));
    W_hh.assign(hid_sz, Vec(hid_sz));
    for (int j = 0; j < hid_sz; ++j) {
        for (int i = 0; i < in_sz; ++i)  W_ih[j][i] = randn(std_dev);
        for (int i = 0; i < hid_sz; ++i) W_hh[j][i] = randn(std_dev);
    }
    b_ih.assign(hid_sz, 0.0);
    b_hh.assign(hid_sz, 0.0);
    zero_grad();
}

Vec RNNCell::forward(const Vec& x, const Vec& h_prev_) {
    x_t = x;
    h_prev = h_prev_;
    tanh_z.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j) {
        double z = b_ih[j] + b_hh[j];
        for (int i = 0; i < input_size; ++i)  z += W_ih[j][i] * x[i];
        for (int i = 0; i < hidden_size; ++i) z += W_hh[j][i] * h_prev[i];
        tanh_z[j] = z;
    }
    h_t.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j)
        h_t[j] = std::tanh(tanh_z[j]);
    return h_t;
}

std::pair<Vec, Vec> RNNCell::backward(const Vec& dh) {
    // d(tanh(z)) = (1 - tanh(z)^2) * dh
    Vec d_z(hidden_size);
    for (int j = 0; j < hidden_size; ++j)
        d_z[j] = (1.0 - h_t[j] * h_t[j]) * dh[j];

    // 梯度
    for (int j = 0; j < hidden_size; ++j) {
        grad_b_ih[j] += d_z[j];
        grad_b_hh[j] += d_z[j];
        for (int i = 0; i < input_size; ++i)
            grad_W_ih[j][i] += d_z[j] * x_t[i];
        for (int i = 0; i < hidden_size; ++i)
            grad_W_hh[j][i] += d_z[j] * h_prev[i];
    }

    // dx, dh_prev
    Vec dx(input_size, 0.0), dh_prev(hidden_size, 0.0);
    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            dx[i] += d_z[j] * W_ih[j][i];
    for (int i = 0; i < hidden_size; ++i)
        for (int j = 0; j < hidden_size; ++j)
            dh_prev[i] += d_z[j] * W_hh[j][i];

    grad_x = dx;
    grad_h_prev = dh_prev;
    return {dx, dh_prev};
}

void RNNCell::zero_grad() {
    grad_W_ih.assign(hidden_size, Vec(input_size, 0.0));
    grad_W_hh.assign(hidden_size, Vec(hidden_size, 0.0));
    grad_b_ih.assign(hidden_size, 0.0);
    grad_b_hh.assign(hidden_size, 0.0);
}

// ============================================================================
//  LSTMCell
// ============================================================================
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

LSTMCell::LSTMCell(int in_sz, int hid_sz) : input_size(in_sz), hidden_size(hid_sz) {
    int gate_sz = 4 * hid_sz;
    double std_dev = std::sqrt(2.0 / (in_sz + hid_sz));
    W_ih.assign(gate_sz, Vec(in_sz));
    W_hh.assign(gate_sz, Vec(hid_sz));
    for (int j = 0; j < gate_sz; ++j) {
        for (int i = 0; i < in_sz; ++i)  W_ih[j][i] = randn(std_dev);
        for (int i = 0; i < hid_sz; ++i) W_hh[j][i] = randn(std_dev);
    }
    b_ih.assign(gate_sz, 0.0);
    b_hh.assign(gate_sz, 0.0);
    // forget gate bias 初始化为 1.0
    for (int i = 0; i < hid_sz; ++i) b_ih[hid_sz + i] = 1.0;
    zero_grad();
}

std::pair<Vec, Vec> LSTMCell::forward(const Vec& x, const Vec& h_prev_,
                                       const Vec& c_prev_) {
    x_t = x; h_prev = h_prev_; c_prev = c_prev_;
    int H = hidden_size;

    // 计算所有门
    i_gate.resize(H); f_gate.resize(H); g_gate.resize(H); o_gate.resize(H);
    i_sig.resize(H);  f_sig.resize(H);  g_tanh.resize(H); o_sig.resize(H);

    for (int j = 0; j < H; ++j) {
        double base_ih = b_ih[j] + b_hh[j];
        for (int k = 0; k < input_size; ++k) base_ih += W_ih[j][k] * x[k];
        for (int k = 0; k < H; ++k)        base_ih += W_hh[j][k] * h_prev[k];
        i_gate[j] = base_ih;
    }
    for (int j = 0; j < H; ++j) {
        double base_fh = b_ih[H + j] + b_hh[H + j];
        for (int k = 0; k < input_size; ++k) base_fh += W_ih[H + j][k] * x[k];
        for (int k = 0; k < H; ++k)        base_fh += W_hh[H + j][k] * h_prev[k];
        f_gate[j] = base_fh;
    }
    for (int j = 0; j < H; ++j) {
        double base_gh = b_ih[2 * H + j] + b_hh[2 * H + j];
        for (int k = 0; k < input_size; ++k) base_gh += W_ih[2 * H + j][k] * x[k];
        for (int k = 0; k < H; ++k)        base_gh += W_hh[2 * H + j][k] * h_prev[k];
        g_gate[j] = base_gh;
    }
    for (int j = 0; j < H; ++j) {
        double base_oh = b_ih[3 * H + j] + b_hh[3 * H + j];
        for (int k = 0; k < input_size; ++k) base_oh += W_ih[3 * H + j][k] * x[k];
        for (int k = 0; k < H; ++k)        base_oh += W_hh[3 * H + j][k] * h_prev[k];
        o_gate[j] = base_oh;
    }

    for (int j = 0; j < H; ++j) {
        i_sig[j] = sigmoid(i_gate[j]);
        f_sig[j] = sigmoid(f_gate[j]);
        g_tanh[j] = std::tanh(g_gate[j]);
        o_sig[j] = sigmoid(o_gate[j]);
    }

    c_t.resize(H); h_t.resize(H);
    for (int j = 0; j < H; ++j) {
        c_t[j] = f_sig[j] * c_prev[j] + i_sig[j] * g_tanh[j];
        h_t[j] = o_sig[j] * std::tanh(c_t[j]);
    }
    return {h_t, c_t};
}

std::tuple<Vec, Vec, Vec> LSTMCell::backward(const Vec& dh, const Vec& dc_next) {
    int H = hidden_size;
    // d(tanh(c)) * dh * o_sig
    Vec d_o_sig(H), d_c(H);
    for (int j = 0; j < H; ++j) {
        double tanh_c = std::tanh(c_t[j]);
        d_o_sig[j] = dh[j] * tanh_c;
        d_c[j] = dh[j] * o_sig[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
    }

    Vec d_i_sig(H), d_f_sig(H), d_g_tanh(H);
    for (int j = 0; j < H; ++j) {
        d_i_sig[j] = d_c[j] * g_tanh[j];
        d_f_sig[j] = d_c[j] * c_prev[j];
        d_g_tanh[j] = d_c[j] * i_sig[j];
    }

    Vec d_i(H), d_f(H), d_g(H), d_o(H);
    for (int j = 0; j < H; ++j) {
        d_i[j] = d_i_sig[j] * i_sig[j] * (1.0 - i_sig[j]);
        d_f[j] = d_f_sig[j] * f_sig[j] * (1.0 - f_sig[j]);
        d_g[j] = d_g_tanh[j] * (1.0 - g_tanh[j] * g_tanh[j]);
        d_o[j] = d_o_sig[j] * o_sig[j] * (1.0 - o_sig[j]);
    }

    // 累加梯度
    for (int j = 0; j < H; ++j) {
        grad_b_ih[j]         += d_i[j]; grad_b_hh[j]         += d_i[j];
        grad_b_ih[H + j]     += d_f[j]; grad_b_hh[H + j]     += d_f[j];
        grad_b_ih[2 * H + j] += d_g[j]; grad_b_hh[2 * H + j] += d_g[j];
        grad_b_ih[3 * H + j] += d_o[j]; grad_b_hh[3 * H + j] += d_o[j];
        for (int k = 0; k < input_size; ++k) {
            grad_W_ih[j][k]         += d_i[j] * x_t[k];
            grad_W_ih[H + j][k]     += d_f[j] * x_t[k];
            grad_W_ih[2 * H + j][k] += d_g[j] * x_t[k];
            grad_W_ih[3 * H + j][k] += d_o[j] * x_t[k];
        }
        for (int k = 0; k < H; ++k) {
            grad_W_hh[j][k]         += d_i[j] * h_prev[k];
            grad_W_hh[H + j][k]     += d_f[j] * h_prev[k];
            grad_W_hh[2 * H + j][k] += d_g[j] * h_prev[k];
            grad_W_hh[3 * H + j][k] += d_o[j] * h_prev[k];
        }
    }

    // dx, dh_prev, dc_prev
    Vec dx(input_size, 0.0), dh_prev(H, 0.0), dc_prev_out(H, 0.0);
    for (int k = 0; k < input_size; ++k)
        for (int j = 0; j < H; ++j)
            dx[k] += d_i[j] * W_ih[j][k] + d_f[j] * W_ih[H + j][k]
                   + d_g[j] * W_ih[2 * H + j][k] + d_o[j] * W_ih[3 * H + j][k];
    for (int k = 0; k < H; ++k) {
        for (int j = 0; j < H; ++j)
            dh_prev[k] += d_i[j] * W_hh[j][k] + d_f[j] * W_hh[H + j][k]
                        + d_g[j] * W_hh[2 * H + j][k] + d_o[j] * W_hh[3 * H + j][k];
        dc_prev_out[k] = d_c[k] * f_sig[k];
    }
    return {dx, dh_prev, dc_prev_out};
}

void LSTMCell::zero_grad() {
    int gs = 4 * hidden_size;
    grad_W_ih.assign(gs, Vec(input_size, 0.0));
    grad_W_hh.assign(gs, Vec(hidden_size, 0.0));
    grad_b_ih.assign(gs, 0.0);
    grad_b_hh.assign(gs, 0.0);
}

// ============================================================================
//  GRUCell
// ============================================================================
GRUCell::GRUCell(int in_sz, int hid_sz) : input_size(in_sz), hidden_size(hid_sz) {
    int gate_sz = 3 * hid_sz;
    double std_dev = std::sqrt(2.0 / (in_sz + hid_sz));
    W_ih.assign(gate_sz, Vec(in_sz));
    W_hh.assign(gate_sz, Vec(hid_sz));
    for (int j = 0; j < gate_sz; ++j) {
        for (int i = 0; i < in_sz; ++i)  W_ih[j][i] = randn(std_dev);
        for (int i = 0; i < hid_sz; ++i) W_hh[j][i] = randn(std_dev);
    }
    b_ih.assign(gate_sz, 0.0);
    b_hh.assign(gate_sz, 0.0);
    zero_grad();
}

Vec GRUCell::forward(const Vec& x, const Vec& h_prev_) {
    x_t = x; h_prev = h_prev_;
    int H = hidden_size;

    z_gate.resize(H); r_gate.resize(H); n_gate.resize(H);
    z_sig.resize(H); r_sig.resize(H); n_tanh.resize(H);

    // r, z: [0, H), [H, 2H)
    for (int j = 0; j < H; ++j) {
        double zr = b_ih[j] + b_hh[j];
        double rr = b_ih[H + j] + b_hh[H + j];
        for (int k = 0; k < input_size; ++k) {
            zr += W_ih[j][k] * x[k];
            rr += W_ih[H + j][k] * x[k];
        }
        for (int k = 0; k < H; ++k) {
            zr += W_hh[j][k] * h_prev[k];
            rr += W_hh[H + j][k] * h_prev[k];
        }
        z_gate[j] = zr;
        r_gate[j] = rr;
    }
    for (int j = 0; j < H; ++j) {
        z_sig[j] = sigmoid(z_gate[j]);
        r_sig[j] = sigmoid(r_gate[j]);
    }

    // n: [2H, 3H)
    for (int j = 0; j < H; ++j) {
        double n = b_ih[2 * H + j] + b_hh[2 * H + j];
        for (int k = 0; k < input_size; ++k) n += W_ih[2 * H + j][k] * x[k];
        for (int k = 0; k < H; ++k) n += W_hh[2 * H + j][k] * r_sig[k] * h_prev[k];
        n_gate[j] = n;
    }
    for (int j = 0; j < H; ++j) n_tanh[j] = std::tanh(n_gate[j]);

    h_t.resize(H);
    for (int j = 0; j < H; ++j)
        h_t[j] = (1.0 - z_sig[j]) * h_prev[j] + z_sig[j] * n_tanh[j];
    return h_t;
}

std::pair<Vec, Vec> GRUCell::backward(const Vec& dh) {
    int H = hidden_size;
    Vec d_z(H), d_n(H), d_r(H);
    for (int j = 0; j < H; ++j) {
        d_z[j] = dh[j] * (n_tanh[j] - h_prev[j]) * z_sig[j] * (1.0 - z_sig[j]);
        d_n[j] = dh[j] * z_sig[j] * (1.0 - n_tanh[j] * n_tanh[j]);
        // d_r via r gate * h_prev affecting n: sum over all k of d_n[k] * W_hh_n[k][j] * h_prev[j] * r_sig'[j]
        d_r[j] = 0.0;
        for (int k = 0; k < H; ++k)
            d_r[j] += d_n[k] * W_hh[2 * H + k][j] * h_prev[j] * r_sig[j] * (1.0 - r_sig[j]);
    }

    // 梯度
    for (int j = 0; j < H; ++j) {
        grad_b_ih[j] += d_z[j];     grad_b_hh[j] += d_z[j];
        grad_b_ih[H + j] += d_r[j]; grad_b_hh[H + j] += d_r[j];
        grad_b_ih[2 * H + j] += d_n[j]; grad_b_hh[2 * H + j] += d_n[j];
    }
    for (int j = 0; j < H; ++j) {
        for (int k = 0; k < input_size; ++k) {
            grad_W_ih[j][k] += d_z[j] * x_t[k];
            grad_W_ih[H + j][k] += d_r[j] * x_t[k];
            grad_W_ih[2 * H + j][k] += d_n[j] * x_t[k];
        }
        for (int k = 0; k < H; ++k) {
            grad_W_hh[j][k] += d_z[j] * h_prev[k];
            grad_W_hh[H + j][k] += d_r[j] * h_prev[k];
            grad_W_hh[2 * H + j][k] += d_n[j] * r_sig[k] * h_prev[k];
        }
    }

    // dx, dh_prev
    Vec dx(input_size, 0.0), dh_prev_out(H, 0.0);
    for (int k = 0; k < input_size; ++k)
        for (int j = 0; j < H; ++j)
            dx[k] += d_z[j] * W_ih[j][k] + d_r[j] * W_ih[H + j][k]
                   + d_n[j] * W_ih[2 * H + j][k];
    for (int k = 0; k < H; ++k) {
        for (int j = 0; j < H; ++j)
            dh_prev_out[k] += d_z[j] * W_hh[j][k] + d_r[j] * W_hh[H + j][k]
                            + d_n[j] * W_hh[2 * H + j][k] * r_sig[k];
        dh_prev_out[k] += dh[k] * (1.0 - z_sig[k]);
    }
    return {dx, dh_prev_out};
}

void GRUCell::zero_grad() {
    int gs = 3 * hidden_size;
    grad_W_ih.assign(gs, Vec(input_size, 0.0));
    grad_W_hh.assign(gs, Vec(hidden_size, 0.0));
    grad_b_ih.assign(gs, 0.0);
    grad_b_hh.assign(gs, 0.0);
}

// ============================================================================
//  RNN / LSTM / GRU 容器
// ============================================================================
RNN::RNN(int in_sz, int hid_sz, int n_layers)
    : input_size(in_sz), hidden_size(hid_sz), num_layers(n_layers) {
    for (int l = 0; l < num_layers; ++l)
        cells.emplace_back(l == 0 ? in_sz : hid_sz, hid_sz);
}

std::pair<std::vector<Vec>, std::vector<Vec>> RNN::forward(const std::vector<Vec>& x_seq) {
    std::vector<Vec> outputs;
    std::vector<Vec> hiddens(num_layers, Vec(hidden_size, 0.0));
    for (size_t t = 0; t < x_seq.size(); ++t) {
        Vec inp = x_seq[t];
        for (int l = 0; l < num_layers; ++l) {
            inp = cells[l].forward(inp, hiddens[l]);
            hiddens[l] = inp;
        }
        outputs.push_back(inp);
    }
    return {outputs, hiddens};
}

std::vector<Vec> RNN::backward(const std::vector<Vec>& dh_seq) {
    int T = (int)dh_seq.size();
    std::vector<Vec> dh_next(num_layers, Vec(hidden_size, 0.0));
    std::vector<Vec> dx_seq(T, Vec(input_size, 0.0));

    for (int t = T - 1; t >= 0; --t) {
        Vec dh = dh_seq[t]; // 来自最终层输出的梯度
        for (int l = num_layers - 1; l >= 0; --l) {
            for (int i = 0; i < hidden_size; ++i)
                dh[i] += dh_next[l][i];
            auto [dx, dh_prev] = cells[l].backward(dh);
            dh_next[l] = dh_prev;
            if (l == 0) dx_seq[t] = dx;
            else dh = dx; // 传递给下一层
        }
    }
    return dx_seq;
}

LSTM::LSTM(int in_sz, int hid_sz, int n_layers)
    : input_size(in_sz), hidden_size(hid_sz), num_layers(n_layers) {
    for (int l = 0; l < num_layers; ++l)
        cells.emplace_back(l == 0 ? in_sz : hid_sz, hid_sz);
}

std::pair<std::vector<Vec>, std::pair<std::vector<Vec>, std::vector<Vec>>>
LSTM::forward(const std::vector<Vec>& x_seq) {
    std::vector<Vec> outputs;
    std::vector<Vec> h(num_layers, Vec(hidden_size, 0.0));
    std::vector<Vec> c(num_layers, Vec(hidden_size, 0.0));
    for (auto& x : x_seq) {
        Vec inp = x;
        for (int l = 0; l < num_layers; ++l) {
            auto [h_out, c_out] = cells[l].forward(inp, h[l], c[l]);
            h[l] = h_out; c[l] = c_out;
            inp = h_out;
        }
        outputs.push_back(inp);
    }
    return {outputs, {h, c}};
}

void LSTM::backward(const std::vector<Vec>& dh_seq) {
    int T = (int)dh_seq.size();
    std::vector<Vec> dh_next(num_layers, Vec(hidden_size, 0.0));
    std::vector<Vec> dc_next(num_layers, Vec(hidden_size, 0.0));

    for (int t = T - 1; t >= 0; --t) {
        Vec dh_cur = dh_seq[t];
        for (int l = num_layers - 1; l >= 0; --l) {
            for (int i = 0; i < hidden_size; ++i)
                dh_cur[i] += dh_next[l][i];
            auto [dx, dh_prev, dc_prev] = cells[l].backward(dh_cur, dc_next[l]);
            dh_next[l] = dh_prev;
            dc_next[l] = dc_prev;
            dh_cur = dx;
        }
    }
}

GRU::GRU(int in_sz, int hid_sz, int n_layers)
    : input_size(in_sz), hidden_size(hid_sz), num_layers(n_layers) {
    for (int l = 0; l < num_layers; ++l)
        cells.emplace_back(l == 0 ? in_sz : hid_sz, hid_sz);
}

std::pair<std::vector<Vec>, std::vector<Vec>> GRU::forward(const std::vector<Vec>& x_seq) {
    std::vector<Vec> outputs;
    std::vector<Vec> h(num_layers, Vec(hidden_size, 0.0));
    for (auto& x : x_seq) {
        Vec inp = x;
        for (int l = 0; l < num_layers; ++l) {
            inp = cells[l].forward(inp, h[l]);
            h[l] = inp;
        }
        outputs.push_back(inp);
    }
    return {outputs, h};
}

std::vector<Vec> GRU::backward(const std::vector<Vec>& dh_seq) {
    int T = (int)dh_seq.size();
    std::vector<Vec> dh_next(num_layers, Vec(hidden_size, 0.0));
    std::vector<Vec> dx_seq(T, Vec(input_size, 0.0));

    for (int t = T - 1; t >= 0; --t) {
        Vec dh = dh_seq[t];
        for (int l = num_layers - 1; l >= 0; --l) {
            for (int i = 0; i < hidden_size; ++i)
                dh[i] += dh_next[l][i];
            auto [dx, dh_prev] = cells[l].backward(dh);
            dh_next[l] = dh_prev;
            if (l == 0) dx_seq[t] = dx;
            else dh = dx;
        }
    }
    return dx_seq;
}

// ============================================================================
//  MultiHeadAttention
// ============================================================================
MultiHeadAttention::MultiHeadAttention(int d_model_, int num_heads_)
    : d_model(d_model_), num_heads(num_heads_), d_k(d_model_ / num_heads_)
{
    double std_dev = std::sqrt(2.0 / d_model);
    auto init_mat = [&](Vec2D& m) {
        m.assign(d_model, Vec(d_model));
        for (auto& r : m) for (auto& v : r) v = randn(std_dev);
    };
    auto init_b = [&](Vec& b) { b.assign(d_model, 0.0); };
    init_mat(W_q); init_mat(W_k); init_mat(W_v); init_mat(W_o);
    init_b(b_q); init_b(b_k); init_b(b_v); init_b(b_o);
    zero_grad();
}

Vec3D MultiHeadAttention::forward(const Vec3D& x) {
    input_cache = x;
    int seq_len = (int)x.size();

    // Q, K, V = x * W + b
    auto matmul = [&](const Vec2D& W, const Vec& b) {
        Vec3D result(seq_len, Vec2D(num_heads, Vec(d_k)));
        for (int s = 0; s < seq_len; ++s) {
            // x[s] * W → 分头
            for (int h = 0; h < num_heads; ++h) {
                for (int k = 0; k < d_k; ++k) {
                    int idx = h * d_k + k;
                    double val = b[idx];
                    for (int i = 0; i < d_model; ++i)
                        val += x[s][0][i] * W[idx][i];
                    result[s][h][k] = val;
                }
            }
        }
        return result;
    };

    q_mat = matmul(W_q, b_q);
    k_mat = matmul(W_k, b_k);
    v_mat = matmul(W_v, b_v);

    // Scaled Dot-Product Attention
    double scale = 1.0 / std::sqrt((double)d_k);
    attn_weights.resize(seq_len, Vec2D(seq_len, Vec(num_heads)));
    Vec3D attn_out(seq_len, Vec2D(num_heads, Vec(d_k)));

    for (int s = 0; s < seq_len; ++s) {
        // attention scores
        Vec scores(seq_len);
        for (int j = 0; j < seq_len; ++j) {
            double dot = 0.0;
            for (int h = 0; h < num_heads; ++h)
                for (int k = 0; k < d_k; ++k)
                    dot += q_mat[s][h][k] * k_mat[j][h][k];
            scores[j] = dot * scale;
        }
        // softmax over j
        double mx = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = std::exp(scores[j] - mx);
            sum += scores[j];
        }
        for (int j = 0; j < seq_len; ++j) {
            scores[j] /= sum;
            attn_weights[s][j] = Vec(num_heads, scores[j]);
        }
        // weighted sum
        for (int h = 0; h < num_heads; ++h) {
            for (int k = 0; k < d_k; ++k) {
                double val = 0.0;
                for (int j = 0; j < seq_len; ++j)
                    val += scores[j] * v_mat[j][h][k];
                attn_out[s][h][k] = val;
            }
        }
    }

    // Concat heads + W_o
    Vec3D out(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s) {
        // concat all heads
        Vec concat(d_model);
        for (int h = 0; h < num_heads; ++h)
            for (int k = 0; k < d_k; ++k)
                concat[h * d_k + k] = attn_out[s][h][k];
        // W_o
        for (int i = 0; i < d_model; ++i) {
            double val = b_o[i];
            for (int j = 0; j < d_model; ++j)
                val += W_o[i][j] * concat[j];
            out[s][0][i] = val;
        }
    }
    return out;
}

Vec3D MultiHeadAttention::backward(const Vec3D& dout) {
    int seq_len = (int)input_cache.size();
    double scale = 1.0 / std::sqrt((double)d_k);

    // === 重建 attention 输出 concat ===
    Vec2D concat(seq_len, Vec(d_model, 0.0));
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_heads; ++h) {
            for (int j = 0; j < seq_len; ++j) {
                double a = attn_weights[s][j][h];
                for (int k = 0; k < d_k; ++k)
                    concat[s][h * d_k + k] += a * v_mat[j][h][k];
            }
        }
    }

    // === 1. W_o 输出投影反向 ===
    Vec2D d_concat(seq_len, Vec(d_model, 0.0));
    for (int s = 0; s < seq_len; ++s) {
        for (int i = 0; i < d_model; ++i) {
            grad_b_o[i] += dout[s][0][i];
            for (int j = 0; j < d_model; ++j) {
                grad_W_o[i][j] += dout[s][0][i] * concat[s][j];
                d_concat[s][j] += dout[s][0][i] * W_o[i][j];
            }
        }
    }

    // === 2. 分割回多头的 d_attn_out ===
    Vec3D d_attn_out(seq_len, Vec2D(num_heads, Vec(d_k, 0.0)));
    for (int s = 0; s < seq_len; ++s)
        for (int h = 0; h < num_heads; ++h)
            for (int k = 0; k < d_k; ++k)
                d_attn_out[s][h][k] = d_concat[s][h * d_k + k];

    // === 3. dV 和 d_scores ===
    // attn_out[s][h][k] = sum_j scores[s][j] * v_mat[j][h][k]
    Vec3D d_v(seq_len, Vec2D(num_heads, Vec(d_k, 0.0)));
    Vec2D d_scores(seq_len, Vec(seq_len, 0.0));
    for (int s = 0; s < seq_len; ++s) {
        for (int j = 0; j < seq_len; ++j) {
            double score = attn_weights[s][j][0]; // 所有 head 共享同一 attention
            for (int h = 0; h < num_heads; ++h) {
                for (int k = 0; k < d_k; ++k) {
                    d_v[j][h][k] += d_attn_out[s][h][k] * score;
                    d_scores[s][j] += d_attn_out[s][h][k] * v_mat[j][h][k];
                }
            }
        }
    }

    // === 4. Softmax 反向 ===
    // score[s] = softmax(raw[s]),  raw[s][j] = scale * dot[s][j]
    Vec2D d_dot(seq_len, Vec(seq_len, 0.0));
    for (int s = 0; s < seq_len; ++s) {
        double sum_ws = 0.0;
        for (int j = 0; j < seq_len; ++j)
            sum_ws += attn_weights[s][j][0] * d_scores[s][j];
        for (int j = 0; j < seq_len; ++j) {
            double d_raw = attn_weights[s][j][0] * (d_scores[s][j] - sum_ws);
            d_dot[s][j] = d_raw * scale;
        }
    }

    // === 5. dQ, dK 从点积 ===
    Vec3D d_quer(seq_len, Vec2D(num_heads, Vec(d_k, 0.0)));
    Vec3D d_key(seq_len, Vec2D(num_heads, Vec(d_k, 0.0)));
    for (int s = 0; s < seq_len; ++s) {
        for (int j = 0; j < seq_len; ++j) {
            double dd = d_dot[s][j];
            for (int h = 0; h < num_heads; ++h) {
                for (int k = 0; k < d_k; ++k) {
                    d_quer[s][h][k] += dd * k_mat[j][h][k];
                    d_key[j][h][k] += dd * q_mat[s][h][k];
                }
            }
        }
    }

    // === 6. 输入投影反向：计算 dW_q/k/v, db_q/k/v, dx ===
    Vec3D dx(seq_len, Vec2D(1, Vec(d_model, 0.0)));
    auto project_backward = [&](const Vec3D& d_proj, Vec2D& W,
                                 Vec2D& gW, Vec& gb) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                for (int k = 0; k < d_k; ++k) {
                    int idx = h * d_k + k;
                    double dp = d_proj[s][h][k];
                    gb[idx] += dp;
                    for (int i = 0; i < d_model; ++i) {
                        gW[idx][i] += dp * input_cache[s][0][i];
                        dx[s][0][i] += dp * W[idx][i];
                    }
                }
            }
        }
    };
    project_backward(d_quer, W_q, grad_W_q, grad_b_q);
    project_backward(d_key, W_k, grad_W_k, grad_b_k);
    project_backward(d_v, W_v, grad_W_v, grad_b_v);

    return dx;
}

void MultiHeadAttention::zero_grad() {
    grad_W_q.assign(d_model, Vec(d_model, 0.0));
    grad_W_k.assign(d_model, Vec(d_model, 0.0));
    grad_W_v.assign(d_model, Vec(d_model, 0.0));
    grad_W_o.assign(d_model, Vec(d_model, 0.0));
    grad_b_q.assign(d_model, 0.0);
    grad_b_k.assign(d_model, 0.0);
    grad_b_v.assign(d_model, 0.0);
    grad_b_o.assign(d_model, 0.0);
}

// ============================================================================
//  FeedForward
// ============================================================================
FeedForward::FeedForward(int d_model_, int d_ff_) : d_model(d_model_), d_ff(d_ff_) {
    double s1 = std::sqrt(2.0 / d_model);
    double s2 = std::sqrt(2.0 / d_ff);
    W1.assign(d_ff, Vec(d_model));
    W2.assign(d_model, Vec(d_ff));
    for (auto& r : W1) for (auto& v : r) v = randn(s1);
    for (auto& r : W2) for (auto& v : r) v = randn(s2);
    b1.assign(d_ff, 0.0);
    b2.assign(d_model, 0.0);
    zero_grad();
}

Vec3D FeedForward::forward(const Vec3D& x) {
    input_cache = x;
    int seq_len = (int)x.size();
    Vec3D out(seq_len, Vec2D(1, Vec(d_model)));
    cache_after_w1.resize(seq_len, Vec2D(1, Vec(d_ff)));
    cache_hidden.resize(seq_len, Vec2D(1, Vec(d_ff)));
    for (int s = 0; s < seq_len; ++s) {
        // W1*x + b1
        for (int j = 0; j < d_ff; ++j) {
            double val = b1[j];
            for (int i = 0; i < d_model; ++i) val += W1[j][i] * x[s][0][i];
            cache_after_w1[s][0][j] = val;
            // ReLU: max(0, z)
            cache_hidden[s][0][j] = val > 0 ? val : 0.0;
        }
        // W2*hidden + b2
        for (int i = 0; i < d_model; ++i) {
            double val = b2[i];
            for (int j = 0; j < d_ff; ++j)
                val += W2[i][j] * cache_hidden[s][0][j];
            out[s][0][i] = val;
        }
    }
    return out;
}

Vec3D FeedForward::backward(const Vec3D& dout) {
    int seq_len = (int)dout.size();
    Vec3D dx(seq_len, Vec2D(1, Vec(d_model, 0.0)));
    for (int s = 0; s < seq_len; ++s) {
        // === Backward through W2 ===
        // out[s][0][i] = b2[i] + sum_j W2[i][j] * hidden[s][0][j]
        Vec d_hidden(d_ff, 0.0);
        for (int i = 0; i < d_model; ++i) {
            grad_b2[i] += dout[s][0][i];
            for (int j = 0; j < d_ff; ++j) {
                grad_W2[i][j] += dout[s][0][i] * cache_hidden[s][0][j];
                d_hidden[j] += dout[s][0][i] * W2[i][j];
            }
        }
        // === Through ReLU ===
        Vec d_pre_relu(d_ff, 0.0);
        for (int j = 0; j < d_ff; ++j)
            d_pre_relu[j] = (cache_after_w1[s][0][j] > 0) ? d_hidden[j] : 0.0;
        // === Backward through W1 ===
        for (int j = 0; j < d_ff; ++j) {
            grad_b1[j] += d_pre_relu[j];
            for (int i = 0; i < d_model; ++i) {
                grad_W1[j][i] += d_pre_relu[j] * input_cache[s][0][i];
                dx[s][0][i] += d_pre_relu[j] * W1[j][i];
            }
        }
    }
    return dx;
}

void FeedForward::zero_grad() {
    grad_W1.assign(d_ff, Vec(d_model, 0.0));
    grad_W2.assign(d_model, Vec(d_ff, 0.0));
    grad_b1.assign(d_ff, 0.0);
    grad_b2.assign(d_model, 0.0);
}

// ============================================================================
//  TransformerEncoderLayer
// ============================================================================
TransformerEncoderLayer::TransformerEncoderLayer(int d_model_, int num_heads, int d_ff)
    : attn(d_model_, num_heads), ff(d_model_, d_ff), ln1(d_model_), ln2(d_model_),
      d_model(d_model_) {}

Vec3D TransformerEncoderLayer::forward(const Vec3D& x) {
    int seq_len = (int)x.size();
    // LayerNorm + Attention + Residual
    Vec3D ln1_out(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        ln1_out[s][0] = ln1.forward(x[s][0]);
    Vec3D attn_out = attn.forward(ln1_out);
    // Residual
    Vec3D res1(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        for (int i = 0; i < d_model; ++i)
            res1[s][0][i] = x[s][0][i] + attn_out[s][0][i];

    // LayerNorm + FF + Residual
    Vec3D ln2_out(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        ln2_out[s][0] = ln2.forward(res1[s][0]);
    Vec3D ff_out = ff.forward(ln2_out);
    Vec3D res2(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        for (int i = 0; i < d_model; ++i)
            res2[s][0][i] = res1[s][0][i] + ff_out[s][0][i];

    return res2;
}

Vec3D TransformerEncoderLayer::backward(const Vec3D& dout) {
    int seq_len = (int)dout.size();

    // forward: res1 = x + attn(ln1(x)); res2 = res1 + ff(ln2(res1))
    //
    // backward 路径:
    //   d_res2 = dout
    //   d_res1 = dout (残差直连) + ln2.backward(ff.backward(dout))
    //   dx = d_res1 (残差直连) + ln1.backward(attn.backward(d_res1))

    // === 第二个子层: FF + LN2 ===
    Vec3D d_ff_out(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        d_ff_out[s][0] = dout[s][0];

    Vec3D d_ln2_out = ff.backward(d_ff_out);

    Vec3D d_res1(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s) {
        // 残差路径直传
        d_res1[s][0] = dout[s][0];
        // LN2 backward (使用 forward 中缓存的 x_centered, inv_std)
        auto [dg, db, dx_ln2] = ln2.backward(d_ln2_out[s][0]);
        for (int i = 0; i < d_model; ++i)
            d_res1[s][0][i] += dx_ln2[i];
        (void)dg; (void)db;
    }

    // === 第一个子层: Attention + LN1 ===
    Vec3D d_attn_out(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s)
        d_attn_out[s][0] = d_res1[s][0];

    Vec3D d_ln1_out = attn.backward(d_attn_out);

    Vec3D dx(seq_len, Vec2D(1, Vec(d_model)));
    for (int s = 0; s < seq_len; ++s) {
        // 残差路径直传
        dx[s][0] = d_res1[s][0];
        // LN1 backward
        auto [dg, db, dxi] = ln1.backward(d_ln1_out[s][0]);
        for (int i = 0; i < d_model; ++i)
            dx[s][0][i] += dxi[i];
        (void)dg; (void)db;
    }

    return dx;
}

void TransformerEncoderLayer::zero_grad() {
    attn.zero_grad();
    ff.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
}

// ============================================================================
//  TransformerEncoder
// ============================================================================
TransformerEncoder::TransformerEncoder(int d_model_, int num_heads, int d_ff,
                                         int num_layers)
    : d_model(d_model_) {
    for (int i = 0; i < num_layers; ++i)
        layers.emplace_back(d_model_, num_heads, d_ff);
}

Vec3D TransformerEncoder::forward(const Vec3D& x) {
    Vec3D out = x;
    for (auto& layer : layers)
        out = layer.forward(out);
    return out;
}

Vec3D TransformerEncoder::backward(const Vec3D& dout) {
    Vec3D d = dout;
    for (int i = (int)layers.size() - 1; i >= 0; --i)
        d = layers[i].backward(d);
    return d;
}

void TransformerEncoder::zero_grad() {
    for (auto& l : layers) l.zero_grad();
}

// ============================================================================
//  PositionalEncoding
// ============================================================================
PositionalEncoding::PositionalEncoding(int d_model_, int max_len)
    : d_model(d_model_) {
    pe.resize(max_len, Vec(d_model));
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            double angle = pos / std::pow(10000.0, 2.0 * (i / 2) / d_model);
            pe[pos][i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
}

Vec3D PositionalEncoding::forward(const Vec3D& x) {
    int seq_len = (int)x.size();
    Vec3D out = x;
    for (int s = 0; s < seq_len; ++s)
        for (int i = 0; i < d_model; ++i)
            out[s][0][i] += pe[s][i];
    return out;
}

// ============================================================================
//  Embedding
// ============================================================================
Embedding::Embedding(int vocab_sz, int embed_dim_)
    : vocab_size(vocab_sz), embed_dim(embed_dim_) {
    double std_dev = std::sqrt(2.0 / vocab_sz);
    weight.assign(vocab_sz, Vec(embed_dim));
    for (auto& r : weight)
        for (auto& v : r) v = randn(std_dev);
    zero_grad();
}

Vec2D Embedding::forward(const std::vector<int>& indices) {
    indices_cache = indices;
    Vec2D out(indices.size(), Vec(embed_dim));
    for (size_t i = 0; i < indices.size(); ++i)
        out[i] = weight[indices[i]];
    return out;
}

Vec2D Embedding::backward(const Vec2D& dout) {
    for (size_t i = 0; i < indices_cache.size(); ++i) {
        int idx = indices_cache[i];
        for (int j = 0; j < embed_dim; ++j)
            grad_w[idx][j] += dout[i][j];
    }
    return {};
}

void Embedding::zero_grad() {
    grad_w.assign(vocab_size, Vec(embed_dim, 0.0));
}

// ============================================================================
//  ResidualBlock
// ============================================================================
ResidualBlock::ResidualBlock(int dim, int hidden_dim) : ln(dim) {
    int hd = (hidden_dim > 0) ? hidden_dim : dim;
    main_path = MLP({dim, hd, dim},
        {Layer::Activation::LEAKY_RELU, Layer::Activation::LINEAR});
}

Vec ResidualBlock::forward(const Vec& x) {
    Vec ln_out = ln.forward(x);
    Vec main_out = main_path.forward(ln_out);
    Vec out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = x[i] + main_out[i];
    return out;
}

Vec ResidualBlock::backward(const Vec& dout) {
    // Forward: out = x + main_path(ln(x))
    // main_path: dim -> hidden_dim (LeakyReLU) -> dim (Linear)
    //
    // Backward:
    //   d_main_out = dout
    //   d_ln_out = main_path backward(d_main_out) [grads stored in mlp_grads]
    //   d_x_from_ln = ln backward(d_ln_out)
    //   dx = dout (residual direct) + d_x_from_ln

    // Step 1: Backprop through MLP layers, store gradients
    Vec delta = dout;
    mlp_grads.clear();
    for (int i = (int)main_path.layers.size() - 1; i >= 0; --i) {
        auto [gw, gb, d] = main_path.layers[i].backward(delta);
        mlp_grads.insert(mlp_grads.begin(), {gw, gb});
        delta = d;
    }
    // delta is now d_ln_out (gradient w.r.t. ln output)

    // Step 2: Backprop through LayerNorm (grads accumulated into ln.grad_gamma/grad_beta)
    auto [dg, db, d_ln_x] = ln.backward(delta);
    (void)dg; (void)db;

    // Step 3: Residual + through ln path
    Vec dx(dout.size());
    for (size_t i = 0; i < dout.size(); ++i)
        dx[i] = dout[i] + d_ln_x[i];

    return dx;
}

void ResidualBlock::zero_grad() {
    mlp_grads.clear();
    ln.zero_grad();
}

// ============================================================================
//  SequentialOp
// ============================================================================
SequentialOp::SequentialOp(OpType t, int d, double p_)
    : type(t), dim(d), p(std::min(p_, 0.9)) {
    if (type == LINEAR) {
        W.assign(dim, Vec(dim));
        for (auto& r : W) for (auto& v : r) v = randn();
        b.assign(dim, 0.0);
        grad_W.assign(dim, Vec(dim, 0.0));
        grad_b.assign(dim, 0.0);
    } else if (type == LAYER_NORM) {
        gamma.assign(dim, 1.0);
        beta.assign(dim, 0.0);
        grad_gamma.assign(dim, 0.0);
        grad_beta.assign(dim, 0.0);
    }
}

Vec SequentialOp::forward(const Vec& x) {
    input_cache = x;
    switch (type) {
    case LINEAR: {
        z_cache.resize(dim);
        Vec out(dim);
        for (int j = 0; j < dim; ++j) {
            z_cache[j] = b[j];
            for (int i = 0; i < dim; ++i) z_cache[j] += W[j][i] * x[i];
            out[j] = z_cache[j];
        }
        return out;
    }
    case RELU: {
        Vec out(dim);
        for (int i = 0; i < dim; ++i) out[i] = x[i] > 0 ? x[i] : 0.0;
        return out;
    }
    case LEAKY_RELU: {
        Vec out(dim);
        for (int i = 0; i < dim; ++i) out[i] = x[i] > 0 ? x[i] : 0.01 * x[i];
        return out;
    }
    case SOFTMAX: {
        Vec out = x;
        double mx = *std::max_element(out.begin(), out.end());
        double sum = 0.0;
        for (auto& v : out) { v = std::exp(v - mx); sum += v; }
        for (auto& v : out) v /= sum;
        return out;
    }
    case DROPOUT: {
        if (p <= 0) return x;
        drop_mask.resize(dim);
        Vec out(dim);
        double scale = 1.0 / (1.0 - p);
        for (int i = 0; i < dim; ++i) {
            drop_mask[i] = (randu() > p) ? scale : 0.0;
            out[i] = x[i] * drop_mask[i];
        }
        return out;
    }
    case LAYER_NORM: {
        double mean = std::accumulate(x.begin(), x.end(), 0.0) / dim;
        double var = 0.0;
        for (auto v : x) var += (v - mean) * (v - mean);
        var = var / dim + 1e-5;
        double inv_std_ = 1.0 / std::sqrt(var);
        ln_mean.assign(1, mean);
        ln_var.assign(1, var);
        ln_x_hat.resize(dim);
        Vec out(dim);
        for (int i = 0; i < dim; ++i) {
            ln_x_hat[i] = (x[i] - mean) * inv_std_;
            out[i] = gamma[i] * ln_x_hat[i] + beta[i];
        }
        return out;
    }
    default: return x;
    }
}

Vec SequentialOp::backward(const Vec& dout) {
    switch (type) {
    case LINEAR: {
        for (int j = 0; j < dim; ++j) {
            grad_b[j] += dout[j];
            for (int i = 0; i < dim; ++i)
                grad_W[j][i] += dout[j] * input_cache[i];
        }
        Vec dx(dim, 0.0);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                dx[i] += dout[j] * W[j][i];
        return dx;
    }
    case RELU: {
        Vec dx(dim);
        for (int i = 0; i < dim; ++i)
            dx[i] = (input_cache[i] > 0) ? dout[i] : 0.0;
        return dx;
    }
    case LEAKY_RELU: {
        Vec dx(dim);
        for (int i = 0; i < dim; ++i)
            dx[i] = (input_cache[i] > 0) ? dout[i] : 0.01 * dout[i];
        return dx;
    }
    case SOFTMAX: {
        // 重新计算 softmax 输出 s = softmax(input_cache)
        Vec s = input_cache;
        double mx = *std::max_element(s.begin(), s.end());
        double sum = 0.0;
        for (auto& v : s) { v = std::exp(v - mx); sum += v; }
        for (auto& v : s) v /= sum;
        // dL/dx_j = s_j * (dL/ds_j - sum_i dL/ds_i * s_i)
        double dot = 0.0;
        for (int i = 0; i < dim; ++i) dot += dout[i] * s[i];
        Vec dx(dim);
        for (int i = 0; i < dim; ++i)
            dx[i] = s[i] * (dout[i] - dot);
        return dx;
    }
    case DROPOUT: {
        Vec dx(dim);
        for (int i = 0; i < dim; ++i)
            dx[i] = dout[i] * drop_mask[i];
        return dx;
    }
    case LAYER_NORM: {
        // d_gamma, d_beta
        for (int i = 0; i < dim; ++i) {
            grad_gamma[i] += dout[i] * ln_x_hat[i];
            grad_beta[i] += dout[i];
        }
        // dx through layer norm:
        // x_hat = (x - mean) * inv_std,  out = gamma * x_hat + beta
        // dx = inv_std * (d_x_hat - mean(d_x_hat) - x_hat * mean(d_x_hat * x_hat))
        double inv_std_ = 1.0 / std::sqrt(ln_var[0]);
        double mean_dxhat = 0.0;
        double mean_dxhat_xhat = 0.0;
        for (int i = 0; i < dim; ++i) {
            double d_xhat = dout[i] * gamma[i];
            mean_dxhat += d_xhat;
            mean_dxhat_xhat += d_xhat * ln_x_hat[i];
        }
        mean_dxhat /= dim;
        mean_dxhat_xhat /= dim;
        Vec dx(dim);
        for (int i = 0; i < dim; ++i) {
            double d_xhat = dout[i] * gamma[i];
            dx[i] = inv_std_ * (d_xhat - mean_dxhat - ln_x_hat[i] * mean_dxhat_xhat);
        }
        return dx;
    }
    default: return dout;
    }
}

void SequentialOp::zero_grad() {
    if (type == LINEAR) {
        for (auto& row : grad_W) std::fill(row.begin(), row.end(), 0.0);
        std::fill(grad_b.begin(), grad_b.end(), 0.0);
    } else if (type == LAYER_NORM) {
        std::fill(grad_gamma.begin(), grad_gamma.end(), 0.0);
        std::fill(grad_beta.begin(), grad_beta.end(), 0.0);
    }
}

// ============================================================================
//  Optimizers
// ============================================================================
Optimizer::Optimizer(MLP* m, double lr_) : model(m), lr(lr_) {}

Adam::Adam(MLP* m, double lr_, double b1, double b2, double eps_)
    : Optimizer(m, lr_), beta1(b1), beta2(b2), eps(eps_), t(0) {
    for (auto& layer : model->layers) {
        m_w.push_back(Vec2D(layer.fan_out, Vec(layer.fan_in, 0.0)));
        v_w.push_back(Vec2D(layer.fan_out, Vec(layer.fan_in, 0.0)));
        m_b.push_back(Vec(layer.fan_out, 0.0));
        v_b.push_back(Vec(layer.fan_out, 0.0));
    }
}

void Adam::step(const std::vector<GradPair>& grads) {
    ++t;
    // 标准 PyTorch 偏置校正: m_hat = m / (1-beta1^t), v_hat = v / (1-beta2^t)
    double bc1 = 1.0 - std::pow(beta1, t);
    double bc2 = 1.0 - std::pow(beta2, t);
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        auto& [gw, gb] = grads[i];
        for (int j = 0; j < layer.fan_out; ++j) {
            for (int k = 0; k < layer.fan_in; ++k) {
                m_w[i][j][k] = beta1 * m_w[i][j][k] + (1 - beta1) * gw[j][k];
                v_w[i][j][k] = beta2 * v_w[i][j][k] + (1 - beta2) * gw[j][k] * gw[j][k];
                double m_hat = m_w[i][j][k] / bc1;
                double v_hat = v_w[i][j][k] / bc2;
                layer.weights[j][k] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
            m_b[i][j] = beta1 * m_b[i][j] + (1 - beta1) * gb[j];
            v_b[i][j] = beta2 * v_b[i][j] + (1 - beta2) * gb[j] * gb[j];
            double m_hat_b = m_b[i][j] / bc1;
            double v_hat_b = v_b[i][j] / bc2;
            layer.biases[j] -= lr * m_hat_b / (std::sqrt(v_hat_b) + eps);
        }
    }
}

SGD::SGD(MLP* m, double lr_) : Optimizer(m, lr_) {}

void SGD::step(const std::vector<GradPair>& grads) {
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        auto& [gw, gb] = grads[i];
        for (int j = 0; j < layer.fan_out; ++j) {
            for (int k = 0; k < layer.fan_in; ++k)
                layer.weights[j][k] -= lr * gw[j][k];
            layer.biases[j] -= lr * gb[j];
        }
    }
}

SGDMomentum::SGDMomentum(MLP* m, double lr_, double momentum_)
    : Optimizer(m, lr_), momentum(momentum_) {
    for (auto& layer : model->layers) {
        v_w.push_back(Vec2D(layer.fan_out, Vec(layer.fan_in, 0.0)));
        v_b.push_back(Vec(layer.fan_out, 0.0));
    }
}

void SGDMomentum::step(const std::vector<GradPair>& grads) {
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        auto& [gw, gb] = grads[i];
        for (int j = 0; j < layer.fan_out; ++j) {
            for (int k = 0; k < layer.fan_in; ++k) {
                v_w[i][j][k] = momentum * v_w[i][j][k] - lr * gw[j][k];
                layer.weights[j][k] += v_w[i][j][k];
            }
            v_b[i][j] = momentum * v_b[i][j] - lr * gb[j];
            layer.biases[j] += v_b[i][j];
        }
    }
}

RMSprop::RMSprop(MLP* m, double lr_, double beta_, double eps_)
    : Optimizer(m, lr_), beta(beta_), eps(eps_) {
    for (auto& layer : model->layers) {
        cache_w.push_back(Vec2D(layer.fan_out, Vec(layer.fan_in, 0.0)));
        cache_b.push_back(Vec(layer.fan_out, 0.0));
    }
}

void RMSprop::step(const std::vector<GradPair>& grads) {
    for (size_t i = 0; i < model->layers.size(); ++i) {
        auto& layer = model->layers[i];
        auto& [gw, gb] = grads[i];
        for (int j = 0; j < layer.fan_out; ++j) {
            for (int k = 0; k < layer.fan_in; ++k) {
                cache_w[i][j][k] = beta * cache_w[i][j][k] + (1 - beta) * gw[j][k] * gw[j][k];
                layer.weights[j][k] -= lr * gw[j][k] / (std::sqrt(cache_w[i][j][k]) + eps);
            }
            cache_b[i][j] = beta * cache_b[i][j] + (1 - beta) * gb[j] * gb[j];
            layer.biases[j] -= lr * gb[j] / (std::sqrt(cache_b[i][j]) + eps);
        }
    }
}

// ============================================================================
//  Loss Functions
// ============================================================================
std::pair<double, Vec> mse_loss(const Vec& pred, const Vec& target) {
    double loss = 0.0;
    Vec grad(pred.size());
    double invN = 1.0 / pred.size();
    for (size_t i = 0; i < pred.size(); ++i) {
        double diff = pred[i] - target[i];
        loss += diff * diff;
        grad[i] = 2.0 * diff * invN;
    }
    return {loss * invN, grad};
}

std::pair<double, Vec> cross_entropy_loss(const Vec& pred, int target_idx) {
    double loss = -std::log(std::max(pred[target_idx], 1e-15));
    Vec grad = pred;
    grad[target_idx] -= 1.0;
    return {loss, grad};
}

std::pair<double, Vec> cross_entropy_loss_vec(const Vec& pred, const Vec& target) {
    double loss = 0.0;
    Vec grad(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        loss -= target[i] * std::log(std::max(pred[i], 1e-15));
        grad[i] = pred[i] - target[i];
    }
    return {loss, grad};
}

// ============================================================================
//  Gradient Clipping
// ============================================================================
double grad_norm(const std::vector<GradPair>& grads) {
    double total = 0.0;
    for (auto& [gw, gb] : grads) {
        for (auto& row : gw)
            for (auto v : row) total += v * v;
        for (auto v : gb) total += v * v;
    }
    return std::sqrt(total);
}

void clip_grad_by_norm(std::vector<GradPair>& grads, double max_norm) {
    double norm = grad_norm(grads);
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (auto& [gw, gb] : grads) {
            for (auto& row : gw) for (auto& v : row) v *= scale;
            for (auto& v : gb) v *= scale;
        }
    }
}

void clip_grad_by_value(std::vector<GradPair>& grads, double clip_val) {
    for (auto& [gw, gb] : grads) {
        for (auto& row : gw)
            for (auto& v : row)
                v = std::max(-clip_val, std::min(clip_val, v));
        for (auto& v : gb)
            v = std::max(-clip_val, std::min(clip_val, v));
    }
}

// ============================================================================
//  LR Schedulers
// ============================================================================
LRScheduler::LRScheduler(Optimizer* opt) : optimizer(opt) {}

StepLR::StepLR(Optimizer* opt, int step_sz, double gamma_)
    : LRScheduler(opt), step_size(step_sz), gamma(gamma_) {}

void StepLR::step(int epoch) {
    if (epoch % step_size == 0)
        optimizer->lr *= gamma;
}

ExponentialLR::ExponentialLR(Optimizer* opt, double gamma_)
    : LRScheduler(opt), gamma(gamma_) {}

void ExponentialLR::step(int epoch) {
    (void)epoch;
    optimizer->lr *= gamma;
}

CosineAnnealingLR::CosineAnnealingLR(Optimizer* opt, int T_max_, double eta_min_)
    : LRScheduler(opt), T_max(T_max_), eta_min(eta_min_), initial_lr(opt->lr) {}

void CosineAnnealingLR::step(int epoch) {
    double progress = (double)((epoch - 1) % T_max) / T_max;
    double lr_curr = eta_min + 0.5 * (initial_lr - eta_min) *
                                 (1.0 + std::cos(M_PI * progress));
    optimizer->lr = lr_curr;
}

// ============================================================================
//  Data Utilities
// ============================================================================
SplitResult split_data(const Dataset& data, double val_ratio,
                        double test_ratio, int seed_) {
    seed(seed_);
    Dataset shuffled = data;
    std::shuffle(shuffled.begin(), shuffled.end(), rng());

    int n = (int)shuffled.size();
    int n_test = (int)(n * test_ratio);
    int n_val = (int)(n * val_ratio);

    SplitResult result;
    result.splits.resize(3);
    result.splits[2] = Dataset(shuffled.begin(), shuffled.begin() + n_test);
    result.splits[1] = Dataset(shuffled.begin() + n_test, shuffled.begin() + n_test + n_val);
    result.splits[0] = Dataset(shuffled.begin() + n_test + n_val, shuffled.end());
    return result;
}

DataLoader::DataLoader(const Dataset& d, int bs, bool shuffle_)
    : data(d), batch_size(bs), do_shuffle(shuffle_), idx(0) {
    indices.resize(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (do_shuffle) reset();
}

bool DataLoader::has_next() { return idx < (int)data.size(); }

Dataset DataLoader::next_batch() {
    int end = std::min(idx + batch_size, (int)data.size());
    Dataset batch;
    for (int i = idx; i < end; ++i)
        batch.push_back(data[indices[i]]);
    idx = end;
    return batch;
}

void DataLoader::reset() {
    idx = 0;
    if (do_shuffle)
        std::shuffle(indices.begin(), indices.end(), rng());
}

int DataLoader::size() const { return (int)data.size(); }

// ============================================================================
//  Text Processing
// ============================================================================
TextProcessor::TextProcessor(const std::string& text) {
    std::set<char> char_set(text.begin(), text.end());
    chars.assign(char_set.begin(), char_set.end());
    for (size_t i = 0; i < chars.size(); ++i) {
        char_to_idx[chars[i]] = (int)i;
        idx_to_char[(int)i] = chars[i];
    }
    vocab_size = (int)chars.size();
}

Vec TextProcessor::encode_one_hot(char c) {
    Vec vec(vocab_size, 0.0);
    if (char_to_idx.count(c))
        vec[char_to_idx[c]] = 1.0;
    return vec;
}

std::vector<int> TextProcessor::encode_indices(const std::string& text) {
    std::vector<int> out;
    for (char c : text)
        out.push_back(char_to_idx.count(c) ? char_to_idx[c] : 0);
    return out;
}

std::string TextProcessor::decode_indices(const std::vector<int>& indices) {
    std::string out;
    for (int i : indices)
        out += idx_to_char.count(i) ? idx_to_char[i] : '?';
    return out;
}

Dataset TextProcessor::prepare_data(const std::string& text, int window_size) {
    Dataset data;
    for (size_t i = 0; i + window_size < text.size(); ++i) {
        Vec inp;
        for (int w = 0; w < window_size; ++w) {
            Vec oh = encode_one_hot(text[i + w]);
            inp.insert(inp.end(), oh.begin(), oh.end());
        }
        Vec target = encode_one_hot(text[i + window_size]);
        data.push_back({inp, target});
    }
    return data;
}

ClsDataset TextProcessor::prepare_index_data(const std::string& text, int window_size) {
    ClsDataset data;
    for (size_t i = 0; i + window_size < text.size(); ++i) {
        auto inp = encode_indices(text.substr(i, window_size));
        int target = char_to_idx[text[i + window_size]];
        data.push_back({Vec(inp.begin(), inp.end()), target});
    }
    return data;
}

// ============================================================================
//  高级工具函数
// ============================================================================
std::vector<GradPair> mlp_backward(MLP& model, const Vec& loss_grad) {
    std::vector<GradPair> grads;
    Vec delta = loss_grad;
    for (int i = (int)model.layers.size() - 1; i >= 0; --i) {
        auto [gw, gb, d] = model.layers[i].backward(delta);
        grads.insert(grads.begin(), {gw, gb});
        delta = d;
    }
    return grads;
}

Vec softmax(const Vec& logits) {
    Vec out = logits;
    double mx = *std::max_element(out.begin(), out.end());
    double sum = 0.0;
    for (auto& v : out) { v = std::exp(v - mx); sum += v; }
    for (auto& v : out) v /= sum;
    return out;
}

Vec softmax_with_temp(const Vec& logits, double temperature) {
    if (temperature == 1.0) return softmax(logits);
    Vec scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        scaled[i] = logits[i] / temperature;
    return softmax(scaled);
}

int sample_index(const Vec& probs) {
    double r = randu();
    double cum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (r < cum) return (int)i;
    }
    return (int)probs.size() - 1;
}

std::string generate_text(MLP& network, TextProcessor& tp,
                          const std::string& start_str,
                          int length, int window_size, double temp) {
    std::string generated = start_str;
    for (int _ = 0; _ < length; ++_) {
        std::string context = generated.substr(
            std::max(0, (int)generated.size() - window_size), window_size);
        while ((int)context.size() < window_size)
            context = std::string(1, tp.chars[0]) + context;

        Vec inp;
        for (char c : context) {
            Vec oh = tp.encode_one_hot(c);
            inp.insert(inp.end(), oh.begin(), oh.end());
        }
        Vec probs = network.forward(inp);
        Vec soft_probs = softmax_with_temp(probs, temp);
        int next_idx = sample_index(soft_probs);
        generated += tp.idx_to_char[next_idx];
    }
    return generated;
}

// ============================================================================
//  Model — High-Level API
// ============================================================================
Model::Model(MLP net, const std::string& loss_fn,
             const std::string& optimizer_name, double lr_,
             const std::string& task_)
    : network(std::move(net)), lr(lr_),
      dropout_rate(0.0), l2_lambda(0.0), test_split(0.0), trained(false)
{
    // 自动推断 task
    if (task_ == "auto" && !network.layers.empty()) {
        auto& last = network.layers.back();
        task = (last.act_type == Layer::Activation::SOFTMAX)
               ? "classification" : "regression";
    } else {
        task = task_;
    }

    // 损失函数
    loss_name = loss_fn;
    if (loss_fn != "mse" && loss_fn != "cross_entropy" && loss_fn != "ce")
        loss_name = task == "classification" ? "cross_entropy" : "mse";

    // 优化器
    opt_name = optimizer_name;
    if (optimizer_name == "adam")
        optimizer = std::make_unique<Adam>(&network, lr_);
    else if (optimizer_name == "sgd")
        optimizer = std::make_unique<SGD>(&network, lr_);
    else if (optimizer_name == "sgd_momentum")
        optimizer = std::make_unique<SGDMomentum>(&network, lr_);
    else if (optimizer_name == "rmsprop")
        optimizer = std::make_unique<RMSprop>(&network, lr_);
    else
        optimizer = std::make_unique<Adam>(&network, lr_);
}

Vec Model::forward(const Vec& x) { return network.forward(x); }
Vec Model::predict_one(const Vec& x) { return forward(x); }

std::vector<Vec> Model::predict(const std::vector<Vec>& x) {
    std::vector<Vec> out;
    for (auto& xi : x) out.push_back(forward(xi));
    return out;
}

int Model::predict_class(const Vec& x) {
    Vec probs = forward(x);
    return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

std::vector<int> Model::predict_classes(const std::vector<Vec>& x) {
    std::vector<int> out;
    for (auto& xi : x) out.push_back(predict_class(xi));
    return out;
}

double Model::evaluate(const Dataset& data) {
    double total = 0.0;
    for (auto& [xi, yi] : data) {
        Vec pred = forward(xi);
        if (loss_name == "mse" || loss_name == "ce") {
            auto [loss, _] = (loss_name == "mse")
                ? mse_loss(pred, yi) : cross_entropy_loss_vec(pred, yi);
            total += loss;
        }
    }
    return total / std::max(1, (int)data.size());
}

double Model::evaluate(const std::vector<Vec>& x, const std::vector<Vec>& y) {
    double total = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        Vec pred = forward(x[i]);
        auto [loss, _] = mse_loss(pred, y[i]);
        total += loss;
    }
    return total / std::max(1, (int)x.size());
}

double Model::score(const std::vector<Vec>& x, const std::vector<Vec>& y) {
    // R² for regression
    auto preds = predict(x);
    double ss_res = 0.0, ss_tot = 0.0;
    double y_mean = 0.0;
    for (auto& yi : y) y_mean += yi[0];
    y_mean /= y.size();
    for (size_t i = 0; i < x.size(); ++i) {
        double diff = preds[i][0] - y[i][0];
        ss_res += diff * diff;
        double diff_tot = y[i][0] - y_mean;
        ss_tot += diff_tot * diff_tot;
    }
    return 1.0 - ss_res / std::max(ss_tot, 1e-15);
}

double Model::score(const std::vector<Vec>& x, const std::vector<int>& y) {
    // accuracy for classification
    int correct = 0;
    for (size_t i = 0; i < x.size(); ++i)
        if (predict_class(x[i]) == y[i]) ++correct;
    return (double)correct / x.size();
}

// === fit (Dataset) ===
void Model::fit(const Dataset& data, int epochs, int batch_size,
                double val_split, const Dataset* val_data,
                int patience, bool verbose, bool shuffle,
                LRScheduler* lr_scheduler, double grad_clip) {
    Dataset train_set = data;
    Dataset val_set;
    Dataset test_set;

    if (val_data) {
        val_set = *val_data;
    } else if (val_split > 0 || test_split > 0) {
        auto splits = split_data(train_set, val_split, test_split);
        train_set = splits.splits[0];
        val_set = splits.splits[1];
        test_set = splits.splits[2];
    }

    if (batch_size <= 0) batch_size = (int)train_set.size();

    train_history.clear();
    val_history.clear();
    test_history.clear();
    trained = false;
    double best_val = DBL_MAX;
    int no_improve = 0;

    // 创建 Dropout 层（每个隐藏层一个）
    int n_layers = (int)network.layers.size();
    std::vector<Dropout> dops;
    if (dropout_rate > 0) {
        for (int li = 0; li < n_layers - 1; ++li)
            dops.emplace_back(dropout_rate);
    }
    bool use_dropout = !dops.empty();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        if (shuffle)
            std::shuffle(train_set.begin(), train_set.end(), rng());

        // 设置 dropout 为训练模式
        for (auto& d : dops) d.training = true;

        double train_loss = 0.0;
        for (int b = 0; b < (int)train_set.size(); b += batch_size) {
            int end = std::min(b + batch_size, (int)train_set.size());
            int B = end - b;
            std::vector<Vec2D> all_gw(n_layers);
            std::vector<Vec> all_gb(n_layers);
            for (int li = 0; li < n_layers; ++li) {
                all_gw[li].assign(network.layers[li].fan_out,
                    Vec(network.layers[li].fan_in, 0.0));
                all_gb[li].assign(network.layers[li].fan_out, 0.0);
            }

            for (int k = b; k < end; ++k) {
                auto& [xi, yi] = train_set[k];

                // === Forward 逐层 + Dropout ===
                Vec act = xi;
                for (int li = 0; li < n_layers; ++li) {
                    act = network.layers[li].forward(act);
                    if (use_dropout && li < n_layers - 1)
                        act = dops[li].forward(act);
                }
                Vec pred = act;

                double loss;
                Vec grad;
                if (task == "classification") {
                    int label = (int)yi[0];
                    auto [l, g] = cross_entropy_loss(pred, label);
                    loss = l; grad = g;
                } else {
                    auto [l, g] = mse_loss(pred, yi);
                    loss = l; grad = g;
                }
                train_loss += loss;

                // === Backward 逐层 + Dropout + L2 ===
                Vec delta = grad;
                for (int li = n_layers - 1; li >= 0; --li) {
                    if (use_dropout && li < n_layers - 1)
                        delta = dops[li].backward(delta);

                    auto [gw, gb, d] = network.layers[li].backward(delta);
                    for (int j = 0; j < network.layers[li].fan_out; ++j) {
                        for (int kk = 0; kk < network.layers[li].fan_in; ++kk)
                            all_gw[li][j][kk] += gw[j][kk];
                        all_gb[li][j] += gb[j];
                    }
                    delta = d;
                }
            }

            // 平均梯度 + L2 权重衰减 + 梯度裁剪
            std::vector<GradPair> avg_grads;
            for (int li = 0; li < n_layers; ++li) {
                for (auto& row : all_gw[li])
                    for (auto& v : row) v /= B;
                for (auto& v : all_gb[li]) v /= B;

                // L2 正则化：对权重梯度添加 l2_lambda * w
                if (l2_lambda > 0) {
                    for (int j = 0; j < network.layers[li].fan_out; ++j)
                        for (int kk = 0; kk < network.layers[li].fan_in; ++kk)
                            all_gw[li][j][kk] += l2_lambda * network.layers[li].weights[j][kk];
                }
                avg_grads.push_back({all_gw[li], all_gb[li]});
            }

            if (grad_clip > 0) clip_grad_by_norm(avg_grads, grad_clip);
            optimizer->step(avg_grads);
        }

        train_loss /= train_set.size();
        train_history.push_back({epoch, train_loss});

        double val_l = -1;
        if (!val_set.empty()) {
            val_l = evaluate(val_set);
            val_history.push_back({epoch, val_l});

            if (patience > 0) {
                if (val_l < best_val - 1e-12) {
                    best_val = val_l; no_improve = 0;
                } else {
                    if (++no_improve >= patience) {
                        if (verbose) std::cout << "Early stopping at epoch " << epoch << "\n";
                        break;
                    }
                }
            }
        }

        if (lr_scheduler) lr_scheduler->step(epoch);

        int report_every = std::max(1, std::min(epochs / 10, 50));
        if (verbose && (epoch == 1 || epoch % report_every == 0 || epoch == epochs)) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " | loss=" << std::fixed << std::setprecision(6) << train_loss;
            if (val_l >= 0) std::cout << " | val_loss=" << val_l;
            std::cout << "\n";
        }
    }

    // === 测试集评估 ===
    if (!test_set.empty()) {
        // 推理时关闭 dropout
        for (auto& d : dops) d.training = false;
        double test_l = evaluate(test_set);
        test_history.push_back({(int)train_history.size(), test_l});
        if (verbose) {
            std::cout << "\n--- Test Set Evaluation ---\n"
                      << "  Test samples: " << test_set.size() << "\n"
                      << "  Test loss: " << test_l << "\n";
            if (task == "classification") {
                int correct = 0;
                for (auto& [xi, yi] : test_set) {
                    Vec pred_t = forward(xi);
                    int pred_cls = (int)(std::max_element(pred_t.begin(), pred_t.end()) - pred_t.begin());
                    if (pred_cls == (int)yi[0]) ++correct;
                }
                std::cout << "  Test accuracy: " << 100.0 * correct / test_set.size() << "%\n";
            }
        }
    }

    trained = true;
}

// === fit (vector<Vec>, vector<Vec>) for regression ===
void Model::fit(const std::vector<Vec>& x, const std::vector<Vec>& y,
                int epochs, int batch_size, double val_split,
                const Dataset* val_data, int patience, bool verbose, bool shuffle,
                LRScheduler* lr_scheduler, double grad_clip) {
    Dataset data;
    for (size_t i = 0; i < x.size(); ++i)
        data.push_back({x[i], y[i]});
    fit(data, epochs, batch_size, val_split, val_data,
        patience, verbose, shuffle, lr_scheduler, grad_clip);
}

// === fit (vector<Vec>, vector<int>) for classification ===
void Model::fit(const std::vector<Vec>& x, const std::vector<int>& y,
                int epochs, int batch_size, double val_split,
                int patience, bool verbose,
                LRScheduler* lr_scheduler, double grad_clip) {
    Dataset data;
    for (size_t i = 0; i < x.size(); ++i)
        data.push_back({x[i], Vec(1, (double)y[i])});
    fit(data, epochs, batch_size, val_split, nullptr,
        patience, verbose, true, lr_scheduler, grad_clip);
}

// === Save / Load ===
void Model::save(const std::string& filepath, int epochs, int batch) {
    std::ofstream f(filepath);
    // 新格式：CONFIG: task loss_name opt_name lr epochs batch dropout l2 test_split
    f << "CONFIG: " << task << " " << loss_name << " " << opt_name << " "
      << lr << " " << epochs << " " << batch << " "
      << dropout_rate << " " << l2_lambda << " " << test_split << "\n";
    // 层数和每层结构
    f << network.layers.size() << "\n";
    for (size_t li = 0; li < network.layers.size(); ++li) {
        auto& L = network.layers[li];
        std::string act_str = "leaky_relu";
        if (L.act_type == Layer::Activation::RELU)    act_str = "relu";
        if (L.act_type == Layer::Activation::SOFTMAX)  act_str = "softmax";
        if (L.act_type == Layer::Activation::LINEAR)   act_str = "linear";
        f << L.fan_in << " " << L.fan_out << " " << act_str << "\n";
        // 权重
        for (int j = 0; j < L.fan_out; ++j) {
            for (int k = 0; k < L.fan_in; ++k) {
                if (k > 0) f << " ";
                f << L.weights[j][k];
            }
            f << "\n";
        }
        // 偏置
        for (int j = 0; j < L.fan_out; ++j) {
            if (j > 0) f << " ";
            f << L.biases[j];
        }
        f << "\n";
    }
    f.close();
    std::cout << "Model saved to " << filepath << "\n";
}

Model Model::load(const std::string& filepath, const std::string& loss_fn,
                  const std::string& optimizer_name, double lr_,
                  int* out_epochs, int* out_batch) {
    std::ifstream f(filepath);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + filepath);

    std::string first_line;
    std::getline(f, first_line);

    std::string task_str = "regression";
    std::string loss_name_loaded = loss_fn;
    std::string opt_name_loaded = optimizer_name;
    double lr_loaded = lr_;
    int epochs_loaded = -1, batch_loaded = -1;
    double dropout_loaded = 0.0, l2_loaded = 0.0, test_split_loaded = 0.0;

    bool is_new_format = (first_line.rfind("CONFIG:", 0) == 0);
    if (is_new_format) {
        // 解析 CONFIG: task loss_name opt_name lr epochs batch dropout l2 test_split
        std::istringstream iss(first_line.substr(7));
        iss >> task_str >> loss_name_loaded >> opt_name_loaded >> lr_loaded;
        if (iss >> epochs_loaded) {
            iss >> batch_loaded;
            // 尝试读取可选字段
            iss >> dropout_loaded;
            iss >> l2_loaded;
            iss >> test_split_loaded;
        }
    }

    int num_layers;
    if (is_new_format)
        f >> num_layers;
    else
        num_layers = std::stoi(first_line);  // 旧格式：第一行就是层数

    std::vector<int> fan_ins, fan_outs;
    std::vector<std::string> acts;
    std::vector<Vec2D> all_w;
    std::vector<Vec> all_b;

    for (int li = 0; li < num_layers; ++li) {
        int fi, fo;
        std::string act;
        f >> fi >> fo >> act;
        fan_ins.push_back(fi);
        fan_outs.push_back(fo);
        acts.push_back(act);

        // 读取权重
        Vec2D w(fo, Vec(fi));
        for (int j = 0; j < fo; ++j)
            for (int k = 0; k < fi; ++k)
                f >> w[j][k];
        all_w.push_back(w);

        // 读取偏置
        Vec b(fo);
        for (int j = 0; j < fo; ++j)
            f >> b[j];
        all_b.push_back(b);
    }

    f.close();

    // 构建 MLP
    MLP mlp;
    std::vector<Layer::Activation> act_enums;
    for (auto& a : acts) {
        if (a == "leaky_relu") act_enums.push_back(Layer::Activation::LEAKY_RELU);
        else if (a == "relu") act_enums.push_back(Layer::Activation::RELU);
        else if (a == "softmax") act_enums.push_back(Layer::Activation::SOFTMAX);
        else act_enums.push_back(Layer::Activation::LINEAR);
    }
    for (size_t i = 0; i < fan_ins.size(); ++i) {
        Layer l(fan_ins[i], fan_outs[i], act_enums[i], i == fan_ins.size() - 1);
        l.weights = all_w[i];
        l.biases = all_b[i];
        mlp.layers.push_back(l);
    }
    // 用文件中的配置构造 Model
    Model m(mlp, loss_name_loaded, opt_name_loaded, lr_loaded, task_str);
    m.dropout_rate = dropout_loaded;
    m.l2_lambda = l2_loaded;
    m.test_split = test_split_loaded;
    if (out_epochs) *out_epochs = epochs_loaded;
    if (out_batch) *out_batch = batch_loaded;
    return m;
}

void Model::summary() {
    std::cout << "==================================================\n"
              << "  Model (" << task << ")\n"
              << "  Loss: " << loss_name << "  |  Optimizer: "
              << opt_name << "(lr=" << lr << ")\n";
    if (dropout_rate > 0) std::cout << "  Dropout: " << dropout_rate << "\n";
    if (l2_lambda > 0) std::cout << "  L2 Lambda: " << l2_lambda << "\n";
    std::cout << "==================================================\n";
    int total = 0;
    for (size_t i = 0; i < network.layers.size(); ++i) {
        auto& L = network.layers[i];
        std::string act = "Linear";
        if (L.act_type == Layer::Activation::LEAKY_RELU) act = "LeakyReLU";
        if (L.act_type == Layer::Activation::RELU) act = "ReLU";
        if (L.act_type == Layer::Activation::SOFTMAX) act = "Softmax";
        int params = L.fan_in * L.fan_out + L.fan_out;
        total += params;
        std::cout << "  [" << i << "] Linear(" << L.fan_in << " -> "
                  << L.fan_out << ") + " << act << "   [" << params << "]\n";
    }
    std::cout << "==================================================\n"
              << "  Total params: " << total << "\n";
}

std::string Model::to_string() const {
    std::ostringstream oss;
    oss << "Model(task=" << task << ", loss=" << loss_name
        << ", opt=" << opt_name << ", lr=" << lr << ")";
    return oss.str();
}

// ============================================================================
//  Factory Functions
// ============================================================================
Model make_classifier(int input_dim, int num_classes,
                       const std::vector<int>& hidden_layers,
                       const std::string& act, double lr,
                       const std::string& optimizer_name) {
    std::vector<int> sizes = {input_dim};
    sizes.insert(sizes.end(), hidden_layers.begin(), hidden_layers.end());
    sizes.push_back(num_classes);

    std::vector<std::string> activations(sizes.size() - 2, act);
    activations.push_back("softmax");

    return Model(MLP(sizes, activations), "cross_entropy", optimizer_name, lr,
                 "classification");
}

Model make_regressor(int input_dim, int output_dim,
                      const std::vector<int>& hidden_layers,
                      const std::string& act, double lr,
                      const std::string& optimizer_name) {
    std::vector<int> sizes = {input_dim};
    sizes.insert(sizes.end(), hidden_layers.begin(), hidden_layers.end());
    sizes.push_back(output_dim);

    std::vector<std::string> activations(sizes.size() - 2, act);
    activations.push_back("linear");

    return Model(MLP(sizes, activations), "mse", optimizer_name, lr, "regression");
}

Model quick_train(const std::vector<Vec>& x, const std::vector<Vec>& y,
                  const std::string& task,
                  const std::vector<int>& hidden_layers,
                  int epochs, int batch_size, double lr,
                  double val_split, int patience,
                  const std::string& optimizer_name, bool verbose) {
    (void)task;
    int input_dim = (int)x[0].size();
    int output_dim = (int)y[0].size();
    Model m = make_regressor(input_dim, output_dim, hidden_layers,
                             "leaky_relu", lr, optimizer_name);
    if (verbose) m.summary();
    m.fit(x, y, epochs, batch_size, val_split, nullptr, patience, verbose);
    return m;
}

Model quick_train(const std::vector<Vec>& x, const std::vector<int>& y,
                  const std::vector<int>& hidden_layers,
                  int epochs, int batch_size, double lr,
                  double val_split, int patience,
                  const std::string& optimizer_name, bool verbose) {
    int input_dim = (int)x[0].size();
    std::set<int> classes(y.begin(), y.end());
    int num_classes = (int)classes.size();
    Model m = make_classifier(input_dim, num_classes, hidden_layers,
                              "leaky_relu", lr, optimizer_name);
    if (verbose) m.summary();
    m.fit(x, y, epochs, batch_size, val_split, patience, verbose);
    return m;
}

} // namespace ai
