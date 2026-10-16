/*
 * c_model.cpp —— 兼容层的实现：全部转调 c/ 下的纯 C 引擎
 */
#include "c_model.hpp"

extern "C" {
#include "ai.h"
#include "layers_conv.h"
#include "model_io.h"
#include "sched.h"
#include "tensor.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ceng {

/* ============================ 小工具 ============================ */

namespace {

Activation to_c_act(const std::string& name)
{
    if (name == "relu")    return ACT_RELU;
    if (name == "softmax") return ACT_SOFTMAX;
    if (name == "linear")  return ACT_LINEAR;
    return ACT_LEAKY_RELU;
}

const char* act_name_of_c(Activation a)
{
    switch (a) {
    case ACT_RELU:    return "relu";
    case ACT_SOFTMAX: return "softmax";
    case ACT_LINEAR:  return "linear";
    default:          return "leaky_relu";
    }
}

Layer::Activation to_view_act(Activation a)
{
    switch (a) {
    case ACT_RELU:    return Layer::Activation::RELU;
    case ACT_SOFTMAX: return Layer::Activation::SOFTMAX;
    case ACT_LINEAR:  return Layer::Activation::LINEAR;
    default:          return Layer::Activation::LEAKY_RELU;
    }
}

OptKind to_c_opt(const std::string& name)
{
    if (name == "sgd")      return OPT_SGD;
    if (name == "momentum") return OPT_SGD_MOMENTUM;
    if (name == "rmsprop")  return OPT_RMSPROP;
    return OPT_ADAM;
}

unsigned long long g_init_seed = 0x9E3779B97F4A7C15ULL;

std::mt19937& rng_engine()
{
    static thread_local std::mt19937 e{ std::random_device{}() };
    return e;
}

std::uniform_real_distribution<double>& rng_dist()
{
    static thread_local std::uniform_real_distribution<double> d(0.0, 1.0);
    return d;
}

} // namespace

/* ============================ Model::Impl ============================ */

struct Model::Impl {
    ::MLP*   net = nullptr;
    Trainer  tr{};
    ::Optimizer opt{};      // C 侧优化器（lr 由回调与 facade 同步）
    bool     trainer_ready = false;
    int      epochs_run = 0;
    int      batch_used = 0;
};

/* 把 facade 的 C++ 调度器（虚函数）桥接给 C 引擎：
 * C 侧每轮调 lrsched_step -> custom_step -> facade->step(epoch) */
struct C_LRSchedulerBridge {
    ::LRScheduler c{};
    ceng::LRScheduler* facade = nullptr;

    static void step_cb(void* ctx, ::Optimizer* /*c_opt*/, int epoch)
    {
        C_LRSchedulerBridge* self = static_cast<C_LRSchedulerBridge*>(ctx);
        if (self->facade) self->facade->step(epoch);
        /* lr 由 facade 的 ceng::Optimizer 承载，on_epoch_cb 会同步进 C 优化器 */
    }

    ::LRScheduler* as_c()
    {
        c.kind = SCHED_STEP;
        c.opt = nullptr;
        c.initial_lr = 0.0;
        c.step_size = 0;
        c.gamma = 1.0;
        c.T_max = 0;
        c.eta_min = 0.0;
        c.custom_step = step_cb;
        c.custom_ctx = this;
        return &c;
    }
};

/* 每轮回调：处理 Stop 按钮 + 把外部改过的学习率同步进 C 引擎 */
int on_epoch_cb(void* ctx, int epoch, double train_loss, double val_loss)
{
    Model* m = static_cast<Model*>(ctx);
    (void)epoch; (void)train_loss; (void)val_loss;
    if (m->optimizer && m->impl) m->impl->tr.opt.lr = m->optimizer->lr;
    if (m->stop_flag && m->stop_flag->load()) return 1;
    return 0;
}


Model::Model()
{
    impl = new Impl();
    optimizer = std::make_shared<Optimizer>();
}

Model::~Model()
{
    if (impl) {
        if (impl->trainer_ready) trainer_release(&impl->tr);
        if (impl->net) mlp_free(impl->net);
        delete impl;
        impl = nullptr;
    }
}

Model::Model(Model&& other) noexcept
{
    impl = other.impl;
    other.impl = nullptr;
    task = std::move(other.task);
    loss_name = std::move(other.loss_name);
    opt_name = std::move(other.opt_name);
    lr = other.lr;
    last_epochs = other.last_epochs;
    last_batch = other.last_batch;
    dropout_rate = other.dropout_rate;
    l2_lambda = other.l2_lambda;
    test_split = other.test_split;
    stop_flag = other.stop_flag;
    optimizer = std::move(other.optimizer);
    network = std::move(other.network);
    train_history = std::move(other.train_history);
    val_history = std::move(other.val_history);
    other.stop_flag = nullptr;
}

Model& Model::operator=(Model&& other) noexcept
{
    if (this == &other) return *this;
    if (impl) {
        if (impl->trainer_ready) trainer_release(&impl->tr);
        if (impl->net) mlp_free(impl->net);
        delete impl;
    }
    impl = other.impl;
    other.impl = nullptr;
    task = std::move(other.task);
    loss_name = std::move(other.loss_name);
    opt_name = std::move(other.opt_name);
    lr = other.lr;
    last_epochs = other.last_epochs;
    last_batch = other.last_batch;
    dropout_rate = other.dropout_rate;
    l2_lambda = other.l2_lambda;
    test_split = other.test_split;
    stop_flag = other.stop_flag;
    optimizer = std::move(other.optimizer);
    network = std::move(other.network);
    train_history = std::move(other.train_history);
    val_history = std::move(other.val_history);
    other.stop_flag = nullptr;
    return *this;
}

int Model::epochs_run() const { return impl ? impl->epochs_run : 0; }
int Model::batch_used() const { return impl ? impl->batch_used : 0; }

void Model::rebuild_from(const int* sizes, const int* acts, int n_sizes)
{
    if (impl->trainer_ready) { trainer_release(&impl->tr); impl->trainer_ready = false; }
    if (impl->net) { mlp_free(impl->net); impl->net = nullptr; }

    Activation cacts[64];
    for (int i = 0; i < n_sizes - 1 && i < 64; i++) cacts[i] = (Activation)acts[i];
    impl->net = mlp_create(sizes, cacts, n_sizes);
    for (int i = 0; i < impl->net->n_layers; i++) he_init(&impl->net->layers[i], &g_init_seed);

    trainer_init(&impl->tr, impl->net, (task == "classification") ? LOSS_CROSS_ENTROPY : LOSS_MSE,
                 to_c_opt(opt_name), lr);
    impl->trainer_ready = true;
    sync_network_view();
}

void Model::sync_network_view()
{
    network.layers.clear();
    if (!impl || !impl->net) return;
    for (int i = 0; i < impl->net->n_layers; i++) {
        ::Layer* cl = &impl->net->layers[i];
        Layer lv;
        lv.fan_in = cl->fan_in;
        lv.fan_out = cl->fan_out;
        lv.act_type = to_view_act(cl->act);
        lv.act_name = act_name_of_c(cl->act);
        lv.weights.assign(cl->fan_out, std::vector<double>(cl->fan_in, 0.0));
        for (int j = 0; j < cl->fan_out; j++)
            for (int k = 0; k < cl->fan_in; k++)
                lv.weights[j][k] = cl->W[(size_t)j * cl->fan_in + k];
        lv.biases.assign(cl->b, cl->b + cl->fan_out);
        network.layers.push_back(std::move(lv));
    }
}

/* ============================ 训练 ============================ */

void Model::fit(const std::vector<Vec>& x, const std::vector<Vec>& y,
                int epochs, int batch_size, double val_split, const Dataset* val_data,
                int patience, bool verbose, bool shuffle,
                LRScheduler* lr_scheduler, double grad_clip)
{
    (void)val_data;
    if (!impl || !impl->net || x.empty()) return;

    const int n = (int)x.size();
    const int in_dim = (int)x[0].size();
    const int out_dim = impl->net->layers[impl->net->n_layers - 1].fan_out;

    std::vector<double> xf((size_t)n * in_dim), yf((size_t)n * out_dim, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < in_dim && j < (int)x[i].size(); j++) xf[(size_t)i * in_dim + j] = x[i][j];
        for (int j = 0; j < out_dim && j < (int)y[i].size(); j++) yf[(size_t)i * out_dim + j] = y[i][j];
    }

    if (impl->trainer_ready) { trainer_release(&impl->tr); impl->trainer_ready = false; }
    trainer_init(&impl->tr, impl->net, LOSS_MSE, to_c_opt(opt_name), lr);
    impl->trainer_ready = true;
    impl->opt = impl->tr.opt;

    ::Dataset data { xf.data(), yf.data(), nullptr, n, in_dim, out_dim };

    std::vector<double> ht((size_t)std::max(1, epochs), 0.0), hv((size_t)std::max(1, epochs), 0.0);
    FitConfig fc;
    std::memset(&fc, 0, sizeof(fc));
    fc.epochs = epochs;
    fc.batch_size = batch_size;
    fc.patience = patience;
    fc.val_split = val_split;
    fc.shuffle = shuffle ? 1 : 0;
    fc.verbose = 0;
    fc.seed = 42;
    fc.grad_clip = grad_clip;
    fc.early_stop_on_train = early_stop_on_train ? 1 : 0;
    fc.on_epoch = on_epoch_cb;
    fc.on_epoch_ctx = this;

    C_LRSchedulerBridge bridge;
    if (lr_scheduler) {
        bridge.facade = lr_scheduler;
        bridge.as_c()->opt = &impl->tr.opt;
        fc.lr_scheduler = bridge.as_c();
    }

    int done = trainer_fit(&impl->tr, &data, nullptr, &fc, ht.data(), hv.data());
    impl->epochs_run = done;
    impl->batch_used = fc.batch_size;

    train_history.clear();
    val_history.clear();
    for (int e = 0; e < done; e++) {
        train_history.push_back({ e + 1, ht[e] });
        if (val_split > 0.0) val_history.push_back({ e + 1, hv[e] });
    }
    last_epochs = epochs;
    last_batch = batch_size;

    if (verbose) {
        std::cout << "  trained " << done << " epochs, final loss = "
                  << (train_history.empty() ? 0.0 : train_history.back().second) << "\n";
    }
}

void Model::fit(const std::vector<Vec>& x, const std::vector<int>& y,
                int epochs, int batch_size, double val_split, int patience,
                bool verbose, LRScheduler* lr_scheduler, double grad_clip)
{
    if (!impl || !impl->net || x.empty()) return;

    const int n = (int)x.size();
    const int in_dim = (int)x[0].size();
    if ((int)y.size() != n) return;

    std::vector<double> xf((size_t)n * in_dim);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < in_dim && j < (int)x[i].size(); j++)
            xf[(size_t)i * in_dim + j] = x[i][j];
    std::vector<int> yi(y.begin(), y.end());

    if (impl->trainer_ready) { trainer_release(&impl->tr); impl->trainer_ready = false; }
    trainer_init(&impl->tr, impl->net, LOSS_CROSS_ENTROPY, to_c_opt(opt_name), lr);
    impl->trainer_ready = true;

    ::Dataset data { xf.data(), nullptr, yi.data(), n, in_dim, 0 };

    std::vector<double> ht((size_t)std::max(1, epochs), 0.0), hv((size_t)std::max(1, epochs), 0.0);
    FitConfig fc;
    std::memset(&fc, 0, sizeof(fc));
    fc.epochs = epochs;
    fc.batch_size = batch_size;
    fc.patience = patience;
    fc.val_split = val_split;
    fc.shuffle = 1;
    fc.verbose = 0;
    fc.seed = 42;
    fc.grad_clip = grad_clip;
    fc.early_stop_on_train = early_stop_on_train ? 1 : 0;
    fc.on_epoch = on_epoch_cb;
    fc.on_epoch_ctx = this;

    C_LRSchedulerBridge bridge;
    if (lr_scheduler) {
        bridge.facade = lr_scheduler;
        bridge.as_c()->opt = &impl->tr.opt;
        fc.lr_scheduler = bridge.as_c();
    }

    int done = trainer_fit(&impl->tr, &data, nullptr, &fc, ht.data(), hv.data());
    impl->epochs_run = done;
    impl->batch_used = fc.batch_size;

    train_history.clear();
    val_history.clear();
    for (int e = 0; e < done; e++) {
        train_history.push_back({ e + 1, ht[e] });
        if (val_split > 0.0) val_history.push_back({ e + 1, hv[e] });
    }
    last_epochs = epochs;
    last_batch = batch_size;

    if (verbose) {
        std::cout << "  trained " << done << " epochs, final loss = "
                  << (train_history.empty() ? 0.0 : train_history.back().second) << "\n";
    }
}

/* ============================ 推理 / 评估 ============================ */

Vec Model::predict_one(const Vec& x)
{
    Vec out;
    if (!impl || !impl->net) return out;
    const int out_dim = impl->net->layers[impl->net->n_layers - 1].fan_out;
    out.assign(out_dim, 0.0);
    mlp_forward(impl->net, x.data(), out.data());
    return out;
}

int Model::predict_class(const Vec& x)
{
    Vec p = predict_one(x);
    if (p.empty()) return 0;
    int best = 0;
    for (int i = 1; i < (int)p.size(); i++) if (p[i] > p[best]) best = i;
    return best;
}

double Model::score(const std::vector<Vec>& x, const std::vector<Vec>& y)
{
    if (!impl || !impl->net || x.empty()) return 0.0;
    const int out_dim = impl->net->layers[impl->net->n_layers - 1].fan_out;
    const int n = (int)x.size();
    std::vector<double> pred((size_t)n * out_dim, 0.0);
    for (int i = 0; i < n; i++) {
        Vec p = predict_one(x[i]);
        for (int j = 0; j < out_dim; j++) pred[(size_t)i * out_dim + j] = p[j];
    }
    /* 与修好后的 Python 版一致：多输出按维度平均 */
    double total = 0.0;
    for (int j = 0; j < out_dim; j++) {
        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += (j < (int)y[i].size()) ? y[i][j] : 0.0;
        mean /= n;
        double ss_res = 0.0, ss_tot = 0.0;
        for (int i = 0; i < n; i++) {
            double yv = (j < (int)y[i].size()) ? y[i][j] : 0.0;
            double d = yv - pred[(size_t)i * out_dim + j];
            ss_res += d * d;
            double dm = yv - mean;
            ss_tot += dm * dm;
        }
        total += 1.0 - ss_res / std::max(ss_tot, 1e-15);
    }
    return total / out_dim;
}

double Model::score(const std::vector<Vec>& x, const std::vector<int>& y)
{
    if (x.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < x.size(); i++)
        if (i < y.size() && predict_class(x[i]) == y[i]) correct++;
    return (double)correct / (double)x.size();
}

void Model::summary()
{
    std::cout << "Model (" << task << ")  optimizer=" << opt_name
              << "  lr=" << lr << "\n";
    int total = 0;
    for (size_t i = 0; i < network.layers.size(); i++) {
        const Layer& L = network.layers[i];
        int p = L.fan_in * L.fan_out + L.fan_out;
        total += p;
        std::cout << "  [" << i << "] Linear(" << L.fan_in << " -> " << L.fan_out
                  << ") + " << L.act_name << "   [" << p << " params]\n";
    }
    std::cout << "  total params = " << total << "\n";
}

/* ==================== 模型瘦身与统计 ==================== */

int Model::weight_count() const
{
    if (!impl || !impl->net) return 0;
    int n = 0;
    for (int i = 0; i < impl->net->n_layers; i++)
        n += impl->net->layers[i].fan_in * impl->net->layers[i].fan_out;
    return n;
}

int Model::zero_count() const
{
    if (!impl || !impl->net) return 0;
    int n = 0;
    for (int i = 0; i < impl->net->n_layers; i++) {
        ::Layer* L = &impl->net->layers[i];
        int cnt = L->fan_in * L->fan_out;
        for (int k = 0; k < cnt; k++) if (L->W[k] == 0.0) n++;
    }
    return n;
}

void Model::reset_init_seed(unsigned long long s)
{
    g_init_seed = s;
}

bool Model::merge_weights(const Model& other, double wSelf, double wOther)
{
    if (!impl || !impl->net || !other.impl || !other.impl->net) return false;
    ::MLP* A = impl->net;
    ::MLP* B = other.impl->net;
    if (A->n_layers != B->n_layers) return false;
    for (int i = 0; i < A->n_layers; i++) {
        ::Layer* La = &A->layers[i];
        ::Layer* Lb = &B->layers[i];
        if (La->fan_in != Lb->fan_in || La->fan_out != Lb->fan_out) return false;
        int cnt = La->fan_in * La->fan_out;
        for (int k = 0; k < cnt; k++) La->W[k] = wSelf * La->W[k] + wOther * Lb->W[k];
        for (int j = 0; j < La->fan_out; j++) La->b[j] = wSelf * La->b[j] + wOther * Lb->b[j];
    }
    sync_network_view();
    return true;
}

int Model::prune_magnitude(double threshold)
{
    if (!impl || !impl->net || threshold <= 0.0) return 0;
    int pruned = 0;
    for (int i = 0; i < impl->net->n_layers; i++) {
        ::Layer* L = &impl->net->layers[i];
        int cnt = L->fan_in * L->fan_out;
        for (int k = 0; k < cnt; k++)
            if (std::fabs(L->W[k]) < threshold) { L->W[k] = 0.0; pruned++; }
    }
    sync_network_view();
    return pruned;
}

double Model::quantize_weights(int bits)
{
    if (!impl || !impl->net || bits < 2 || bits > 24) return -1.0;
    double mn = 1e300, mx = -1e300;
    for (int i = 0; i < impl->net->n_layers; i++) {
        ::Layer* L = &impl->net->layers[i];
        int cnt = L->fan_in * L->fan_out;
        for (int k = 0; k < cnt; k++) {
            double v = L->W[k];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    if (mx <= mn) return -1.0;
    int levels = (1 << bits) - 1;
    double step = (mx - mn) / levels;
    if (step <= 0.0) return -1.0;
    for (int i = 0; i < impl->net->n_layers; i++) {
        ::Layer* L = &impl->net->layers[i];
        int cnt = L->fan_in * L->fan_out;
        for (int k = 0; k < cnt; k++) {
            double q = std::round((L->W[k] - mn) / step);
            L->W[k] = mn + q * step;
        }
    }
    sync_network_view();
    return step;
}

/* ==================== 与 Python 版 JSON 格式互通 ==================== */

bool Model::save_python_json(const std::string& filepath)
{
    if (!impl || !impl->net) return false;
    sync_network_view();
    return mlp_save_json(impl->net, filepath.c_str()) == 0;
}

Model Model::load_python_json(const std::string& filepath)
{
    Model m;
    ::MLP* net = mlp_load_json(filepath.c_str());
    if (!net) return m;

    if (m.impl->trainer_ready) { trainer_release(&m.impl->tr); m.impl->trainer_ready = false; }
    if (m.impl->net) mlp_free(m.impl->net);
    m.impl->net = net;

    // 按输出层激活推断任务类型（JSON 里不存 task）
    ::Layer* last = &net->layers[net->n_layers - 1];
    m.task = (last->act == ACT_SOFTMAX) ? "classification" : "regression";
    m.loss_name = (last->act == ACT_SOFTMAX) ? "cross_entropy" : "mse";
    trainer_init(&m.impl->tr, net,
                 (last->act == ACT_SOFTMAX) ? LOSS_CROSS_ENTROPY : LOSS_MSE,
                 to_c_opt(m.opt_name), m.lr);
    m.impl->trainer_ready = true;
    m.sync_network_view();
    return m;
}

/* ============================ 存档（沿用旧引擎的文本格式） ============================ */

void Model::save(const std::string& filepath, int epochs, int batch)
{
    sync_network_view();
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cout << "save failed: " << filepath << "\n";
        return;
    }
    f << std::setprecision(17);
    f << "CONFIG: " << task << " " << loss_name << " " << opt_name << " "
      << lr << " " << epochs << " " << batch << " "
      << dropout_rate << " " << l2_lambda << " " << test_split << "\n";
    f << network.layers.size() << "\n";
    for (size_t li = 0; li < network.layers.size(); ++li) {
        Layer& L = network.layers[li];
        f << L.fan_in << " " << L.fan_out << " " << L.act_name << "\n";
        for (int j = 0; j < L.fan_out; ++j) {
            for (int k = 0; k < L.fan_in; ++k) {
                if (k > 0) f << " ";
                f << L.weights[j][k];
            }
            f << "\n";
        }
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
                  int* out_epochs, int* out_batch)
{
    Model m;
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cout << "load failed: " << filepath << "\n";
        return m;
    }

    std::string line;
    std::getline(f, line);
    {
        std::istringstream ss(line);
        std::string tag;
        double ep = -1, bt = -1;
        ss >> tag;                       // "CONFIG:"
        ss >> m.task >> m.loss_name >> m.opt_name >> m.lr >> ep >> bt
           >> m.dropout_rate >> m.l2_lambda >> m.test_split;
        m.last_epochs = (int)ep;
        m.last_batch = (int)bt;
        if (out_epochs) *out_epochs = (int)ep;
        if (out_batch)  *out_batch  = (int)bt;
    }
    /* 旧文件里 opt/lr 可能是占位，允许调用方覆盖 */
    if (!optimizer_name.empty()) m.opt_name = optimizer_name;
    if (lr_ > 0.0) m.lr = lr_;
    if (!loss_fn.empty()) m.loss_name = loss_fn;
    m.optimizer->lr = m.lr;
    m.optimizer->name = m.opt_name;

    int n_layers = 0;
    f >> n_layers;
    if (n_layers <= 0) return m;

    std::vector<int> sizes;
    std::vector<int> acts;
    struct RawLayer { int fan_in, fan_out; std::vector<std::vector<double>> w; std::vector<double> b; std::string act; };
    std::vector<RawLayer> raws;
    for (int li = 0; li < n_layers; ++li) {
        RawLayer R;
        f >> R.fan_in >> R.fan_out >> R.act;
        R.w.assign(R.fan_out, std::vector<double>(R.fan_in, 0.0));
        for (int j = 0; j < R.fan_out; ++j)
            for (int k = 0; k < R.fan_in; ++k) f >> R.w[j][k];
        R.b.assign(R.fan_out, 0.0);
        for (int j = 0; j < R.fan_out; ++j) f >> R.b[j];
        raws.push_back(std::move(R));
    }
    sizes.push_back(raws[0].fan_in);
    for (auto& R : raws) { sizes.push_back(R.fan_out); acts.push_back((int)to_c_act(R.act)); }

    m.rebuild_from(sizes.data(), acts.data(), (int)sizes.size());
    for (int li = 0; li < n_layers; ++li) {
        ::Layer* cl = &m.impl->net->layers[li];
        for (int j = 0; j < cl->fan_out; j++) {
            for (int k = 0; k < cl->fan_in; k++) cl->W[(size_t)j * cl->fan_in + k] = raws[li].w[j][k];
            cl->b[j] = raws[li].b[j];
        }
    }
    m.sync_network_view();
    std::cout << "Model loaded from " << filepath << "\n";
    return m;
}

/* ============================ StepLR ============================ */

StepLR::StepLR(Optimizer* opt, int step_size, double gamma)
    : opt_(opt), step_size_(step_size > 0 ? step_size : 1), gamma_(gamma)
{
    initial_lr_ = opt_ ? opt_->lr : 0.01;
}

void StepLR::step(int epoch)
{
    if (!opt_) return;
    if (epoch > 0 && epoch % step_size_ == 0)
        opt_->lr = initial_lr_ * std::pow(gamma_, (double)(epoch / step_size_));
}

/* ============================ 构造辅助 ============================ */

Model make_regressor(int input_dim, int output_dim, std::vector<int> hidden,
                     std::string act, double lr, std::string opt)
{
    Model m;
    m.task = "regression";
    m.loss_name = "mse";
    m.opt_name = opt;
    m.lr = lr;
    m.optimizer->lr = lr;
    m.optimizer->name = opt;

    std::vector<int> sizes;
    std::vector<int> acts;
    sizes.push_back(input_dim);
    for (int h : hidden) sizes.push_back(h);
    sizes.push_back(output_dim);
    for (size_t i = 0; i + 1 < sizes.size(); i++) {
        bool is_last = (i + 2 == sizes.size());
        acts.push_back((int)(is_last ? ACT_LINEAR : to_c_act(act)));
    }
    m.rebuild_from(sizes.data(), acts.data(), (int)sizes.size());
    return m;
}

Model make_classifier(int input_dim, int num_classes, std::vector<int> hidden,
                      std::string act, double lr, std::string opt)
{
    Model m;
    m.task = "classification";
    m.loss_name = "cross_entropy";
    m.opt_name = opt;
    m.lr = lr;
    m.optimizer->lr = lr;
    m.optimizer->name = opt;

    std::vector<int> sizes;
    std::vector<int> acts;
    sizes.push_back(input_dim);
    for (int h : hidden) sizes.push_back(h);
    sizes.push_back(num_classes);
    for (size_t i = 0; i + 1 < sizes.size(); i++) {
        bool is_last = (i + 2 == sizes.size());
        acts.push_back((int)(is_last ? ACT_SOFTMAX : to_c_act(act)));
    }
    m.rebuild_from(sizes.data(), acts.data(), (int)sizes.size());
    return m;
}

Model quick_train(const std::vector<Vec>& x, const std::vector<Vec>& y,
                  const std::string& task, std::vector<int> hidden,
                  int epochs, int batch)
{
    int in_dim = x.empty() ? 1 : (int)x[0].size();
    int out_dim = y.empty() ? 1 : (int)y[0].size();
    Model m = (task == "classification")
              ? make_classifier(in_dim, out_dim, hidden)
              : make_regressor(in_dim, out_dim, hidden);
    m.fit(x, y, epochs, batch, 0.0, nullptr, 0, true, true, nullptr, 0.0);
    return m;
}

Model quick_train(const std::vector<Vec>& x, const std::vector<int>& y,
                  std::vector<int> hidden, int epochs, int batch)
{
    int in_dim = x.empty() ? 1 : (int)x[0].size();
    int classes = 1;
    for (int v : y) if (v + 1 > classes) classes = v + 1;
    Model m = make_classifier(in_dim, classes, hidden);
    m.fit(x, y, epochs, batch, 0.0, 0, true, nullptr, 0.0);
    return m;
}

/* ============================ 卷积 / 池化 / 展平 / 前向 MLP ============================ */

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding)
{
    impl_ = conv2d_create(in_channels, out_channels, kernel_size, stride, padding);
}

Conv2d::~Conv2d()
{
    if (impl_) conv2d_free((::Conv2d*)impl_);
    impl_ = nullptr;
}

Vec3D Conv2d::forward(const Vec3D& x)
{
    Vec3D out;
    if (x.empty() || x[0].empty()) return out;
    int C = (int)x.size(), H = (int)x[0].size(), W = (int)x[0][0].size();
    int xs[3] = { C, H, W };
    ::Conv2d* cv = (::Conv2d*)impl_;
    conv2d_bind_shape(cv, C, H, W);
    int ho = 0, wo = 0;
    conv2d_out_shape(cv, H, W, &ho, &wo);
    int os[3] = { cv->out_channels, ho, wo };
    Tensor* tx = tensor_new(3, xs);
    Tensor* ty = tensor_new(3, os);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                tx->data[(size_t)c * H * W + (size_t)h * W + w] = x[c][h][w];
    conv2d_forward(cv, tx, ty);
    out.assign(cv->out_channels, Vec2D(ho, Vec(wo, 0.0)));
    for (int c = 0; c < cv->out_channels; c++)
        for (int h = 0; h < ho; h++)
            for (int w = 0; w < wo; w++)
                out[c][h][w] = ty->data[((size_t)c * ho + h) * wo + w];
    tensor_free(tx);
    tensor_free(ty);
    return out;
}

MaxPool2d::MaxPool2d(int kernel_size, int stride)
{
    impl_ = maxpool2d_create(kernel_size, stride < 0 ? kernel_size : stride);
}

MaxPool2d::~MaxPool2d()
{
    if (impl_) maxpool2d_free((::MaxPool2d*)impl_);
    impl_ = nullptr;
}

Vec3D MaxPool2d::forward(const Vec3D& x)
{
    Vec3D out;
    if (x.empty() || x[0].empty()) return out;
    int C = (int)x.size(), H = (int)x[0].size(), W = (int)x[0][0].size();
    int xs[3] = { C, H, W };
    ::MaxPool2d* mp = (::MaxPool2d*)impl_;
    maxpool2d_bind_shape(mp, C, H, W);
    int ho = 0, wo = 0;
    maxpool2d_out_shape(mp, H, W, &ho, &wo);
    int os[3] = { C, ho, wo };
    Tensor* tx = tensor_new(3, xs);
    Tensor* ty = tensor_new(3, os);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                tx->data[(size_t)c * H * W + (size_t)h * W + w] = x[c][h][w];
    maxpool2d_forward(mp, tx, ty);
    out.assign(C, Vec2D(ho, Vec(wo, 0.0)));
    for (int c = 0; c < C; c++)
        for (int h = 0; h < ho; h++)
            for (int w = 0; w < wo; w++)
                out[c][h][w] = ty->data[((size_t)c * ho + h) * wo + w];
    tensor_free(tx);
    tensor_free(ty);
    return out;
}

Flatten::Flatten() { impl_ = flatten_create(); }
Flatten::~Flatten() { if (impl_) flatten_free((::Flatten*)impl_); impl_ = nullptr; }

Vec Flatten::forward(const Vec3D& x)
{
    Vec out;
    if (x.empty() || x[0].empty()) return out;
    int C = (int)x.size(), H = (int)x[0].size(), W = (int)x[0][0].size();
    int xs[3] = { C, H, W };
    int os[1] = { C * H * W };
    Tensor* tx = tensor_new(3, xs);
    Tensor* ty = tensor_new(1, os);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                tx->data[(size_t)c * H * W + (size_t)h * W + w] = x[c][h][w];
    flatten_forward((::Flatten*)impl_, tx, ty);
    out.assign(ty->data, ty->data + ty->size);
    tensor_free(tx);
    tensor_free(ty);
    return out;
}

MLP::MLP(std::vector<int> sizes, std::vector<std::string> activations)
{
    const int n = (int)sizes.size();
    std::vector<Activation> acts((size_t)std::max(0, n - 1), ACT_LEAKY_RELU);
    for (int i = 0; i < n - 1; i++)
        acts[i] = (i < (int)activations.size()) ? to_c_act(activations[i]) : ACT_LEAKY_RELU;
    ::MLP* net = mlp_create(sizes.data(), acts.data(), n);
    for (int i = 0; i < net->n_layers; i++) he_init(&net->layers[i], &g_init_seed);
    impl_ = net;
}

MLP::~MLP()
{
    if (impl_) mlp_free((::MLP*)impl_);
    impl_ = nullptr;
}

Vec MLP::forward(const Vec& x)
{
    ::MLP* net = (::MLP*)impl_;
    Vec out((size_t)net->layers[net->n_layers - 1].fan_out, 0.0);
    mlp_forward(net, x.data(), out.data());
    return out;
}

/* ============================ 随机数辅助 ============================ */

void seed(int s) { rng_engine().seed((unsigned)s); }
double randu() { return rng_dist()(rng_engine()); }

} // namespace ceng
