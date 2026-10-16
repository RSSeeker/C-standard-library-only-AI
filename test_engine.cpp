/*
 * test_engine.cpp —— C++ 引擎（AI.hpp / AI.cpp）的最小回归测试
 *
 * 背景：这个库此前完全没有测试，改动只能靠人工核对。这里先立一个最小但有效的基线：
 *   1) 回归能收敛（R²）
 *   2) 分类能收敛（准确率）
 *   3) 训练损失单调下降（不是发散的）
 *   4) 存档往返：存盘再读回，同一输入的预测必须一致
 *   5) 模型瘦身：param_count / sparsity / prune_magnitude / quantize_weights / merge_weights
 *   6) 早停：patience 生效（训练轮数被截断）
 *
 * 编译（不依赖 Win32）：
 *   g++ -std=c++17 -O2 -I. -o test_engine.exe test_engine.cpp AI.cpp
 * 运行：test_engine.exe      -> 全部通过返回 0
 */
#include "AI.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ai;

static int g_pass = 0, g_fail = 0;

static void check(const char* name, bool ok, const std::string& detail = "")
{
    if (ok) { g_pass++; std::printf("  [OK]   %-42s %s\n", name, detail.c_str()); }
    else    { g_fail++; std::printf("  [FAIL] %-42s %s\n", name, detail.c_str()); }
}

static std::string f2s(double v, int p = 6)
{
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", p, v);
    return b;
}

/* ---------------- 1/2/3) 训练收敛 ---------------- */
static void test_training()
{
    std::printf("\n== 训练收敛 ==\n");
    std::vector<Vec> x;
    std::vector<Vec> y;
    for (int i = 0; i < 40; ++i) {
        double a = -1.0 + 2.0 * i / 39.0;
        x.push_back({ a, 0.5 * a });
        y.push_back({ a * a + 0.5 * a });
    }

    Model m = make_regressor(2, 1, { 16, 16 }, "leaky_relu", 0.02);
    m.fit(x, y, 800, 8, 0.1, nullptr, 50, false, true, nullptr, 0.0);
    double r2 = m.score(x, y);
    check("回归收敛 R² > 0.95", r2 > 0.95, "R² = " + f2s(r2));

    // 损失应当整体下降（比较前 10 轮均值与最后 10 轮均值）
    if (m.train_history.size() > 20) {
        double head = 0, tail = 0;
        for (size_t i = 0; i < 10; ++i) head += m.train_history[i].second;
        for (size_t i = m.train_history.size() - 10; i < m.train_history.size(); ++i) tail += m.train_history[i].second;
        check("训练损失下降", tail < head, "首段 " + f2s(head / 10) + " -> 末段 " + f2s(tail / 10));
    } else {
        check("训练损失下降", false, "history 太短");
    }

    std::vector<Vec> cx;
    std::vector<int> cy;
    for (int i = 0; i < 60; ++i) {
        double a = -1.0 + 2.0 * i / 59.0;
        cx.push_back({ a, a * a });
        cy.push_back(a > 0 ? 1 : 0);
    }
    Model c = make_classifier(2, 2, { 16, 16 }, "leaky_relu", 0.02);
    c.fit(cx, cy, 500, 10, 0.1, 50, false, nullptr, 0.0);
    double acc = c.score(cx, cy);
    check("分类收敛 准确率 > 0.9", acc > 0.9, "acc = " + f2s(acc * 100, 2) + "%");
}

/* ---------------- 4) 存档往返 ---------------- */
static void test_save_load()
{
    std::printf("\n== 存档往返 ==\n");
    std::vector<Vec> x;
    std::vector<Vec> y;
    for (int i = 0; i < 20; ++i) {
        double a = -1.0 + 2.0 * i / 19.0;
        x.push_back({ a });
        y.push_back({ 3.0 * a - 0.5 });
    }
    Model m = make_regressor(1, 1, { 8, 8 }, "leaky_relu", 0.03);
    m.fit(x, y, 600, 4, 0.0, nullptr, 0, false, true, nullptr, 0.0);

    Vec before = m.predict({ Vec{ 0.25 } })[0];
    const char* path = "test_engine_model.txt";
    m.save(path, 600, 4);

    int ep = -1, bt = -1;
    Model m2 = Model::load(path, "mse", "adam", 0.03, &ep, &bt);
    Vec after = m2.predict({ Vec{ 0.25 } })[0];

    double d = std::fabs(before[0] - after[0]);
    check("存盘读回后预测一致", d < 1e-9,
          f2s(before[0], 10) + " vs " + f2s(after[0], 10) + " (Δ=" + f2s(d, 2) + ")");
    check("读回的 epochs/batch 正确", ep == 600 && bt == 4,
          "epochs=" + std::to_string(ep) + " batch=" + std::to_string(bt));
    std::remove(path);
}

/* ---------------- 5) 模型瘦身 ---------------- */
static void test_slimming()
{
    std::printf("\n== 模型瘦身 ==\n");
    Model m = make_regressor(3, 2, { 5 }, "leaky_relu", 0.02);
    int pc = m.network.param_count();
    check("param_count = (3*5+5)+(5*2+2) = 32", pc == 32, "实得 " + std::to_string(pc));

    check("初始稀疏度 = 0", m.network.sparsity() == 0.0,
          "实得 " + f2s(m.network.sparsity()));

    double first = m.network.layers[0].weights[0][0];
    m.network.layers[0].weights[0][0] = 9.0;         // 放一个大权重
    int pruned = m.network.prune_magnitude(1e9);     // 阈值极大 -> 全剪
    int totalW = 3 * 5 + 5 * 2;
    check("全量剪枝个数正确", pruned == totalW,
          "剪掉 " + std::to_string(pruned) + " / " + std::to_string(totalW));
    check("剪枝后稀疏度 = 1.0", m.network.sparsity() == 1.0,
          "实得 " + f2s(m.network.sparsity()));
    (void)first;

    Model q = make_regressor(2, 2, { 4 }, "leaky_relu", 0.02);
    double step = q.network.quantize_weights(8);
    check("8 位量化返回正步长", step > 0, "step = " + f2s(step, 8));
    check("非法 bits 返回 -1", q.network.quantize_weights(1) < 0 && q.network.quantize_weights(64) < 0);

    // 量化后所有值应落在 lo + k*step 的格点上
    double lo = 1e300;
    for (auto& L : q.network.layers)
        for (auto& row : L.weights)
            for (double w : row) if (w < lo) lo = w;
    double worst = 0.0;
    for (auto& L : q.network.layers)
        for (auto& row : L.weights)
            for (double w : row) {
                double k = (w - lo) / step;
                worst = std::max(worst, std::fabs(k - std::round(k)));
            }
    check("量化值落在格点上", worst < 1e-9, "最大偏离 " + f2s(worst, 2));

    // 权重合并：一半一半
    Model a = make_regressor(1, 1, { 2 }, "linear", 0.01);
    Model b = make_regressor(1, 1, { 2 }, "linear", 0.01);
    a.network.layers[0].weights[0][0] = 0.0;
    b.network.layers[0].weights[0][0] = 2.0;
    a.network.merge_weights(b.network, 0.5, 0.5);
    check("merge_weights 半步合并 = 1.0",
          std::fabs(a.network.layers[0].weights[0][0] - 1.0) < 1e-12,
          "实得 " + f2s(a.network.layers[0].weights[0][0], 10));

    bool threw = false;
    try {
        Model c = make_regressor(2, 1, { 2 }, "linear", 0.01);
        a.network.merge_weights(c.network, 0.5, 0.5);
    } catch (const std::exception&) {
        threw = true;
    }
    check("结构不一致时抛异常", threw);
}

/* ---------------- 6) 早停 ---------------- */
static void test_early_stop()
{
    std::printf("\n== 早停 ==\n");
    std::vector<Vec> x;
    std::vector<Vec> y;
    for (int i = 0; i < 30; ++i) {
        double a = -1.0 + 2.0 * i / 29.0;
        x.push_back({ a });
        y.push_back({ 2.0 * a + 1.0 });
    }
    Model m = make_regressor(1, 1, { 8 }, "leaky_relu", 0.05);
    m.fit(x, y, 500, 8, 0.1, nullptr, 20, false, true, nullptr, 0.0);
    int ran = (int)m.train_history.size();
    check("patience 触发提前停止", ran < 500, "实际训练 " + std::to_string(ran) + " 轮");
}

int main()
{
    std::printf("===== C++ 引擎（AI.hpp / AI.cpp）回归测试 =====\n");
    test_training();
    test_save_load();
    test_slimming();
    test_early_stop();
    std::printf("\n汇总: 通过 %d，失败 %d  ->  %s\n",
                g_pass, g_fail, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
