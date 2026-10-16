/*
 * cli_main.cpp —— AI_C 的命令行版
 *
 * 与 GUI 版（main.cpp + c_model.cpp）共用同一个纯 C 引擎（c/ 目录），
 * 区别只是没有 Win32 界面，适合脚本化、服务器、CI 里跑。
 *
 * 用法：
 *   ai_cli                      跑全部内置示例（对应 GUI 里的 Examples）
 *   ai_cli demo                同上
 *   ai_cli selftest             训练收敛自检（回归 + 分类 + CNN 前向）
 *   ai_cli train [选项]         用 CSV 训练
 *        --csv <文件>           数据文件，一行一个样本，逗号/空白分隔
 *        --task regression|classification   默认 regression
 *        --hidden 16,8          隐藏层（默认 16,8）
 *        --epochs N             默认 500
 *        --lr X                 默认 0.01
 *        --batch N              默认 16
 *        --val X                验证集比例，默认 0.1
 *        --patience N           默认 50
 *        --header               跳过首行（表头）
 *        --save <文件>          训练后保存模型
 *   分类任务：最后一列是类别索引；回归任务：最后一列起是目标（可多列）
 *   ai_cli predict --model <文件> --input "0.5,0.3,0.1"
 */
#include "c_model.hpp"

#include <chrono>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX   /* 必须放在 windows.h 之前：否则 min/max 宏会破坏 std::min/std::max
                    * （MSVC 报 C2589: '(' illegal token on right side of '::'） */
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ceng;

namespace {

void usage()
{
    std::cout <<
        "AI_C 命令行版（与 GUI 版共用纯 C 引擎）\n\n"
        "  ai_cli                     跑全部内置示例\n"
        "  ai_cli selftest            训练收敛自检\n"
        "  ai_cli train --csv <文件> [--task regression|classification]\n"
        "               [--hidden 16,8] [--epochs 500] [--lr 0.01] [--batch 16]\n"
        "               [--val 0.1] [--patience 50] [--header] [--save <文件>]\n"
        "  ai_cli predict --model <文件> --input \"0.5,0.3\"\n"
        "  ai_cli predict --model <文件> --csv <文件> [--out <文件>] [--header]\n"
        "  ai_cli bench [--samples 1000] [--epochs 5]\n"
        "  ai_cli cross-validate --csv <文件> [--k 5] [--hidden 16,8] [--epochs 400]\n"
        "                        [--lr 0.02] [--batch 16] [--task ...] [--header]\n"
        "  ai_cli grid-search --csv <文件> [--hidden \"8,8;16,16\"] [--lr 0.01,0.03]\n"
        "                     [--epochs 300] [--val 0.1] [--task ...] [--header]\n"
        "  ai_cli data --csv <文件> [--header]                     数据体检（列统计+任务猜测）\n"
        "  ai_cli split --csv <文件> [--ratio 0.2] [--seed N] [--header]\n"
        "               [--out-train a.csv] [--out-test b.csv]  可复现地切分数据集\n"
        "  ai_cli describe --model <文件>                          模型结构与参数量\n"
        "  ai_cli plot --curve <文件> [--svg out.svg] [--csv out.csv]\n"
        "  ai_cli train --csv <文件> --kfold 5 [--save <前缀>] [--final <文件>]\n"
        "  ai_cli prune --model <文件> [--ratio 0.5|--threshold 0.01] [--out <文件>]\n"
        "                [--test <文件>] [--header]                  幅度剪枝 + 精度对账\n"
        "  ai_cli quantize --model <文件> [--bits 8] [--out <文件>] [--test <文件>]\n"
        "  ai_cli compare --model a.txt --model-b b.txt --csv <测试集> [--tol 1e-6]\n"
        "  ai_cli serve --model <文件> [--port 8080] [--token <串>]  HTTP 推理服务\n"
        "  ai_cli distill --teacher <文件> --csv <文件> [--hidden 4] [--epochs 400]\n"
        "                 [--lr 0.02] [--out student.txt] [--test t.csv]  小模型学大模型\n"
        "  ai_cli export-svg --model <文件> [--out model.svg]     模型结构图\n"
        "  ai_cli train ... [--early-stop-metric val|train] [--kfold 5 --aggregate]\n"
        "  ai_cli export-weights --model <文件> --out <文件.json>   （与 Python 版互通）\n"
        "  ai_cli import-weights --json <文件.json> --out <文件>    （转成本地文本格式）\n\n"
        "  CSV：一行一个样本，逗号或空白分隔；分类任务最后一列是类别索引，\n"
        "       回归任务最后一列（起）是目标。\n";
}

std::vector<double> parseList(const std::string& s)
{
    std::vector<double> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        try { v.push_back(std::stod(tok)); } catch (...) { }
    }
    return v;
}

std::vector<int> parseIntList(const std::string& s)
{
    std::vector<int> v;
    for (double d : parseList(s)) v.push_back((int)d);
    return v;
}

bool loadCSV(const std::string& path, bool hasHeader, int nTarget,
             std::vector<Vec>& X, std::vector<Vec>& Y)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "打不开数据文件: " << path << "\n";
        return false;
    }
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (hasHeader && first) { first = false; continue; }
        first = false;
        for (char& ch : line) if (ch == ',' || ch == '\t' || ch == ';') ch = ' ';
        std::stringstream ss(line);
        std::vector<double> row;
        double v;
        while (ss >> v) row.push_back(v);
        if (row.empty()) continue;
        int cut = (int)row.size() - nTarget;
        if (cut < 1) continue;
        X.push_back(Vec(row.begin(), row.begin() + cut));
        Y.push_back(Vec(row.begin() + cut, row.end()));
    }
    return !X.empty();
}

std::string fmt(double v, int prec = 6)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

/* ---------------- 内置示例（与 GUI 的 Examples 对应） ---------------- */

int demoRegression()
{
    std::cout << "\n===== 回归示例：y = x0^2 + x1 =====\n";
    std::vector<Vec> X, Y;
    for (int i = 0; i < 40; ++i) {
        double a = -1.0 + 2.0 * i / 39.0;
        double b = 0.5 * a;
        X.push_back({ a, b });
        Y.push_back({ a * a + b });
    }
    Model m = quick_train(X, Y, "regression", { 16, 16 }, 600, 8);
    double r2 = m.score(X, Y);
    std::cout << "  R² = " << fmt(r2, 6) << (r2 > 0.95 ? "   [OK]" : "   [偏低]") << "\n";
    Vec probe{ 0.6, 0.3 };
    Vec pred = m.predict_one(probe);
    std::cout << "  predict(0.6, 0.3) = " << fmt(pred[0]) << "   期望 "
              << fmt(0.36 + 0.3) << "\n";
    return r2 > 0.95 ? 0 : 1;
}

int demoClassification()
{
    std::cout << "\n===== 分类示例：a > 0 ? =====\n";
    std::vector<Vec> X;
    std::vector<int> Y;
    for (int i = 0; i < 60; ++i) {
        double a = -1.0 + 2.0 * i / 59.0;
        X.push_back({ a, a * a });
        Y.push_back(a > 0 ? 1 : 0);
    }
    Model m = quick_train(X, Y, { 16, 16 }, 400, 10);
    double acc = m.score(X, Y);
    std::cout << "  准确率 = " << fmt(acc * 100, 2) << "%" << (acc > 0.95 ? "   [OK]" : "   [偏低]") << "\n";
    return acc > 0.95 ? 0 : 1;
}

int demoGradientClip()
{
    std::cout << "\n===== 梯度裁剪 + StepLR =====\n";
    std::vector<Vec> X, Y;
    for (int i = 0; i < 20; ++i) {
        double a = 0.2 + 1.1 * i / 19.0;
        X.push_back({ a, a * 1.5 });
        Y.push_back({ a * a, a * a * 1.5 });
    }
    Model m = make_regressor(2, 2, { 8, 8, 4 }, "leaky_relu", 0.01);
    StepLR sch(m.optimizer.get(), 100, 0.5);
    m.fit(X, Y, 300, 2, 0.0, nullptr, 0, true, true, &sch, 1.0);
    double r2 = m.score(X, Y);
    std::cout << "  300 轮后 LR = " << fmt(m.optimizer->lr, 6)
              << "   R² = " << fmt(r2, 6) << "\n";
    return r2 > 0.9 ? 0 : 1;
}

int demoCNN()
{
    std::cout << "\n===== CNN 前向：Conv2d -> MaxPool2d -> Flatten -> MLP =====\n";
    Conv2d conv(1, 2, 2, 1);
    MaxPool2d pool(2);
    Flatten flat;
    MLP fc({ 2, 4, 2 }, { "leaky_relu", "softmax" });

    Vec3D img(1, Vec2D(4, Vec(4, 0.0)));
    seed(12345);
    for (int h = 0; h < 4; ++h)
        for (int w = 0; w < 4; ++w)
            img[0][h][w] = randu();

    Vec3D c = conv.forward(img);
    Vec3D p = pool.forward(c);
    Vec f = flat.forward(p);
    Vec out = fc.forward(f);
    std::cout << "  (1,4,4) -> Conv " << c.size() << "x" << c[0].size() << "x" << c[0][0].size()
              << " -> Pool " << p.size() << "x" << p[0].size() << "x" << p[0][0].size()
              << " -> Flatten " << f.size() << " -> MLP " << out.size() << "\n";
    std::cout << "  MLP 输出 = (" << fmt(out[0]) << ", " << fmt(out[1]) << ")\n";
    return 0;
}

/* ---------------- 训练收敛自检 ---------------- */

int selftest()
{
    std::cout << "===== AI_C 命令行版自检 =====\n";
    int fail = 0;

    // 1) 回归：拟合已知函数，检查 R²
    {
        std::vector<Vec> X, Y;
        for (int i = 0; i < 30; ++i) {
            double a = -1.0 + 2.0 * i / 29.0;
            X.push_back({ a, 0.5 * a });
            Y.push_back({ a * a + 0.5 * a });
        }
        Model m = quick_train(X, Y, "regression", { 24, 24 }, 500, 8);
        double r2 = m.score(X, Y);
        bool ok = r2 > 0.95;
        std::cout << "  [1] 回归 y=x0^2+0.5*x1  R² = " << fmt(r2, 6)
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n";
        if (!ok) fail++;
    }

    // 2) 分类：线性可分数据，检查准确率
    {
        std::vector<Vec> X; std::vector<int> Y;
        for (int i = 0; i < 40; ++i) {
            double a = -1.0 + 2.0 * i / 39.0;
            X.push_back({ a, a * a });
            Y.push_back(a > 0 ? 1 : 0);
        }
        Model m = quick_train(X, Y, { 16 }, 300, 8);
        double acc = m.score(X, Y);
        bool ok = acc > 0.9;
        std::cout << "  [2] 分类 a>0        准确率 = " << fmt(acc * 100, 2) << "%"
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n";
        if (!ok) fail++;
    }

    // 3) 存档往返：存盘再读回，预测值必须一致
    {
        std::vector<Vec> X, Y;
        for (int i = 0; i < 12; ++i) {
            double a = -1.0 + 2.0 * i / 11.0;
            X.push_back({ a });
            Y.push_back({ 2.0 * a + 1.0 });
        }
        Model m = quick_train(X, Y, "regression", { 8 }, 300, 4);
        Vec before = m.predict_one(Vec{ 0.25 });
        m.save("cli_selftest_model.txt", 300, 4);
        Model m2 = Model::load("cli_selftest_model.txt", "mse", "adam", 0.01);
        Vec after = m2.predict_one(Vec{ 0.25 });
        bool ok = !before.empty() && !after.empty() && std::fabs(before[0] - after[0]) < 1e-9;
        std::cout << "  [3] 存档往返        " << fmt(before.empty() ? 0 : before[0], 10)
                  << " vs " << fmt(after.empty() ? 0 : after[0], 10)
                  << (ok ? "  [OK]" : "  [FAIL]") << "\n";
        if (!ok) fail++;
        std::remove("cli_selftest_model.txt");
    }

    std::cout << (fail == 0 ? "全部通过\n" : "存在失败项\n");
    return fail;
}

/* ---------------- 性能基准（与 c/bench.c 同负载，便于横向对比） ---------------- */

int bench(int samples, int epochs)
{
    const int in_dim = 64, out_dim = 8;
    std::cout << "===== 性能基准 =====\n";
    std::cout << "  MLP[64,256,128,8]，" << samples << " 样本 × " << epochs
              << " 轮，batch=32，Adam(0.001)\n";

    std::vector<Vec> X(samples, Vec(in_dim)), Y(samples, Vec(out_dim));
    for (int i = 0; i < samples; ++i) {
        for (int k = 0; k < in_dim; ++k)
            X[i][k] = (double)((i * 7 + k * 13) % 100) / 50.0 - 1.0;
        for (int k = 0; k < out_dim; ++k) {
            double sacc = 0.0;
            for (int t = 0; t < 8; ++t) sacc += X[i][(k * 8 + t) % in_dim];
            Y[i][k] = sacc / 8.0;
        }
    }

    Model m = make_regressor(in_dim, out_dim, { 256, 128 }, "leaky_relu", 0.001);
    int params = 0;
    for (auto& L : m.network.layers) params += L.fan_in * L.fan_out + L.fan_out;

    auto t0 = std::chrono::steady_clock::now();
    m.fit(X, Y, epochs, 32, 0.0, nullptr, 0, false, true, nullptr, 0.0);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double r2 = m.score(X, Y);
    std::cout << "  参数 " << params << "\n";
    std::cout << "  耗时 " << fmt(sec, 3) << " s（" << fmt(sec * 1000.0 / epochs, 1)
              << " ms/轮，" << fmt(sec * 1e6 / ((double)epochs * samples), 1) << " us/样本轮）\n";
    std::cout << "  训练集 R² = " << fmt(r2, 6) << "\n";
    std::cout << "  对照：同样负载 Python 版约 77 s（约 58–68 倍）\n";
    return 0;
}

/* ---------------- 批量预测：读 CSV -> 预测 -> 指标 + 可选写出 ---------------- */

int batchPredict(const std::string& modelPath, const std::string& csvPath,
                 const std::string& outPath, bool header)
{
    Model m = Model::load(modelPath, "", "", 0.0);
    std::vector<Vec> X, Y;
    if (!loadCSV(csvPath, header, 1, X, Y)) {
        std::cout << "数据为空或格式不对: " << csvPath << "\n";
        return 2;
    }
    bool cls = (m.task == "classification");
    std::cout << "模型任务: " << m.task << "，样本 " << X.size()
              << "，输入维度 " << X[0].size() << "\n";

    std::ofstream out;
    bool writeOut = !outPath.empty();
    if (writeOut) {
        out.open(outPath);
        if (!out.is_open()) { std::cout << "写不出文件: " << outPath << "\n"; return 2; }
        out << "pred";
        for (int j = 1; j < (int)m.network.layers.back().fan_out; ++j) out << ",pred" << j;
        if (cls) out << ",pred_class";
        out << ",target\n";
    }

    std::vector<int> yi;
    double firstErr = -1.0;
    for (size_t i = 0; i < X.size(); ++i) {
        Vec p = m.predict_one(X[i]);
        if (writeOut) {
            out << std::setprecision(10);
            for (size_t j = 0; j < p.size(); ++j) out << (j ? "," : "") << p[j];
            if (cls) out << "," << m.predict_class(X[i]);
            out << "," << (Y[i].empty() ? 0.0 : Y[i][0]) << "\n";
        }
        if (cls) yi.push_back((int)std::lround(Y[i].empty() ? 0.0 : Y[i][0]));
        if (i == 0 && !p.empty() && !Y[i].empty()) firstErr = std::fabs(p[0] - Y[i][0]);
    }
    if (writeOut) { out.close(); std::cout << "已写出: " << outPath << "\n"; }

    if (cls) {
        double acc = m.score(X, yi);
        std::cout << "准确率 = " << fmt(acc * 100, 2) << "%\n";
    } else {
        double r2 = m.score(X, Y);
        std::cout << "R² = " << fmt(r2, 6) << "（多输出按各维平均）\n";
    }
    if (firstErr >= 0) std::cout << "首个样本 |pred-target| = " << fmt(firstErr, 6) << "\n";
    return 0;
}

/* ---------------- K 折交叉验证 ---------------- */

int crossValidate(const std::string& csv, int k, std::vector<int> hidden,
                  int epochs, double lr, int batch, bool header, bool cls)
{
    std::vector<Vec> X, Y;
    int nTarget = 1;
    if (!loadCSV(csv, header, nTarget, X, Y)) {
        std::cout << "数据为空或格式不对: " << csv << "\n";
        return 2;
    }
    const int n = (int)X.size();
    if (k < 2) k = 2;
    if (k > n) k = n;
    std::cout << "===== " << k << " 折交叉验证（" << n << " 样本，hidden ";
    for (size_t i = 0; i < hidden.size(); ++i) std::cout << (i ? "-" : "") << hidden[i];
    std::cout << "，epochs " << epochs << "）=====\n";

    std::vector<double> scores;
    for (int fold = 0; fold < k; ++fold) {
        std::vector<Vec> TX, VX, TY, VY;
        std::vector<int> tyi, vyi;
        for (int i = 0; i < n; ++i) {
            if (i % k == fold) { VX.push_back(X[i]); VY.push_back(Y[i]); }
            else               { TX.push_back(X[i]); TY.push_back(Y[i]); }
        }
        double sc = 0.0;
        if (cls) {
            for (auto& r : TY) tyi.push_back((int)std::lround(r.empty() ? 0.0 : r[0]));
            for (auto& r : VY) vyi.push_back((int)std::lround(r.empty() ? 0.0 : r[0]));
            int classes = 1;
            for (int v : tyi) if (v + 1 > classes) classes = v + 1;
            Model m = make_classifier((int)X[0].size(), classes, hidden, "leaky_relu", lr);
            m.fit(TX, tyi, epochs, batch, 0.0, 0, false, nullptr, 0.0);
            sc = m.score(VX, vyi);
            std::cout << "  fold " << fold + 1 << ": 准确率 = " << fmt(sc * 100, 2) << "%"
                      << "（验证 " << VX.size() << "）\n";
        } else {
            Model m = make_regressor((int)X[0].size(), (int)Y[0].size(), hidden, "leaky_relu", lr);
            m.fit(TX, TY, epochs, batch, 0.0, nullptr, 0, false, true, nullptr, 0.0);
            sc = m.score(VX, VY);
            std::cout << "  fold " << fold + 1 << ": R² = " << fmt(sc, 6)
                      << "（验证 " << VX.size() << "）\n";
        }
        scores.push_back(sc);
    }
    double mean = 0.0;
    for (double v : scores) mean += v;
    mean /= scores.size();
    double var = 0.0;
    for (double v : scores) var += (v - mean) * (v - mean);
    var = scores.size() > 1 ? var / (scores.size() - 1) : 0.0;
    std::cout << "  平均 = " << fmt(cls ? mean * 100 : mean, 6)
              << (cls ? "%" : "") << "   标准差 = " << fmt(std::sqrt(var), 6) << "\n";
    return 0;
}

/* ---------------- 超参网格搜索 ---------------- */

int gridSearch(const std::string& csv, const std::vector<std::vector<int>>& hiddenSets,
               const std::vector<double>& lrs, int epochs, int batch, double val,
               bool header, bool cls)
{
    std::vector<Vec> X, Y;
    if (!loadCSV(csv, header, 1, X, Y)) {
        std::cout << "数据为空或格式不对: " << csv << "\n";
        return 2;
    }
    std::cout << "===== 网格搜索：hidden " << hiddenSets.size() << " 种 × lr "
              << lrs.size() << " 种 = " << hiddenSets.size() * lrs.size() << " 组 =====\n";

    double best = -1e18;
    std::string bestDesc;
    for (auto& h : hiddenSets) {
        for (double lr : lrs) {
            double sc = 0.0;
            if (cls) {
                std::vector<int> yi;
                int classes = 1;
                for (auto& r : Y) { int v = (int)std::lround(r.empty() ? 0.0 : r[0]); yi.push_back(v); if (v + 1 > classes) classes = v + 1; }
                Model m = make_classifier((int)X[0].size(), classes, h, "leaky_relu", lr);
                m.fit(X, yi, epochs, batch, val, 50, false, nullptr, 0.0);
                sc = m.score(X, yi);
            } else {
                Model m = make_regressor((int)X[0].size(), (int)Y[0].size(), h, "leaky_relu", lr);
                m.fit(X, Y, epochs, batch, val, nullptr, 50, false, true, nullptr, 0.0);
                sc = m.score(X, Y);
            }
            std::string desc = "hidden=";
            for (size_t i = 0; i < h.size(); ++i) desc += (i ? "-" : "") + std::to_string(h[i]);
            desc += "  lr=" + fmt(lr, 4);
            std::cout << "  " << desc << "  ->  " << (cls ? "准确率 " : "R² ")
                      << fmt(cls ? sc * 100 : sc, 4) << (cls ? "%" : "") << "\n";
            if (sc > best) { best = sc; bestDesc = desc; }
        }
    }
    std::cout << "  最佳: " << bestDesc << "  ->  " << (cls ? "准确率 " : "R² ")
              << fmt(cls ? best * 100 : best, 4) << (cls ? "%" : "") << "\n";
    return 0;
}

/* ---------------- 数据体检：行列统计 + 任务类型猜测 ---------------- */

int dataInfo(const std::string& csv, bool header)
{
    std::ifstream f(csv);
    if (!f.is_open()) { std::cout << "打不开数据文件: " << csv << "\n"; return 2; }
    std::vector<std::vector<double>> rows;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (header && first) { first = false; continue; }
        first = false;
        for (char& ch : line) if (ch == ',' || ch == '\t' || ch == ';') ch = ' ';
        std::stringstream ss(line);
        std::vector<double> row; double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(row);
    }
    if (rows.empty()) { std::cout << "没有解析到数据\n"; return 2; }

    const int n = (int)rows.size();
    int cols = 0;
    for (auto& r : rows) cols = std::max(cols, (int)r.size());
    std::cout << "===== 数据体检 =====\n";
    std::cout << "  文件: " << csv << "\n";
    std::cout << "  样本 " << n << " 行，列数 " << cols << "\n\n";
    std::cout << "  列      最小        最大        均值        标准差\n";
    for (int c = 0; c < cols; ++c) {
        double mn = 1e300, mx = -1e300, sum = 0.0;
        int cnt = 0;
        for (auto& r : rows) if (c < (int)r.size()) { double v = r[c]; mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ++cnt; }
        if (cnt == 0) continue;
        double mean = sum / cnt, var = 0.0;
        for (auto& r : rows) if (c < (int)r.size()) { double d = r[c] - mean; var += d * d; }
        var = cnt > 1 ? var / (cnt - 1) : 0.0;
        char buf[128];
        snprintf(buf, sizeof(buf), "  %-6d %-11.6g %-11.6g %-11.6g %-11.6g", c, mn, mx, mean, std::sqrt(var));
        std::cout << buf << "\n";
    }

    // 猜任务：最后一列若全为非负整数且取值很少，多半是分类标签
    bool allInt = true, allNonNeg = true;
    std::vector<double> uniq;
    for (auto& r : rows) {
        if (r.empty()) continue;
        double v = r.back();
        if (std::fabs(v - std::round(v)) > 1e-9) allInt = false;
        if (v < 0) allNonNeg = false;
        bool seen = false;
        for (double u : uniq) if (std::fabs(u - v) < 1e-9) { seen = true; break; }
        if (!seen) uniq.push_back(v);
    }
    std::cout << "\n  末列去重值个数 = " << uniq.size();
    if (allInt && allNonNeg && uniq.size() <= 20 && uniq.size() >= 2)
        std::cout << "  ->  看起来是分类标签（建议 --task classification）\n";
    else
        std::cout << "  ->  看起来是回归目标（建议 --task regression）\n";
    return 0;
}

/* ---------------- 数据集切分（可复现） ---------------- */

int splitData(const std::string& csv, const std::string& trainOut, const std::string& testOut,
              double testRatio, int seedVal, bool header)
{
    std::ifstream f(csv);
    if (!f.is_open()) { std::cout << "打不开数据文件: " << csv << "\n"; return 2; }
    std::vector<std::string> lines;
    std::string line, head;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (header && first) { head = line; first = false; continue; }
        first = false;
        lines.push_back(line);
    }
    if (lines.empty()) { std::cout << "没有数据\n"; return 2; }

    std::vector<int> idx(lines.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::mt19937 rng((unsigned)(seedVal != 0 ? seedVal : 42));
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t nTest = (size_t)(lines.size() * testRatio);
    if (nTest < 1 && lines.size() > 1) nTest = 1;
    if (nTest >= lines.size()) nTest = lines.size() - 1;

    std::ofstream ft(trainOut), fv(testOut);
    if (!ft.is_open() || !fv.is_open()) { std::cout << "写不出文件\n"; return 2; }
    if (!head.empty()) { ft << head << "\n"; fv << head << "\n"; }
    for (size_t k = 0; k < idx.size(); ++k) {
        if (k < nTest) fv << lines[idx[k]] << "\n";
        else           ft << lines[idx[k]] << "\n";
    }
    ft.close(); fv.close();
    std::cout << "已切分: 训练 " << lines.size() - nTest << " -> " << trainOut
              << "，测试 " << nTest << " -> " << testOut
              << "（种子 " << (seedVal != 0 ? seedVal : 42) << "）\n";
    return 0;
}

/* ---------------- 模型信息 ---------------- */

int describeModel(const std::string& modelPath)
{
    Model m = Model::load(modelPath, "", "", 0.0);
    if (m.network.layers.empty()) { std::cout << "加载失败: " << modelPath << "\n"; return 1; }
    int total = 0;
    std::cout << "===== 模型信息 =====\n";
    std::cout << "  文件: " << modelPath << "\n";
    std::cout << "  任务: " << m.task << "   损失: " << m.loss_name
              << "   优化器: " << m.opt_name << "   lr: " << fmt(m.lr, 6) << "\n";
    std::cout << "  结构:\n";
    for (size_t i = 0; i < m.network.layers.size(); ++i) {
        auto& L = m.network.layers[i];
        int p = L.fan_in * L.fan_out + L.fan_out;
        total += p;
        std::cout << "    [" << i << "] Linear(" << L.fan_in << " -> " << L.fan_out
                  << ") + " << L.act_name << "   [" << p << " 参数]\n";
    }
    std::cout << "  参数总量: " << total << "\n";
    return 0;
}

/* ---------------- 训练曲线导出：CSV + 零依赖 SVG ---------------- */

int plotCurve(const std::string& curvePath, const std::string& svgOut, const std::string& csvOut)
{
    std::ifstream f(curvePath);
    if (!f.is_open()) {
        std::cout << "打不开曲线文件: " << curvePath << "（用 train --curve 生成）\n";
        return 2;
    }
    std::vector<std::pair<int, double>> tr, va;
    std::string line;
    std::getline(f, line);                       // 表头 epoch,train[,val]
    while (std::getline(f, line)) {
        for (char& ch : line) if (ch == ',') ch = ' ';
        std::stringstream ss(line);
        int ep; double a = 0, b = 0;
        if (!(ss >> ep >> a)) continue;
        tr.push_back({ ep, a });
        if (ss >> b) va.push_back({ ep, b });
    }
    if (tr.empty()) { std::cout << "曲线为空\n"; return 2; }

    if (!csvOut.empty() && csvOut != curvePath) {
        std::ofstream o(csvOut);
        o << "epoch,train,val\n";
        for (size_t i = 0; i < tr.size(); ++i) {
            o << tr[i].first << "," << tr[i].second;
            if (i < va.size()) o << "," << va[i].second;
            o << "\n";
        }
        std::cout << "曲线 CSV 已写出: " << csvOut << "\n";
    }
    if (svgOut.empty()) return 0;

    const int W = 800, H = 400, ML = 70, MR = 20, MT = 30, MB = 45;
    double mn = 1e300, mx = -1e300;
    for (auto& p : tr) { mn = std::min(mn, p.second); mx = std::max(mx, p.second); }
    for (auto& p : va) { mn = std::min(mn, p.second); mx = std::max(mx, p.second); }
    if (mx - mn < 1e-12) { mx += 1; mn -= 1; }
    double pad = (mx - mn) * 0.08; mn -= pad; mx += pad;
    int maxEp = tr.back().first;
    if (maxEp < 1) maxEp = 1;

    auto X = [&](int ep) { return ML + (double)(ep - 1) / std::max(1, maxEp - 1) * (W - ML - MR); };
    auto Y = [&](double v) { return MT + (mx - v) / (mx - mn) * (H - MT - MB); };

    std::ofstream o(svgOut);
    if (!o.is_open()) { std::cout << "写不出: " << svgOut << "\n"; return 2; }
    o << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
      << "\" viewBox=\"0 0 " << W << " " << H << "\">\n";
    o << "  <rect width=\"" << W << "\" height=\"" << H << "\" fill=\"#0d1117\"/>\n";
    // 坐标轴 + 网格
    for (int t = 0; t <= 4; ++t) {
        double v = mn + (mx - mn) * t / 4.0;
        double y = Y(v);
        o << "  <line x1=\"" << ML << "\" y1=\"" << y << "\" x2=\"" << (W - MR)
          << "\" y2=\"" << y << "\" stroke=\"#30363d\" stroke-width=\"1\"/>\n";
        o << "  <text x=\"" << (ML - 8) << "\" y=\"" << (y + 4)
          << "\" fill=\"#8b949e\" font-size=\"11\" text-anchor=\"end\">" << fmt(v, 5) << "</text>\n";
    }
    o << "  <line x1=\"" << ML << "\" y1=\"" << MT << "\" x2=\"" << ML << "\" y2=\"" << (H - MB)
      << "\" stroke=\"#8b949e\"/>\n";
    o << "  <line x1=\"" << ML << "\" y1=\"" << (H - MB) << "\" x2=\"" << (W - MR)
      << "\" y2=\"" << (H - MB) << "\" stroke=\"#8b949e\"/>\n";
    o << "  <text x=\"" << (W / 2) << "\" y=\"" << (H - 12)
      << "\" fill=\"#8b949e\" font-size=\"12\" text-anchor=\"middle\">epoch</text>\n";
    o << "  <text x=\"" << ML << "\" y=\"18\" fill=\"#c9d1d9\" font-size=\"13\">Loss curve</text>\n";
    auto poly = [&](std::vector<std::pair<int, double>>& pts, const char* color) {
        o << "  <polyline fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2\" points=\"";
        for (auto& p : pts) o << fmt(X(p.first), 1) << "," << fmt(Y(p.second), 1) << " ";
        o << "\"/>\n";
    };
    if (!tr.empty()) poly(tr, "#58a6ff");
    if (!va.empty()) poly(va, "#f0883e");
    if (!va.empty())
        o << "  <text x=\"" << (W - MR - 120) << "\" y=\"18\" fill=\"#58a6ff\" font-size=\"12\">train</text>\n"
          << "  <text x=\"" << (W - MR - 60) << "\" y=\"18\" fill=\"#f0883e\" font-size=\"12\">val</text>\n";
    o << "</svg>\n";
    o.close();
    std::cout << "曲线 SVG 已写出: " << svgOut << "（" << tr.size() << " 个点，"
              << (va.empty() ? "仅 train" : "train + val") << "）\n";
    return 0;
}

/* ---------------- 评估辅助：按模型任务算指标 ---------------- */

double evalOn(Model& m, const std::vector<Vec>& X, const std::vector<Vec>& Y, bool cls)
{
    if (cls) {
        std::vector<int> yi;
        for (auto& r : Y) yi.push_back((int)std::lround(r.empty() ? 0.0 : r[0]));
        return m.score(X, yi);
    }
    return m.score(X, Y);
}

/* ---------------- K 折训练：每折存模型，最后可选全量重训 ---------------- */

int trainKfold(const std::string& csv, int k, std::vector<int> hidden, int epochs,
               double lr, int batch, double val, bool header, bool cls,
               const std::string& savePrefix, const std::string& finalSave,
               bool aggregate = false, bool aggregateWeights = false)
{
    std::vector<Vec> X, Y;
    if (!loadCSV(csv, header, 1, X, Y)) {
        std::cout << "数据为空或格式不对: " << csv << "\n";
        return 2;
    }
    const int n = (int)X.size();
    if (k < 2) k = 2;
    if (k > n) k = n;
    std::cout << "===== K 折训练（k=" << k << "，" << n << " 样本）=====\n";

    std::vector<double> scores;
    std::vector<Model> foldModels;
    for (int fold = 0; fold < k; ++fold) {
        std::vector<Vec> TX, VX, TY, VY;
        for (int i = 0; i < n; ++i) {
            if (i % k == fold) { VX.push_back(X[i]); VY.push_back(Y[i]); }
            else               { TX.push_back(X[i]); TY.push_back(Y[i]); }
        }
        if (aggregateWeights) Model::reset_init_seed();   // 各折同一初值，权重平均才有意义
        Model m = cls ? make_classifier((int)X[0].size(), 2, hidden, "leaky_relu", lr)
                      : make_regressor((int)X[0].size(), (int)Y[0].size(), hidden, "leaky_relu", lr);
        if (cls) {
            std::vector<int> tyi;
            int classes = 1;
            for (auto& r : TY) { int v = (int)std::lround(r.empty() ? 0.0 : r[0]); tyi.push_back(v); if (v + 1 > classes) classes = v + 1; }
            m = make_classifier((int)X[0].size(), classes, hidden, "leaky_relu", lr);
            m.fit(TX, tyi, epochs, batch, val, 50, false, nullptr, 0.0);
        } else {
            m.fit(TX, TY, epochs, batch, val, nullptr, 50, false, true, nullptr, 0.0);
        }
        double sc = evalOn(m, VX, VY, cls);
        scores.push_back(sc);

        std::cout << "  fold " << fold + 1 << "/" << k << ": "
                  << (cls ? "准确率 " : "R² ") << fmt(cls ? sc * 100 : sc, 5) << (cls ? "%" : "")
                  << "   （训练 " << TX.size() << " / 验证 " << VX.size() << "）";
        if (!savePrefix.empty()) {
            std::string path = savePrefix + "_fold" + std::to_string(fold + 1) + ".txt";
            m.save(path, epochs, batch);          // 必须先存：下面 move 走之后 m 就空了
            std::cout << "  -> " << path;
        }
        if (aggregateWeights) foldModels.push_back(std::move(m));
        std::cout << "\n";
    }
    double mean = 0.0;
    for (double v : scores) mean += v;
    mean /= scores.size();
    double var = 0.0;
    for (double v : scores) var += (v - mean) * (v - mean);
    var = scores.size() > 1 ? var / (scores.size() - 1) : 0.0;
    std::cout << "  交叉验证平均 = " << fmt(cls ? mean * 100 : mean, 5) << (cls ? "%" : "")
              << "   标准差 = " << fmt(std::sqrt(var), 5) << "\n";

    // K 折权重平均：把各折权重线性平均成一个模型（单模型部署，无需推理时集成）
    if (aggregateWeights && foldModels.size() >= 2) {
        Model avg = make_regressor(1, 1, { 1 }, "linear", lr);
        avg = std::move(foldModels[0]);
        int okCnt = 1;
        for (size_t i = 1; i < foldModels.size(); ++i) {
            if (avg.merge_weights(foldModels[i], 1.0, 1.0)) okCnt++;
        }
        // 合并了 okCnt 份，除以 okCnt 得到平均
        if (okCnt > 1) {
            Model unit = make_regressor(1, 1, { 1 }, "linear", lr);
            (void)unit;
            avg.merge_weights(avg, 1.0 / okCnt, 0.0);   // 自乘缩放：avg *= 1/okCnt
        }
        std::cout << "  ---- 权重平均（" << okCnt << " 份，各折同初值）----\n";
        double scAvg = evalOn(avg, X, Y, cls);
        std::cout << "  单折平均 = " << fmt(cls ? mean * 100 : mean, 5) << (cls ? "%" : "")
                  << "   权重平均模型 = " << fmt(cls ? scAvg * 100 : scAvg, 5) << (cls ? "%" : "")
                  << "   提升 " << fmt((scAvg - mean) * (cls ? 100 : 1), 5) << "\n";
        std::string outAvg = (savePrefix.empty() ? std::string("kfold_avg") : savePrefix + "_avg") + ".txt";
        avg.save(outAvg, epochs, batch);
    }

    // K 折集成：把各折模型在同一批数据上的预测平均起来
    if (aggregate) {
        if (savePrefix.empty()) {
            std::cout << "  集成需要 --save <前缀> 才能找到各折模型\n";
        } else {
            std::vector<Model> folds;
            for (int i = 1; i <= k; ++i) {
                std::string path = savePrefix + "_fold" + std::to_string(i) + ".txt";
                Model fm = Model::load(path, "", "", 0.0);
                if (!fm.network.layers.empty()) folds.push_back(std::move(fm));
            }
            if (folds.size() >= 2) {
                std::cout << "  ---- K 折集成（" << folds.size() << " 个模型）----\n";
                std::vector<double> ens;
                for (int i = 0; i < n; ++i) {
                    Vec acc;
                    for (auto& fm : folds) {
                        Vec p = fm.predict_one(X[i]);
                        if (acc.empty()) acc.assign(p.size(), 0.0);
                        for (size_t j = 0; j < p.size() && j < acc.size(); ++j) acc[j] += p[j];
                    }
                    for (double& v : acc) v /= (double)folds.size();
                    if (cls) {
                        int best = 0;
                        for (size_t j = 1; j < acc.size(); ++j) if (acc[j] > acc[best]) best = (int)j;
                        ens.push_back((double)best);
                    } else {
                        ens.push_back(acc.empty() ? 0.0 : acc[0]);
                    }
                }
                // 集成指标
                double sc = 0.0;
                if (cls) {
                    int ok = 0;
                    for (int i = 0; i < n; ++i) if ((int)std::lround(Y[i].empty() ? 0.0 : Y[i][0]) == (int)ens[i]) ok++;
                    sc = (double)ok / n;
                } else {
                    double mean = 0.0;
                    for (auto& r : Y) mean += r.empty() ? 0.0 : r[0];
                    mean /= n;
                    double res = 0.0, tot = 0.0;
                    for (int i = 0; i < n; ++i) {
                        double yv = Y[i].empty() ? 0.0 : Y[i][0];
                        res += (yv - ens[i]) * (yv - ens[i]);
                        tot += (yv - mean) * (yv - mean);
                    }
                    sc = 1.0 - res / std::max(tot, 1e-15);
                }
                std::cout << "  单折平均 = " << fmt(cls ? mean * 100 : mean, 5) << (cls ? "%" : "")
                          << "   集成(全量数据) = " << fmt(cls ? sc * 100 : sc, 5) << (cls ? "%" : "")
                          << "   提升 " << fmt((sc - mean) * (cls ? 100 : 1), 5) << "\n";
            }
        }
    }

    if (!finalSave.empty()) {
        // 常规做法：用全部数据重训一个最终模型
        Model fm = cls ? make_classifier((int)X[0].size(), 2, hidden, "leaky_relu", lr)
                       : make_regressor((int)X[0].size(), (int)Y[0].size(), hidden, "leaky_relu", lr);
        if (cls) {
            std::vector<int> yi;
            int classes = 1;
            for (auto& r : Y) { int v = (int)std::lround(r.empty() ? 0.0 : r[0]); yi.push_back(v); if (v + 1 > classes) classes = v + 1; }
            fm = make_classifier((int)X[0].size(), classes, hidden, "leaky_relu", lr);
            fm.fit(X, yi, epochs, batch, val, 50, true, nullptr, 0.0);
            std::cout << "  全量重训: 训练准确率 = " << fmt(fm.score(X, yi) * 100, 2) << "%\n";
        } else {
            fm.fit(X, Y, epochs, batch, val, nullptr, 50, true, true, nullptr, 0.0);
            std::cout << "  全量重训: 训练集 R² = " << fmt(fm.score(X, Y), 6) << "\n";
        }
        fm.save(finalSave, epochs, batch);
    }
    return 0;
}

/* ---------------- 剪枝 ---------------- */

int pruneModel(const std::string& modelPath, const std::string& outPath,
               double threshold, double ratio, const std::string& testPath, bool header)
{
    Model m = Model::load(modelPath, "", "", 0.0);
    if (m.network.layers.empty()) { std::cout << "加载失败\n"; return 1; }
    bool cls = (m.task == "classification");
    int total = m.weight_count();

    std::vector<Vec> TX, TY;
    bool hasTest = !testPath.empty() && loadCSV(testPath, header, 1, TX, TY);
    double before = hasTest ? evalOn(m, TX, TY, cls) : 0.0;

    if (ratio > 0.0) {
        // 按比例剪：取 |w| 的分位数当阈值
        std::vector<double> absW;
        absW.reserve(total);
        for (auto& L : m.network.layers)
            for (auto& row : L.weights)
                for (double w : row) absW.push_back(std::fabs(w));
        if (absW.empty()) { std::cout << "没有权重\n"; return 1; }
        std::sort(absW.begin(), absW.end());
        size_t k = (size_t)(absW.size() * std::min(1.0, std::max(0.0, ratio)));
        if (k >= absW.size()) k = absW.size() - 1;
        threshold = absW[k];
        std::cout << "按比例 " << fmt(ratio * 100, 1) << "% 剪枝 -> 阈值 " << fmt(threshold, 6) << "\n";
    }

    int pruned = m.prune_magnitude(threshold);
    std::cout << "===== 幅度剪枝 =====\n";
    std::cout << "  参数总量 " << total << "，剪掉 " << pruned
              << "，稀疏度 " << fmt(100.0 * pruned / std::max(1, total), 2) << "%\n";
    if (hasTest) {
        double after = evalOn(m, TX, TY, cls);
        std::cout << "  " << (cls ? "准确率" : "R²") << "（测试集）: 剪枝前 "
                  << fmt(cls ? before * 100 : before, 5) << "  ->  剪枝后 "
                  << fmt(cls ? after * 100 : after, 5) << (cls ? "%" : "") << "\n";
    }
    if (!outPath.empty()) m.save(outPath, -1, -1);
    return 0;
}

/* ---------------- 量化 ---------------- */

int quantizeModel(const std::string& modelPath, const std::string& outPath, int bits,
                  const std::string& testPath, bool header)
{
    Model m = Model::load(modelPath, "", "", 0.0);
    if (m.network.layers.empty()) { std::cout << "加载失败\n"; return 1; }
    bool cls = (m.task == "classification");

    std::vector<Vec> TX, TY;
    bool hasTest = !testPath.empty() && loadCSV(testPath, header, 1, TX, TY);
    double before = hasTest ? evalOn(m, TX, TY, cls) : 0.0;

    double step = m.quantize_weights(bits);
    std::cout << "===== 权重量化 =====\n";
    if (step <= 0) { std::cout << "  量化失败（bits 需在 2..24）\n"; return 1; }
    int levels = (1 << bits) - 1;
    std::cout << "  " << bits << " 位定点：档位 " << levels << "，步长 " << fmt(step, 8) << "\n";
    std::cout << "  理论存储：float64 -> " << bits << " bit，压缩到 "
              << fmt(100.0 * bits / 64.0, 1) << "%（"
              << fmt((double)m.weight_count() * 64 / 8 / 1024, 1) << " KB -> "
              << fmt((double)m.weight_count() * bits / 8 / 1024, 2) << " KB）\n";
    if (hasTest) {
        double after = evalOn(m, TX, TY, cls);
        std::cout << "  " << (cls ? "准确率" : "R²") << "（测试集）: 量化前 "
                  << fmt(cls ? before * 100 : before, 5) << "  ->  量化后 "
                  << fmt(cls ? after * 100 : after, 5) << (cls ? "%" : "")
                  << "   变化 " << fmt((after - before) * (cls ? 100 : 1), 5) << "\n";
    }
    if (!outPath.empty()) m.save(outPath, -1, -1);
    return 0;
}

/* ---------------- 两个模型对账 ---------------- */

int compareModels(const std::string& pathA, const std::string& pathB,
                  const std::string& testPath, bool header, double tol)
{
    Model a = Model::load(pathA, "", "", 0.0);
    Model b = Model::load(pathB, "", "", 0.0);
    if (a.network.layers.empty() || b.network.layers.empty()) { std::cout << "模型加载失败\n"; return 1; }
    std::vector<Vec> X, Y;
    if (!loadCSV(testPath, header, 1, X, Y)) { std::cout << "测试集读取失败\n"; return 2; }

    bool cls = (a.task == "classification");
    double sa = evalOn(a, X, Y, cls), sb = evalOn(b, X, Y, cls);
    std::cout << "===== 模型对账（" << X.size() << " 个测试样本）=====\n";
    std::cout << "  A: " << pathA << "\n      参数 " << a.weight_count()
              << "   稀疏度 " << fmt(100.0 * a.zero_count() / std::max(1, a.weight_count()), 2) << "%"
              << "   " << (cls ? "准确率 " : "R² ") << fmt(cls ? sa * 100 : sa, 5) << (cls ? "%" : "") << "\n";
    std::cout << "  B: " << pathB << "\n      参数 " << b.weight_count()
              << "   稀疏度 " << fmt(100.0 * b.zero_count() / std::max(1, b.weight_count()), 2) << "%"
              << "   " << (cls ? "准确率 " : "R² ") << fmt(cls ? sb * 100 : sb, 5) << (cls ? "%" : "") << "\n";

    double diff = sb - sa;
    std::cout << "  差异 (B-A) = " << fmt(cls ? diff * 100 : diff, 5) << (cls ? "%" : "")
              << "   ->  " << (std::fabs(diff) < 1e-9 ? "持平" : (diff > 0 ? "B 更好" : "A 更好")) << "\n";

    // 逐样本一致性
    int agree = 0, disagree = 0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        Vec pa = a.predict_one(X[i]), pb = b.predict_one(X[i]);
        size_t nOut = std::min(pa.size(), pb.size());
        double md = 0.0;
        for (size_t j = 0; j < nOut; ++j) md = std::max(md, std::fabs(pa[j] - pb[j]));
        maxAbs = std::max(maxAbs, md);
        bool same = cls ? (a.predict_class(X[i]) == b.predict_class(X[i])) : (md <= tol);
        if (same) agree++; else disagree++;
    }
    std::cout << "  预测一致: " << agree << " / " << X.size() << "（不一致 " << disagree << "，"
              << "判定容差 " << (cls ? std::string("按类别") : fmt(tol, 6)) << "）\n";
    std::cout << "  逐样本最大输出差 = " << fmt(maxAbs, 8) << "\n";
    return 0;
}

/* ---------------- HTTP 推理服务（WinSock，零第三方依赖） ---------------- */

#ifdef _WIN32
int serveModel(const std::string& modelPath, int port, const std::string& token = "")
{
    Model m = Model::load(modelPath, "", "", 0.0);
    if (m.network.layers.empty()) { std::cout << "加载失败: " << modelPath << "\n"; return 1; }
    bool cls = (m.task == "classification");

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { std::cout << "WSAStartup 失败\n"; return 1; }
    SOCKET srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (srv == INVALID_SOCKET) { std::cout << "socket 失败\n"; WSACleanup(); return 1; }
    BOOL yes = TRUE;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 只监听本机，避免无意暴露
    addr.sin_port = htons((u_short)port);
    if (bind(srv, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::cout << "bind 失败（端口被占用？）\n"; closesocket(srv); WSACleanup(); return 1;
    }
    if (listen(srv, 8) == SOCKET_ERROR) {
        std::cout << "listen 失败\n"; closesocket(srv); WSACleanup(); return 1;
    }
    std::cout << "===== HTTP 推理服务 =====\n";
    std::cout << "  模型: " << modelPath << "（task=" << m.task << "）\n";
    std::cout << "  监听: http://127.0.0.1:" << port << "/\n";
    std::cout << "  用法: 浏览器打开上面的地址，或\n";
    std::cout << "        curl \"http://127.0.0.1:" << port << "/predict?x=0.5,0.3\"\n";
    std::cout << "        curl -d \"0.5,0.3\" http://127.0.0.1:" << port << "/predict\n";
    std::cout << "  Ctrl+C 结束\n";

    for (;;) {
        SOCKET cli = accept(srv, nullptr, nullptr);
        if (cli == INVALID_SOCKET) break;
        std::string req, body, resp;
        char buf[8192];
        // 1) 先读请求头（可能分多个 TCP 段到达）
        size_t sep = std::string::npos;
        while (sep == std::string::npos && req.size() < 65536) {
            int got = recv(cli, buf, sizeof(buf) - 1, 0);
            if (got <= 0) break;
            buf[got] = 0;
            req.append(buf, got);
            sep = req.find("\r\n\r\n");
        }
        if (!req.empty()) {
            std::string head = (sep == std::string::npos) ? req : req.substr(0, sep);
            std::string line = head.substr(0, head.find("\r\n"));

            // 2) 标准客户端（curl / .NET）会先发 Expect: 100-continue 等确认，
            //    不回应就会双方互等直到连接被关 —— POST 必须在读 body 前先回 100
            if (head.find("100-continue") != std::string::npos) {
                const char* cont = "HTTP/1.1 100 Continue\r\n\r\n";
                send(cli, cont, (int)strlen(cont), 0);
            }

            // 2.5) 简单鉴权：--token 非空时要求 ?token=xxx 或 X-Token 头
            if (!token.empty() &&
                head.find("token=" + token) == std::string::npos &&
                head.find("X-Token: " + token) == std::string::npos) {
                std::string e = "{\"error\":\"unauthorized: pass ?token=... or X-Token header\"}";
                resp = "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: "
                     + std::to_string(e.size()) + "\r\nConnection: close\r\n\r\n" + e;
                send(cli, resp.c_str(), (int)resp.size(), 0);
                closesocket(cli);
                continue;
            }

            // 3) 按 Content-Length 把 body 读全
            if (line.rfind("POST", 0) == 0) {
                size_t clPos = head.find("Content-Length:");
                if (clPos == std::string::npos) clPos = head.find("content-length:");
                size_t want = 0;
                if (clPos != std::string::npos) want = (size_t)strtoul(head.c_str() + clPos + 15, nullptr, 10);
                if (sep != std::string::npos) body = req.substr(sep + 4);
                while (body.size() < want && body.size() < 65536) {
                    int got = recv(cli, buf, sizeof(buf) - 1, 0);
                    if (got <= 0) break;
                    buf[got] = 0;
                    body.append(buf, got);
                }
            } else if (line.find("x=") != std::string::npos) {
                size_t q = line.find("x=");
                size_t end = line.find(' ', q);
                body = line.substr(q + 2, end == std::string::npos ? std::string::npos : end - q - 2);
            } else if (line.find("GET / ") == 0 || line.find("GET /index") == 0) {
                std::string html =
                    "<!doctype html><meta charset=utf-8><title>AI_C 推理</title>"
                    "<body style='font-family:system-ui;max-width:640px;margin:40px auto'>"
                    "<h2>AI_C 推理服务</h2><p>模型: " + modelPath + "（" + m.task + "）</p>"
                    "<form onsubmit=\"location='/predict?x='+encodeURIComponent(document.getElementById('v').value);return false\">"
                    "<input id=v size=40 placeholder='0.5,0.3'><button>预测</button></form>"
                    "<p>或 <code>curl -d \"0.5,0.3\" http://127.0.0.1:" + std::to_string(port) + "/predict</code></p>"
                    "<div id=r></div></body>";
                resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: "
                     + std::to_string(html.size()) + "\r\nConnection: close\r\n\r\n" + html;
                send(cli, resp.c_str(), (int)resp.size(), 0);
                closesocket(cli);
                continue;
            }
            for (char& ch : body) if (ch == '{' || ch == '}' || ch == '[' || ch == ']' ||
                                      ch == '"' || ch == ':' || ch == 'x') ch = ' ';
            Vec x = parseList(body);
            if (x.empty()) {
                std::string e = "{\"error\":\"no input; use ?x=0.5,0.3 or POST body\"}";
                resp = "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: "
                     + std::to_string(e.size()) + "\r\nConnection: close\r\n\r\n" + e;
            } else {
                Vec p = m.predict_one(x);
                std::string j = "{\"pred\":[";
                for (size_t i = 0; i < p.size(); ++i)
                    j += (i ? "," : "") + fmt(p[i], 10);
                j += "]";
                if (cls) j += ",\"class\":" + std::to_string(m.predict_class(x));
                j += "}";
                resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n"
                       "Access-Control-Allow-Origin: *\r\nContent-Length: "
                     + std::to_string(j.size()) + "\r\nConnection: close\r\n\r\n" + j;
            }
        }
        send(cli, resp.c_str(), (int)resp.size(), 0);
        closesocket(cli);
    }
    closesocket(srv);
    WSACleanup();
    return 0;
}
#endif

/* ---------------- 模型结构图（零依赖 SVG） ---------------- */

int exportSvg(const std::string& modelPath, const std::string& svgOut)
{
    Model m = Model::load(modelPath, "", "", 0.0);
    if (m.network.layers.empty()) { std::cout << "加载失败\n"; return 1; }

    const int BW = 150, BH = 54, GAP = 46, MT = 60;
    int n = (int)m.network.layers.size();
    int W = 40 + n * BW + (n - 1) * GAP + 40;
    int H = MT + 150;
    int total = m.weight_count();

    std::ofstream o(svgOut);
    if (!o.is_open()) { std::cout << "写不出: " << svgOut << "\n"; return 2; }
    o << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
      << "\" viewBox=\"0 0 " << W << " " << H << "\">\n";
    o << "  <rect width=\"" << W << "\" height=\"" << H << "\" fill=\"#0d1117\"/>\n";
    o << "  <text x=\"40\" y=\"34\" fill=\"#c9d1d9\" font-size=\"17\">" << modelPath
      << "   (" << m.task << ", " << total << " params)</text>\n";
    int x = 40, prevRight = -1;
    for (int i = 0; i < n; ++i) {
        auto& L = m.network.layers[i];
        int y = MT + 20;
        if (prevRight >= 0) {
            o << "  <line x1=\"" << prevRight << "\" y1=\"" << (y + BH / 2) << "\" x2=\"" << x
              << "\" y2=\"" << (y + BH / 2) << "\" stroke=\"#8b949e\" stroke-width=\"1.5\" marker-end=\"url(#a)\"/>\n";
        }
        o << "  <rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << BW << "\" height=\"" << BH
          << "\" rx=\"8\" fill=\"#161b22\" stroke=\"#58a6ff\"/>\n";
        o << "  <text x=\"" << (x + BW / 2) << "\" y=\"" << (y + 22)
          << "\" fill=\"#c9d1d9\" font-size=\"13\" text-anchor=\"middle\">Linear "
          << L.fan_in << " → " << L.fan_out << "</text>\n";
        o << "  <text x=\"" << (x + BW / 2) << "\" y=\"" << (y + 40)
          << "\" fill=\"#8b949e\" font-size=\"12\" text-anchor=\"middle\">" << L.act_name << "</text>\n";
        o << "  <text x=\"" << (x + BW / 2) << "\" y=\"" << (y + BH + 18)
          << "\" fill=\"#6e7681\" font-size=\"11\" text-anchor=\"middle\">"
          << (L.fan_in * L.fan_out + L.fan_out) << " params</text>\n";
        prevRight = x + BW + 2;
        x += BW + GAP;
    }
    o << "  <defs><marker id=\"a\" markerWidth=\"8\" markerHeight=\"8\" refX=\"7\" refY=\"3\" orient=\"auto\">"
      << "<path d=\"M0,0 L7,3 L0,6 z\" fill=\"#8b949e\"/></marker></defs>\n";
    o << "</svg>\n";
    o.close();
    std::cout << "结构图已写出: " << svgOut << "（" << n << " 层，" << total << " 参数）\n";
    return 0;
}

/* ---------------- 蒸馏：让小模型去拟合大模型的输出 ---------------- */

int distill(const std::string& teacherPath, const std::string& csv, std::vector<int> hidden,
            int epochs, double lr, int batch, double val, bool header,
            const std::string& outPath, const std::string& testPath)
{
    Model teacher = Model::load(teacherPath, "", "", 0.0);
    if (teacher.network.layers.empty()) { std::cout << "教师模型加载失败\n"; return 1; }
    std::vector<Vec> X, Y;
    if (!loadCSV(csv, header, 1, X, Y)) { std::cout << "数据读取失败\n"; return 2; }
    bool cls = (teacher.task == "classification");

    std::cout << "===== 蒸馏 =====\n";
    std::cout << "  教师: " << teacherPath << "（" << teacher.weight_count() << " 参数）\n";

    // 用教师的输出当软目标
    std::vector<Vec> soft(X.size());
    std::vector<int> softCls(X.size(), 0);
    for (size_t i = 0; i < X.size(); ++i) {
        soft[i] = teacher.predict_one(X[i]);
        softCls[i] = teacher.predict_class(X[i]);
    }

    Model stu = cls ? make_classifier((int)X[0].size(), (int)soft[0].size(), hidden, "leaky_relu", lr)
                    : make_regressor((int)X[0].size(), (int)soft[0].size(), hidden, "leaky_relu", lr);
    std::cout << "  学生: hidden ";
    for (size_t i = 0; i < hidden.size(); ++i) std::cout << (i ? "-" : "") << hidden[i];
    std::cout << "（" << stu.weight_count() << " 参数，压缩到 "
              << fmt(100.0 * stu.weight_count() / std::max(1, teacher.weight_count()), 1) << "%）\n";

    if (cls) stu.fit(X, softCls, epochs, batch, val, 50, true, nullptr, 0.0);
    else     stu.fit(X, soft, epochs, batch, val, nullptr, 50, true, true, nullptr, 0.0);

    // 与教师的一致率（这才是蒸馏的直接指标）
    int agree = 0;
    double maxDiff = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        Vec p = stu.predict_one(X[i]);
        if (cls) { if (stu.predict_class(X[i]) == softCls[i]) agree++; }
        else {
            double md = 0.0;
            for (size_t j = 0; j < p.size() && j < soft[i].size(); ++j)
                md = std::max(md, std::fabs(p[j] - soft[i][j]));
            maxDiff = std::max(maxDiff, md);
            if (md <= 1e-2) agree++;
        }
    }
    std::cout << "  与教师一致率 = " << fmt(100.0 * agree / X.size(), 2) << "%"
              << (cls ? "" : "（回归按误差 ≤1e-2 判定，最大偏差 " + fmt(maxDiff, 6) + "）") << "\n";

    if (!testPath.empty()) {
        std::vector<Vec> TX, TY;
        if (loadCSV(testPath, header, 1, TX, TY)) {
            std::cout << "  学生（测试集）: " << (cls ? "准确率 " : "R² ")
                      << fmt(evalOn(stu, TX, TY, cls) * (cls ? 100 : 1), 5) << (cls ? "%" : "")
                      << "    教师（测试集）: " << (cls ? "准确率 " : "R² ")
                      << fmt(evalOn(teacher, TX, TY, cls) * (cls ? 100 : 1), 5) << (cls ? "%" : "") << "\n";
        }
    }
    if (!outPath.empty()) stu.save(outPath, epochs, batch);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string cmd = (argc > 1) ? argv[1] : "demo";

    if (cmd == "-h" || cmd == "--help" || cmd == "help") { usage(); return 0; }
    if (cmd == "selftest") return selftest();

    if (cmd == "demo" || cmd == "all") {
        int fail = 0;
        fail += demoRegression();
        fail += demoClassification();
        fail += demoGradientClip();
        fail += demoCNN();
        std::cout << "\n" << (fail == 0 ? "全部示例通过" : "有示例未达标") << "\n";
        return fail == 0 ? 0 : 1;
    }

    if (cmd == "train") {
        std::string csv, task = "regression", savePath, testPath, curvePath, finalSave;
        int kfold = 0;
        std::string esMetric = "val";
        bool aggregate = false, aggregateWeights = false;
        int seedVal = 0;
        std::vector<int> hidden{ 16, 8 };
        int epochs = 500, batch = 16, patience = 50;
        double lr = 0.01, val = 0.1;
        bool header = false;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char* def) -> std::string {
                return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
            };
            if (a == "--csv") csv = next("");
            else if (a == "--task") task = next("regression");
            else if (a == "--hidden") hidden = parseIntList(next("16,8"));
            else if (a == "--epochs") epochs = std::stoi(next("500"));
            else if (a == "--lr") lr = std::stod(next("0.01"));
            else if (a == "--batch") batch = std::stoi(next("16"));
            else if (a == "--val") val = std::stod(next("0.1"));
            else if (a == "--patience") patience = std::stoi(next("50"));
            else if (a == "--save") savePath = next("");
            else if (a == "--test") testPath = next("");
            else if (a == "--curve") curvePath = next("");
            else if (a == "--kfold") kfold = std::stoi(next("5"));
            else if (a == "--final") finalSave = next("");
            else if (a == "--early-stop-metric") esMetric = next("val");
            else if (a == "--aggregate") aggregate = true;
            else if (a == "--aggregate-weights") aggregateWeights = true;
            else if (a == "--seed") seedVal = std::stoi(next("0"));
            else if (a == "--header") header = true;
        }
        if (csv.empty()) { std::cout << "缺少 --csv\n\n"; usage(); return 2; }
        if (hidden.empty()) hidden = { 16 };

        if (seedVal != 0) { seed(seedVal); std::cout << "随机种子 = " << seedVal << "\n"; }
        bool cls = (task == "classification");
        std::vector<Vec> X, Y;
        if (!loadCSV(csv, header, 1, X, Y)) {
            std::cout << "数据为空或格式不对: " << csv << "\n";
            return 2;
        }
        std::cout << "数据: " << X.size() << " 个样本，输入维度 " << X[0].size()
                  << "，任务 " << task << "\n";

        if (kfold >= 2) {
            std::cout << "随机种子 = " << (seedVal != 0 ? seedVal : 42) << "\n";
            return trainKfold(csv, kfold, hidden, epochs, lr, batch, val, header, cls,
                              savePath, finalSave, aggregate, aggregateWeights);
        }

        if (cls) {
            std::vector<int> yi;
            int classes = 1;
            for (auto& r : Y) { int v = (int)std::lround(r[0]); yi.push_back(v); if (v + 1 > classes) classes = v + 1; }
            Model m = make_classifier((int)X[0].size(), classes, hidden, "leaky_relu", lr);
            m.early_stop_on_train = (esMetric == "train");
            if (m.early_stop_on_train) std::cout << "早停判据 = 训练损失\n";
            m.fit(X, yi, epochs, batch, val, patience, true, nullptr, 0.0);
            double acc = m.score(X, yi);
            std::cout << "训练准确率 = " << fmt(acc * 100, 2) << "%\n";
            if (!curvePath.empty()) {
                std::ofstream cf(curvePath);
                cf << "epoch,train,val\n";
                for (size_t k = 0; k < m.train_history.size(); ++k) {
                    cf << m.train_history[k].first << "," << m.train_history[k].second;
                    if (k < m.val_history.size()) cf << "," << m.val_history[k].second;
                    cf << "\n";
                }
                cf.close();
                std::cout << "训练曲线已写出: " << curvePath << "（" << m.train_history.size() << " 个点）\n";
            }
            if (!testPath.empty()) {
                std::vector<Vec> TX, TY;
                std::vector<int> tyi;
                if (loadCSV(testPath, header, 1, TX, TY)) {
                    for (auto& r : TY) tyi.push_back((int)std::lround(r.empty() ? 0.0 : r[0]));
                    std::cout << "测试集准确率 = " << fmt(m.score(TX, tyi) * 100, 2)
                              << "%（" << TX.size() << " 个样本）\n";
                } else {
                    std::cout << "测试集读取失败: " << testPath << "\n";
                }
            }
            if (!savePath.empty()) m.save(savePath, epochs, batch);
        } else {
            int nOut = (int)Y[0].size();
            Model m = make_regressor((int)X[0].size(), nOut, hidden, "leaky_relu", lr);
            m.early_stop_on_train = (esMetric == "train");
            if (m.early_stop_on_train) std::cout << "早停判据 = 训练损失\n";
            m.fit(X, Y, epochs, batch, val, nullptr, patience, true, true, nullptr, 0.0);
            double r2 = m.score(X, Y);
            std::cout << "训练集 R² = " << fmt(r2, 6) << "\n";
            if (!curvePath.empty()) {
                std::ofstream cf(curvePath);
                cf << "epoch,train,val\n";
                for (size_t k = 0; k < m.train_history.size(); ++k) {
                    cf << m.train_history[k].first << "," << m.train_history[k].second;
                    if (k < m.val_history.size()) cf << "," << m.val_history[k].second;
                    cf << "\n";
                }
                cf.close();
                std::cout << "训练曲线已写出: " << curvePath << "（" << m.train_history.size() << " 个点）\n";
            }
            if (!testPath.empty()) {
                std::vector<Vec> TX, TY;
                if (loadCSV(testPath, header, 1, TX, TY))
                    std::cout << "测试集 R² = " << fmt(m.score(TX, TY), 6)
                              << "（" << TX.size() << " 个样本）\n";
                else
                    std::cout << "测试集读取失败: " << testPath << "\n";
            }
            if (!savePath.empty()) m.save(savePath, epochs, batch);
        }
        return 0;
    }

    if (cmd == "export-weights" || cmd == "import-weights") {
        std::string modelPath, jsonPath, outPath;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
            else if (a == "--json" && i + 1 < argc) jsonPath = argv[++i];
            else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
        }
        if (cmd == "export-weights") {
            if (modelPath.empty() || outPath.empty()) { usage(); return 2; }
            Model m = Model::load(modelPath, "", "", 0.0);
            if (!m.save_python_json(outPath)) { std::cout << "导出失败\n"; return 1; }
            std::cout << "已导出 Python 版 JSON: " << outPath << "\n"
                      << "  可在 Python 版里用 load_model(\"" << outPath << "\") 直接加载\n";
        } else {
            if (jsonPath.empty() || outPath.empty()) { usage(); return 2; }
            Model m = Model::load_python_json(jsonPath);
            if (m.network.layers.empty()) { std::cout << "导入失败\n"; return 1; }
            m.save(outPath, -1, -1);
            std::cout << "已导入并转成本地格式: " << outPath
                      << "（task=" << m.task << "，层数 " << m.network.layers.size() << "）\n";
        }
        return 0;
    }

    if (cmd == "cross-validate" || cmd == "grid-search") {
        std::string csv, task = "regression";
        std::vector<int> hidden{ 16, 8 };
        std::vector<std::vector<int>> hiddenSets;
        std::vector<double> lrs{ 0.01 };
        int k = 5, epochs = 400, batch = 16;
        double lr = 0.02, val = 0.1;
        bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char* def) -> std::string {
                return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
            };
            if (a == "--csv") csv = next("");
            else if (a == "--task") task = next("regression");
            else if (a == "--hidden") {
                std::string hs = next("16,8");
                if (hs.find(';') != std::string::npos) {          // 多个配置："8,8;16,16"
                    std::stringstream ss(hs);
                    std::string one;
                    while (std::getline(ss, one, ';')) {
                        if (one.empty()) continue;
                        auto v = parseIntList(one);
                        if (!v.empty()) hiddenSets.push_back(v);
                    }
                } else {
                    hidden = parseIntList(hs);
                }
            }
            else if (a == "--lr") {
                std::string ls = next("0.01");
                if (ls.find(',') != std::string::npos) {
                    lrs = parseList(ls);
                } else {
                    lr = std::stod(ls);
                }
            }
            else if (a == "--k") k = std::stoi(next("5"));
            else if (a == "--epochs") epochs = std::stoi(next("400"));
            else if (a == "--batch") batch = std::stoi(next("16"));
            else if (a == "--val") val = std::stod(next("0.1"));
            else if (a == "--header") header = true;
        }
        if (csv.empty()) { usage(); return 2; }
        bool cls = (task == "classification");
        if (cmd == "cross-validate")
            return crossValidate(csv, k, hidden, epochs, lr, batch, header, cls);
        if (hiddenSets.empty()) hiddenSets.push_back(hidden);
        return gridSearch(csv, hiddenSets, lrs, epochs, batch, val, header, cls);
    }

    if (cmd == "distill") {
        std::string teacherPath, csv, outPath, testPath;
        std::vector<int> hidden{ 8 };
        int epochs = 400, batch = 16;
        double lr = 0.02, val = 0.1;
        bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char* d) -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(d); };
            if (a == "--teacher") teacherPath = next("");
            else if (a == "--csv") csv = next("");
            else if (a == "--out") outPath = next("");
            else if (a == "--test") testPath = next("");
            else if (a == "--hidden") hidden = parseIntList(next("8"));
            else if (a == "--epochs") epochs = std::stoi(next("400"));
            else if (a == "--lr") lr = std::stod(next("0.02"));
            else if (a == "--batch") batch = std::stoi(next("16"));
            else if (a == "--val") val = std::stod(next("0.1"));
            else if (a == "--header") header = true;
        }
        if (teacherPath.empty() || csv.empty()) { usage(); return 2; }
        return distill(teacherPath, csv, hidden, epochs, lr, batch, val, header, outPath, testPath);
    }

    if (cmd == "serve") {
        std::string modelPath, token; int port = 8080;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
            else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
            else if (a == "--token" && i + 1 < argc) token = argv[++i];
        }
        if (modelPath.empty()) { usage(); return 2; }
#ifdef _WIN32
        return serveModel(modelPath, port, token);
#else
        std::cout << "serve 目前只在 Windows（WinSock）下实现\n";
        return 2;
#endif
    }

    if (cmd == "export-svg") {
        std::string modelPath, out = "model.svg";
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
            else if (a == "--out" && i + 1 < argc) out = argv[++i];
        }
        if (modelPath.empty()) { usage(); return 2; }
        return exportSvg(modelPath, out);
    }

    if (cmd == "prune" || cmd == "quantize" || cmd == "compare") {
        std::string mA, mB, out, test, csv;
        double ratio = 0.0, threshold = 0.0, tol = 1e-6;
        int bits = 8;
        bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) mA = argv[++i];
            else if (a == "--model-b" && i + 1 < argc) mB = argv[++i];
            else if (a == "--out" && i + 1 < argc) out = argv[++i];
            else if (a == "--test" && i + 1 < argc) test = argv[++i];
            else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (a == "--ratio" && i + 1 < argc) ratio = std::stod(argv[++i]);
            else if (a == "--threshold" && i + 1 < argc) threshold = std::stod(argv[++i]);
            else if (a == "--bits" && i + 1 < argc) bits = std::stoi(argv[++i]);
            else if (a == "--tol" && i + 1 < argc) tol = std::stod(argv[++i]);
            else if (a == "--header") header = true;
        }
        if (cmd == "compare") {
            if (mA.empty() || mB.empty() || csv.empty()) { usage(); return 2; }
            return compareModels(mA, mB, csv, header, tol);
        }
        if (mA.empty()) { usage(); return 2; }
        if (cmd == "prune") {
            if (ratio <= 0.0 && threshold <= 0.0) ratio = 0.5;   // 默认剪一半
            return pruneModel(mA, out, threshold, ratio, test, header);
        }
        return quantizeModel(mA, out, bits, test, header);
    }

    if (cmd == "data") {
        std::string csv; bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (a == "--header") header = true;
        }
        if (csv.empty()) { usage(); return 2; }
        return dataInfo(csv, header);
    }

    if (cmd == "split") {
        std::string csv, tr, te; double ratio = 0.2; int sd = 42; bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--csv" && i + 1 < argc) csv = argv[++i];
            else if (a == "--out-train" && i + 1 < argc) tr = argv[++i];
            else if (a == "--out-test" && i + 1 < argc) te = argv[++i];
            else if (a == "--ratio" && i + 1 < argc) ratio = std::stod(argv[++i]);
            else if (a == "--seed" && i + 1 < argc) sd = std::stoi(argv[++i]);
            else if (a == "--header") header = true;
        }
        if (csv.empty()) { usage(); return 2; }
        if (tr.empty()) tr = "train.csv";
        if (te.empty()) te = "test.csv";
        return splitData(csv, tr, te, ratio, sd, header);
    }

    if (cmd == "describe") {
        std::string modelPath;
        for (int i = 2; i < argc; ++i)
            if (std::string(argv[i]) == "--model" && i + 1 < argc) modelPath = argv[++i];
        if (modelPath.empty()) { usage(); return 2; }
        return describeModel(modelPath);
    }

    if (cmd == "plot") {
        std::string curve, svg, csv;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--curve" && i + 1 < argc) curve = argv[++i];
            else if (a == "--svg" && i + 1 < argc) svg = argv[++i];
            else if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        }
        if (curve.empty()) { usage(); return 2; }
        if (svg.empty() && csv.empty()) svg = "loss_curve.svg";
        return plotCurve(curve, svg, csv);
    }

    if (cmd == "bench") {
        int samples = 1000, epochs = 5;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--samples" && i + 1 < argc) samples = std::stoi(argv[++i]);
            else if (a == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        }
        return bench(samples, epochs);
    }

    if (cmd == "predict") {
        std::string modelPath, input, csvPath, outPath;
        bool header = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model" && i + 1 < argc) modelPath = argv[++i];
            else if (a == "--input" && i + 1 < argc) input = argv[++i];
            else if (a == "--csv" && i + 1 < argc) csvPath = argv[++i];
            else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
            else if (a == "--header") header = true;
        }
        if (modelPath.empty()) { usage(); return 2; }
        if (!csvPath.empty()) return batchPredict(modelPath, csvPath, outPath, header);
        if (input.empty()) { usage(); return 2; }
        Model m = Model::load(modelPath, "", "", 0.0);
        Vec x = parseList(input);
        if (x.empty()) { std::cout << "输入解析失败\n"; return 2; }
        Vec p = m.predict_one(x);
        std::cout << "输出: [";
        for (size_t i = 0; i < p.size(); ++i) std::cout << (i ? ", " : "") << fmt(p[i]);
        std::cout << "]\n";
        if (m.task == "classification") std::cout << "类别索引: " << m.predict_class(x) << "\n";
        return 0;
    }

    usage();
    return 2;
}
