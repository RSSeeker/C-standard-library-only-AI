# AI_C — 纯 C++ 神经网络工具箱（零第三方依赖）

[![Release](https://img.shields.io/badge/release-v1.1.0-blue)](https://github.com/RSSeeker/C-standard-library-only-AI/releases/tag/v1.1.0)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个**纯 STL 实现**的深度学习框架，无需任何第三方库，仅依赖 C++17 标准库。附带原生 Win32 图形界面，支持模型训练、可视化、保存与加载。

---

## 特性

### 神经网络层
| 组件 | 说明 |
|------|------|
| `Layer` / `MLP` | 全连接层 + LeakyReLU / ReLU / Softmax / Linear 激活 |
| `Conv2d` / `MaxPool2d` | 二维卷积与最大池化 |
| `RNN` / `LSTM` / `GRU` | 循环神经网络，完整 BPTT 反向传播 |
| `TransformerEncoder` | Multi-Head Attention + FeedForward + LayerNorm |
| `Dropout` / `BatchNorm1d` / `LayerNorm` | 正则化与归一化 |
| `Embedding` / `ResidualBlock` / `Sequential` | 基础组件 |

### 训练组件
- **优化器**: Adam / SGD / SGD+Momentum / RMSprop
- **损失函数**: MSE / CrossEntropy
- **梯度裁剪**: 按范数 / 按值
- **学习率调度**: StepLR / ExponentialLR / CosineAnnealingLR
- **Early Stopping** + 数据划分 + DataLoader

### 高层 API（sklearn 风格）
```cpp
// 分类
auto model = ai::make_classifier(2, 3, {16, 8});
model.fit(X_train, y_train, 500);

// 回归
auto model = ai::make_regressor(1, 1, {32, 16});
model.fit(X_train, y_train, 300);

// 快速训练
auto model = ai::quick_train(X, y, "regression", {32}, 200);
```

### 文件操作
- **保存**: 完整网络结构 + 权重 + 偏置 + 训练配置 (文本格式)
- **加载**: 从文件恢复模型，支持继续训练

---

## 编译与运行

### 环境要求
- C++17 编译器 (GCC 8+ / MSVC 2019+ / Clang 7+)
- Windows SDK（GUI 程序使用 Win32 API）

### 命令行编译（推荐 MinGW-w64）

```bash
g++ -std=c++17 -O2 -Wall -mwindows -o ai.exe main.cpp AI.cpp -lcomctl32 -lcomdlg32
```

### CMake 编译

```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

### 纯控制台模式

如果不需要 GUI，可直接去除 `main.cpp`，仅使用 `AI.hpp` / `AI.cpp` 作为头文件库：

```bash
g++ -std=c++17 -O2 your_app.cpp AI.cpp
```

---

## 快速开始

```cpp
#include "AI.hpp"
using namespace ai;

int main() {
    // 1. 准备数据 — XOR 分类
    std::vector<Vec> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<int> Y = {0, 1, 1, 0};

    // 2. 创建模型
    auto model = make_classifier(2, 2, {8}, "leaky_relu");

    // 3. 训练
    model.fit(X, Y, 500);

    // 4. 预测
    for (auto& xi : X) {
        int c = model.predict_class(xi);
        printf("[%.0f, %.0f] -> %d\n", xi[0], xi[1], c);
    }

    // 5. 保存 / 加载
    model.save("model.txt");
    auto loaded = Model::load("model.txt");

    return 0;
}
```

---

## GUI 使用说明

运行 `ai.exe` 启动图形界面：

| 区域 | 功能 |
|------|------|
| **左侧面板** | 选择示例任务、配置模型参数、开始训练 |
| **Loss Curve** | 实时显示训练/验证 Loss 折线图 |
| **输出区** | 训练日志与结果输出 |
| **底部** | 保存模型到文件、加载文件中的模型做预测 |

---

## 项目结构

```
.
├── AI.hpp          # 头文件 — 全部组件声明与高层 API
├── AI.cpp          # 实现文件 — 前向/反向传播、优化器、模型存取
├── main.cpp        # Win32 GUI 入口
├── CMakeLists.txt  # CMake 构建配置
└── README.md
```

---

## 许可证

MIT License

---

## 版本

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.1.0 | 2026-07 | 修复窗口缩放滚动渲染问题 |
| v1.0.0 | 2026 | 初始发布：完整神经网络工具箱 + Win32 GUI |
