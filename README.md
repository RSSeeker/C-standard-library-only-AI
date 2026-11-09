# AI_C — 纯 C 神经网络工具箱（零第三方依赖）

[![Release](https://img.shields.io/badge/release-v1.2.0-blue)](https://github.com/RSSeeker/C-standard-library-only-AI/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个**不依赖任何第三方库**的深度学习框架：引擎是 `c/` 目录下的**纯 C17 实现**，
C++ 侧（`c_model.cpp/hpp`）提供与旧版一致的面向对象封装。同一份引擎提供两种交付：

| 版本 | 可执行文件 | 说明 |
|------|-----------|------|
| **GUI 图形界面版** | `ai.exe` | 原生 Win32 界面，训练 / 可视化 / 保存加载 |
| **CLI 命令行版** | `ai_cli.exe` | 21 条子命令，不含界面代码，适合脚本化 / 服务器 / CI |

两者**共用同一个引擎与兼容层**，行为与模型存档格式完全一致（CLI 存的模型 GUI 能加载，反之亦然）。
模型存档还与 Python 版 `AI.py` 的 JSON 格式双向互通（`export-weights` / `import-weights`）。

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

### 文本评估指标
- **METEOR 完整版**: `tm_meteor_score` / `tm_meteor_score_full`（精确→词干→同义词贪心对齐、chunk 双连续惩罚）
- **chrF**: `tm_chrf_score` / `tm_chrf_score_tokens`（字符 n-gram F-beta，按有效 n 平均）
- **错误率**: `tm_word_error_rate` / `tm_character_error_rate`（Levenshtein 编辑距离，含 tokens 版本）

### 音频质量指标
- **STOI**: `am_stoi_score`（梅尔频带包络相关系数，返回 [0,1]，1 = 完全相同）
- **PESQ 轻量代理**: `am_pesq_score`（频带对数谱失真 + 帧能量包络相关，返回 [1.0,4.5]）

### 感知 / 视频指标
- **LPIPS 轻量代理**: `vm_lpips_lite`（多尺度池化灰色梯度特征 + 单位化 RMSE，越小越相似）
- **FVD 轻量代理**: `vm_fvd_lite`（默认帧表征的对角协方差 Fréchet 距离，越小分布越近）

### ASR 解码
- **CTC 前缀 beam search**: `ctc_prefix_beam_search`（Graves 的 p_blank / p_nonblank 合并，可注入字符级 LM 与长度惩罚）
- **logits 入口**: `ctc_beam_decode`（内部逐帧 log_softmax 后调用上面的搜索）

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

### ONNX 导出
- **导出**: `onnx_export_mlp` / `onnx_export_mlp_file`（MLP → ModelProto 字节/文件；Gemm + Relu/LeakyRelu/Softmax/Identity 节点，手写 protobuf，无第三方依赖）
- **解析**: `onnx_mlp_parse`（读回 Graph、节点属性、初始化器、输入输出签名）
- **执行**: `onnx_mlp_forward`（纯 C 解释执行导出的计算图，不需要 onnxruntime）
- 边界：导出只覆盖 C 的四种激活（linear/relu/leaky_relu/softmax）；解析/执行额外识别 Tanh/Sigmoid 节点

### 离线强化学习（IQL / CQL）
- **IQL**: `iql_create` / `iql_update` / `iql_act` / `iql_param_count`（expectile 回归 V + 无 max 的 Q 回归 + AWR 策略提取）
- **CQL**: `cql_create` / `cql_update` / `cql_act`（TD 损失 + `alpha*(logsumexp Q - Q(s,a))` 保守项 + AWR）
- **对拍辅助**: `iql_net_q` / `iql_net_v` / `iql_net_policy` / `cql_net_q` / `cql_net_policy` 可直接载入参考权重
- 边界：离散动作；批次由调用方提供（无 ReplayBuffer）；目标网络按 target_tau 软更新

---

## 编译与运行

### E(3) 等变图神经网络（EGNN）

- **层**: `egnn_layer_create(d_model, d_ff)` / `egnn_layer_forward(l, h, x, n, ...)` / `egnn_layer_backward`（phi_e / phi_x / phi_h 三张 MLP；边特征 `[h_i, h_j, ||x_i-x_j||²]`，坐标更新只沿差值方向）
- **模型**: `egnn_model_create(d_model, num_layers, d_ff)` / `egnn_model_loss_and_backward`（均值池化 readout，MSE 标量输出）
- **边界**: 全连接边（n² 对），仅支持三维坐标；旋转/平移/置换等变由结构与反向公式保证，`tests/compare_egnn.c` 同时校验等变性与梯度。

### 投机解码（Medusa / EAGLE）
- **Medusa**: `medusa_heads_create` / `medusa_heads_forward` / `medusa_heads_loss_and_backward` / `medusa_decode`（K 个 linear 头同时猜未来 K 个 token，再交给目标模型整段前向验证）
- **EAGLE**: `eagle_drafter_create` / `eagle_drafter_forward` / `eagle_decode`（隐藏状态 + 上一 token 词嵌入 → 草稿链，逐 token 贪心验证）
- **统计**: `SpecStats` 记录 drafted / accepted / target_calls / accept_rate；`llm_hidden` 返回最近一次前向的隐藏状态供草稿器使用
- 边界：草稿器随机初始化、未训练；验证严格按目标模型贪心一致；EAGLE 草稿器只做特征级预测、无反传

### 扩散采样器（EDM / DPM-Solver / Consistency）
- **调度**: `diffusion_sigmas`（对数间隔 + rho 弯曲，输出含末尾 0）
- **EDM**: `edm_euler_sampler` / `edm_heun_sampler`（VE 参数化，去噪函数以 `DiffDenoiseFn` 回调提供）
- **DPM-Solver**: `dpm_solver_sampler`（一阶指数积分器）
- **一致性模型**: `consistency_training_loss`（student 对齐 teacher 两步 Euler 目标）、`consistency_sampler`
- 边界：纯数值采样器，不依赖模型结构；sigma 调度与步数由调用方给定

### Flow Matching（整流流）
- **模型**: `flowmatching_create` / `flowmatching_free` / `flowmatching_param_count`（三层 MLP，时间特征 sin/cos）
- **路径**: `flowmatching_interpolate`（直线插值）、`flowmatching_target`（目标速度 noise-x0）
- **训练**: `flowmatching_velocity` 前向 + `flowmatching_train_step`（显式 t/noise，累积梯度）、`flowmatching_zero_grad` / `flowmatching_sgd_step`
- **采样**: `flowmatching_sample`（给定初值，Euler / Heun 两种积分器）
- 边界：只移植确定性的前向/反向/采样，RNG 由调用方提供初值（与 Python 的随机采样入口不同）

### 语音合成（TextToMel + Griffin-Lim 声码器）
- **Griffin-Lim**: `tts_griffin_lim`（由幅度谱迭代估计相位还原波形，窗平方和归一化；`tts_wave_free` 释放）；RNG 为与 Python `random.Random(seed)` 逐位一致的 MT19937
- **Mel 反演**: `tts_mel_to_linear`（Mel 滤波器组的最小二乘逆，负值截零）
- **声学模型**: `text_to_mel_create` / `text_to_mel_forward` / `text_to_mel_param_count`（Embedding + 两次 same-padding Conv1d + ReLU + 线性输出，按 duration 展开 token）
- 边界：TextToMel 只移植推理前向，不移植训练反向；Griffin-Lim 为纯数值算法，不依赖模型结构

### Pix2Pix（可训练图像到图像）
- **判别器**: `p2p_disc_create` / `p2p_disc_forward`（三层 PatchGAN，LeakyReLU(0.2)）/ `p2p_disc_backward`（返回输入图像梯度）/ `p2p_disc_sgd_step`
- **生成器**: `p2p_gen_create` / `p2p_gen_forward`（两次 stride-2 卷积编码 + 两次转置卷积解码，ReLU + tanh）/ `p2p_gen_backward` / `p2p_gen_sgd_step`
- **训练**: `p2p_train_step`（先判别器真图目标 1 / 假图目标 0，再生成器 BCE + λ·L1）
- **对拍**: `p2p_disc_set_weights` / `p2p_gen_set_weights` 可从外部灌入权重（跨语言逐位比对）
- 边界：只支持 H/W 为偶数且能被 4 整除的方形输入；无 U-Net skip / InstanceNorm / 多尺度 PatchGAN；训练收敛不保证

### GAN 变体辅助函数（Progressive / AdaIN / 循环损失 / R1·R2）

`gan_variants.*` 把 GAN 变体族的纯数值构件集中到一处（图像按 `k*H*W + i*W + j` 展平）：

| 函数 | 说明 |
|------|------|
| `gv_fade_in(low, high, c, lh, lw, alpha, out)` | Progressive growing 淡入：`(1-alpha)*upsample2x(low) + alpha*high` |
| `gv_progressive_alpha(step, total, &alpha)` | 阶段系数 `min(1, max(0, step/total))` |
| `gv_progressive_stages(...)` | 把递增分辨率展开成 `(low, high, alpha)` 调度 |
| `gv_adain_style(features, c, h, w, style, eps, out, mean, std)` | 逐通道 AdaIN，`style` 长度 `2C` |
| `gv_truncation_trick(w, w_avg, n, psi, out)` | StyleGAN 截断 `w_avg + psi*(w-w_avg)` |
| `gv_cycle_consistency_loss(...)` | CycleGAN 循环一致性损失与两侧梯度 |
| `gv_identity_mapping_loss(...)` | CycleGAN identity 映射损失 |
| `gv_patchgan_targets(h, w, real, out)` | PatchGAN 全 1/全 0 目标网格 |
| `gv_pix2pix_g_loss(...)` | Pix2Pix 生成器损失 `BCE(D(fake),1) + λ·L1` |
| `gv_r1_penalty` / `gv_r2_penalty` | R1/R2 梯度罚项诊断量 `0.5*lam*mean(||g||²)` |

所有函数返回 0 表示成功、-1 表示参数非法；参考数据由 `tools/gan_variants_reference.py` 从 Python 侧生成，`tests/compare_gan_variants.c` 逐项对拍（容差 1e-12）。

### StyleGAN（映射网络 + AdaIN，前向）

`stylegan.*` 提供简化 StyleGAN 生成器 `StyleGanGenerator` 的单向推理：

- `z` 经两层 Mapping MLP（`relu -> linear`）得到 `w`；
- 常数 `4×4×base` 特征 `conv1 -> AdaIN(style1) -> ReLU -> 最近邻上采样 8×8`；
- `conv2 -> AdaIN(style2) -> ReLU -> 上采样 16×16`；
- `to_rgb` 后 `tanh` 到 `[-1, 1]`。

API：`sg_create(latent_dim, d_model, base, out_channels)`、`sg_free`、`sg_param_count`、`sg_forward(g, z, out)`、`sg_average_w(g, zs, n, out_w)`、`sg_last_w`。对拍参考由 `tools/stylegan_reference.py` 生成（`StyleGANGenerator(latent_dim=8, d_model=8, base=4)`，参数量 685），`tests/compare_stylegan.c` 比较前向输出与平均 `w`（容差 1e-12）；训练/反向未移植（README 边界）。

### Progressive Growing（逐阶段增长，前向）

`progressive.*` 提供 ProGAN 风格生成器 `ProgressiveGenerator` 的单向推理：

- 阶段 0：`z -> fc(relu) -> (c0, base, base)`，再用 1×1 卷积得到低分辨率 RGB；
- 每次 `pg_grow()` 追加一组 `ConvTranspose2d(k=4, s=2, p=1) + ReLU + 1×1 RGB 头`（分辨率翻倍）；
- `pg_forward(g, z, alpha, out)` 中 `alpha<1` 时按 `fade_in(上一阶段 RGB, 当前 RGB, alpha)` 平滑混合，最后 `tanh`。

API：`pg_create(latent_dim, channels, n_stages, base, out_channels)`、`pg_free`、`pg_out_shape`、`pg_grow`、`pg_param_count`、`pg_forward`。对拍参考由 `tools/progressive_reference.py` 生成（`latent_dim=8, channels=(8,4), base=2`；参数 818，grow 后 1083），`tests/compare_progressive.c` 比较 base/grow 前向与淡入（容差 1e-12，实测 11/11 逐位相同）；训练/反向未移植。

### CycleGAN（双向循环一致性训练步）

`cyclegan.*` 复用 Pix2Pix 的生成器/判别器，实现一步 CycleGAN 训练：

- 判别器阶段：`D_A` 对 `real_a` 目标 1、对 `fake_a` 目标 0；`D_B` 对 `real_b` 目标 1、对 `fake_b` 目标 0；各自 SGD 更新；
- 生成器阶段：对抗（`D_A(fake_a)`、`D_B(fake_b)` 目标 1）+ 循环一致（`G_BA(G_AB(a))≈a`、`G_AB(G_BA(b))≈b`，权重 `lambda_cycle`）+ identity（`G_AB(b)≈b`、`G_BA(a)≈a`，权重 `lambda_identity`）；
- 参数：`cyclegan_train_step(g_ab, g_ba, d_a, d_b, real_a, real_b, C, H, W, lambda_cycle, lambda_identity, lr_g, lr_d, &step)`，`CycleGanStep` 返回 `d_loss/g_loss/adv_loss/cycle_loss/identity_loss`。

对拍参考由 `tools/cyclegan_reference.py` 生成（两个 `Pix2PixGenerator` + 两个 `PatchDiscriminator`、8×8 图像、一步训练），`tests/compare_cyclegan.c` 比较五项损失、四个网络更新后的全部权重与更新后 fake 输出（容差 1e-9，实测 38/38 逐位相同）。`fit_cyclegan` 的多步循环与随机采样未移植（由调用方组织。

### CLIP 双塔（相似度 / 对称 InfoNCE / 检索 / 线性投影头）

`clip.*` 移植 Python 的多模态双塔组件：

- `cm_clip_similarity(imgs,n,d,txts,n_txt,d2,sim)`：行归一化后的余弦相似度矩阵；
- `cm_clip_loss(imgs,n,d,txts,d2,temperature,g_img,g_txt)`：对称 InfoNCE（第 i 对为正样本），返回损失并可回填两侧梯度（含归一化反向）；
- `cm_clip_retrieval_accuracy(...)`：图文检索 i2t / t2i 命中率；
- `cm_projector_create/free/param_count/set_weights/project/loss/loss_and_grads/apply_sgd`：`ClipProjector` 两个线性投影头（温度、权重布局 `[embed][输入维]`）。

对拍参考由 `tools/clip_reference.py` 生成（`image_dim=3, text_dim=4, embed_dim=5, temperature=0.07`，参数 45），`tests/compare_clip.c` 比较投影、损失、梯度、SGD 更新后权重、相似度、clip_loss 梯度与检索率（容差 1e-12，实测 147/147 逐位级一致）。边界：只做线性投影头，无图像/文本编码器与预训练权重。

### 环境要求
- **C17 编译器**（GCC 8+ / Clang 7+ / MSVC 2019+）—— 引擎 `c/` 是 C17
- **C++17 编译器** —— 界面与命令行版
- Windows SDK（GUI 程序使用 Win32 API）

### CMake 编译（推荐）

```bash
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
# 产出 build/ai.exe（GUI）与 build/ai_cli.exe（CLI）
```

CMake 里必须写 `project(AI CXX C)` 显式启用 C 语言：只写 `CXX` 时 CMake 会**静默忽略所有
`.c` 文件**，直到链接期才报 undefined reference。

### 命令行编译（MinGW-w64）

C 源用 `gcc` 单独编译，再用 `g++` 链接（不要把 C 代码交给 g++ 当 C++ 编）：

```bash
# 1) 编译引擎（C17）
gcc -std=c17 -O2 -Wall -c -I. -Ic \
  c/ai.c c/tensor.c c/layers_conv.c c/layers_norm.c c/rnn.c \
  c/attention.c c/containers.c c/json.c c/model_io.c c/model_high.c \
  c/text.c c/sched.c

# 2) GUI 版
g++ -std=c++17 -O2 -Wall -mwindows -static -o ai.exe -I. -Ic \
  main.cpp c_model.cpp *.o -lcomctl32 -lcomdlg32

# 3) CLI 版（只需要 ws2_32：serve 子命令用 WinSock）
g++ -std=c++17 -O2 -Wall -static -o ai_cli.exe -I. -Ic \
  cli_main.cpp c_model.cpp *.o -lws2_32
```

### 作为库使用

`c/` 是完整的 C 引擎，`c_model.hpp` 是 C++ 封装。只要引擎（C 项目）：

```bash
gcc -std=c17 -O2 -I. -Ic your_app.c c/*.c       # 排除含 main() 的 demo.c / bench.c
```

要 C++ 封装：

```bash
g++ -std=c++17 -O2 -I. -Ic your_app.cpp c_model.cpp <上面那批 .o>
```

### 命令行版用法

```bash
ai_cli                     # 跑全部内置示例
ai_cli selftest            # 训练收敛自检
ai_cli help                # 全部子命令
```

21 条子命令按用途分：训练（`train` 含 K 折 / 网格搜索 / 蒸馏 / 早停判据、`cross-validate`、
`grid-search`、`distill`）、推理（`predict` 单样本与批量、`serve` HTTP 推理服务）、
模型（`describe` / `prune` / `quantize` / `compare` / `export-svg` /
`export-weights` / `import-weights`）、数据（`data` / `split`）、工具（`bench` / `plot`）。

### 测试

```bash
build/test_engine.exe     # C++ 引擎 21 项回归测试（训练收敛 / 存档往返 / 瘦身 / 早停 / 蒸馏）
cd c && make test         # C 引擎测试（含有限差分梯度校验、跨语言逐位比对、零分配验证）
cd c && make check        # 与 Python 版逐位对拍（文本 41 例 + 音频 15 例 + 视觉 10 例 + CTC 16 例 + ONNX 5 例 + 离线 RL 2 例 + EGNN 1 例 + 投机解码 6 例 + 扩散 17 例 + 整流流 7 例 + 语音合成 1 例 + Pix2Pix 1 例 + 变体 15 例 + StyleGAN 1 例 + ProGAN 1 例 + CycleGAN 1 例 + CLIP 1 例等）
```

---

## 快速开始

```cpp
#include "c_model.hpp"   // C++ 封装；引擎实现在 c/ 目录（纯 C17）
using namespace ceng;

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

### Transformer 端到端训练

`TransformerModel` 提供可端到端训练的 Transformer 分类器：
Embedding → 位置编码 → TransformerEncoder → 平均池化 → 线性分类头。
Attention、FeedForward、LayerNorm 的 γ/β、Embedding 与分类头的所有参数都会参与训练
（通过通用参数优化器 `ParamOptimizer`，也可用它对 BatchNorm / RNN 等组件的参数做自定义训练）。

```cpp
// 序列分类：token 索引序列 -> 类别
TransformerModel model(/*vocab=*/20, /*d_model=*/32, /*heads=*/2,
                       /*d_ff=*/64, /*layers=*/1, /*num_classes=*/3,
                       /*lr=*/0.002, "adam");

std::vector<std::vector<int>> X = {{1,2,3,4}, {5,6,7,8}, /* ... */};
std::vector<int> Y = {0, 1, /* ... */};

model.fit(X, Y, /*epochs=*/50, /*batch_size=*/16, /*val_split=*/0.2);
int cls = model.predict_class({1, 2, 3, 4});
```

`RecurrentModel` 提供可端到端训练的 RNN / LSTM / GRU 序列分类器（最后一步隐状态 → 分类头），
同样支持 `fit` / `predict` / `save` / `load`：

```cpp
RecurrentModel rnn("lstm", /*input_size=*/2, /*hidden=*/16, /*layers=*/1,
                   /*num_classes=*/3, /*lr=*/0.01, "adam");

std::vector<std::vector<Vec>> X = {{{0.1,0.2},{0.3,0.4}}, /* ... */};
std::vector<int> Y = {0, /* ... */};
rnn.fit(X, Y, 50, 16, 0.2);
```

两个模型的 `save` / `load` 均为自定义文本格式，往返精度为双精度全精度。

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
├── c/                  # 纯 C17 引擎（81 个文件，约 27,000 行）
│   ├── ai.c/.h         #   Layer / MLP / 损失 / 4 种优化器 / 训练循环
│   ├── tensor.c/.h     #   张量基础
│   ├── layers_conv.*   #   Conv2d / MaxPool2d / Flatten
│   ├── layers_norm.*   #   BatchNorm1d / LayerNorm / Dropout
│   ├── rnn.*           #   RNN / LSTM / GRU（完整 BPTT）
│   ├── attention.*     #   多头注意力 / Transformer
│   ├── ctc_beam.*      #   CTC 前缀 beam search / LM 浅融合解码
│   ├── onnx_mlp.*      #   MLP ↔ ONNX（手写 protobuf 子集：导出/解析/执行）
│   ├── offline_rl.*    #   IQL / CQL 离线强化学习（离散动作，手写反向）
│   ├── egnn.*          #   E(3) 等变图神经网络 EGNN（三张 MLP + 坐标更新）
│   ├── medusa.*        #   Medusa 多头 / EAGLE 特征草稿投机解码
│   ├── diffusion.*      #   EDM / DPM-Solver / Consistency 纯数值采样器
│   ├── flowmatching.*   #   Flow Matching（整流流速度场 + Euler/Heun 采样）
│   ├── tts.*            #   语音合成（TextToMel 前向 / Griffin-Lim / Mel 反演）
│   ├── pix2pix.*       #   图像到图像 Pix2Pix（PatchGAN + 卷积/转置卷积生成器）
│   ├── gan_variants.*  #   GAN 辅助：淡入/逐阶段/AdaIN/截断/循环损失/R1-R2
│   ├── stylegan.*      #   简化 StyleGAN 生成器前向（映射网络 + AdaIN）
│   ├── progressive.*   #   ProGAN 风格逐阶段增长生成器前向（fade_in）
│   ├── cyclegan.*      #   CycleGAN 一步训练（循环一致性 + identity）
│   ├── clip.*          #   CLIP 双塔相似度 / 对称 InfoNCE / 线性投影头
│   ├── model_io.*      #   与 Python 版互通的 JSON 存档
│   ├── model_high.*    #   高层封装
│   ├── text.*          #   文本处理与生成
│   ├── text_metrics.*  #   文本指标（METEOR / chrF / WER / CER）
│   ├── audio_metrics.* #   音频指标（STOI / PESQ 轻量代理）
│   ├── visual_metrics.*#   感知指标（LPIPS / FVD 轻量代理）
│   ├── sched.*         #   学习率调度
│   ├── tests/          #   有限差分梯度校验、跨语言逐位比对、零分配验证
│   └── tools/          #   与 Python 版对拍的参考实现导出脚本
├── c_model.hpp/.cpp    # C++ 兼容层：把 C 引擎包成原来的面向对象 API
├── main.cpp            # Win32 GUI 入口（ai.exe）
├── cli_main.cpp        # 命令行版入口（ai_cli.exe，21 条子命令）
├── test_engine.cpp     # C++ 引擎回归测试（21 项）
├── AI.hpp/.cpp         # 早期纯 C++ 引擎，已不在构建中（保留作参考）
├── CMakeLists.txt      # CMake 构建配置（ai 与 ai_cli 两个目标）
└── README.md
```

---

## 许可证

MIT License
