# AI_C / c —— Python 版神经网络工具箱的 C 移植

[Python-standard-library-only-AI](../Python-standard-library-only-AI-main) 的 **C17 实现**，
纯 libc + libm，**零第三方依赖**（与该 Python 项目和本目录 C++ 版 AI_C 的定位一致）。

核心目标不是"照抄一遍"，而是**可验证的等价**：给定相同权重，C 侧的计算结果与 Python 版
逐位一致；并且在等价的前提下拿到数量级的速度提升。

---

## 一句话状态

| 项目 | 状态 |
|------|------|
| 数值等价性 | MLP 前向/梯度、Conv2d/MaxPool2d/Flatten、BatchNorm1d/LayerNorm/Dropout、注意力组件、MHA/FFN/TransformerEncoderLayer 前向 **全部与 Python 逐位一致**（0.000e+00） |
| 训练循环 | **零分配**（预热后 `malloc` 调用次数 = 0，用链接器 `--wrap` 统计，与机器负载无关） |
| 性能 | MLP[64,256,128,8]、5 万参数、1000 样本 × 5 轮：C 约 1.3 s vs Python 约 77 s（**约 58–68 倍**） |
| 存档互通 | 与 Python 的 `save_model` / `load_model` **双向互通**，两个方向都逐位一致 |
| 编译 | `-std=c17 -O2 -Wall -Wextra` 下**零警告** |
| 副产品 | 移植中在 **Python 版发现并修复了一个真实梯度 bug**（见文末） |

### 已移植模块

| Python | C | 状态 |
|--------|---|------|
| `Layer` / `MLP` / 四个优化器 / `Model.fit` | `ai.c` | 已完成，逐位一致 + 零分配 |
| `Conv2d` / `MaxPool2d` / `Flatten` | `layers_conv.c` | 已完成，12 项逐位一致 |
| `Dropout` / `BatchNorm1d` / `LayerNorm` | `layers_norm.c` | 已完成，21 项逐位一致 |
| `RNNCell/RNN` / `LSTMCell/LSTM` / `GRUCell/GRU` + step cache | `rnn.c` | 已完成 |
| `MultiHeadAttention` / `FeedForward` / `TransformerEncoder(Layer)` / `PositionalEncoding` / `Embedding` | `attention.c` | 已完成 |
| `Sequential` / `ResidualBlock` | `containers.c` | 已完成（Module vtable） |
| `TextProcessor` / `generate_text` / `softmax_with_temp` | `text.c` | 已完成，逐位一致 |
| `save_model` / `load_model`（JSON） | `json.c` + `model_io.c` | 已完成，双向互通 |
| `StepLR` / `ExponentialLR` / `CosineAnnealingLR` / `clip_grad_by_*` | `sched.c` | 已完成（`pow` 有 1 ULP 差异，见下） |
| `make_classifier` / `make_regressor` / `quick_train` | `model_high.c` | 已完成 |

---

## 构建与运行

```bash
cd c

# 没有 make 时（Windows/MinGW 与 Linux/macOS 通用）
gcc -std=c17 -O2 -Wall -Wextra -o demo demo.c ai.c tensor.c layers_conv.c \
    layers_norm.c rnn.c attention.c containers.c json.c model_io.c text.c sched.c -lm
./demo

# 有 make 时
make            # 构建全部程序
make test       # 运行全部 C 自测
make check      # 与 Python 版做数值等价比对
```

`tools/` 下的 Python 脚本需要能找到本 Python 版仓库，查找顺序是
**环境变量 `AI_PY_DIR`** → 默认的 `../../Python-standard-library-only-AI-main`（与 AI_C 同级）。

---

## 文件结构

```
c/
├── ai.h / ai.c            全连接层、MLP、损失、四个优化器、训练循环、预测/评分
├── tensor.h / tensor.c    张量类型 + 通用模块 vtable（供容器类使用）
├── numeric.h              与 Python 对齐的 Neumaier 补偿求和原语
├── layers_conv.h/.c       Conv2d / MaxPool2d / Flatten
├── layers_norm.h/.c       BatchNorm1d / LayerNorm / Dropout
├── rnn.h / rnn.c          RNNCell/RNN、LSTMCell/LSTM、GRUCell/GRU（含 step cache）
├── attention.h / .c       MultiHeadAttention / FeedForward / TransformerEncoder / PositionalEncoding / Embedding
├── containers.h / .c      Sequential / ResidualBlock（基于 Module vtable）
├── json.h / json.c        极简 JSON 读写（供存档互通）
├── model_io.h / .c        与 Python 的 save_model/load_model 格式互通
├── model_high.h / .c      make_classifier / make_regressor / quick_train
├── text.h / text.c        TextProcessor / generate_text / softmax_with_temp / sample_index
├── sched.h / sched.c      梯度裁剪 + StepLR / ExponentialLR / CosineAnnealingLR
├── demo.c                 对应 README 快速开始的示例
├── bench.c                性能基准
├── compare_python.c       跨语言数值等价比对
├── check_noalloc.c        训练循环零分配验证
├── tests/                 各模块自测（含有限差分梯度校验）
└── tools/                 参考数据导出、对等基准、互通比对脚本
```

---

## 验证：等价性是怎么测的

### 1. 逐位一致（`compare_python`）

`tools/c_reference.py` 用 Python 版建一个固定种子的模型，导出**初始权重、输入、参考前向输出、
参考梯度、5 步 Adam 后的权重**；`compare_python.c` 用 C 实现重算并逐项比对：

```
1) 前向传播（6 个样本）      forward output        最大差异 0.000e+00  [OK]
2) 梯度累加（6 个样本）      accumulated grad W    最大差异 0.000e+00  [OK]
                            accumulated grad b    最大差异 0.000e+00  [OK]
3) 训练 5 步全批量 Adam      weights after training 最大差异 2.220e-16  [OK]
                            biases  after training 最大差异 5.551e-17  [OK]
```

**为什么能做到逐位一致**：Python 3.12+ 的 `sum()` 对浮点使用 Neumaier 补偿求和，不是朴素累加。
C 侧的点积与 MSE 求和照搬了同一算法（见 `numeric.h`）。若用普通 `s += a[i]*b[i]`，
会差 1 ULP 量级并随层数累积。

### 2. 零分配（`check_noalloc`）

用链接器 `--wrap` 把 `malloc/calloc/realloc` 换成计数版，预热一次后统计第二次 `trainer_fit`
期间的分配次数。**这个检查与机器负载无关**（数的是调用次数而不是时间），所以在 CPU 被别的
进程占满时依然可信：

```
training 3 epochs x 200 samples x MLP[64,256,128,8]
  allocations during fit (after warmup): 0  (0 bytes)
  -> zero-allocation hot path: OK
```

### 3. 存档互通（`tools/json_interop.py` + `tests/test_json`）

```
C 写 -> Python 读：  最大差异 0.000e+00  PASS   （权重、激活名、前向输出全部一致）
Python 写 -> C 读：  最大差异 0.000e+00  PASS   （含 softmax 输出层）
```

### 4. 文本与调度器互通（`tools/text_interop.py`、`tools/sched_interop.py`）

- TextProcessor：词表码点序列、one-hot、索引编解码、`softmax_with_temp` 全部一致（逐位）
- 调度器：StepLR / CosineAnnealingLR / `clip_grad_by_norm` / `clip_grad_by_value` 逐位一致
  （`ExponentialLR` 因 `pow` 实现不同差 1 ULP，见下方"已知差异"）

### 5. 逐模块等价性（`tools/*_reference.py` + `tests/compare_*.c`）

```
python tools/norm_reference.py && ./tests/compare_norm     # 21 项，全部 0.000e+00
python tools/conv_reference.py && ./tests/compare_conv     # 12 项，全部 0.000e+00
```

覆盖：BatchNorm1d（训练 forward/反向 dx/grad_gamma/grad_beta/running 统计量/mean/var/inv_std/x_hat、
推理 forward）、LayerNorm（forward/反向/参数梯度/中间量）、Dropout（灌入 Python 的 mask 后逐位比对）、
Conv2d（两种配置的 forward/grad_w/grad_b/dx）、MaxPool2d（两种配置）、Flatten。

> 注意 `Conv2d.forward` 在 Python 里走的是 `_matmul`（内部 `sum(map(mul, ...))` = Neumaier 补偿求和），
> 不是朴素三重循环累加——C 侧照此实现才能逐位一致。

---

## 移植过程中在 Python 版发现的 bug

**`TransformerEncoderLayer.backward` 的梯度是错的**（`seq_len > 1` 时），已定位、修复并加回归测试。

原因：`LayerNorm` 只保存「最后一次 forward」的单槽缓存，而 `TransformerEncoderLayer.forward`
是逐 token 调 `self.ln1.forward(v)` 的（缓存里只剩最后一个 token）；`backward` 又逐 token 调
`ln.backward` → token `0..T-2` 全部用最后一个 token 的 `mean/var` 反传。

独立验证（中心差分，`L = Σ coef·out`）：

```
修复前：seq_len=1 误差 3.3e-10（正确）   seq_len=2 误差 4.9e-01   seq_len=4 误差 1.19e+00
修复后：seq_len=1/2/4 全部回到 1e-10 量级
```

`seq_len=1` 时恰好正确，所以在小序列上完全看不出来。修法是在逐 token 反传前重放该 token 的
前向缓存（`res1`/`res2` 已经缓存着，正好够用）。

**这个 bug 是被"跨语言移植"逼出来的**：C 侧的数值梯度检查一直通过，而 Python 侧没有
`TransformerEncoderLayer` 的数值梯度测试，两边一比对才暴露。Python 仓库已修复并新增
6 个回归测试（测试数 171 → 177），修复后 demo 里 Transformer 的训练损失全面下降
（Epoch 50：0.4579 → 0.3390）。

## 与 Python 版的已知差异

| 差异 | 原因 | 影响 |
|------|------|------|
| Adam 训练若干步后权重差 1 ULP | **libm 的 `pow` 实现质量不同**：MinGW 的 `pow(0.9,4)` = `0.65610000000000013`，MSVC（Python 用的）返回正确舍入的 `0.6561`。glibc 的 `pow` 是正确舍入的，Linux 上不会有这个差异 | 只影响含 `pow` 的地方（Adam 偏差修正、ExponentialLR），量级 1e-16 |
| 随机数与 Python 不同 | Python 用 Mersenne Twister + 极坐标法高斯；C 用 xorshift64\* + Box-Muller | 影响初始化与 shuffle 的轨迹，**不影响**给定权重下的数值等价性 |
| Dropout 的 mask | C 侧用自带 RNG，但提供 `dropout_set_mask()` 供测试灌入确定 mask | 同上 |
| 存档的浮点文本 | C 用 `%.17g`，Python 用 `repr`（最短表示） | 文本不同，**解析回 double 完全相同** |
| `pow` 相关的 1 ULP | MinGW 的 `pow` 非正确舍入（`pow(0.9,4)` = 0.65610000000000013），MSVC/glibc 返回正确舍入的 `0.6561` | 只影响 ExponentialLR 与 Adam 偏差修正；换 glibc 即消失 |
| **libm 的超越函数**（`exp` / `tanh` / `pow`） | Windows 上 Python 用 UCRT，MinGW 自带一套实现：实测 `exp` 4010 个样本中 15 个差 1 ULP、`tanh` 2 个 | **Windows 上跨语言比对不可能天然逐位一致**，容易被误判成移植 bug。实测把 C 侧 libm 指到 `ucrtbase.dll`（`-Wl,--wrap=exp/tanh/pow`）后，RNN 的 56 项全部逐位相同。Linux/glibc 下两边同一套 libm，可直接逐位比对 |
| `attention.c` 自带私有 `TfLayerNorm` / `TfDropout` | Attention 模块先于归一化模块完成，为避免跨文件依赖各自实现了一份 | **待统一**：应改为复用 `layers_norm.h` 的 `LayerNorm` / `Dropout` |
| RNN 的 step cache 存副本 | Python 的 `_StepCacheMixin` 存的是引用（`self.x = x`） | C 侧存副本，调用方原地改写输入缓冲时不会污染缓存（有意差异） |

## 所有权契约（容易踩，务必先读）

`tensor.h` 的 `module_free(m)` 会先调用 `vt->free(m)` 再 `free(m)`。各模块的 `as_module()` 有两种实现：

- **嵌入式**（`layers_conv.c`）：结构体首成员是 `Module mod`，`as_module()` 只返回 `&self->mod`，
  `vt->free` 只释放内部缓冲，结构体由 `module_free` 里的 `free(m)` 释放
- **包装式**（`layers_norm.c`）：`as_module()` 用 `malloc` 造一个包装，`vt->free` 释放底层对象，
  包装由 `module_free` 释放

两种都自洽，但**`xxx_free(x)` 与 `module_free(xxx_as_module(x))` 只能二选一**，都调就是 double-free。
容器（`Sequential` / `ResidualBlock`）**拥有**其子模块，会在自己的 `free` 里递归 `module_free`——
所以把模块交给容器后就不要再手动释放它。

---

## 设计说明

### 数值对齐优先
所有"需要与 Python 的 `sum()` 对齐"的位置都用 `numeric.h` 里的 `dot_neumaier` / `sum_neumaier`；
而 Python 侧本来就是朴素 `+=` 的地方（梯度累加、优化器更新）用普通 `+`。
判断依据是**逐行核对 Python 源码**，不是凭经验。

### 两层结构
- **扁平快速路径**（`ai.c` 的 `Layer`/`MLP`）：权重要连续数组、行主序，这是比 Python 嵌套 list
  快近两个数量级的主要原因之一；梯度、优化器状态、反向 delta 乒乓缓冲全部预分配
- **张量/模块层**（`tensor.h` + 各 `layers_*`、`rnn`、`attention`、`containers`）：
  用于 CNN / RNN / Transformer 这类多维模块，容器类通过 vtable 装任意模块，
  对应 Python 的鸭子类型

### 反向传播的契约
Python 的模块有两种 backward 返回约定（元组取末位 / 非元组整体），C 侧统一为
"把输入梯度写进 `dx`，参数梯度累加到内部缓冲"——消除了这个歧义。
（Python 的 `ResidualBlock` 曾因为没处理元组约定而在 sublayer 为 `Layer` 时抛 TypeError，
这个 bug 在 Python 侧已修，C 侧靠 vtable 天然规避。）

### 与 Python 一致的两个"坑"
1. **softmax 只能配 cross-entropy**：`Layer._softmax_deriv` 恒返回 1，只有与 CE 搭配才正确
   （CE 直接返回 `pred - one_hot` 的联合梯度）。C 侧保留同一约定。
2. **BPTT 的梯度缓冲是内部数组**：`get_grads()` 返回的就是 cell 内部缓冲，
   调用方不能再手工累加，否则等于自己加自己（Python 侧曾因此让梯度按 2 的幂放大）。

### 读参考数据必须用 strtod，不能用 `fscanf("%lf")`

msvcrt 的 `scanf("%lf")` 对某些 17 位十进制串会**解析错 1 ULP**（实测
`-0.01863766757260517` 被读成 `...168`，而 `strtod` 与 Python 的 `float()` 都得到 `...172`）。
参考数据是 Python 用 `repr()` 写出的最短往返表示，所以 `compare_python.c` 一律用
`fscanf("%63s")` + `strtod` 读取（`-D__USE_MINGW_ANSI_STDIO=1` 修不了这个问题）。
`json.c` 内部本来就用 `strtod`，因此 `compare_norm` / `compare_conv` 不受影响。

### 编译器选项提醒

GCC 默认 `-ffp-contract=fast`，在支持 FMA 的机器上 `s += W*x` 可能被收缩成 FMA，
而 Python 永远不会 —— 逐位一致会被破坏。若要加 `-march=native`，请同时加
`-ffp-contract=off`（当前 `-O2` 的 x86-64 基线没有 FMA，不受影响）。

---

## 未完成 / 待办

- `containers.c`（Sequential / ResidualBlock）正在实现中
- 部分模块的跨语言等价性比对仍在补充（卷积/归一化/RNN/注意力的参考数据导出）
- 未做 SIMD / OpenMP（本机 gcc 支持 `-fopenmp`），大矩阵还有数倍空间
