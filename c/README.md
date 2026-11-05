# AI_C / c —— Python 版神经网络工具箱的 C 移植

[Python-standard-library-only-AI](../Python-standard-library-only-AI-main) 的 **C17 实现**，
纯 libc + libm，**零第三方依赖**（与该 Python 项目和本目录 C++ 版 AI_C 的定位一致）。

核心目标不是"照抄一遍"，而是**可验证的等价**：给定相同权重，C 侧的计算结果与 Python 版
逐位一致；并且在等价的前提下拿到数量级的速度提升。

---

## 一句话状态

| 项目 | 状态 |
|------|------|
| 数值等价性 | MLP 前向/梯度、Conv2d/MaxPool2d/Flatten、BatchNorm1d/LayerNorm/Dropout、注意力组件、MHA/FFN/TransformerEncoderLayer 前向/反向 **全部与 Python 逐位一致**（0.000e+00，含 window/RoPE/GQA/MQA、ALiBi 与注意力权重 dropout 的注入 mask 对拍，见第 8 节） |
| 训练循环 | **零分配**（预热后 `malloc` 调用次数 = 0，用链接器 `--wrap` 统计，与机器负载无关） |
| 性能 | MLP[64,256,128,8]、5 万参数、1000 样本 × 5 轮：C 约 1.3 s vs Python 约 77 s（**约 58–68 倍**） |
| 存档互通 | 与 Python 的 `save_model` / `load_model` **双向互通**，两个方向都逐位一致 |
| 因果语言模型 | `CausalLM`（因果掩码 + GPT 式并行训练）：前向 logits、loss、19 个梯度块、一步 Adam、贪心生成、6 组带罚分/min_p 的确定性生成 **全部与 Python 逐位一致**；多步训练受 libm `exp/log` 的 1 ULP 影响（见"已知差异"） |
| 扩展特性 | 滑动窗口注意力、RoPE、GQA/MQA、ALiBi、nucleus（top-p）采样、min_p、repetition/presence/frequency 罚分、no_repeat_ngram、困惑度：语义与 Python 版逐项对齐；window/RoPE/GQA/MQA/ALiBi 已纳入 `compare_attention` 逐位对拍，top_p（确定性 nucleus）、罚分/min_p 的确定性生成与候选数已纳入 `compare_llm`；**默认路径（不做任何 setter 调用、不传新参数时）仍与 Python 逐位一致**——`compare_llm` 的判定未变 |
| 编译 | `-std=c17 -O2 -Wall -Wextra` 下**零警告** |
| 副产品 | 移植中在 **Python 版发现并修复了一个真实梯度 bug**（见文末） |

### 已移植模块

| Python | C | 状态 |
|--------|---|------|
| `Layer` / `MLP` / 四个优化器 / `Model.fit` | `ai.c` | 已完成，逐位一致 + 零分配 |
| `Conv2d` / `MaxPool2d` / `Flatten` | `layers_conv.c` | 已完成，12 项逐位一致 |
| `Dropout` / `BatchNorm1d` / `LayerNorm` | `layers_norm.c` | 已完成，21 项逐位一致 |
| `RNNCell/RNN` / `LSTMCell/LSTM` / `GRUCell/GRU` + step cache | `rnn.c` | 已完成 |
| `MultiHeadAttention` / `FeedForward` / `TransformerEncoder(Layer)` / `PositionalEncoding` / `Embedding` | `attention.c` | 已完成（含 `causal` 因果掩码开关、`mha_set_window` 滑动窗口、`mha_set_rope` 旋转位置编码、`mha_create_gqa` 的 GQA/MQA、`mha_set_alibi` + `tel_set_alibi` / `te_set_alibi` 的 ALiBi 距离偏置、`mha_forward_step` 增量前向、`mha_set_training` + `mha_set_attn_mask` 的注意力权重 dropout） |
| `MultiHeadAttention(causal=True)` / `CausalLM`（LLM） | `attention.c` + `llm.c` | 已完成：因果掩码、teacher forcing 训练、自回归生成、`llm_generate_p`（nucleus 采样）、`llm_generate_ex` / `llm_generate_cached_ex`（min_p + repetition/presence/frequency 罚分 + no_repeat_ngram）、`llm_perplexity` |
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
    layers_norm.c rnn.c attention.c llm.c containers.c json.c model_io.c text.c sched.c -lm
./demo

# 有 make 时
make            # 构建全部程序
make test       # 运行全部 C 自测（含 tests/test_llm：因果性 / 数值梯度 / 训练 / 生成）
make check      # 与 Python 版做数值等价比对
```

没有 make 时自测/比对程序也照「同一份源码列表 + 目标 .c」编译（`tests/test_llm_ext.c` 是新增的
扩展特性自测，不需要额外的 c/ 源文件）：

```bash
SRC="ai.c tensor.c layers_conv.c layers_norm.c rnn.c attention.c llm.c containers.c \
     json.c model_io.c text.c sched.c model_high.c"
gcc -std=c17 -O2 -Wall -Wextra -o tests/compare_llm tests/compare_llm.c $SRC -lm
gcc -std=c17 -O2 -Wall -Wextra -o tests/test_llm_ext tests/test_llm_ext.c $SRC -lm
python tools/llm_reference.py && ./tests/compare_llm   # 6 段比对
./tests/test_llm_ext self                              # 6 段扩展特性自测
```

`./demo` 输出三段：分类、回归、**因果语言模型**（字符级「012012…」，约 0.06 s 把 loss 从 2.2 降到 0.003，贪心续写 `0120120120120`）。

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
├── attention.h / .c       MultiHeadAttention / FeedForward / TransformerEncoder / PositionalEncoding / Embedding（含 causal 掩码、滑动窗口、RoPE、GQA/MQA）
├── llm.h / llm.c          CausalLM：因果语言模型（复用 attention.c 的编码器 + 词表头 + Adam + nucleus 采样 + 困惑度）
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
├── tests/                 各模块自测（含有限差分梯度校验、test_llm_ext 的扩展特性自测）
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
python tools/llm_reference.py && ./tests/compare_llm      # 6 段，见下
python tools/attention_reference.py && ./tests/compare_attention  # 13 个用例，见第 8 节
```

覆盖：BatchNorm1d（训练 forward/反向 dx/grad_gamma/grad_beta/running 统计量/mean/var/inv_std/x_hat、
推理 forward）、LayerNorm（forward/反向/参数梯度/中间量）、Dropout（灌入 Python 的 mask 后逐位比对）、
Conv2d（两种配置的 forward/grad_w/grad_b/dx）、MaxPool2d（两种配置）、Flatten。

`compare_llm` 比对 CausalLM 的八段：前向 logits、loss + 19 个梯度块、一步 Adam 后的参数、
训练 3 轮后的参数与 eval loss（含 perplexity）、贪心生成的 token、top_p 确定性生成、
六组带罚分/min_p 的 `top_k=1` 生成（`llm_generate_ex` 与 `llm_generate_cached_ex` 各比一遍）、
8 个 `min_p` 取值下的候选集合大小。第 1/2/3/5/6/7/8 段**逐位相同**；第 4 段的前两轮也逐位相同，
从第 3 轮起 3 个块出现 ≤3.7e-13 的差异（msvcrt 与 UCRT 的 `exp`/`log` 差 1 ULP，随步数累积），
因此这一段单独用 `1e-12` 容差并在输出里注明。`perplexity = exp(loss)` 的 loss 逐位相同，
`exp` 本身差 8.9e-16（1 ULP），同样用 `1e-12`；`top_p=1e-9` 时 nucleus 只含 argmax，
采样与 RNG 无关，token 序列逐位相同（任意 `top_p` 的随机采样不能跨语言逐位比较，
见"已知差异"）。

> 注意 `Conv2d.forward` 在 Python 里走的是 `_matmul`（内部 `sum(map(mul, ...))` = Neumaier 补偿求和），
> 不是朴素三重循环累加——C 侧照此实现才能逐位一致。

### 6. 扩展特性自测（`tests/test_llm_ext.c`）

`./tests/test_llm_ext self`，六段，全部通过（有限差分统一 eps=1e-6、rel tol=1e-5）：

```
[1] 滑动窗口（window=3，未开 causal）  窗口外权重 0.000e+00 / 每行和 2.220e-16 /
                                       未来位置 0.000e+00 / window>=T 与全因果逐位一致 0.000e+00
[2] RoPE 开关                        rope 关掉后与默认路径逐位一致 0.000e+00 / 打开后输出确实改变 /
                                       位置 0 不变（theta=0）0.000e+00 / 不同 rope_base 输出不同
[3] RoPE 梯度                         W_q[46] 2.370e-10 / W_v[12] 1.613e-10 / W_k[26] 1.570e-10
[4] GQA（4 头 / 2 KV 头）            W_q[13] 3.075e-10 / W_k[26] 8.197e-10 / b_k[5] 4.347e-311 /
                                       KV 头复制后与普通多头逐位一致 0.000e+00 / 冒烟 25.386758→1.294080
[5] nucleus 采样                     top_p=1.0 与 llm_generate 逐位一致 / top_p=0.5 在尖分布上
                                       （argmax 概率 0.996495）5 个 seed 全部等价于贪心
[6] 困惑度                            perplexity == exp(evaluate) 0.000e+00（evaluate 0.445626 → 1.561467）
```

[1] 的「每行和」用 `1e-12` 容差：softmax 后各行和与 1 的偏差就是 `2.2e-16`（一个 ULP 量级），
不可能更小——这是浮点性质，不是实现误差。[4] 的 GQA 训练冒烟用 `lr=0.002`：损失取
`0.5*||out||²` 时步长上界是 `2/λmax`，`lr=0.05` 会直接发散到 nan（不是 bug）。

### 7. 增量 KV cache 生成（`tests/test_llm_cache.c`）

`llm_generate` / `llm_generate_p` 每生成一个 token 都把整个窗口重算一遍（O(T²)）。
新增的 `llm_generate_cached` 走增量路径：预填充一次，之后每个新 token 只做一次单 token 前向
（`llm_forward_step`，内部是各层的 `te_forward_step` / `tel_forward_step` / `mha_forward_step`），
KV 存在每层的 `cache_K` / `cache_V` 里，位置取「当前窗口内下标」。

- **默认行为不变**：老接口原样保留；新接口是 `llm_cache_reset` / `llm_forward_step` /
  `llm_generate_cached` 三个新函数。`llm_generate_p` 只是把采样部分抽成 `sample_row` 供两条路径共用
  （同一份 top_k / top_p / 温度 / RNG 消耗顺序），原有数值路径未改。
- **逐位相同**：`./tests/test_llm_cache` 共 382 项检查、0 失败，容差 0。覆盖
  `max_len` = 20/16/8/5/4（生成长度 16，即大于/等于/小于三种情形）× 绝对位置编码/RoPE，
  每一步的 logits 都与 `llm_forward` 对当前窗口的整段重算逐位对照；token 序列另在
  5 种 `max_len` × 2 种位置编码 × 5 组采样参数（贪心、top_k=3、top_p=0.9×2、top_k=5、
  top_p=0.5、temperature=0.8/1.2）下与 `llm_generate_p` 完全一致，且同 seed 重复调用可复现。
  第 [3b] 节再把 `llm_generate_ex` 与 `llm_generate_cached_ex` 放在同一张表上比：7 组配置覆盖
  `min_p`、`repetition_penalty`（>1 与 <1 两分支）、`presence/frequency_penalty`、
  `no_repeat_ngram_size`（2/3）及其全开组合，`max_len` 取 4/5/8/16/64（4 与 5 保证每步滑动），
  7×2 项 token 序列检查全部逐位相同；全默认参数时旧包装 `llm_generate_cached` 与
  `llm_generate_cached_ex` 也返回同一串 token。
  窗口滑动时会重建 `max_len` 个 token 的 cache（与 Python `_generate_cached` 的重建分支一致），
  这就是「位置重基」——重算路径每步也把窗口当位置 0..T-1，二者语义相同。
- **实测加速比**（`clock()`，prompt 8 + 新增 40，d_model=32，heads=4，d_ff=64，贪心；
  每项重复 20 次取平均）：

  | 层数 | max_len | 重算 | cached | 加速 |
  |---|---|---|---|---|
  | 2 | 64（不滑动） | 56.700 ms | 2.800 ms | **20.25×** |
  | 4 | 64（不滑动） | 99.950 ms | 4.800 ms | **20.82×** |
  | 2 | 16（每步都滑动） | 24.300 ms | 21.150 ms | 1.15× |
  | 4 | 16（每步都滑动） | 44.950 ms | 41.000 ms | 1.10× |

  > 上表是某一轮本机实测；绝对毫秒数随机器与负载变化。加入罚分/min_p 分支后复测，
  > 同样的四组加速比为 20.91× / 21.67× / 1.31× / 1.00×，量级与趋势不变
  > （罚分与 min_p 只在非默认参数时进入，贪心基准路径未改）。

  加速比与 Python 侧报告的 17.8× 同量级。窗口远大于生成长度时收益最大；`max_len` 等于窗口、
  每步都触发重建时，增量路径也要重算 `max_len` 个 token（单 token 前向的循环开销还略高于
  批量 `mm_nt`），所以只剩 ~1.1×——这是算法本身的性质，不是退化 bug。

### 8. 注意力 / TEL 跨语言对拍（`tests/compare_attention.c`）

```
python tools/attention_reference.py && ./tests/compare_attention   # 13 个用例，全部 0.000e+00
```

`attention_reference.py` 用固定种子在 Python 侧建模型，把权重、输入 `x`、上游梯度 `dout`、
输出、`dx` 和全部参数梯度写进 `attention_reference.json`；C 侧用 `mha_set_params` /
`ff_set_params` / 直接灌 LN 的 `gamma`/`beta` 复原同一模型后重算。覆盖：

| 用例 | 配置 | 比对项 |
|---|---|---|
| `mha_causal` | d=8, h=2, T=4, causal | out / dx / 8 组梯度 / `forward_step` 逐步输出 |
| `mha_window2` | T=6, window=2 | 同上 |
| `mha_rope` | T=5, causal + RoPE | 同上 |
| `mha_gqa` | h=4, kv=2, T=4 | 同上 |
| `mha_mqa` | h=4, kv=1, T=5 | 同上 |
| `mha_combo` | h=4, kv=2, window=2, RoPE, T=6 | 同上 |
| `mha_alibi` | d=8, h=2, T=4, causal + ALiBi | out / dx / 8 组梯度 / `forward_step` 逐步输出 |
| `mha_alibi_win` | h=4, kv=2, window=2, ALiBi, T=6 | 同上 |
| `mha_alibi_rope` | h=2, T=5, causal + RoPE + ALiBi | 同上 |
| `tel_causal` | d=8, h=2, ff=16, T=4 | out / dx / 注意力 8 组 / FFN 4 组 / LN 4 组梯度 |
| `tel_alibi` | 同上 + ALiBi | 同上 |
| `mha_dropout_p50` | d=8, h=2, T=4, p=0.5, training=1 | out / dx / 8 组梯度 / `forward_step`（注入 Python 生成的 mask） |
| `mha_dropout_ctrl` | 同上权重，p=0, training=1 | 同上（dropout=0 对照，确认默认路径不变） |

全部用例容差 0（`0.000e+00`）。RoPE 的 `pow`/`cos`/`sin` 在 msvcrt 与 CPython 3.14 上实测
逐位相同（`%a` 与 `float.hex()` 只是格式不同），所以这一组不需要放宽容差。

**注意力权重 dropout 怎么对拍的（不依赖 RNG 同源）**：Python 的 mask 来自
`self.rng.random()`，C 用自己的 splitmix64，两边天然不同源。所以这一组不用 RNG 对拍，
而是**注入 mask**：导出脚本把模型的 `rng` 换成一个只有 `.random()` 的假对象（按预生成
列表返回），于是 Python 实际用的 `(num_heads,T,T)` mask 完全确定，一并写进 JSON；C 侧
`mha_set_training(1)` + `mha_set_attn_mask()` 灌入同一份 mask，再比较 out / dx / 8 组梯度。
两边乘的是同一串 double，任何差异都只能来自实现（mask 作用位置、dV 用哪份权重、softmax
雅可比是否用干净权重），容差仍是 0。实测 `mha_dropout_p50` 全部 `0.000e+00`；
`mha_dropout_ctrl`（同权重、p=0）也全部 `0.000e+00`，说明默认路径没有被动到。

**ALiBi 怎么对拍**：`mha_set_alibi(m, 1)` 后，前向在缩放后的 scores 上叠加
`slope_h * (t - j)`（仅 `j <= t`），位置在 `softmax` 之前、因果/窗口掩码之前；反向不需要任何
改动——常数偏置在 softmax 雅可比里自然抵消，`d_scores` 用的是已经含偏置的 `attn`。
`slope_h = 2^(-8*(h+1)/num_heads)`；对拍用例取 `h = 2/4`，斜率恰为 2 的整数次幂
（2^-4、2^-8 或 2^-2..2^-8），二进制可精确表示，所以 `pow` 的 1 ULP 问题不会出现，
容差维持 0。4 个 ALiBi 用例（含 window、RoPE 组合与 TEL 透传）的 out / dx / 8 组梯度 /
`forward_step` 全部 `0.000e+00`。单元测试另在 `test_attention` 第 2c 节把 `W_q`/`W_k` 置 0，
直接验证 `attn[t][j] == softmax_j(slope*(t-j))` 的闭式解（实测 0.000e+00）、关闭后退化为
均匀权重、`forward_step` 与整段一致，以及 `tel_set_alibi` / `te_set_alibi` 的逐层透传。

> **顺带修掉一个真实的 GQA/MQA bug**：`mha_forward` 给 `v_full` 加 bias 时循环上界误用了
> `d_model`，而 `b_v` 只有 `kv_dim` 长。GQA/MQA 下这会越界读 `b_v`，并把值写进 `v_full`
> 下一行的开头（最后一行的写甚至越过缓冲区）。`mha_mqa` 用例第一次跑就暴露：输出差
> `3.902e-01`，`grad_W_o` 差 `4.114e-01`；组合用例差到 `2.460e+00`。改成按 `kv_dim` 循环后
> 修复后全部用例逐位相同；默认 `kv_dim == d_model` 时循环次数与顺序不变，`compare_llm` /
> `test_llm_ext` 的既有结果未变。这个 bug 之前的 C 侧自测没抓到，是因为默认路径和
> （碰巧拿到清零堆内存的）GQA 用例都没有触发可见差异。

### 9. 生成采样增强：min_p 与五种罚分（`llm_generate_ex`）

在原有 `temperature` / `top_k` / `top_p` 之外，新增与 Python `CausalLM.generate` 对齐的五个参数：

```c
int llm_generate_ex(CausalLM *m, const int *prompt, int prompt_len, int max_new,
                    double temperature, int top_k, double top_p, double min_p,
                    double repetition_penalty, double presence_penalty,
                    double frequency_penalty, int no_repeat_ngram_size,
                    int *out, unsigned long long seed);
int llm_generate_cached_ex(/* 同参数 */);
```

旧的 `llm_generate` / `llm_generate_p` / `llm_generate_cached` 签名与行为一位没变：
它们只是用默认值 `min_p=0, repetition_penalty=1, presence=0, frequency=0, ngram=0`
转调 `*_ex`，而 `*_ex` 在「全部默认」时直接跳过罚分与 min_p 分支。`test_llm` 第 [10] 节用
同 seed 比较新旧接口的 token 序列（贪心与 top_p 两组）逐位相同；`compare_llm` 第 [1]-[6] 段
（前向、loss/梯度、Adam、训练、生成、top_p）结论也未变。

**运算顺序**与 Python 完全一致：先用**完整历史 ids**（不只是当前窗口）做罚分，再 `top_k`
（对原始 logits 排序）→ 除以 `temperature` → `top_p` 截尾 → `min_p` 截尾 → softmax → 逆变换采样。
增量路径每步在 `llm_forward_step` 得到的 logits 上做同一套处理，因此 cached 与整段重算逐位一致。

- `repetition_penalty`：对出现过的 token，`logit > 0` 时除以系数、`logit <= 0` 时乘以系数
  （`== 0` 走乘法分支，结果仍为 0）；`1.0` 表示不惩罚。
- `presence_penalty`：每个出现过的 token 扣一次；`frequency_penalty`：按出现次数扣
  `count * frequency_penalty`。
- `no_repeat_ngram_size = n >= 2`：取最近 `n-1` 个 token 作 prefix，扫描历史中所有相同
  prefix，把其后继 token 的 logit 置 `-INFINITY`；`n = 0` 关闭，`n = 1` 非法。
- `min_p ∈ [0,1)`：先对 `logits/temperature` 做 softmax，`thresh = min_p * max(prob)`，
  概率严格小于阈值的 token 被置 `-INFINITY`；`min_p = 0` 时整段跳过。
- 参数校验（`fprintf(stderr, ...)` + `exit(1)`，与其它入口一致）：`min_p` 必须为
  `[0,1)` 内的有限数、`repetition_penalty > 0`、`presence/frequency` 有限、
  `no_repeat_ngram_size >= 0 且 != 1`。

**跨语言验证（容差 0）**：`tools/llm_reference.py` 新增 6 组 `gen_penalty_cases`，
全部走 `top_k=1`（argmax 确定性，绕开 C/Python RNG 不同源的问题），每组同时导出
`tokens_full` 与 `tokens_cached`（Python 侧先自检两条路径一致）。覆盖
`repetition_penalty` 的 >1 / <1、`presence`、`frequency`、`ngram=2`、以及
`rp+presence+frequency+ngram=3+min_p` 全开组合；其中 `rp=0.5, max_new=8` 与组合用例
会触发窗口滑动。`compare_llm` 第 [7] 段用 `llm_generate_ex` 与 `llm_generate_cached_ex`
分别比 `tokens_full` / `tokens_cached`，第 [8] 段用 `llm_min_p_candidates()` 比 8 个
`min_p` 取值下的候选集合大小（11/5/4/3/3/1/1/1），全部 `[OK] (逐位相同)`。
`llm_apply_generation_penalties()` 与 `llm_min_p_candidates()` 也作为公开诊断函数导出，
`test_llm` 第 [9] 节直接用它们测正/负 logit 两分支、presence 只扣一次、frequency 按次数、
ngram 禁后继、min_p 候选数（含整行 `-inf` 的边界）。第 [11] 节则把编码器第一层的
W_q/W_k 清零、W_v/W_o 设为单位阵，经 `te_set_alibi` 透传后断言注意力权重严格等于
`softmax(slope*(t-j))`（关闭时退化为 `1/(t+1)`），把 TEL/TE 的透传也纳入 C 侧单测。

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
| `attention.c` 自带私有 `TfLayerNorm` / `TfDropout` | Attention 模块先于归一化模块完成；两者缓存布局不同：Tf 版按 `max_seq` 一次缓存所有行的 `x_hat` 供整段反向使用，`layers_norm` 版是单样本 + Module vtable | 保留两份（取舍见"未完成/待办"）；`test_attention` 第 11 节逐位对照两份 LayerNorm，TEL 梯度由 `compare_attention` 覆盖 |
| 注意力权重 dropout 的**随机 mask 本身** | C 侧用 splitmix64 生成 mask（`mha_set_training(1)` 且未注入 mask 时），Python 用 `self.rng.random()`（Mersenne Twister）；两者不可能逐位相同 | `p>0` 且**未注入 mask** 时，C 与 Python 抽到的 mask 不同，因此这一模式**不能**做逐位对比（只保证分布语义相同：丢弃概率 `p`、保留位置乘 `1/(1-p)`）。逐位对拍一律走注入 mask：Python 导出假 RNG 产生的 mask，C 侧 `mha_set_attn_mask()` 灌入（第 8 节，实测 0.000e+00）。`p=0` 或 `training=0` 时完全不生成 mask，默认路径与 Python 逐位一致 |
| **多步训练后的参数轨迹**（`compare_llm` 第 4 段） | 每步 `exp`/`log` 的 1 ULP 差异随步数累积：实测 3 轮后 `L0.attn.b_k` 差 3.7e-13（该参数本身约 1e-11，抵消把相对误差放大）、`out.W` 差 5.6e-17 | 结构错误是 1e-2 级，1e-13 可安全区分；比对程序对该段用 1e-12 容差 |
| RNN 的 step cache 存副本 | Python 的 `_StepCacheMixin` 存的是引用（`self.x = x`） | C 侧存副本，调用方原地改写输入缓冲时不会污染缓存（有意差异） |
| **任意 top_p / 随机采样的 token 不能跨语言逐位比较** | C 用 splitmix64，Python 用 `random.MT19937`；即使 logits 逐位相同，`_sample` 消耗的随机数也不同 | 确定性部分（logits、`perplexity`、`top_p` 只含 argmax 时的 token）已逐位对拍；随机采样只在 C 侧自测（同 seed 可复现、cached 与重算一致） |
| **带罚分/min_p 的随机采样 token 仍不能跨语言比较** | 同上：罚分与 min_p 只改 logits，最后的采样仍消耗不同 RNG | 新增对拍只取 `top_k=1`（argmax 与 RNG 无关）的完整生成序列，以及 `llm_min_p_candidates` 的候选数；这两类都逐位相同。随机路径由 `test_llm_cache` 的 `generate_ex` vs `cached_ex` 覆盖 |
| ALiBi 斜率在**任意头数**下可能差 1 ULP | `slope_h = pow(2, -8*(h+1)/H)`；`H` 不是 2 的幂时指数是无理数，MinGW 与 UCRT 的 `pow` 可能不同 | 对拍用例 `H = 2/4`，斜率恰为 2 的整数次幂，二进制精确，容差 0；其它 `H` 下偏置只进 softmax，误差量级 ~1e-16 |
| 困惑度的 `loss > 700` 保护分支不可达 | `llm_evaluate` 与 Python 的 `evaluate` 一样把单位置概率夹在 `1e-12`，平均 loss 上界 `-log(1e-12) = 27.631021` | 纯防御代码（与 Python 的 `perplexity()` 完全对应），公开 API 无法触发；测试里不做行为断言并注明原因 |
| nucleus 采样的并列概率 | Python 用 `sorted(..., reverse=True)` 的稳定排序（并列时小下标在前）；C 用 O(V²) 的「未保留中概率严格更大」扫描 | 选出的 token 集合完全相同；`top_p < 1` 时多 O(V²) 扫描，`top_p >= 1` 根本进不到该分支（故默认路径逐位一致） |

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

### 输入校验：该报错就报错，不静默

C 侧的错误风格是 `fprintf(stderr, ...)` + `exit(1)`（Python 版抛异常）。两类"报了也白报"的
情况专门补了防护，都是先把错翻成一句话再退出：

| 位置 | 检查 | 为什么必须查 |
|------|------|--------------|
| `check_ids` | 序列长度 1..`max_len`，每个 id 在 `[0, vocab_size)` | 越界会读坏缓冲 |
| `llm_generate_p` / `llm_generate_cached` | `temperature > 0`、`top_p ∈ (0,1]`、**`top_k >= 0`** | `top_k < 0` 原来等价于"不过滤"，调用方以为自己在做 top-k |
| `llm_generate_ex` / `llm_generate_cached_ex` | 除上面三条外：`min_p ∈ [0,1)`、`repetition_penalty > 0`、`presence/frequency` 有限、`no_repeat_ngram_size >= 0 且 != 1` | 非法值会让采样退化（如 `min_p >= 1` 全掩码、`rp <= 0` 除零/翻符号）或让 ngram 规则失去意义；与 Python `_check_min_p` / `_check_penalty_args` 的报错条件一一对应 |
| `llm_loss_and_backward` / `llm_evaluate` | 前向后的 logits 必须全部有限 | 权重混进 NaN/Inf 时，loss 会被夹成 `-log(1e-12)=27.63`，训练照跑、梯度全 NaN，最难查 |
| `softmax_row_inplace` | 整行 `-inf` 时直接返回全 0 | `exp(-inf - -inf)` 是 NaN。C 没有 `pad_mask`，正常路径进不到；与 Python 的 `_softmax_row` 保持同一语义 |

注意这与"默认路径逐位一致"不冲突：上面每条都只在非法输入或原本会出 NaN 的分支上触发，
合法输入的浮点运算顺序一位没动（11 个 `test_*.exe`、`compare_attention` 146 项全部
0.000e+00、`compare_llm` 八段的判定与数值修复前一致，第 4 段仍是 ≤3.7e-13 / 容差 1e-12）。

### 五个注意力扩展与"默认路径不变"

五条扩展都遵守同一条纪律：**不打开就不进新代码路径**，所以默认配置下的运算序列与旧版完全相同
（`compare_llm` 八段的判定就是这条纪律的回归测试）。

- **滑动窗口**（`mha_set_window`）：`window > 0` 时隐含因果——先无条件屏蔽未来，再把
  `j < t - window + 1` 也置 `-INFINITY`；`window <= 0` 时整个分支不进。掩码在 softmax 之前，
  被屏蔽位置权重恰好为 0。增量路径 `mha_forward_step` 只对已见位置打分，再按同一个
  `lo` 屏蔽更早的位置；两条路径都有回归（`compare_attention` 的 window 用例、
  `test_llm_cache` 的滑动用例）。
- **RoPE**（`mha_set_rope`）：GPT-NeoX 版式，`freq_i = 1/(base^(2i/d_k))`，把维度 `i` 与
  `i + d_k/2` 配对。位置编码在**拆头之后、打分之前**就地在 `m->Q` / `m->K` 上施加（与 Python
  缓存旋转后的 `self.Q` 一致），反向在拼回 `dQ_full` / `dK_full` 之前用 `sign = -1` 逆旋转。
  打开 RoPE 时 `llm_set_rope` 同时把 `use_abs_pos` 置 0，`llm_forward` 用 memcpy 跳过
  `PositionalEncoding`（两套位置信息叠加会互相干扰）。
- **GQA / MQA**（`mha_create_gqa`）：`W_k` / `W_v` 变成 `(kv_dim, d_model)`，但前向在拆头时
  把共享 KV 按 `kv_off = (head / kv_group) * d_k` **复制成每头一份**，于是打分与反向代码与普通
  多头完全共用。反向里必须先 `memset` 再对共享头累加（`dK_full` / `dV_full` 是累加缓冲，
  不清零会跨调用累积——这个 bug 曾把梯度放大到发散）。默认 `num_kv_heads == num_heads` 时
  `kv_dim == d_model`，随机初始化序列与旧版相同，因此逐位一致。
- **为什么没有 `mha_set_num_kv_heads`**：`W_k`/`W_v`、`k_full`/`v_full`、`grad_W_k`/`grad_W_v`
  的尺寸都在 create 时定死，事后改要么重分配丢掉训练好的权重，要么静默越界；Python 侧
  `num_kv_heads` 同样是构造参数，所以 C 侧用第二个构造函数。
- **注意力权重 dropout**（`mha_set_training` + `dropout_p`）：只有 `training=1 && p>0` 才进
  新分支，默认 `training=0`（构造即 eval）时 `used == attn`，`m->dropout_active` 为 0，
  `mha_forward`/`mha_backward` 的运算序列与旧版完全相同。`mha_forward_step` 与
  增量路径都不做 dropout（与 Python 的 `forward_step` 一致）。mask 缓冲按
  `(num_heads, max_seq, max_seq)` 一次性预分配（与 `attn`/`d_scores` 同一套 `h*T*T` 布局，
  构造时定死，不动态重分配）；`T < max_seq` 时只用前 `T` 行/列。反向严格照 Python：
  `dV` 用乘过 mask 的 `attn_drop`，`d_attn` 先乘 mask，而 softmax 雅可比的 `s_t` 与
  `attn[t][j]*(...)` 仍用干净 `attn`。跨语言验证走 `mha_set_attn_mask()` 注入，
  不依赖两边 RNG 同源（见第 8 节）。
- **ALiBi**（`mha_set_alibi` / `tel_set_alibi` / `te_set_alibi`）：`m->alibi == 0` 时整段
  跳过；打开后只在 softmax 之前给 `j <= t` 的分数加 `slope_h*(t-j)`（`slope_h` 按
  `2^(-8*(h+1)/num_heads)` 现算），因果/窗口掩码仍在其后。反向无需改动：偏置对分数的导数
  在 softmax 雅可比里抵消，`d_scores` 用含偏置的 `attn` 计算即可。`mha_forward_step`
  只对已见位置（天然 `j <= pos`）加同一偏置，所以增量与整段一致。`test_attention` 第 2c 节
  验证闭式解、关闭退化、增量一致与 TEL/TE 透传；`compare_attention` 的 4 个 ALiBi 用例
  覆盖 RoPE/window 组合，全部 0.000e+00。
- **nucleus 采样**：`llm_generate` 只是一行委托 `llm_generate_p(..., 1.0, ...)`；`top_p >= 1.0`
  时那段代码**整块跳过**（不多抽一次随机数、不做额外运算），这才是"默认逐位一致"的原因，
  而不是"数值上恰好相等"。

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

- `attention.c` 自带的私有 `TfLayerNorm` / `TfDropout` **有意保留两份**：`TfLayerNorm` 按
  `max_seq` 缓存所有 token 的 `x_hat`，配合 `TransformerEncoderLayer` 的整段反向；而
  `layers_norm.h` 的 `LayerNorm` 是单样本 + Module vtable（缓存只有一份 `x_hat`），
  `TfDropout` 也绑定 `(max_seq,d)` 布局，`Dropout` 则是任意 size + `reserve`/`set_mask`。
  要合并就得改结构体和分配路径，属于"可能改变浮点运算顺序"的高风险改动；合并后还必须用
  `compare_attention`（TEL 梯度）与 `compare_norm`（21 项）同时证明 delta 0。当前两份代码
  各自有逐位对拍覆盖，收益不足以抵消风险，因此只记录取舍、不做清理。`test_attention`
  第 11 节现在用同一组 `gamma`/`beta`/`x`/`dout`（d=4/8/16、3 行）逐步比对两份
  LayerNorm 的 forward、dx、`grad_gamma`、`grad_beta`，要求逐位相同（实测 0.000e+00）——
  以后谁只改一份、把运算顺序改出 1 ULP，这个测试会立刻报错。
- 跨语言等价性比对已覆盖归一化 / 卷积 / LLM / MHA+TEL（含 window/RoPE/GQA/MQA）；RNN 仍缺
  `rnn_reference.py` + `compare_rnn.c` 这一组前向/梯度对拍。
- 未做 SIMD / OpenMP（本机 gcc 支持 `-fopenmp`），大矩阵还有数倍空间。
- 增量 KV cache 已落地（第 7 节），并在贪心、top_k/top_p、min_p 与各项罚分
  （`llm_generate_cached_ex` vs `llm_generate_ex`，含窗口滑动）上逐位验证；
  beam search 与 `llm_fit` 不走 cache，暂无计划。
