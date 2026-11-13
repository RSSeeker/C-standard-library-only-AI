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
- 边界：离散动作；训练数据由调用方提供（无 ReplayBuffer）；目标网络按 target_tau 软更新

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
- 边界：只实现确定性的前向/反向/采样，RNG 由调用方提供初值（与 Python 的随机采样入口不同）

### 语音合成（TextToMel + Griffin-Lim 声码器）
- **Griffin-Lim**: `tts_griffin_lim`（由幅度谱迭代估计相位还原波形，窗平方和归一化；`tts_wave_free` 释放）；RNG 为与 Python `random.Random(seed)` 逐位一致的 MT19937
- **Mel 反演**: `tts_mel_to_linear`（Mel 滤波器组的最小二乘逆，负值截零）
- **声学模型**: `text_to_mel_create` / `text_to_mel_forward` / `text_to_mel_param_count`（Embedding + 两次 same-padding Conv1d + ReLU + 线性输出，按 duration 展开 token）
- 边界：TextToMel 只实现推理前向，不实现训练反向；Griffin-Lim 为纯数值算法，不依赖模型结构

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
| `gv_progressive_alpha(step, total, &alpha)` | 淡化系数 `min(1, max(0, step/total))` |
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

API：`sg_create(latent_dim, d_model, base, out_channels)`、`sg_free`、`sg_param_count`、`sg_forward(g, z, out)`、`sg_average_w(g, zs, n, out_w)`、`sg_last_w`。对拍参考由 `tools/stylegan_reference.py` 生成（`StyleGANGenerator(latent_dim=8, d_model=8, base=4)`，参数量 685），`tests/compare_stylegan.c` 比较前向输出与平均 `w`（容差 1e-12）；训练/反向未实现（README 边界）。

### Progressive Growing（逐分辨率增长，前向）

`progressive.*` 提供 ProGAN 风格生成器 `ProgressiveGenerator` 的单向推理：

- 初始分辨率：`z -> fc(relu) -> (c0, base, base)`，再用 1×1 卷积得到低分辨率 RGB；
- 每次 `pg_grow()` 追加一组 `ConvTranspose2d(k=4, s=2, p=1) + ReLU + 1×1 RGB 头`（分辨率翻倍）；
- `pg_forward(g, z, alpha, out)` 中 `alpha<1` 时按 `fade_in(上一分辨率输出, 当前 RGB, alpha)` 平滑混合，最后 `tanh`。

API：`pg_create(latent_dim, channels, n_stages, base, out_channels)`、`pg_free`、`pg_out_shape`、`pg_grow`、`pg_param_count`、`pg_forward`。对拍参考由 `tools/progressive_reference.py` 生成（`latent_dim=8, channels=(8,4), base=2`；参数 818，grow 后 1083），`tests/compare_progressive.c` 比较 base/grow 前向与淡入（容差 1e-12，实测 11/11 逐位相同）；训练/反向未实现。

### CycleGAN（双向循环一致性训练步）

`cyclegan.*` 复用 Pix2Pix 的生成器/判别器，实现一步 CycleGAN 训练：

- 先更新判别器：`D_A` 对 `real_a` 目标 1、对 `fake_a` 目标 0；`D_B` 对 `real_b` 目标 1、对 `fake_b` 目标 0；各自 SGD 更新；
- 再更新生成器：对抗（`D_A(fake_a)`、`D_B(fake_b)` 目标 1）+ 循环一致（`G_BA(G_AB(a))≈a`、`G_AB(G_BA(b))≈b`，权重 `lambda_cycle`）+ identity（`G_AB(b)≈b`、`G_BA(a)≈a`，权重 `lambda_identity`）；
- 参数：`cyclegan_train_step(g_ab, g_ba, d_a, d_b, real_a, real_b, C, H, W, lambda_cycle, lambda_identity, lr_g, lr_d, &step)`，`CycleGanStep` 返回 `d_loss/g_loss/adv_loss/cycle_loss/identity_loss`。

对拍参考由 `tools/cyclegan_reference.py` 生成（两个 `Pix2PixGenerator` + 两个 `PatchDiscriminator`、8×8 图像、一步训练），`tests/compare_cyclegan.c` 比较五项损失、四个网络更新后的全部权重与更新后 fake 输出（容差 1e-9，实测 38/38 逐位相同）。`fit_cyclegan` 的多步循环与随机采样未实现（由调用方组织）。

### CLIP 双塔（相似度 / 对称 InfoNCE / 检索 / 线性投影头）

`clip.*` 实现 Python 的多模态双塔组件：

- `cm_clip_similarity(imgs,n,d,txts,n_txt,d2,sim)`：行归一化后的余弦相似度矩阵；
- `cm_clip_loss(imgs,n,d,txts,d2,temperature,g_img,g_txt)`：对称 InfoNCE（第 i 对为正样本），返回损失并可回填两侧梯度（含归一化反向）；
- `cm_clip_retrieval_accuracy(...)`：图文检索 i2t / t2i 命中率；
- `cm_projector_create/free/param_count/set_weights/project/loss/loss_and_grads/apply_sgd`：`ClipProjector` 两个线性投影头（温度、权重布局 `[embed][输入维]`）。

对拍参考由 `tools/clip_reference.py` 生成（`image_dim=3, text_dim=4, embed_dim=5, temperature=0.07`，参数 45），`tests/compare_clip.c` 比较投影、损失、梯度、SGD 更新后权重、相似度、clip_loss 梯度与检索率（容差 1e-12，实测 147/147 逐位级一致）。边界：只做线性投影头，无图像/文本编码器与预训练权重。

### 分布式训练与低精度数值模拟

`parallel.*` 对应 Python 的分布式训练模拟：

- `par_split_ranges` / `par_fsdp_shard_flat` / `par_fsdp_all_gather`：连续分片与 all-gather；
- `par_zero_step`：ZeRO-1 式「每 rank 只更新自己分片」；
- `par_fsdp_reduce_scatter`：各 rank 梯度求平均后按分片切回；
- `par_tensor_parallel`：Megatron 式列并行 / 行并行线性层前向（mode=0 列切、mode=1 行切）；
- `par_sequence_parallel_layer_norm`：逐 token LayerNorm（序列并行数值对照）。

`quant.*` 对应 Python 的低精度舍入：

- `quant_fp16_round` / `quant_bf16_round` / `quant_fp8_round`（e4m3 / e5m2，饱和处理）；
- `quant_fake_quantize`：对称均匀伪量化（可选自动算 scale，round-half-even）。

对拍参考由 `tools/train_sim_reference.py` 生成，`tests/compare_train_sim.c` 比较分片/更新/并行前向/LayerNorm 与四种舍入及伪量化（容差 1e-12，实测 126/126 逐位相同）。

### LLaMA 式模型（前向）

`llama.*` 把 C 侧既有构件拼成 LLaMA 解码器：

- `LlamaBlock`：`RMSNorm → GQA 因果注意力（RoPE 内置）→ 残差 → RMSNorm → SwiGLU → 残差`（pre-norm）；
- `LlamaModel`：`Embedding → N × LlamaBlock → RMSNorm → 词表头`；
- SwiGLU 单独暴露：`out = (silu(xWg)+bg) ⊙ (xW1+b1) 后接 W2`。

API：`llama_create / llama_free / llama_param_count / llama_forward`；`llama_emb / llama_block / llama_norm / llama_set_head` 用于灌入跨语言权重；`swiglu_create / swiglu_set_params / swiglu_forward / swiglu_param_count`。对拍参考由 `tools/llama_reference.py` 生成（`vocab=6, d_model=8, heads=2, kv_heads=1, d_ff=12, layers=2, max_len=8, rope=True`，参数 1214），`tests/compare_llama.c` 灌入同一组权重后比较前向 logits（容差 1e-12，实测 maxdiff 1.1e-15）。

边界：只实现前向推理，不含反向、Adam 与生成循环；dropout 固定 0；不提供权重初始化等价性（跨语言对拍必须用访问器灌权重）。

### BERT 式模型（前向）

`bert.*` 复用 `Embedding` + `PositionalEncoding` + `TransformerEncoder`，拼出 BERT 式编码器并接两个线性头：

- `bert_forward`：token embedding + segment embedding → 正弦位置编码 → TransformerEncoder，返回 `(T, d_model)` hidden；
- `bert_mlm_logits`：`hidden @ W_mlm^T + b_mlm`，输出 `(T, vocab)`；
- `bert_nsp_logits`：取第 0 个位置过 `W_nsp/b_nsp`，输出长度 2。

对拍参考由 `tools/bert_reference.py` 生成（`vocab=6, d_model=8, heads=2, d_ff=12, layers=2, max_len=8`，参数 1200），`tests/compare_bert.c` 按 `_named_params` 名字灌入权重后比较 hidden/MLM/NSP（容差 1e-12，实测 hidden 与 NSP 逐位相同，MLM 最大差 3.3e-16）。

边界：只实现前向推理；无反向、无训练循环；dropout 固定 0；不包含 SQuAD 等下游任务头。

### ViT 式视觉 Transformer（前向）

`vit.*` 把 C 侧构件拼成视觉 Transformer：

- `PatchEmbedding`：kernel=stride=patch_size 的 Conv2d，把 `(C,H,W)` 切成 `(d_model, gh, gw)` 再按行优先摊平成 token；
- 前置可学习 `cls` + `pos`（num_patches+1 行）；
- `TransformerEncoder` 编码后，仅取 CLS 位置做 LayerNorm，再接线性分类头。

API：`vit_create / vit_free / vit_param_count / vit_forward`；`vit_patch_conv / vit_encoder / vit_norm` 用于灌入跨语言权重，`vit_set_cls_pos / vit_set_head` 设置前置向量与分类头。对拍参考由 `tools/vit_reference.py` 生成（`in_channels=1, image_size=8, patch_size=4, d_model=8, heads=2, d_ff=12, layers=1, num_classes=3`，参数 759），`tests/compare_vit.c` 灌权重后比较 logits（容差 1e-12，实测 maxdiff 8.3e-17）。

边界：只实现前向；无反向、训练与 MAE 预训练；dropout 固定 0；`image_size` 必须能被 `patch_size` 整除。

### ELECTRA 判别器（前向）

`electra.*` 实现 ELECTRA 判别器的前向：Embedding 词表为 `vocab_size+1`（多一行 mask token），
`PositionalEncoding → TransformerEncoder`，最后用两个输出的差值做逐位置二分类打分：

- `score[t] = (b_d[1]-b_d[0]) + Σ_j (W_d[1][j]-W_d[0][j]) * h[t][j]`；
- `prob[t] = sigmoid(score[t])`。

API：`electra_disc_create / electra_disc_free / electra_disc_param_count / electra_disc_forward`；
`electra_disc_embedding / positional / encoder` 与 `electra_disc_set_head` 用于灌入跨语言权重。
对拍参考由 `tools/electra_reference.py` 生成（vocab=6, d=8, heads=2, d_ff=12, layers=2, max_len=8；
参数 1138；输入含 mask token id=6），`tests/compare_electra.c` 比较 score/prob（逐位相同）。

边界：只实现判别器前向；生成器（MaskedLM）训练、BCE 反向与预训练流程未实现；dropout 固定 0。

### RWKV 式模型（前向）

`rwkv.*` 实现 RWKV 语言模型的单向推理（Embedding → N×RWKVBlock → LayerNorm → 词表头）：

- time-mixing：`ln1` 后投影出 r/k/v；逐通道状态 `state = decay*state + k*v`（`decay = exp(-exp(w_raw))`），
  输出 `y = r*(state + u*v)`，再经 `Wo` 加残差；
- channel-mixing：`ln2` 后投影到 `d_ff`，`y2 = r2*relu(k2)^2*v2`，经 `Wo2` 加残差；
- 末尾 `LayerNorm + W_out` 得到 logits。

API：`rwkv_create(vocab_size, d_model, d_ff, num_layers, max_len, seed)`、`rwkv_free`、
`rwkv_param_count`、`rwkv_forward(ids, T, logits)`（`RwkvModel`/`RwkvBlock` 结构体与权重数组公开，
便于灌入 `tools/rwkv_reference.py` 导出的 Python 权重）。对拍参考
（`RWKVModel(vocab=6, d=8, d_ff=12, layers=1, max_len=8, seed=3)`，参数 882）由
`tools/rwkv_reference.py` 生成，`tests/compare_rwkv.c` 灌权重后比较 logits（容差 1e-12；
C 侧 LayerNorm 用 Neumaier 补偿求和，与 Python 普通求和最多差几个 1e-15）。

边界：只实现前向；无反向、训练与生成；dropout 固定 0；序列长度不能超过 `max_len`。

### RetNet 式模型（前向）

`retnet.*` 实现 RetNet 语言模型的单向推理（Embedding → N×RetNetBlock → LayerNorm → 词表头）：

- 保留层（retention）：`ln1` 后投影出 q/k/v；逐头 `gamma_h = 1 - 2^(-5-h)`（固定、不训练），
  `out[t] = Σ_{j<=t} gamma^(t-j)·(q_t·k_j)·v_j / denom_t`，其中 `denom_t = Σ gamma^(t-j)`；
  多头拼接后过 `W_o` 加残差；
- FFN：`ln2 → W1(relu) → W2` 加残差。

API：`retnet_create(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, seed)`、
`retnet_free`、`retnet_param_count`、`retnet_forward(ids, T, logits)`（结构与权重数组公开，
可直接灌入 `tools/retnet_reference.py` 导出的 Python 权重）。对拍参考
（`RetNetModel(vocab=6, d=8, heads=2, d_ff=12, layers=1, max_len=8, seed=3)`，参数 650）由
`tools/retnet_reference.py` 生成，`tests/compare_retnet.c` 比较 logits（容差 1e-12；
C 侧 LayerNorm 用 Neumaier 补偿求和）。

边界：只实现前向；无反向、训练与生成；dropout 固定 0；序列长度不能超过 `max_len`。

### xLSTM 式模型（前向）

`xlstm.*` 实现 xLSTM 语言模型的单向推理（Embedding → N×XLSTMBlock → LayerNorm → 词表头）：

- sLSTM cell（简化版）：`Wg` 投影出 i/f/z/o 四组门；
  `i = exp(clamp(i_pre,-10,5))`、`f = sigmoid(f_pre)`、`z = tanh(z_pre)`、`o = sigmoid(o_pre)`；
  逐通道记忆 `c_t = f*c_prev + i*z`、`n_t = f*n_prev + i`、`h_t = o*c_t/max(|n_t|,eps)`；
- block：`ln1 → sLSTM → 残差`，再 `ln2 → W1(relu) → W2 → 残差`。

API：`xlstm_create(vocab_size, d_model, d_ff, num_layers, max_len, seed)`、`xlstm_free`、
`xlstm_param_count`、`xlstm_forward(ids, T, logits)`（结构与权重数组公开）。对拍参考
（`XLSTMModel(vocab=6, d=8, d_ff=12, layers=1, max_len=8, seed=3)`，参数 722）由
`tools/xlstm_reference.py` 生成，`tests/compare_xlstm.c` 比较 logits（容差 1e-12；
C 侧 LayerNorm 用 Neumaier 补偿求和）。

边界：只实现前向；门只由当前输入计算（不拼接 h_{t-1}）；无反向/训练/生成；dropout 固定 0。

### Mamba 式模型（前向）

`mamba.*` 实现 Mamba 语言模型的单向推理（Embedding → N×MambaBlock → 词表头）：

- block：`in_proj` 投影出 x/z 两路；x 经 `_CausalConv1d`（权重 `[O][C][k]`、左补 `k-1`）与 SiLU 得 u；
- `delta = softplus(u @ delta_proj)`、`B = u @ b_proj`、`C = u @ c_proj`，选择性扫描逐时刻递推
  `h_t = exp(delta_t ⊙ A) ⊙ h_{t-1} + (delta_t ⊙ B_t) ⊙ u_t`、`y_t = C_t · h_t + D ⊙ u_t`；
- 输出 `x + out_proj(y ⊙ SiLU(z))`。

API：`mamba_create(vocab_size, d_model, d_state, d_conv, expand, num_layers, max_len, seed)`、
`mamba_free`、`mamba_param_count`、`mamba_forward(ids, T, logits)`（结构与权重数组公开）。对拍参考
（`MambaModel(vocab=6, d=8, d_state=4, d_conv=4, expand=2, layers=1, max_len=8, seed=3)`，参数 2054）由
`tools/mamba_reference.py` 生成，`tests/compare_mamba.c` 比较 logits（容差 1e-12；
C 侧卷积求和顺序与 Python 略有差异）。

边界：只实现前向；`A` 直接作为参数（不做 `-exp(A_log)` 参数化）；顺序递推而非并行扫描；
无反向/训练/生成；dropout 固定 0。

### DeBERTa 式模型（前向）

`deberta.*` 实现 DeBERTa 风格语言模型的单向推理（Embedding → N×DeBERTaBlock → LayerNorm → 词表头）：

- 单头解耦相对注意力（因果，`j <= t`）：`s(t,j) = scale · (q_t·k_j + q_t·Rk[d] + k_j·Rq[d])`，
  `d = min(t-j, max_rel)`；`Rk` 为 content-to-position 表、`Rq` 为 position-to-content 表；
- block：`ln1 → (Q/K/V + 解耦注意力 + Wo) 残差 → ln2 → W1(relu) → W2 残差`。

API：`deberta_create(vocab_size, d_model, d_ff, num_layers, max_len, max_rel, seed)`、
`deberta_free`、`deberta_param_count`、`deberta_forward(ids, T, logits)`。对拍参考
（`DeBERTaModel(vocab=6, d=8, d_ff=12, layers=2, max_len=8, max_rel=4, seed=21)`，参数 1470）由
`tools/deberta_reference.py` 生成，`tests/compare_deberta.c` 比较 logits（容差 1e-12）。

边界：只实现前向；单头（正式 DeBERTa 为多头），每个相对距离共享一对位置向量；只建模因果的
过去距离；无反向/训练/生成；dropout 固定 0。

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
cd c && make check        # 与 Python 版逐位对拍（文本 41 例 + 音频 15 例 + 视觉 10 例 + CTC 16 例 + ONNX 5 例 + 离线 RL 2 例 + EGNN 1 例 + 投机解码 6 例 + 扩散 17 例 + 整流流 7 例 + 语音合成 1 例 + Pix2Pix 1 例 + 变体 15 例 + StyleGAN 1 例 + ProGAN 1 例 + CycleGAN 1 例 + CLIP 1 例 + 训练模拟 1 例 + LLaMA 1 例 + BERT 1 例 + ViT 1 例 + ELECTRA 1 例 + RWKV 1 例 + RetNet 1 例 + xLSTM 1 例 + Mamba 1 例 + DeBERTa 1 例 + 训练辅助 10 例 + RL/μP 工具 40 例 + SAE 3 例 + 树状注意力 5 例 + 幻觉检测 12 例 + 块分配器 2 例 + KAN 2 例 + FIM/span corruption 68 例 + QK-Norm/DyT 35 例 + RoPE 扩展 96 例 + 线性注意力/长卷积 53 例 + RWKV-5/6/7 time-mixing 34 例等）
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
├── c/                  # 纯 C17 引擎（103 个文件，约 30,300 行）
│   ├── ai.c/.h         #   Layer / MLP / 损失 / 4 种优化器 / 训练循环
│   ├── tensor.c/.h     #   张量基础
│   ├── layers_conv.*   #   Conv2d / MaxPool2d / Flatten
│   ├── layers_norm.*   #   BatchNorm1d / LayerNorm / Dropout
│   ├── qk_norm.*       #   QK-Norm 逐行 RMSNorm 前向/反向 + DynamicTanh 归一化激活层
│   ├── rope_ext.*      #   ABF / LongRoPE 逆频率、2D / 3D RoPE 旋转与逆变换
│   ├── rnn.*           #   RNN / LSTM / GRU（完整 BPTT）
│   ├── attention.*     #   多头注意力 / Transformer
│   ├── linear_attn.*   #   线性注意力（naive / lightning / GLA / DeltaNet）与 Hyena 长卷积
│   ├── ctc_beam.*      #   CTC 前缀 beam search / LM 浅融合解码
│   ├── onnx_mlp.*      #   MLP ↔ ONNX（手写 protobuf 子集：导出/解析/执行）
│   ├── offline_rl.*    #   IQL / CQL 离线强化学习（离散动作，手写反向）
│   ├── egnn.*          #   E(3) 等变图神经网络 EGNN（三张 MLP + 坐标更新）
│   ├── train_utils.*   #   NF4 量化/双重量化 / WeightStandardization / GradScaler / MoE 负载均衡
│   ├── rl_utils.*      #   GRPO 组相对优势 / Best-of-N / 拒绝采样 / 自洽性 / μP 工具
│   ├── sae.*           #   稀疏自编码器（relu 编码 + L1 稀疏 + 手写反向）
│   ├── fim.*           #   FIM / T5 span corruption（目标序列构造与掩码，固定 seed 可复现）
│   ├── tree_attn.*     #   树状注意力：草稿树 trie / 祖先可见偏置 / 布尔掩码
│   ├── hallu.*         #   幻觉检测：余弦一致性 / NLI 三分类 / token 级缺口
│   ├── block_mgr.*     #   PagedAttention 块分配器（fork / COW / swap / 统计）
│   ├── kan.*           #   KAN：B 样条边函数层/多层模型（前向 + 手写反向）
│   ├── medusa.*        #   Medusa 多头 / EAGLE 特征草稿投机解码
│   ├── diffusion.*      #   EDM / DPM-Solver / Consistency 纯数值采样器
│   ├── flowmatching.*   #   Flow Matching（整流流速度场 + Euler/Heun 采样）
│   ├── tts.*            #   语音合成（TextToMel 前向 / Griffin-Lim / Mel 反演）
│   ├── pix2pix.*       #   图像到图像 Pix2Pix（PatchGAN + 卷积/转置卷积生成器）
│   ├── gan_variants.*  #   GAN 辅助：淡入/逐分辨率/AdaIN/截断/循环损失/R1-R2
│   ├── stylegan.*      #   简化 StyleGAN 生成器前向（映射网络 + AdaIN）
│   ├── progressive.*   #   ProGAN 风格逐分辨率增长生成器前向（fade_in）
│   ├── cyclegan.*      #   CycleGAN 一步训练（循环一致性 + identity）
│   ├── clip.*          #   CLIP 双塔相似度 / 对称 InfoNCE / 线性投影头
│   ├── parallel.*      #   分布式训练数值模拟（FSDP / ZeRO / 张量并行 / 序列并行）
│   ├── quant.*         #   fp16 / bf16 / fp8 舍入与对称伪量化
│   ├── llama.*         #   LLaMA 式模型前向（RMSNorm + GQA/RoPE + SwiGLU）
│   ├── bert.*          #   BERT 式模型前向（MLM / NSP 双头）
│   ├── vit.*           #   ViT 式视觉 Transformer 前向
│   ├── electra.*       #   ELECTRA 判别器前向（逐位置二分类）
│   ├── rwkv.*          #   RWKV 式模型前向（时间混合 + 通道混合）
│   ├── rwkv_mix.*      #   RWKV-5/6/7 time-mixing（逐 token 状态递推前向）
│   ├── retnet.*        #   RetNet 式模型前向（多尺度保留 + FFN）
│   ├── xlstm.*         #   xLSTM 式模型前向（sLSTM 记忆 + FFN）
│   ├── mamba.*         #   Mamba 式模型前向（选择性扫描）
│   ├── deberta.*       #   DeBERTa 式模型前向（解耦相对注意力）
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

## 与 Python 版本的功能对照

本项目对应 Python 主仓库（`D:scode\Python-standard-library-only-AI-main`）的 C17 引擎；下表能力均提供 C 实现，并通过 `make check` 与 Python 参考逐位比对（无法逐位一致的浮点仅因 libm 差异，按 JSON 中声明的容差判定）。

- **指标族**：文本 METEOR 完整版/chrF/WER/CER；音频 STOI/PESQ；视觉 LPIPS/FVD；CLIP 双塔相似度/对称 InfoNCE/检索/线性投影头；
- **解码与导出**：CTC 前缀 beam search；MLP → ONNX 导出/解析/执行；Medusa 多头 / EAGLE 特征草稿投机解码；
- **训练与 RL**：IQL/CQL 离线强化学习；E(3) 等变图神经网络（EGNN）；EDM/DPM-Solver/Consistency 扩散采样；Flow Matching 整流流；
- **生成模型**：Pix2Pix 生成器/判别器与训练步；CycleGAN 一步训练；StyleGAN 前向；Progressive Growing 前向；GAN 变体辅助函数与 R1/R2 罚项；
- **训练技术**：FSDP/ZeRO/张量并行/序列并行数值模拟；fp16/bf16/fp8 舍入与对称伪量化；
- **训练辅助**：NF4 量化/双重量化、WeightStandardization（沿 fan_in 标准化，精确反向）、动态 GradScaler（loss scaling 状态机）、MoE 负载均衡损失；
- **归一化扩展**：QK-Norm（逐行 RMSNorm 前向/反向、gamma 梯度）与 DynamicTanh（tanh 缩放激活层的前向/反向/梯度清零/SGD 单步/参数量；`seed` 构造参数在 Python 中未使用故省略，行长不一致/非有限值等校验错误在 C 扁平入参下未对拍）；
- **位置编码扩展**：ABF（base×factor 缩放）与 LongRoPE（低频段逆频率缩放）逆频率计算、2D / 3D RoPE 分轴旋转与逆变换（行主序网格，纯数值，C 扁平入参下行长不一致/非有限值类校验未对拍）；
- **线性注意力与长卷积**：naive O(T²) 展开、lightning 标量衰减递推、GLA（`exp(-exp(g))` 门控）、DeltaNet（beta 门控 delta 规则），以及 Hyena 因果长卷积与逐元素门控（纯数值；`decay**(t-j)` 的 pow 与 CPython UCRT 存在 ≤2 ULP 的 libm 差异，按 1e-12 容差对拍，其余用例逐位相同）；
- **RWKV time-mixing**：RWKV-5 固定逐 channel 衰减 `S=diag(decay)S+k⊗v`、RWKV-6 数据相关衰减（`alpha=exp(-exp(w))`）、RWKV-7 的 delta 修正增量写入（先读旧状态再修正）；纯前向，无反向/生成。`exp(-exp(w))` 走 libm 有 ≤1 ULP 差异，RWKV-6/7 用例按 1e-12 容差对拍，RWKV-5 逐位相同；
- **训练目标**：FIM（Fill-in-the-Middle，PSM/SPM 两种排列）与 T5 span corruption（span 掩码 + 目标序列构造，含 Python `random.Random` 兼容 MT19937；C 侧要求显式 seed，`seed=None` 的随机路径未移植）；
- **RL/μP 工具**：GRPO 组相对优势、Best-of-N、拒绝采样过滤、整数样本自洽性投票；μP 的 target_std / 分层学习率 / MuPAdam 单步更新；
- **可解释性**：稀疏自编码器 SAE（重建 MSE + L1 稀疏、手写反向、特征激活统计）；
- **服务与推理**：树状注意力（共享前缀草稿树、祖先可见偏置、布尔掩码，供投机解码批处理）；
- **块管理**：PagedAttention 块分配器（分配/扩块/释放、fork 共享引用计数、写时复制、swap_out/swap_in、统计）；
- **幻觉检测**：余弦相似度 / 句向量相似度矩阵 / NLI 三分类比例 / token 级支撑检测与报告（纯数值，不调用模型）；
- **语音**：TextToMel 前向、Griffin-Lim 声码器、Mel 反演（含 Python `random.Random` 兼容 MT19937）。
- **KAN**：Kolmogorov-Arnold Network（均匀节点 B 样条边函数、前向 + 手写反向、参数量统计）；
- **模型类（前向）**：LLaMA（RMSNorm + GQA/RoPE + SwiGLU）、BERT（MLM/NSP）、ViT、ELECTRA、RWKV、RetNet、xLSTM、Mamba（选择性扫描）、DeBERTa（解耦相对注意力）。

未包含的部分（可按需扩展）：T5/DETR 等需要解码器/交叉注意力的模型类、真实分布式执行框架、PEFT 包装类（LoRA / QLoRA 等）、实验追踪/GUI 附属组件；新增训练型模块（Mamba-2、DiT、FNO / DeepONet、Graphormer、Jamba、Titans、VMamba、DINOv2 / iBOT）只有 Python 版本；需要文件系统或回调接口的组件（命名数据版本库 VersionedStore、MCTS / Tree-of-Thought、LLM-as-judge 等）以及梯度检查点等训练内存管理能力也未移植。基础组件（GPT/LLM 推理、训练循环等）见上文各节。

## 许可证

MIT License
