/*
 * model_io.h —— 与 Python 版 save_model / load_model 的 JSON 格式互通
 *
 * Python 侧格式（AI.py 的 save_model）：
 * {
 *   "type": "MLP",
 *   "layer_sizes": [2, 4, 1],
 *   "layers": [
 *     {"fan_in":2, "fan_out":4, "is_output":false,
 *      "activation":"leaky_relu", "weights":[[...]], "biases":[...]},
 *     ...
 *   ]
 * }
 *
 * 说明：
 *   - 权重按 Python 的 layer.weights[fan_out][fan_in] 顺序对应 C 侧行主序 W
 *   - 数字用 %.17g 输出，与 Python 的 repr 文本不同但解析回 double 完全一致
 *   - load 时对 activation 缺失/未知的情况按 Python 的旧版兼容策略处理
 */
#ifndef AI_MODEL_IO_H
#define AI_MODEL_IO_H

#include "ai.h"

/* 返回 0 成功，-1 失败（并打印原因） */
int  mlp_save_json(const MLP *m, const char *path);
MLP *mlp_load_json(const char *path);

/* 激活名 <-> 枚举（"leaky_relu"/"relu"/"softmax"/"linear"） */
const char *act_name_of(Activation a);
int         act_from_name(const char *name, Activation *out);

#endif /* AI_MODEL_IO_H */
