/*
 * onnx_mlp.h —— MLP ⇄ ONNX ModelProto 子集（与 Python AI.py 的
 * export_onnx_mlp / onnx_mlp_to_dict / onnx_mlp_forward 对齐）
 *
 * 只覆盖本项目的 MLP：每层一个 Gemm(transB=1) + 一个单目激活
 * （Identity/Relu/LeakyRelu/Tanh/Sigmoid/Softmax），无 Conv/Attention/RNN。
 *
 * 编码是手写的 protobuf 子集（零第三方依赖）：字段按 Python 侧的写入顺序
 * 输出，因此同样的 MLP 在两侧应得到逐字节相同的字节流。
 */
#ifndef ONNX_MLP_H
#define ONNX_MLP_H

#include <stddef.h>

#include "ai.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ONNX_MLP_IR_VERSION 8
#define ONNX_MLP_OPSET 13
#define ONNX_MLP_PRODUCER "python-standard-library-only-ai"

typedef struct {
    unsigned char *data;
    size_t len;
} OnnxMlpBytes;

/* 导出：成功 0 / 失败 -1（不支持的激活、内存不足等） */
int  onnx_export_mlp(const MLP *net, OnnxMlpBytes *out);
int  onnx_export_mlp_file(const MLP *net, const char *path);
void onnx_mlp_bytes_free(OnnxMlpBytes *b);

typedef enum {
    OX_OP_GEMM = 0,
    OX_OP_MATMUL,
    OX_OP_ADD,
    OX_OP_IDENTITY,
    OX_OP_RELU,
    OX_OP_LEAKY_RELU,
    OX_OP_TANH,
    OX_OP_SIGMOID,
    OX_OP_SOFTMAX,
    OX_OP_UNKNOWN
} OxOpKind;

typedef struct {
    OxOpKind op;
    char op_type[20];
    char name[24];
    int n_inputs;
    char inputs[3][24];
    char output[24];
    int has_transb;
    int transb;
    int has_alpha;
    double alpha;
    int has_axis;
    int axis;
} OxNode;

typedef struct {
    char name[24];
    int n_dims;
    int dims[2];
    int count;
    double *values;   /* 扁平行主序 */
} OxTensor;

typedef struct {
    int ir_version;
    int opset;
    char producer[48];
    char producer_version[16];
    char graph_name[24];
    int n_nodes;
    OxNode *nodes;
    int n_tensors;
    OxTensor *tensors;
    char input_name[24];
    int n_input_dims;
    int input_dims[2];       /* 字符串维（batch）记为 -2 */
    char output_name[24];
    int n_output_dims;
    int output_dims[2];
} OnnxMlpGraph;

/* 解析 export_onnx_mlp 产出的字节；成功 0 / 失败 -1 */
int  onnx_mlp_parse(const unsigned char *data, size_t len, OnnxMlpGraph *out);
void onnx_mlp_graph_free(OnnxMlpGraph *g);

/* 执行解析后的图：rows 为 n_rows × 输入维度行主序；输出 malloc 到 *out，
 * 由调用方 free；成功 0 / 失败 -1。 */
int  onnx_mlp_forward(const OnnxMlpGraph *g, const double *rows, int n_rows,
                      double **out, int *out_cols);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_MLP_H */
