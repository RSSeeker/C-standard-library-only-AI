/*
 * onnx_mlp.c —— MLP ⇄ ONNX ModelProto 子集的编码 / 解析 / 执行
 *
 * 编码端与 Python 的 export_onnx_mlp 逐字段同序写入（含节点名、初始化器顺序、
 * 属性顺序），因此同一个 MLP 两侧应得到逐字节一致的字节流。
 * 解析端是通用 protobuf wire 解码（0/1/2/5），只提取本模块用到的子结构。
 */
#include "onnx_mlp.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================== 编码 ===================== */

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
    int err;
} OxBuf;

static int oxb_reserve(OxBuf *b, size_t extra)
{
    if (b->err) return -1;
    if (b->len + extra > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        unsigned char *p;
        while (cap < b->len + extra) cap *= 2;
        p = (unsigned char *)realloc(b->data, cap);
        if (!p) {
            b->err = 1;
            return -1;
        }
        b->data = p;
        b->cap = cap;
    }
    return 0;
}

static int oxb_put(OxBuf *b, const void *src, size_t n)
{
    if (oxb_reserve(b, n) != 0) return -1;
    if (n) memcpy(b->data + b->len, src, n);
    b->len += n;
    return 0;
}

static int oxb_varint(OxBuf *b, unsigned long long v)
{
    unsigned char tmp[10];
    int n = 0;
    do {
        unsigned char byte = (unsigned char)(v & 0x7Fu);
        v >>= 7;
        if (v) byte |= 0x80u;
        tmp[n++] = byte;
    } while (v);
    return oxb_put(b, tmp, (size_t)n);
}

static int oxb_tag(OxBuf *b, int field, int wire)
{
    return oxb_varint(b, ((unsigned long long)field << 3) | (unsigned)wire);
}

static int oxb_int_field(OxBuf *b, int field, long long value)
{
    if (oxb_tag(b, field, 0) != 0) return -1;
    return oxb_varint(b, (unsigned long long)value);
}

static int oxb_bytes_field(OxBuf *b, int field, const unsigned char *data, size_t n)
{
    if (oxb_tag(b, field, 2) != 0) return -1;
    if (oxb_varint(b, (unsigned long long)n) != 0) return -1;
    return oxb_put(b, data, n);
}

static int oxb_str_field(OxBuf *b, int field, const char *s)
{
    return oxb_bytes_field(b, field, (const unsigned char *)s, strlen(s));
}

static int oxb_float32_field(OxBuf *b, int field, double value)
{
    float f = (float)value;
    unsigned char raw[4];
    uint32_t u;
    memcpy(&u, &f, 4);
    raw[0] = (unsigned char)(u & 0xFFu);
    raw[1] = (unsigned char)((u >> 8) & 0xFFu);
    raw[2] = (unsigned char)((u >> 16) & 0xFFu);
    raw[3] = (unsigned char)((u >> 24) & 0xFFu);
    if (oxb_tag(b, field, 5) != 0) return -1;
    return oxb_put(b, raw, 4);
}

static int oxb_float32_array(OxBuf *b, const double *vals, size_t n)
{
    if (oxb_reserve(b, n * 4) != 0) return -1;
    for (size_t i = 0; i < n; i++) {
        float f = (float)vals[i];
        uint32_t u;
        memcpy(&u, &f, 4);
        b->data[b->len++] = (unsigned char)(u & 0xFFu);
        b->data[b->len++] = (unsigned char)((u >> 8) & 0xFFu);
        b->data[b->len++] = (unsigned char)((u >> 16) & 0xFFu);
        b->data[b->len++] = (unsigned char)((u >> 24) & 0xFFu);
    }
    return 0;
}

static void oxb_free(OxBuf *b)
{
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

static const char *ox_activation_op(Activation act, int *ok)
{
    *ok = 1;
    switch (act) {
    case ACT_LINEAR: return "Identity";
    case ACT_RELU: return "Relu";
    case ACT_LEAKY_RELU: return "LeakyRelu";
    case ACT_SOFTMAX: return "Softmax";
    default: *ok = 0; return NULL;
    }
}

/* TensorProto: dims(packed) + data_type=FLOAT + raw_data + name */
static int ox_build_tensor(const char *name, const int *dims, int n_dims,
                           const double *values, int count, OxBuf *out)
{
    OxBuf packed = {0, 0, 0, 0};
    for (int i = 0; i < n_dims; i++) {
        if (oxb_varint(&packed, (unsigned long long)dims[i]) != 0) {
            oxb_free(&packed);
            return -1;
        }
    }
    if (oxb_bytes_field(out, 1, packed.data, packed.len) != 0) {
        oxb_free(&packed);
        return -1;
    }
    oxb_free(&packed);
    if (oxb_int_field(out, 2, 1) != 0) return -1;
    if (oxb_tag(out, 9, 2) != 0) return -1;
    if (oxb_varint(out, (unsigned long long)count * 4u) != 0) return -1;
    if (oxb_float32_array(out, values, (size_t)count) != 0) return -1;
    return oxb_str_field(out, 8, name);
}

/* Dimension: dim_value(int field 1) 或 dim_param(string field 2) */
static int ox_build_dim(OxBuf *out, int is_param, const char *param_name, int value)
{
    if (is_param) return oxb_str_field(out, 2, param_name);
    return oxb_int_field(out, 1, value);
}

/* ValueInfoProto: name + TypeProto{tensor_type{elem_type, shape{dims}}} */
static int ox_build_value_info(const char *name, const char *first_param,
                               int last_dim_value, OxBuf *out)
{
    OxBuf shape = {0, 0, 0, 0};
    OxBuf tensor_type = {0, 0, 0, 0};
    OxBuf type_proto = {0, 0, 0, 0};
    OxBuf dim0 = {0, 0, 0, 0};
    OxBuf dim1 = {0, 0, 0, 0};
    int rc = -1;

    if (ox_build_dim(&dim0, 1, first_param, 0) != 0) goto done;
    if (ox_build_dim(&dim1, 0, NULL, last_dim_value) != 0) goto done;
    if (oxb_bytes_field(&shape, 1, dim0.data, dim0.len) != 0) goto done;
    if (oxb_bytes_field(&shape, 1, dim1.data, dim1.len) != 0) goto done;
    if (oxb_int_field(&tensor_type, 1, 1) != 0) goto done;   /* elem_type = FLOAT */
    if (oxb_bytes_field(&tensor_type, 2, shape.data, shape.len) != 0) goto done;
    if (oxb_bytes_field(&type_proto, 1, tensor_type.data, tensor_type.len) != 0) goto done;
    if (oxb_str_field(out, 1, name) != 0) goto done;
    if (oxb_bytes_field(out, 2, type_proto.data, type_proto.len) != 0) goto done;
    rc = 0;

done:
    oxb_free(&shape);
    oxb_free(&tensor_type);
    oxb_free(&type_proto);
    oxb_free(&dim0);
    oxb_free(&dim1);
    return rc;
}

static int ox_build_attribute_int(const char *name, long long value, OxBuf *out)
{
    if (oxb_str_field(out, 1, name) != 0) return -1;
    if (oxb_int_field(out, 3, value) != 0) return -1;
    return oxb_int_field(out, 20, 2);
}

static int ox_build_attribute_float(const char *name, double value, OxBuf *out)
{
    if (oxb_str_field(out, 1, name) != 0) return -1;
    if (oxb_float32_field(out, 2, value) != 0) return -1;
    return oxb_int_field(out, 20, 1);
}

static int ox_build_node(const char *op_type, const char *const *inputs,
                         int n_inputs, const char *output, const char *name,
                         const OxBuf *attrs, int n_attrs, OxBuf *out)
{
    for (int i = 0; i < n_inputs; i++)
        if (oxb_str_field(out, 1, inputs[i]) != 0) return -1;
    if (oxb_str_field(out, 2, output) != 0) return -1;
    if (oxb_str_field(out, 3, name) != 0) return -1;
    if (oxb_str_field(out, 4, op_type) != 0) return -1;
    for (int i = 0; i < n_attrs; i++)
        if (oxb_bytes_field(out, 5, attrs[i].data, attrs[i].len) != 0) return -1;
    return 0;
}

int onnx_export_mlp(const MLP *net, OnnxMlpBytes *out)
{
    OxBuf nodes_msgs = {0, 0, 0, 0};
    OxBuf inits_msgs = {0, 0, 0, 0};
    OxBuf input_vi = {0, 0, 0, 0};
    OxBuf output_vi = {0, 0, 0, 0};
    OxBuf graph = {0, 0, 0, 0};
    OxBuf opset = {0, 0, 0, 0};
    OxBuf model = {0, 0, 0, 0};
    char current[24];
    int rc = -1;

    if (!net || !out) return -1;
    if (net->n_layers < 1) return -1;
    out->data = NULL;
    out->len = 0;

    snprintf(current, sizeof current, "%s", "input");
    for (int i = 0; i < net->n_layers; i++) {
        const Layer *layer = &net->layers[i];
        char w_name[24], b_name[24], z_name[24], y_name[24];
        char gemm_name[24], act_name[24];
        const int w_dims[2] = {layer->fan_out, layer->fan_in};
        const int b_dims[1] = {layer->fan_out};
        const double *w_flat = layer->W;
        const char *gemm_inputs[3];
        const char *act_inputs[1];
        OxBuf node = {0, 0, 0, 0};
        OxBuf attr_transb = {0, 0, 0, 0};
        OxBuf inits[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        int ok_act = 0;
        const char *act_op;
        int rc_node = -1;

        snprintf(w_name, sizeof w_name, "W%d", i);
        snprintf(b_name, sizeof b_name, "b%d", i);
        snprintf(z_name, sizeof z_name, "z%d", i);
        snprintf(y_name, sizeof y_name, "y%d", i);
        snprintf(gemm_name, sizeof gemm_name, "gemm%d", i);
        snprintf(act_name, sizeof act_name, "act%d", i);

        act_op = ox_activation_op(layer->act, &ok_act);
        if (!ok_act) goto node_done;

        if (ox_build_tensor(w_name, w_dims, 2, w_flat,
                            layer->fan_out * layer->fan_in, &inits[0]) != 0)
            goto node_done;
        if (ox_build_tensor(b_name, b_dims, 1, layer->b, layer->fan_out,
                            &inits[1]) != 0)
            goto node_done;
        if (oxb_bytes_field(&inits_msgs, 5, inits[0].data, inits[0].len) != 0)
            goto node_done;
        if (oxb_bytes_field(&inits_msgs, 5, inits[1].data, inits[1].len) != 0)
            goto node_done;

        if (ox_build_attribute_int("transB", 1, &attr_transb) != 0) goto node_done;
        {
            gemm_inputs[0] = current;
            gemm_inputs[1] = w_name;
            gemm_inputs[2] = b_name;
            if (ox_build_node("Gemm", gemm_inputs, 3, z_name, gemm_name,
                              &attr_transb, 1, &node) != 0)
                goto node_done;
            if (oxb_bytes_field(&nodes_msgs, 1, node.data, node.len) != 0)
                goto node_done;
            node.len = 0;
        }
        {
            OxBuf act_attrs[2] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            int n_act_attrs = 0;
            act_inputs[0] = z_name;
            if (layer->act == ACT_LEAKY_RELU) {
                if (ox_build_attribute_float("alpha", 0.01, &act_attrs[0]) != 0)
                    goto node_done;
                n_act_attrs = 1;
            } else if (layer->act == ACT_SOFTMAX) {
                if (ox_build_attribute_int("axis", -1, &act_attrs[0]) != 0)
                    goto node_done;
                n_act_attrs = 1;
            }
            if (ox_build_node(act_op, act_inputs, 1, y_name, act_name,
                              act_attrs, n_act_attrs, &node) != 0) {
                oxb_free(&act_attrs[0]);
                goto node_done;
            }
            oxb_free(&act_attrs[0]);
            if (oxb_bytes_field(&nodes_msgs, 1, node.data, node.len) != 0)
                goto node_done;
        }
        snprintf(current, sizeof current, "%s", y_name);
        rc_node = 0;

node_done:
        oxb_free(&node);
        oxb_free(&attr_transb);
        oxb_free(&inits[0]);
        oxb_free(&inits[1]);
        if (rc_node != 0) goto done;
    }

    if (oxb_put(&graph, nodes_msgs.data, nodes_msgs.len) != 0) goto done;
    if (oxb_str_field(&graph, 2, "mlp") != 0) goto done;
    if (oxb_put(&graph, inits_msgs.data, inits_msgs.len) != 0) goto done;
    if (ox_build_value_info("input", "batch", net->layers[0].fan_in, &input_vi) != 0)
        goto done;
    if (ox_build_value_info(current, "batch",
                            net->layers[net->n_layers - 1].fan_out,
                            &output_vi) != 0)
        goto done;
    if (oxb_bytes_field(&graph, 11, input_vi.data, input_vi.len) != 0) goto done;
    if (oxb_bytes_field(&graph, 12, output_vi.data, output_vi.len) != 0) goto done;

    if (oxb_str_field(&opset, 1, "") != 0) goto done;
    if (oxb_int_field(&opset, 2, ONNX_MLP_OPSET) != 0) goto done;

    if (oxb_int_field(&model, 1, ONNX_MLP_IR_VERSION) != 0) goto done;
    if (oxb_bytes_field(&model, 2, opset.data, opset.len) != 0) goto done;
    if (oxb_str_field(&model, 3, ONNX_MLP_PRODUCER) != 0) goto done;
    if (oxb_str_field(&model, 4, "1.0") != 0) goto done;
    if (oxb_bytes_field(&model, 7, graph.data, graph.len) != 0) goto done;

    out->data = model.data;
    out->len = model.len;
    model.data = NULL;
    model.len = model.cap = 0;
    rc = 0;

done:
    oxb_free(&nodes_msgs);
    oxb_free(&inits_msgs);
    oxb_free(&input_vi);
    oxb_free(&output_vi);
    oxb_free(&graph);
    oxb_free(&opset);
    oxb_free(&model);
    return rc;
}

int onnx_export_mlp_file(const MLP *net, const char *path)
{
    OnnxMlpBytes bytes = {NULL, 0};
    FILE *f;
    if (!path) return -1;
    if (onnx_export_mlp(net, &bytes) != 0) return -1;
    f = fopen(path, "wb");
    if (!f) {
        onnx_mlp_bytes_free(&bytes);
        return -1;
    }
    if (bytes.len && fwrite(bytes.data, 1, bytes.len, f) != bytes.len) {
        fclose(f);
        onnx_mlp_bytes_free(&bytes);
        return -1;
    }
    fclose(f);
    onnx_mlp_bytes_free(&bytes);
    return 0;
}

void onnx_mlp_bytes_free(OnnxMlpBytes *b)
{
    if (!b) return;
    free(b->data);
    b->data = NULL;
    b->len = 0;
}

/* ===================== 解析 ===================== */

typedef struct {
    const unsigned char *p;
    size_t len;
    size_t pos;
} OxReader;

typedef struct {
    int field;
    int wire;
    unsigned long long u;
    const unsigned char *bytes;
    size_t blen;
    float f32;
} OxField;

static int oxr_varint(OxReader *r, unsigned long long *v)
{
    unsigned long long result = 0;
    int shift = 0;
    while (1) {
        unsigned char byte;
        if (r->pos >= r->len) return -1;
        byte = r->p[r->pos++];
        result |= (unsigned long long)(byte & 0x7Fu) << shift;
        if (!(byte & 0x80u)) {
            *v = result;
            return 0;
        }
        shift += 7;
        if (shift > 63) return -1;
    }
}

static int oxr_next(OxReader *r, OxField *f)
{
    unsigned long long key;
    memset(f, 0, sizeof(*f));
    if (r->pos >= r->len) return 0;
    if (oxr_varint(r, &key) != 0) return -1;
    f->field = (int)(key >> 3);
    f->wire = (int)(key & 7u);
    switch (f->wire) {
    case 0:
        if (oxr_varint(r, &f->u) != 0) return -1;
        return 1;
    case 1:
        if (r->pos + 8 > r->len) return -1;
        r->pos += 8;
        return 1;
    case 2: {
        unsigned long long n;
        if (oxr_varint(r, &n) != 0) return -1;
        if (r->pos + (size_t)n > r->len) return -1;
        f->bytes = r->p + r->pos;
        f->blen = (size_t)n;
        r->pos += (size_t)n;
        return 1;
    }
    case 5: {
        unsigned char raw[4];
        if (r->pos + 4 > r->len) return -1;
        memcpy(raw, r->p + r->pos, 4);
        r->pos += 4;
        {
            uint32_t u = (uint32_t)raw[0] | ((uint32_t)raw[1] << 8)
                         | ((uint32_t)raw[2] << 16) | ((uint32_t)raw[3] << 24);
            memcpy(&f->f32, &u, 4);
        }
        return 1;
    }
    default:
        return -1;
    }
}

static void ox_copy_text(char *dst, size_t cap, const unsigned char *src, size_t n)
{
    if (cap == 0) return;
    if (n >= cap) n = cap - 1;
    if (n && src) memcpy(dst, src, n);
    dst[n] = '\0';
}

static int ox_parse_dim(OxReader r, int *is_param, char *param, size_t param_cap,
                        int *value)
{
    OxField f;
    while (oxr_next(&r, &f) > 0) {
        if (f.field == 1 && f.wire == 0) {
            *is_param = 0;
            *value = (int)f.u;
        } else if (f.field == 2 && f.wire == 2) {
            *is_param = 1;
            ox_copy_text(param, param_cap, f.bytes, f.blen);
        }
    }
    return 0;
}

static int ox_parse_value_info(OxReader r, char *name, size_t name_cap,
                               int *n_dims, int *dims, int *has_param)
{
    OxField f;
    *n_dims = 0;
    *has_param = 0;
    while (oxr_next(&r, &f) > 0) {
        if (f.field == 1 && f.wire == 2) {
            ox_copy_text(name, name_cap, f.bytes, f.blen);
        } else if (f.field == 2 && f.wire == 2) {
            OxReader type_r = {f.bytes, f.blen, 0};
            OxField tf;
            while (oxr_next(&type_r, &tf) > 0) {
                if (tf.field == 1 && tf.wire == 2) {
                    OxReader tensor_r = {tf.bytes, tf.blen, 0};
                    OxField tt;
                    while (oxr_next(&tensor_r, &tt) > 0) {
                        if (tt.field == 2 && tt.wire == 2) {
                            OxReader shape_r = {tt.bytes, tt.blen, 0};
                            OxField sf;
                            while (oxr_next(&shape_r, &sf) > 0) {
                                if (sf.field == 1 && sf.wire == 2) {
                                    OxReader dim_r = {sf.bytes, sf.blen, 0};
                                    int is_param = 0, value = 0;
                                    char param[24] = {0};
                                    ox_parse_dim(dim_r, &is_param, param,
                                                 sizeof param, &value);
                                    if (*n_dims < 2) {
                                        dims[*n_dims] = is_param ? -2 : value;
                                        if (is_param) *has_param = 1;
                                        (*n_dims)++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static OxOpKind ox_op_from_name(const char *op)
{
    if (strcmp(op, "Gemm") == 0) return OX_OP_GEMM;
    if (strcmp(op, "MatMul") == 0) return OX_OP_MATMUL;
    if (strcmp(op, "Add") == 0) return OX_OP_ADD;
    if (strcmp(op, "Identity") == 0) return OX_OP_IDENTITY;
    if (strcmp(op, "Relu") == 0) return OX_OP_RELU;
    if (strcmp(op, "LeakyRelu") == 0) return OX_OP_LEAKY_RELU;
    if (strcmp(op, "Tanh") == 0) return OX_OP_TANH;
    if (strcmp(op, "Sigmoid") == 0) return OX_OP_SIGMOID;
    if (strcmp(op, "Softmax") == 0) return OX_OP_SOFTMAX;
    return OX_OP_UNKNOWN;
}

static int ox_parse_attribute(OxReader r, OxNode *node)
{
    OxField f;
    char name[24] = {0};
    int attr_type = 0;
    long long ival = 0;
    double fval = 0.0;
    int has_int = 0, has_float = 0;
    while (oxr_next(&r, &f) > 0) {
        if (f.field == 1 && f.wire == 2) {
            ox_copy_text(name, sizeof name, f.bytes, f.blen);
        } else if (f.field == 2 && f.wire == 5) {
            has_float = 1;
            fval = (double)f.f32;
        } else if (f.field == 3 && f.wire == 0) {
            has_int = 1;
            ival = (long long)f.u;
        } else if (f.field == 20 && f.wire == 0) {
            attr_type = (int)f.u;
        }
    }
    if (strcmp(name, "transB") == 0 && has_int) {
        node->has_transb = 1;
        node->transb = (int)ival;
    } else if (strcmp(name, "alpha") == 0 && has_float) {
        node->has_alpha = 1;
        node->alpha = fval;
    } else if (strcmp(name, "axis") == 0 && has_int) {
        node->has_axis = 1;
        node->axis = (int)ival;
    }
    (void)attr_type;
    return 0;
}

static int ox_parse_node(OxReader r, OxNode *node)
{
    OxField f;
    memset(node, 0, sizeof(*node));
    node->op = OX_OP_UNKNOWN;
    while (oxr_next(&r, &f) > 0) {
        if (f.field == 1 && f.wire == 2) {
            if (node->n_inputs < 3)
                ox_copy_text(node->inputs[node->n_inputs++], 24,
                             f.bytes, f.blen);
        } else if (f.field == 2 && f.wire == 2) {
            ox_copy_text(node->output, sizeof node->output, f.bytes, f.blen);
        } else if (f.field == 3 && f.wire == 2) {
            ox_copy_text(node->name, sizeof node->name, f.bytes, f.blen);
        } else if (f.field == 4 && f.wire == 2) {
            ox_copy_text(node->op_type, sizeof node->op_type, f.bytes, f.blen);
            node->op = ox_op_from_name(node->op_type);
        } else if (f.field == 5 && f.wire == 2) {
            OxReader ar = {f.bytes, f.blen, 0};
            ox_parse_attribute(ar, node);
        }
    }
    return 0;
}

static int ox_parse_tensor(OxReader r, OxTensor *t)
{
    OxField f;
    int dim_cap = 0;
    memset(t, 0, sizeof(*t));
    while (oxr_next(&r, &f) > 0) {
        if (f.field == 1) {
            if (f.wire == 2) {
                OxReader pr = {f.bytes, f.blen, 0};
                unsigned long long v;
                while (oxr_varint(&pr, &v) == 0) {
                    if (dim_cap < 2) t->dims[dim_cap++] = (int)v;
                    if (pr.pos >= pr.len) break;
                }
            } else if (f.wire == 0) {
                if (dim_cap < 2) t->dims[dim_cap++] = (int)f.u;
            }
        } else if (f.field == 8 && f.wire == 2) {
            ox_copy_text(t->name, sizeof t->name, f.bytes, f.blen);
        } else if (f.field == 9 && f.wire == 2) {
            const size_t count = f.blen / 4;
            t->values = (double *)malloc(sizeof(double) * (count ? count : 1));
            if (!t->values) return -1;
            for (size_t i = 0; i < count; i++) {
                unsigned char raw[4];
                uint32_t u;
                float fv;
                memcpy(raw, f.bytes + i * 4, 4);
                u = (uint32_t)raw[0] | ((uint32_t)raw[1] << 8)
                    | ((uint32_t)raw[2] << 16) | ((uint32_t)raw[3] << 24);
                memcpy(&fv, &u, 4);
                t->values[i] = (double)fv;
            }
            t->count = (int)count;
        }
    }
    t->n_dims = dim_cap;
    return 0;
}

void onnx_mlp_graph_free(OnnxMlpGraph *g)
{
    if (!g) return;
    for (int i = 0; i < g->n_tensors; i++) free(g->tensors[i].values);
    free(g->tensors);
    free(g->nodes);
    memset(g, 0, sizeof(*g));
}

int onnx_mlp_parse(const unsigned char *data, size_t len, OnnxMlpGraph *out)
{
    OxReader top = {data, len, 0};
    OxField f;
    int rc = -1;

    if (!data || !out) return -1;
    memset(out, 0, sizeof(*out));

    while (oxr_next(&top, &f) > 0) {
        if (f.field == 1 && f.wire == 0) {
            out->ir_version = (int)f.u;
        } else if (f.field == 2 && f.wire == 2) {
            OxReader or_ = {f.bytes, f.blen, 0};
            OxField of;
            while (oxr_next(&or_, &of) > 0) {
                if (of.field == 2 && of.wire == 0) out->opset = (int)of.u;
            }
        } else if (f.field == 3 && f.wire == 2) {
            ox_copy_text(out->producer, sizeof out->producer, f.bytes, f.blen);
        } else if (f.field == 4 && f.wire == 2) {
            ox_copy_text(out->producer_version, sizeof out->producer_version,
                         f.bytes, f.blen);
        } else if (f.field == 7 && f.wire == 2) {
            OxReader gr = {f.bytes, f.blen, 0};
            OxField gf;
            int node_cap = 0, tensor_cap = 0;
            while (oxr_next(&gr, &gf) > 0) {
                if (gf.field == 1 && gf.wire == 2) {
                    if (out->n_nodes == node_cap) {
                        int cap = node_cap ? node_cap * 2 : 8;
                        OxNode *nodes = (OxNode *)realloc(
                            out->nodes, sizeof(OxNode) * (size_t)cap);
                        if (!nodes) goto done;
                        out->nodes = nodes;
                        node_cap = cap;
                    }
                    {
                        OxReader nr = {gf.bytes, gf.blen, 0};
                        if (ox_parse_node(nr, &out->nodes[out->n_nodes]) != 0)
                            goto done;
                        out->n_nodes++;
                    }
                } else if (gf.field == 2 && gf.wire == 2) {
                    ox_copy_text(out->graph_name, sizeof out->graph_name,
                                 gf.bytes, gf.blen);
                } else if (gf.field == 5 && gf.wire == 2) {
                    if (out->n_tensors == tensor_cap) {
                        int cap = tensor_cap ? tensor_cap * 2 : 8;
                        OxTensor *ts = (OxTensor *)realloc(
                            out->tensors, sizeof(OxTensor) * (size_t)cap);
                        if (!ts) goto done;
                        out->tensors = ts;
                        tensor_cap = cap;
                    }
                    {
                        OxReader tr = {gf.bytes, gf.blen, 0};
                        if (ox_parse_tensor(tr, &out->tensors[out->n_tensors]) != 0)
                            goto done;
                        out->n_tensors++;
                    }
                } else if (gf.field == 11 && gf.wire == 2) {
                    OxReader vr = {gf.bytes, gf.blen, 0};
                    int has_param = 0;
                    ox_parse_value_info(vr, out->input_name,
                                        sizeof out->input_name,
                                        &out->n_input_dims, out->input_dims,
                                        &has_param);
                } else if (gf.field == 12 && gf.wire == 2) {
                    OxReader vr = {gf.bytes, gf.blen, 0};
                    int has_param = 0;
                    ox_parse_value_info(vr, out->output_name,
                                        sizeof out->output_name,
                                        &out->n_output_dims, out->output_dims,
                                        &has_param);
                }
            }
            rc = 0;
        }
    }

done:
    if (rc != 0) onnx_mlp_graph_free(out);
    return rc;
}

/* ===================== 执行 ===================== */

typedef struct {
    char name[24];
    double *data;
    int rows;
    int cols;
} Val;

typedef struct {
    Val *items;
    int count;
    int cap;
} ValMap;

static Val *val_find(ValMap *m, const char *name)
{
    for (int i = 0; i < m->count; i++) {
        if (strcmp(m->items[i].name, name) == 0) return &m->items[i];
    }
    return NULL;
}

static int val_add(ValMap *m, const char *name, double *data, int rows, int cols)
{
    if (m->count == m->cap) {
        int cap = m->cap ? m->cap * 2 : 16;
        Val *items = (Val *)realloc(m->items, sizeof(Val) * (size_t)cap);
        if (!items) return -1;
        m->items = items;
        m->cap = cap;
    }
    memset(&m->items[m->count], 0, sizeof(Val));
    snprintf(m->items[m->count].name, sizeof m->items[m->count].name, "%s", name);
    m->items[m->count].data = data;
    m->items[m->count].rows = rows;
    m->items[m->count].cols = cols;
    m->count++;
    return 0;
}

static void valmap_free(ValMap *m)
{
    for (int i = 0; i < m->count; i++) free(m->items[i].data);
    free(m->items);
    m->items = NULL;
    m->count = m->cap = 0;
}

int onnx_mlp_forward(const OnnxMlpGraph *g, const double *rows, int n_rows,
                     double **out, int *out_cols)
{
    ValMap vals = {0, 0, 0};
    int in_dim;
    int rc = -1;

    if (!g || !out || !out_cols || n_rows < 1) return -1;
    if (g->n_input_dims < 1) return -1;
    in_dim = g->input_dims[g->n_input_dims - 1];
    if (in_dim < 1) return -1;

    /* 初始化：常量和输入 */
    for (int i = 0; i < g->n_tensors; i++) {
        const OxTensor *t = &g->tensors[i];
        int r = (t->n_dims == 2) ? t->dims[0] : 1;
        int c = (t->n_dims == 2) ? t->dims[1] : t->count;
        double *data = (double *)malloc(sizeof(double) * (size_t)t->count);
        if (!data) goto done;
        memcpy(data, t->values, sizeof(double) * (size_t)t->count);
        if (val_add(&vals, t->name, data, r, c) != 0) {
            free(data);
            goto done;
        }
    }
    {
        double *data = (double *)malloc(sizeof(double) * (size_t)n_rows
                                        * (size_t)in_dim);
        if (!data) goto done;
        memcpy(data, rows, sizeof(double) * (size_t)n_rows * (size_t)in_dim);
        if (val_add(&vals, g->input_name[0] ? g->input_name : "input",
                    data, n_rows, in_dim) != 0) {
            free(data);
            goto done;
        }
    }

    for (int ni = 0; ni < g->n_nodes; ni++) {
        const OxNode *node = &g->nodes[ni];
        Val *a = NULL, *b = NULL, *c = NULL;
        double *res = NULL;
        int r = 0, m = 0;
        if (node->n_inputs < 1) goto done;
        a = val_find(&vals, node->inputs[0]);
        if (!a) goto done;
        if (node->n_inputs > 1) {
            b = val_find(&vals, node->inputs[1]);
            if (!b) goto done;
        }
        if (node->n_inputs > 2) {
            c = val_find(&vals, node->inputs[2]);
            if (!c) goto done;
        }

        switch (node->op) {
        case OX_OP_GEMM: {
            const int n = a->rows, k = a->cols;
            int transb = node->has_transb ? node->transb : 0;
            const int mw = transb ? b->rows : b->cols; /* 输出列数 */
            if (!b || (transb ? b->cols != k : b->rows != k)) goto done;
            r = n;
            m = mw;
            res = (double *)calloc((size_t)r * (size_t)m, sizeof(double));
            if (!res) goto done;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    double acc = 0.0;
                    for (int t = 0; t < k; t++) {
                        const double wv = transb ? b->data[(size_t)j * b->cols + t]
                                                 : b->data[(size_t)t * b->cols + j];
                        acc += a->data[(size_t)i * a->cols + t] * wv;
                    }
                    if (c && j < c->cols) acc += c->data[j];
                    res[(size_t)i * (size_t)m + j] = acc;
                }
            }
            break;
        }
        case OX_OP_MATMUL: {
            const int n = a->rows, k = a->cols;
            if (!b || b->rows != k) goto done;
            r = n;
            m = b->cols;
            res = (double *)calloc((size_t)r * (size_t)m, sizeof(double));
            if (!res) goto done;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++) {
                    double acc = 0.0;
                    for (int t = 0; t < k; t++)
                        acc += a->data[(size_t)i * (size_t)k + t]
                               * b->data[(size_t)t * (size_t)b->cols + j];
                    res[(size_t)i * (size_t)m + j] = acc;
                }
            break;
        }
        case OX_OP_ADD: {
            if (!b) goto done;
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < m; j++) {
                    const double bv = (b->rows == 1) ? b->data[j]
                                                     : b->data[(size_t)i * (size_t)m + j];
                    res[(size_t)i * (size_t)m + j] =
                        a->data[(size_t)i * (size_t)m + j] + bv;
                }
            break;
        }
        case OX_OP_IDENTITY: {
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            memcpy(res, a->data, sizeof(double) * (size_t)r * (size_t)m);
            break;
        }
        case OX_OP_RELU: {
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (size_t i = 0; i < (size_t)r * (size_t)m; i++)
                res[i] = (a->data[i] > 0.0) ? a->data[i] : 0.0;
            break;
        }
        case OX_OP_LEAKY_RELU: {
            const double alpha = node->has_alpha ? node->alpha : 0.01;
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (size_t i = 0; i < (size_t)r * (size_t)m; i++)
                res[i] = (a->data[i] > 0.0) ? a->data[i] : alpha * a->data[i];
            break;
        }
        case OX_OP_TANH: {
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (size_t i = 0; i < (size_t)r * (size_t)m; i++)
                res[i] = tanh(a->data[i]);
            break;
        }
        case OX_OP_SIGMOID: {
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (size_t i = 0; i < (size_t)r * (size_t)m; i++)
                res[i] = 1.0 / (1.0 + exp(-a->data[i]));
            break;
        }
        case OX_OP_SOFTMAX: {
            r = a->rows;
            m = a->cols;
            res = (double *)malloc(sizeof(double) * (size_t)r * (size_t)m);
            if (!res) goto done;
            for (int i = 0; i < r; i++) {
                double mx = a->data[(size_t)i * m];
                double tot = 0.0;
                for (int j = 1; j < m; j++) {
                    const double v = a->data[(size_t)i * m + j];
                    if (v > mx) mx = v;
                }
                for (int j = 0; j < m; j++) {
                    res[(size_t)i * m + j] =
                        exp(a->data[(size_t)i * m + j] - mx);
                    tot += res[(size_t)i * m + j];
                }
                for (int j = 0; j < m; j++) res[(size_t)i * m + j] /= tot;
            }
            break;
        }
        default:
            goto done;
        }

        if (val_add(&vals, node->output, res, r, m) != 0) {
            free(res);
            goto done;
        }
        res = NULL;
    }

    {
        Val *res = val_find(&vals, g->output_name[0] ? g->output_name : "output");
        if (!res) goto done;
        *out = (double *)malloc(sizeof(double) * (size_t)res->rows
                                * (size_t)res->cols);
        if (!*out) goto done;
        memcpy(*out, res->data, sizeof(double) * (size_t)res->rows
                                  * (size_t)res->cols);
        *out_cols = res->cols;
        rc = 0;
    }

done:
    valmap_free(&vals);
    return rc;
}
