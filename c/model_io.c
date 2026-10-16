/*
 * model_io.c —— 与 Python 版 JSON 存档互通的实现
 */
#include "model_io.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *act_name_of(Activation a)
{
    switch (a) {
    case ACT_LEAKY_RELU: return "leaky_relu";
    case ACT_RELU:       return "relu";
    case ACT_SOFTMAX:    return "softmax";
    case ACT_LINEAR:     return "linear";
    }
    return "linear";
}

int act_from_name(const char *name, Activation *out)
{
    if (!name) return 0;
    if (strcmp(name, "leaky_relu") == 0) { *out = ACT_LEAKY_RELU; return 1; }
    if (strcmp(name, "relu") == 0)       { *out = ACT_RELU;       return 1; }
    if (strcmp(name, "softmax") == 0)    { *out = ACT_SOFTMAX;    return 1; }
    if (strcmp(name, "linear") == 0)     { *out = ACT_LINEAR;     return 1; }
    return 0;
}

int mlp_save_json(const MLP *m, const char *path)
{
    JsonWriter w;
    jw_init(&w, 1);

    jw_begin_object(&w);
    jw_key(&w, "type");
    jw_string(&w, "MLP");

    jw_key(&w, "layer_sizes");
    jw_begin_array(&w);
    jw_int(&w, m->layers[0].fan_in);
    for (int i = 0; i < m->n_layers; i++) {
        jw_comma(&w);
        jw_int(&w, m->layers[i].fan_out);
    }
    jw_end_array(&w);

    jw_key(&w, "layers");
    jw_begin_array(&w);
    for (int i = 0; i < m->n_layers; i++) {
        const Layer *l = &m->layers[i];
        if (i) jw_comma(&w);
        jw_begin_object(&w);

        jw_key(&w, "fan_in");   jw_int(&w, l->fan_in);
        jw_key(&w, "fan_out");  jw_int(&w, l->fan_out);
        jw_key(&w, "is_output"); jw_bool(&w, i == m->n_layers - 1);
        jw_key(&w, "activation"); jw_string(&w, act_name_of(l->act));

        jw_key(&w, "weights");
        jw_begin_array(&w);
        for (int j = 0; j < l->fan_out; j++) {
            if (j) jw_comma(&w);
            jw_begin_array(&w);
            for (int k = 0; k < l->fan_in; k++) {
                if (k) jw_comma(&w);
                jw_number(&w, l->W[(size_t)j * l->fan_in + k]);
            }
            jw_end_array(&w);
        }
        jw_end_array(&w);

        jw_key(&w, "biases");
        jw_begin_array(&w);
        for (int j = 0; j < l->fan_out; j++) {
            if (j) jw_comma(&w);
            jw_number(&w, l->b[j]);
        }
        jw_end_array(&w);

        jw_end_object(&w);
    }
    jw_end_array(&w);
    jw_end_object(&w);

    int rc = jw_write_file(&w, path);
    jw_free(&w);
    if (rc != 0) {
        fprintf(stderr, "mlp_save_json: 写入 %s 失败\n", path);
        return -1;
    }
    return 0;
}

MLP *mlp_load_json(const char *path)
{
    char err[256];
    JsonValue *root = json_parse_file(path, err, sizeof(err));
    if (!root) {
        fprintf(stderr, "mlp_load_json: %s 解析失败: %s\n", path, err);
        return NULL;
    }
    if (root->kind != JSON_OBJECT) {
        fprintf(stderr, "mlp_load_json: 顶层不是对象\n");
        json_free(root);
        return NULL;
    }

    const char *type = json_str(json_get(root, "type"), "");
    if (strcmp(type, "MLP") != 0) {
        fprintf(stderr, "mlp_load_json: 不支持的类型 \"%s\"\n", type);
        json_free(root);
        return NULL;
    }

    const JsonValue *sizes_j = json_get(root, "layer_sizes");
    int n_sizes = json_array_len(sizes_j);
    if (n_sizes < 2) {
        fprintf(stderr, "mlp_load_json: layer_sizes 至少要有 2 项\n");
        json_free(root);
        return NULL;
    }
    const JsonValue *layers_j = json_get(root, "layers");
    int n_layers = json_array_len(layers_j);
    if (n_layers != n_sizes - 1) {
        fprintf(stderr, "mlp_load_json: layers 数量 %d 与 layer_sizes %d 不匹配\n",
                n_layers, n_sizes);
        json_free(root);
        return NULL;
    }

    int *sizes = malloc((size_t)n_sizes * sizeof(int));
    Activation *acts = malloc((size_t)n_layers * sizeof(Activation));
    if (!sizes || !acts) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int i = 0; i < n_sizes; i++) sizes[i] = json_int(json_at(sizes_j, i), 0);

    for (int i = 0; i < n_layers; i++) {
        const JsonValue *ld = json_at(layers_j, i);
        const char *an = json_str(json_get(ld, "activation"), NULL);
        Activation a;
        if (an && act_from_name(an, &a)) {
            acts[i] = a;
        } else {
            /* 与 Python 的旧版兼容策略一致 */
            int fan_out = json_int(json_get(ld, "fan_out"), 0);
            if (fan_out == sizes[n_sizes - 1] && fan_out > 1) acts[i] = ACT_LINEAR;
            else acts[i] = ACT_LEAKY_RELU;
        }
    }

    MLP *m = mlp_create(sizes, acts, n_sizes);

    for (int i = 0; i < n_layers; i++) {
        const JsonValue *ld = json_at(layers_j, i);
        Layer *l = &m->layers[i];
        if (json_int(json_get(ld, "fan_in"), -1) != l->fan_in ||
            json_int(json_get(ld, "fan_out"), -1) != l->fan_out) {
            fprintf(stderr, "mlp_load_json: 第 %d 层维度与 layer_sizes 不一致\n", i);
            free(sizes); free(acts); mlp_free(m); json_free(root);
            return NULL;
        }
        const JsonValue *wj = json_get(ld, "weights");
        const JsonValue *bj = json_get(ld, "biases");
        if (json_array_len(wj) != l->fan_out || json_array_len(bj) != l->fan_out) {
            fprintf(stderr, "mlp_load_json: 第 %d 层 weights/biases 形状不对\n", i);
            free(sizes); free(acts); mlp_free(m); json_free(root);
            return NULL;
        }
        for (int j = 0; j < l->fan_out; j++) {
            const JsonValue *row = json_at(wj, j);
            if (json_array_len(row) != l->fan_in) {
                fprintf(stderr, "mlp_load_json: 第 %d 层第 %d 行长度不对\n", i, j);
                free(sizes); free(acts); mlp_free(m); json_free(root);
                return NULL;
            }
            for (int k = 0; k < l->fan_in; k++)
                l->W[(size_t)j * l->fan_in + k] = json_num(json_at(row, k), 0.0);
            l->b[j] = json_num(json_at(bj, j), 0.0);
        }
    }

    free(sizes);
    free(acts);
    json_free(root);
    return m;
}
