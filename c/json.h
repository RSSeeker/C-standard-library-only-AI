/*
 * json.h —— 极简 JSON 读写（只为与 Python 版的存档格式互通）
 *
 * 为什么自己写：Python 版用 json.dump/json.load 存模型，C 侧要能读写同一份文件，
 * 而本项目坚持零第三方依赖，所以自带一个够用的实现。
 *
 * 支持：对象 / 数组 / 字符串（含 \" \\ \/ \b \f \n \r \t \uXXXX）/ 数字 / true / false / null。
 * 不支持：注释、NaN/Infinity、尾随逗号、流式解析——遇到非法输入返回 NULL 并给出错误位置。
 *
 * 数值打印用 %.17g（可无损失往返），Python 用 repr（最短表示）：
 * 两者文本不同，但解析回 double 完全相同。
 */
#ifndef AI_JSON_H
#define AI_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonKind;

typedef struct JsonValue JsonValue;

struct JsonValue {
    JsonKind kind;
    double num;              /* JSON_NUMBER */
    int boolean;             /* JSON_BOOL */
    char *str;               /* JSON_STRING */
    JsonValue **items;       /* JSON_ARRAY */
    int n_items;
    char **keys;             /* JSON_OBJECT */
    JsonValue **values;
    int n_pairs;
};

/* ---------- 解析 ---------- */
/* 失败返回 NULL，并把原因写进 err（err 长度 errlen）；errbuf 可为 NULL */
JsonValue *json_parse(const char *text, char *errbuf, size_t errlen);
JsonValue *json_parse_file(const char *path, char *errbuf, size_t errlen);
void       json_free(JsonValue *v);

/* ---------- 访问（越界/类型不符返回默认值，不报错，便于取可选字段） ---------- */
const JsonValue *json_get(const JsonValue *obj, const char *key);
int              json_array_len(const JsonValue *arr);
const JsonValue *json_at(const JsonValue *arr, int idx);
double           json_num(const JsonValue *v, double dflt);
int              json_int(const JsonValue *v, int dflt);
const char      *json_str(const JsonValue *v, const char *dflt);
int              json_is_num(const JsonValue *v);

/* ---------- 生成 ---------- */
typedef struct {
    char *buf;
    size_t len;
    size_t cap;
    int pretty;              /* 1 = 两空格缩进（对齐 Python 的 indent=2） */
    int depth;
} JsonWriter;

void jw_init(JsonWriter *w, int pretty);
void jw_free(JsonWriter *w);
void jw_raw(JsonWriter *w, const char *text);      /* 追加原文（调用方保证合法） */
void jw_string(JsonWriter *w, const char *s);      /* 带转义的字符串字面量 */
void jw_number(JsonWriter *w, double v);
void jw_int(JsonWriter *w, int v);
void jw_bool(JsonWriter *w, int v);
void jw_null(JsonWriter *w);
void jw_key(JsonWriter *w, const char *key);       /* 写 "key": ，自动补逗号/换行/缩进 */
void jw_begin_object(JsonWriter *w);
void jw_end_object(JsonWriter *w);
void jw_begin_array(JsonWriter *w);
void jw_end_array(JsonWriter *w);
void jw_comma(JsonWriter *w);                      /* 数组元素之间的分隔（对象用 jw_key 自动处理） */
int  jw_write_file(const JsonWriter *w, const char *path);

#endif /* AI_JSON_H */
