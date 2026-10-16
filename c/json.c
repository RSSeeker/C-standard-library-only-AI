/*
 * json.c —— 极简 JSON 读写实现（见 json.h 的说明）
 */
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "json: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static void *xrealloc(void *p, size_t n)
{
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "json: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return q;
}

/* ============================ 解析 ============================ */

typedef struct {
    const char *p;
    char *err;
    size_t errlen;
    int failed;
    int depth;
} Parser;

#define JSON_MAX_DEPTH 64

static void fail(Parser *ps, const char *msg)
{
    if (ps->failed) return;
    ps->failed = 1;
    if (ps->err && ps->errlen) snprintf(ps->err, ps->errlen, "%s (偏移 %ld)", msg, (long)(ps->p - ps->p));
}

static void skip_ws(Parser *ps)
{
    while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r') ps->p++;
}

static JsonValue *parse_value(Parser *ps);

static JsonValue *new_value(JsonKind k)
{
    JsonValue *v = xmalloc(sizeof(JsonValue));
    memset(v, 0, sizeof(*v));
    v->kind = k;
    return v;
}

/* 把 \uXXXX 编码成 UTF-8 追加到 out */
static void append_utf8(char **out, size_t *len, size_t *cap, unsigned cp)
{
    char b[4];
    int n;
    if (cp < 0x80) { b[0] = (char)cp; n = 1; }
    else if (cp < 0x800) {
        b[0] = (char)(0xC0 | (cp >> 6)); b[1] = (char)(0x80 | (cp & 0x3F)); n = 2;
    } else {
        b[0] = (char)(0xE0 | (cp >> 12));
        b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    }
    if (*len + (size_t)n + 1 > *cap) {
        *cap = (*len + (size_t)n + 1) * 2;
        *out = xrealloc(*out, *cap);
    }
    for (int i = 0; i < n; i++) (*out)[(*len)++] = b[i];
    (*out)[*len] = '\0';
}

static int hex4(const char *s, unsigned *out)
{
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static char *parse_string_raw(Parser *ps)
{
    if (*ps->p != '"') { fail(ps, "期望字符串"); return NULL; }
    ps->p++;
    size_t cap = 32, len = 0;
    char *out = xmalloc(cap);
    out[0] = '\0';
    while (*ps->p && *ps->p != '"') {
        if (*ps->p == '\\') {
            ps->p++;
            char c = *ps->p;
            if (c == 'u') {
                unsigned cp;
                if (!hex4(ps->p + 1, &cp)) { fail(ps, "\\u 转义非法"); free(out); return NULL; }
                ps->p += 5;
                /* 代理对 */
                if (cp >= 0xD800 && cp <= 0xDBFF && ps->p[0] == '\\' && ps->p[1] == 'u') {
                    unsigned lo;
                    if (hex4(ps->p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        ps->p += 6;
                    }
                }
                append_utf8(&out, &len, &cap, cp);
                continue;
            }
            ps->p++;
            char d;
            switch (c) {
            case '"': d = '"'; break;
            case '\\': d = '\\'; break;
            case '/': d = '/'; break;
            case 'b': d = '\b'; break;
            case 'f': d = '\f'; break;
            case 'n': d = '\n'; break;
            case 'r': d = '\r'; break;
            case 't': d = '\t'; break;
            default: fail(ps, "非法转义"); free(out); return NULL;
            }
            if (len + 2 > cap) { cap *= 2; out = xrealloc(out, cap); }
            out[len++] = d;
            out[len] = '\0';
            continue;
        }
        if (len + 2 > cap) { cap *= 2; out = xrealloc(out, cap); }
        out[len++] = *ps->p++;
        out[len] = '\0';
    }
    if (*ps->p != '"') { fail(ps, "字符串未闭合"); free(out); return NULL; }
    ps->p++;
    return out;
}

static JsonValue *parse_object(Parser *ps)
{
    JsonValue *v = new_value(JSON_OBJECT);
    int cap = 8;
    v->keys = xmalloc((size_t)cap * sizeof(char *));
    v->values = xmalloc((size_t)cap * sizeof(JsonValue *));
    ps->p++;                     /* { */
    skip_ws(ps);
    if (*ps->p == '}') { ps->p++; return v; }
    for (;;) {
        skip_ws(ps);
        char *key = parse_string_raw(ps);
        if (!key) { json_free(v); return NULL; }
        skip_ws(ps);
        if (*ps->p != ':') { fail(ps, "对象缺少冒号"); free(key); json_free(v); return NULL; }
        ps->p++;
        JsonValue *val = parse_value(ps);
        if (!val) { free(key); json_free(v); return NULL; }
        if (v->n_pairs == cap) {
            cap *= 2;
            v->keys = xrealloc(v->keys, (size_t)cap * sizeof(char *));
            v->values = xrealloc(v->values, (size_t)cap * sizeof(JsonValue *));
        }
        v->keys[v->n_pairs] = key;
        v->values[v->n_pairs] = val;
        v->n_pairs++;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; return v; }
        fail(ps, "对象未闭合");
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_array(Parser *ps)
{
    JsonValue *v = new_value(JSON_ARRAY);
    int cap = 8;
    v->items = xmalloc((size_t)cap * sizeof(JsonValue *));
    ps->p++;                     /* [ */
    skip_ws(ps);
    if (*ps->p == ']') { ps->p++; return v; }
    for (;;) {
        JsonValue *item = parse_value(ps);
        if (!item) { json_free(v); return NULL; }
        if (v->n_items == cap) {
            cap *= 2;
            v->items = xrealloc(v->items, (size_t)cap * sizeof(JsonValue *));
        }
        v->items[v->n_items++] = item;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; skip_ws(ps); continue; }
        if (*ps->p == ']') { ps->p++; return v; }
        fail(ps, "数组未闭合");
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_number(Parser *ps)
{
    char *end = NULL;
    double d = strtod(ps->p, &end);
    if (end == ps->p) { fail(ps, "数字格式非法"); return NULL; }
    ps->p = end;
    JsonValue *v = new_value(JSON_NUMBER);
    v->num = d;
    return v;
}

static JsonValue *parse_value(Parser *ps)
{
    if (++ps->depth > JSON_MAX_DEPTH) { fail(ps, "嵌套过深"); return NULL; }
    skip_ws(ps);
    JsonValue *v = NULL;
    char c = *ps->p;
    if (c == '{') v = parse_object(ps);
    else if (c == '[') v = parse_array(ps);
    else if (c == '"') {
        char *s = parse_string_raw(ps);
        if (s) { v = new_value(JSON_STRING); v->str = s; }
    } else if (c == 't' && strncmp(ps->p, "true", 4) == 0) {
        ps->p += 4; v = new_value(JSON_BOOL); v->boolean = 1;
    } else if (c == 'f' && strncmp(ps->p, "false", 5) == 0) {
        ps->p += 5; v = new_value(JSON_BOOL); v->boolean = 0;
    } else if (c == 'n' && strncmp(ps->p, "null", 4) == 0) {
        ps->p += 4; v = new_value(JSON_NULL);
    } else if (c == '-' || c == '+' || (c >= '0' && c <= '9')) {
        v = parse_number(ps);
    } else {
        fail(ps, "遇到非法字符");
    }
    ps->depth--;
    return v;
}

JsonValue *json_parse(const char *text, char *errbuf, size_t errlen)
{
    Parser ps;
    memset(&ps, 0, sizeof(ps));
    ps.p = text;
    ps.err = errbuf;
    ps.errlen = errlen;
    if (errbuf && errlen) errbuf[0] = '\0';
    JsonValue *v = parse_value(&ps);
    if (!v) return NULL;
    skip_ws(&ps);
    if (*ps.p != '\0') {
        if (errbuf && errlen) snprintf(errbuf, errlen, "JSON 末尾有多余内容");
        json_free(v);
        return NULL;
    }
    return v;
}

JsonValue *json_parse_file(const char *path, char *errbuf, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errbuf && errlen) snprintf(errbuf, errlen, "打不开文件 %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); if (errbuf && errlen) snprintf(errbuf, errlen, "文件长度读取失败"); return NULL; }
    char *buf = xmalloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    JsonValue *v = json_parse(buf, errbuf, errlen);
    free(buf);
    return v;
}

void json_free(JsonValue *v)
{
    if (!v) return;
    free(v->str);
    for (int i = 0; i < v->n_items; i++) json_free(v->items[i]);
    free(v->items);
    for (int i = 0; i < v->n_pairs; i++) {
        free(v->keys[i]);
        json_free(v->values[i]);
    }
    free(v->keys);
    free(v->values);
    free(v);
}

/* ============================ 访问 ============================ */

const JsonValue *json_get(const JsonValue *obj, const char *key)
{
    if (!obj || obj->kind != JSON_OBJECT) return NULL;
    for (int i = 0; i < obj->n_pairs; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->values[i];
    return NULL;
}

int json_array_len(const JsonValue *arr)
{
    return (arr && arr->kind == JSON_ARRAY) ? arr->n_items : 0;
}

const JsonValue *json_at(const JsonValue *arr, int idx)
{
    if (!arr || arr->kind != JSON_ARRAY || idx < 0 || idx >= arr->n_items) return NULL;
    return arr->items[idx];
}

int json_is_num(const JsonValue *v)
{
    return v && v->kind == JSON_NUMBER;
}

double json_num(const JsonValue *v, double dflt)
{
    return (v && v->kind == JSON_NUMBER) ? v->num : dflt;
}

int json_int(const JsonValue *v, int dflt)
{
    return (v && v->kind == JSON_NUMBER) ? (int)(v->num < 0 ? v->num - 0.5 : v->num + 0.5) : dflt;
}

const char *json_str(const JsonValue *v, const char *dflt)
{
    return (v && v->kind == JSON_STRING) ? v->str : dflt;
}

/* ============================ 生成 ============================ */

void jw_init(JsonWriter *w, int pretty)
{
    w->cap = 256;
    w->buf = xmalloc(w->cap);
    w->len = 0;
    w->buf[0] = '\0';
    w->pretty = pretty;
    w->depth = 0;
}

void jw_free(JsonWriter *w)
{
    free(w->buf);
    w->buf = NULL;
    w->len = w->cap = 0;
}

static void jw_reserve(JsonWriter *w, size_t extra)
{
    if (w->len + extra + 1 > w->cap) {
        while (w->len + extra + 1 > w->cap) w->cap *= 2;
        w->buf = xrealloc(w->buf, w->cap);
    }
}

void jw_raw(JsonWriter *w, const char *text)
{
    size_t n = strlen(text);
    jw_reserve(w, n);
    memcpy(w->buf + w->len, text, n);
    w->len += n;
    w->buf[w->len] = '\0';
}

static void jw_indent(JsonWriter *w)
{
    if (!w->pretty) return;
    jw_raw(w, "\n");
    for (int i = 0; i < w->depth; i++) jw_raw(w, "  ");
}

/* 判断当前是否在一个"非空容器"里，需要补逗号 */
static int need_comma(const JsonWriter *w)
{
    if (w->len == 0) return 0;
    char last = w->buf[w->len - 1];
    return !(last == '{' || last == '[' || last == ':' || last == ' ' || last == '\n');
}

void jw_string(JsonWriter *w, const char *s)
{
    jw_raw(w, "\"");
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '"':  jw_raw(w, "\\\""); break;
        case '\\': jw_raw(w, "\\\\"); break;
        case '\b': jw_raw(w, "\\b"); break;
        case '\f': jw_raw(w, "\\f"); break;
        case '\n': jw_raw(w, "\\n"); break;
        case '\r': jw_raw(w, "\\r"); break;
        case '\t': jw_raw(w, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                jw_raw(w, tmp);
            } else {
                /* 直接输出 UTF-8 字节，对应 Python 的 ensure_ascii=False */
                char tmp[2] = { (char)c, '\0' };
                jw_raw(w, tmp);
            }
        }
    }
    jw_raw(w, "\"");
}

void jw_number(JsonWriter *w, double v)
{
    char tmp[40];
    if (v == (double)(long long)v && fabs(v) < 1e15) {
        snprintf(tmp, sizeof(tmp), "%lld", (long long)v);   /* 整数写成 1 而不是 1.0 */
    } else {
        snprintf(tmp, sizeof(tmp), "%.17g", v);
    }
    jw_raw(w, tmp);
}

void jw_int(JsonWriter *w, int v)
{
    char tmp[24];
    snprintf(tmp, sizeof(tmp), "%d", v);
    jw_raw(w, tmp);
}

void jw_bool(JsonWriter *w, int v) { jw_raw(w, v ? "true" : "false"); }
void jw_null(JsonWriter *w)        { jw_raw(w, "null"); }

void jw_key(JsonWriter *w, const char *key)
{
    if (need_comma(w)) jw_raw(w, ",");
    jw_indent(w);
    jw_string(w, key);
    jw_raw(w, w->pretty ? ": " : ":");
}

void jw_begin_object(JsonWriter *w) { jw_raw(w, "{"); w->depth++; }
void jw_end_object(JsonWriter *w)
{
    int empty = (w->len > 0 && w->buf[w->len - 1] == '{');
    w->depth--;
    if (!empty) jw_indent(w);
    jw_raw(w, "}");
}

void jw_begin_array(JsonWriter *w) { jw_raw(w, "["); w->depth++; }
void jw_end_array(JsonWriter *w)
{
    int empty = (w->len > 0 && w->buf[w->len - 1] == '[');
    w->depth--;
    if (!empty) jw_indent(w);
    jw_raw(w, "]");
}

void jw_comma(JsonWriter *w)
{
    jw_raw(w, ",");
}

int jw_write_file(const JsonWriter *w, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(w->buf, 1, w->len, f);
    fputc('\n', f);
    fclose(f);
    return 0;
}
