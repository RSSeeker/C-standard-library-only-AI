/*
 * compare_audio_metrics.c —— C 的 STOI / PESQ 与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_audio_metrics [audio_metrics_reference.json]
 *
 * 参考信号数组由 tools/audio_metrics_reference.py 从 AI.py 现场算出并写入
 * JSON；本程序只负责喂给 C 实现并做双精度逐位比较（容差 0）。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../audio_metrics.h"
#include "../json.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_audio_metrics] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_audio_metrics] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/audio_metrics_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

/* JSON 数字数组 -> double 数组（调用方 free） */
static double *nums_from(const JsonValue *v, int *out_n)
{
    if (json_array_len(v) < 0) die("期望数字数组");
    const int n = json_array_len(v);
    double *arr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    if (!arr) die("out of memory");
    for (int i = 0; i < n; i++) {
        arr[i] = json_num(json_at(v, i), NAN);
    }
    *out_n = n;
    return arr;
}

static void report(const char *what, double got, double want, double tol)
{
    g_checks++;
    if (memcmp(&got, &want, sizeof(double)) == 0) {
        g_exact++;
        return;
    }
    const double d = fabs(got - want);
    if (d > g_worst) g_worst = d;
    if (d <= tol) {
        printf("  %-20s C=%.17g py=%.17g 差=%.3e [容差 %.0e 内]\n",
               what, got, want, d, tol);
        return;
    }
    printf("  %-20s C=%.17g py=%.17g 差=%.3e [不等]\n", what, got, want, d);
    g_fail++;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "audio_metrics_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_audio_metrics] 读不了 %s：%s\n"
                        "请先运行 python tools/audio_metrics_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_audio_metrics] %s：%d 个用例\n", path, n_cases);

    for (int c = 0; c < n_cases; c++) {
        const JsonValue *cs = json_at(cases, c);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *func = json_str(json_get(cs, "func"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        const double want = json_num(need(cs, "value", name), NAN);
        int n_clean = 0, n_deg = 0;
        double *clean = nums_from(need(cs, "clean", name), &n_clean);
        double *degraded = nums_from(need(cs, "degraded", name), &n_deg);
        const int sample_rate = (int)json_num(need(cs, "sample_rate", name), 16000.0);
        const int frame_length = (int)json_num(need(cs, "frame_length", name), 256.0);
        const int hop_length = (int)json_num(need(cs, "hop_length", name), 128.0);
        const int n_bands = (int)json_num(need(cs, "n_bands", name), 8.0);
        const double fmin = json_num(need(cs, "fmin", name), 0.0);
        const double fmax = json_num(need(cs, "fmax", name), -1.0);
        const double eps = json_num(need(cs, "eps", name), 1e-12);
        printf("\n用例 %-22s func=%s\n", name, func);

        if (strcmp(func, "stoi") == 0) {
            report("stoi",
                   am_stoi_score(clean, n_clean, degraded, n_deg,
                                 sample_rate, frame_length, hop_length,
                                 n_bands, fmin, fmax, eps),
                   want, tol);
        } else if (strcmp(func, "pesq") == 0) {
            report("pesq",
                   am_pesq_score(clean, n_clean, degraded, n_deg,
                                 sample_rate, frame_length, hop_length,
                                 n_bands, fmin, fmax, eps),
                   want, tol);
        } else {
            fprintf(stderr, "[compare_audio_metrics] 未知 func \"%s\"（用例 %s）\n",
                    func, name);
            free(clean);
            free(degraded);
            json_free(root);
            return 2;
        }
        free(clean);
        free(degraded);
    }

    json_free(root);
    printf("\n[compare_audio_metrics] 共 %d 项检查：逐位一致 %d，失败 %d，最大绝对差 %.3e\n",
           g_checks, g_exact, g_fail, g_worst);
    return g_fail == 0 ? 0 : 1;
}
