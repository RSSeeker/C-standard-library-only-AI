/*
 * compare_visual_metrics.c —— C 的 LPIPS / FVD 与 AI.py 逐位对拍
 *
 * 用法：./tests/compare_visual_metrics [visual_metrics_reference.json]
 * 参考图像以通道主序扁平数组写在 JSON 里；本程序只负责喂给 C 实现并比较。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../visual_metrics.h"

static int g_checks;
static int g_exact;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_visual_metrics] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_visual_metrics] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/visual_metrics_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static VmImage read_image(const JsonValue *v, double **owned)
{
    *owned = NULL;
    const int c = (int)json_num(need(v, "channels", "image"), 0.0);
    const int h = (int)json_num(need(v, "height", "image"), 0.0);
    const int w = (int)json_num(need(v, "width", "image"), 0.0);
    const JsonValue *arr = need(v, "pixels", "image");
    const int n = json_array_len(arr);
    if (c <= 0 || h <= 0 || w <= 0 || n != c * h * w) die("图像字段不一致");
    double *pix = (double *)malloc(sizeof(double) * (size_t)n);
    if (!pix) die("out of memory");
    for (int i = 0; i < n; i++) pix[i] = json_num(json_at(arr, i), NAN);
    *owned = pix;
    VmImage img;
    img.pixels = pix;
    img.channels = c;
    img.height = h;
    img.width = w;
    return img;
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
    const char *path = (argc > 1) ? argv[1] : "visual_metrics_reference.json";
    char err[256] = {0};
    JsonValue *root = json_parse_file(path, err, sizeof err);
    if (!root) {
        fprintf(stderr, "[compare_visual_metrics] 读不了 %s：%s\n"
                        "请先运行 python tools/visual_metrics_reference.py\n", path, err);
        return 2;
    }
    const JsonValue *cases = need(root, "cases", "root");
    const int n_cases = json_array_len(cases);
    printf("[compare_visual_metrics] %s：%d 个用例\n", path, n_cases);

    for (int ci = 0; ci < n_cases; ci++) {
        const JsonValue *cs = json_at(cases, ci);
        const char *name = json_str(json_get(cs, "name"), "?");
        const char *func = json_str(json_get(cs, "func"), "?");
        const double tol = json_num(json_get(cs, "tol"), 0.0);
        const double want = json_num(need(cs, "value", name), NAN);
        printf("\n用例 %-22s func=%s\n", name, func);

        if (strcmp(func, "lpips") == 0) {
            double *pa = NULL, *pb = NULL;
            const VmImage a = read_image(need(cs, "a", name), &pa);
            const VmImage b = read_image(need(cs, "b", name), &pb);
            const JsonValue *js = json_get(cs, "scales");
            int scales[8];
            int n_scales = 0;
            if (js && json_array_len(js) > 0) {
                n_scales = json_array_len(js);
                if (n_scales > (int)(sizeof scales / sizeof scales[0])) die("scales 太多");
                for (int i = 0; i < n_scales; i++) scales[i] = (int)json_num(json_at(js, i), 0.0);
            } else {
                scales[0] = 1;
                scales[1] = 2;
                scales[2] = 4;
                n_scales = 3;
            }
            const JsonValue *jw = json_get(cs, "weights");
            double weights[8];
            const double *wp = NULL;
            if (jw && jw->kind == JSON_ARRAY) {
                const int nw = json_array_len(jw);
                if (nw != n_scales) die("weights 长度与 scales 不一致");
                for (int i = 0; i < nw; i++) weights[i] = json_num(json_at(jw, i), 0.0);
                wp = weights;
            }
            report("lpips", vm_lpips_lite(&a, &b, scales, n_scales, wp, 1e-8),
                   want, tol);
            free(pa);
            free(pb);
        } else if (strcmp(func, "fvd") == 0) {
            const JsonValue *ja = need(cs, "frames_a", name);
            const JsonValue *jb = need(cs, "frames_b", name);
            const int na = json_array_len(ja);
            const int nb = json_array_len(jb);
            if (na < 2 || nb < 2) die("FVD 帧数不足");
            VmImage *fa = (VmImage *)malloc(sizeof(VmImage) * (size_t)na);
            VmImage *fb = (VmImage *)malloc(sizeof(VmImage) * (size_t)nb);
            double **owners_a = (double **)calloc((size_t)na, sizeof(double *));
            double **owners_b = (double **)calloc((size_t)nb, sizeof(double *));
            if (!fa || !fb || !owners_a || !owners_b) die("out of memory");
            for (int i = 0; i < na; i++) {
                fa[i] = read_image(json_at(ja, i), &owners_a[i]);
            }
            for (int i = 0; i < nb; i++) {
                fb[i] = read_image(json_at(jb, i), &owners_b[i]);
            }
            report("fvd", vm_fvd_lite(fa, na, fb, nb, 1e-10), want, tol);
            for (int i = 0; i < na; i++) free(owners_a[i]);
            for (int i = 0; i < nb; i++) free(owners_b[i]);
            free(owners_a);
            free(owners_b);
            free(fa);
            free(fb);
        } else {
            fprintf(stderr, "[compare_visual_metrics] 未知 func \"%s\"（用例 %s）\n",
                    func, name);
            json_free(root);
            return 2;
        }
    }

    json_free(root);
    printf("\n[compare_visual_metrics] 共 %d 项检查：逐位一致 %d，失败 %d，"
           "最大绝对差 %.3e\n", g_checks, g_exact, g_fail, g_worst);
    return g_fail == 0 ? 0 : 1;
}
