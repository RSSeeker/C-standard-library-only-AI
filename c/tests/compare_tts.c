/*
 * compare_tts.c —— C 侧 TTS（griffin_lim / mel_to_linear / TextToMel）与 AI.py 对拍
 *
 * 用法：./tests/compare_tts [tts_reference.json]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../json.h"
#include "../tts.h"

static int g_checks;
static int g_fail;
static double g_worst;

static void die(const char *msg)
{
    fprintf(stderr, "[compare_tts] %s\n", msg);
    exit(2);
}

static const JsonValue *need(const JsonValue *o, const char *key, const char *who)
{
    const JsonValue *v = json_get(o, key);
    if (!v) {
        fprintf(stderr, "[compare_tts] %s 缺少字段 \"%s\"，"
                        "请先运行 python tools/tts_reference.py\n", who, key);
        exit(2);
    }
    return v;
}

static double jnum(const JsonValue *v) { return json_num(v, 0.0); }
static int jint(const JsonValue *v) { return json_int(v, 0); }
static const char *jstr(const JsonValue *v) { return json_str(v, ""); }

static double *darr(const JsonValue *v, int *n_out)
{
    int n = v ? json_array_len(v) : 0;
    double *out = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jnum(json_at(v, i));
    if (n_out) *n_out = n;
    return out;
}

static int *iarr(const JsonValue *v, int *n_out)
{
    int n = v ? json_array_len(v) : 0;
    int *out = (int *)malloc(sizeof(int) * (size_t)(n ? n : 1));
    if (!out) die("out of memory");
    for (int i = 0; i < n; i++) out[i] = jint(json_at(v, i));
    if (n_out) *n_out = n;
    return out;
}

static void check(const char *name, int idx, double got, double want, double tol)
{
    g_checks++;
    double diff = fabs(got - want);
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    double rel = diff / scale;
    if (diff > g_worst) g_worst = diff;
    if (rel > tol) {
        g_fail++;
        printf("  %-26s [%d] C=%.17g py=%.17g rel=%.3e [不等]\n",
               name, idx, got, want, rel);
    }
}

static void run_rng(const JsonValue *cases)
{
    int n = json_array_len(cases);
    for (int c = 0; c < n; c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *name = jstr(need(cv, "name", "rng"));
        unsigned int seed = (unsigned int)jint(need(cv, "seed", name));
        const JsonValue *want = need(cv, "expected", name);
        TtsRng rng;
        tts_rng_init(&rng, seed);
        for (int i = 0; i < json_array_len(want); i++)
            check(name, i, tts_rng_random(&rng), jnum(json_at(want, i)), 1e-15);
    }
}

static void run_griffin(const JsonValue *cases)
{
    int n = json_array_len(cases);
    for (int c = 0; c < n; c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *name = jstr(need(cv, "name", "griffin"));
        int mn = 0;
        double *mag = darr(need(cv, "magnitude", name), &mn);
        int n_frames = jint(need(cv, "n_frames", name));
        int n_bins = jint(need(cv, "n_bins", name));
        int n_fft = jint(need(cv, "n_fft", name));
        int hop = jint(need(cv, "hop", name));
        int iterations = jint(need(cv, "iterations", name));
        unsigned int seed = (unsigned int)jint(need(cv, "seed", name));
        double tol = json_num(need(cv, "tol", name), 1e-12);
        double *wave = NULL;
        int wave_len = 0;
        int rc;
        if (mn != n_frames * n_bins) die("magnitude 长度不一致");
        rc = tts_griffin_lim(mag, n_frames, n_bins, n_fft, hop, iterations,
                             seed, NULL, &wave, &wave_len);
        if (rc != 0 || wave == NULL) {
            g_fail++;
            printf("  %-26s tts_griffin_lim 失败 rc=%d\n", name, rc);
        } else {
            const JsonValue *want = need(cv, "expected", name);
            if (json_array_len(want) != wave_len) {
                g_fail++;
                printf("  %-26s 长度不一致 C=%d py=%d\n", name, wave_len,
                       json_array_len(want));
            } else {
                for (int i = 0; i < wave_len; i++)
                    check(name, i, wave[i], jnum(json_at(want, i)), tol);
            }
        }
        tts_wave_free(wave);
        free(mag);
    }
}

static void run_griffin_errors(const JsonValue *cases)
{
    int n = json_array_len(cases);
    for (int c = 0; c < n; c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *name = jstr(need(cv, "name", "griffin_err"));
        int mn = 0;
        double *mag = darr(need(cv, "magnitude", name), &mn);
        double *wave = NULL;
        int wave_len = 0;
        int rc = tts_griffin_lim(mag, jint(need(cv, "n_frames", name)),
                                 jint(need(cv, "n_bins", name)),
                                 jint(need(cv, "n_fft", name)),
                                 jint(need(cv, "hop", name)),
                                 jint(need(cv, "iterations", name)),
                                 (unsigned int)jint(need(cv, "seed", name)),
                                 NULL, &wave, &wave_len);
        g_checks++;
        if (rc == 0) {
            g_fail++;
            printf("  %-26s 期望失败但返回 0\n", name);
        }
        tts_wave_free(wave);
        free(mag);
    }
}

static void run_mel(const JsonValue *cases)
{
    int n = json_array_len(cases);
    for (int c = 0; c < n; c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *name = jstr(need(cv, "name", "mel"));
        int m_n = 0;
        int fb_n = 0;
        double *mel = darr(need(cv, "mel", name), &m_n);
        double *fb = darr(need(cv, "fb", name), &fb_n);
        int n_frames = jint(need(cv, "n_frames", name));
        int n_mels = jint(need(cv, "n_mels", name));
        int n_bins = jint(need(cv, "n_bins", name));
        double eps = jnum(need(cv, "eps", name));
        double tol = json_num(need(cv, "tol", name), 1e-12);
        double *out = (double *)malloc(sizeof(double) * (size_t)n_frames
                                       * (size_t)n_bins);
        if (!out) die("out of memory");
        int rc = tts_mel_to_linear(mel, n_frames, n_mels, fb, n_bins, eps, out);
        if (rc != 0) {
            g_fail++;
            printf("  %-26s tts_mel_to_linear 失败 rc=%d\n", name, rc);
        } else {
            const JsonValue *want = need(cv, "expected", name);
            for (int i = 0; i < n_frames * n_bins; i++)
                check(name, i, out[i], jnum(json_at(want, i)), tol);
        }
        free(out);
        free(mel);
        free(fb);
    }
}

static void run_mel_errors(const JsonValue *cases)
{
    int n = json_array_len(cases);
    for (int c = 0; c < n; c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *name = jstr(need(cv, "name", "mel_err"));
        int m_n = 0;
        int fb_n = 0;
        double *mel = darr(need(cv, "mel", name), &m_n);
        double *fb = darr(need(cv, "fb", name), &fb_n);
        int n_frames = jint(need(cv, "n_frames", name));
        int n_bins = jint(need(cv, "n_bins", name));
        double *out = (double *)malloc(sizeof(double) * (size_t)n_frames
                                       * (size_t)n_bins);
        if (!out) die("out of memory");
        int rc = tts_mel_to_linear(mel, n_frames, jint(need(cv, "n_mels", name)),
                                   fb, n_bins, jnum(need(cv, "eps", name)), out);
        g_checks++;
        if (rc == 0) {
            g_fail++;
            printf("  %-26s 期望失败但返回 0\n", name);
        }
        free(out);
        free(mel);
        free(fb);
    }
}

static void run_text_to_mel(const JsonValue *doc)
{
    const char *name = "text_to_mel";
    int vocab = jint(need(doc, "vocab_size", name));
    int d_model = jint(need(doc, "d_model", name));
    int n_mels = jint(need(doc, "n_mels", name));
    int kernel = jint(need(doc, "kernel_size", name));
    TextToMel *m = text_to_mel_create(vocab, d_model, n_mels, kernel);
    int emb_n = 0, c1w_n = 0, c1b_n = 0, c2w_n = 0, c2b_n = 0;
    int ow_n = 0, ob_n = 0;
    double *emb = NULL;
    double *c1w = NULL;
    double *c1b = NULL;
    double *c2w = NULL;
    double *c2b = NULL;
    double *ow = NULL;
    double *ob = NULL;
    if (!m) die("text_to_mel_create 失败");
    emb = darr(need(doc, "emb", name), &emb_n);
    c1w = darr(need(doc, "c1w", name), &c1w_n);
    c1b = darr(need(doc, "c1b", name), &c1b_n);
    c2w = darr(need(doc, "c2w", name), &c2w_n);
    c2b = darr(need(doc, "c2b", name), &c2b_n);
    ow = darr(need(doc, "ow", name), &ow_n);
    ob = darr(need(doc, "ob", name), &ob_n);
    if (emb_n != vocab * d_model || c1w_n != d_model * d_model * kernel
            || c2w_n != d_model * d_model * kernel || ow_n != n_mels * d_model
            || c1b_n != d_model || c2b_n != d_model || ob_n != n_mels)
        die("TextToMel 权重长度不一致");
    text_to_mel_set_weights(m, emb, c1w, c1b, c2w, c2b, ow, ob);
    g_checks++;
    if (text_to_mel_param_count(m) != jint(need(doc, "param_count", name))) {
        g_fail++;
        printf("  text_to_mel param_count 不一致 C=%d\n",
               text_to_mel_param_count(m));
    }
    const JsonValue *cases = need(doc, "cases", name);
    for (int c = 0; c < json_array_len(cases); c++) {
        const JsonValue *cv = json_at(cases, c);
        const char *cn = jstr(need(cv, "name", name));
        int ids_n = 0, dur_n = 0;
        int *ids = iarr(need(cv, "ids", cn), &ids_n);
        int *dur = iarr(need(cv, "durations", cn), &dur_n);
        double *mel = NULL;
        int frames = 0;
        int rc;
        if (ids_n != dur_n || ids_n < 1) die("ids/durations 长度不一致");
        rc = text_to_mel_forward(m, ids, dur, ids_n, &mel, &frames);
        if (rc != 0 || mel == NULL) {
            g_fail++;
            printf("  %-26s text_to_mel_forward 失败 rc=%d\n", cn, rc);
        } else {
            const JsonValue *want = need(cv, "expected", cn);
            if (json_array_len(want) != frames * n_mels) {
                g_fail++;
                printf("  %-26s 帧数不一致 C=%d\n", cn, frames);
            } else {
                for (int i = 0; i < frames * n_mels; i++)
                    check(cn, i, mel[i], jnum(json_at(want, i)), 1e-12);
            }
        }
        free(mel);
        free(ids);
        free(dur);
    }
    free(emb); free(c1w); free(c1b); free(c2w); free(c2b); free(ow); free(ob);
    text_to_mel_free(m);
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "tts_reference.json";
    char err[256] = {0};
    JsonValue *doc = json_parse_file(path, err, sizeof err);
    if (!doc) {
        fprintf(stderr, "[compare_tts] JSON 解析失败：%s\n", err);
        return 2;
    }
    const JsonValue *rng = json_get(doc, "rng_cases");
    const JsonValue *gr = json_get(doc, "griffin_cases");
    const JsonValue *glerr = json_get(doc, "griffin_errors");
    const JsonValue *mel = json_get(doc, "mel_cases");
    const JsonValue *melerr = json_get(doc, "mel_errors");
    const JsonValue *ttm = json_get(doc, "text_to_mel");
    if (!rng || !gr || !glerr || !mel || !melerr || !ttm)
        die("JSON 缺少顶层字段");
    run_rng(rng);
    run_griffin(gr);
    run_griffin_errors(glerr);
    run_mel(mel);
    run_mel_errors(melerr);
    run_text_to_mel(ttm);
    printf("\n检查 %d 项：失败 %d，最大绝对差 %.3e\n", g_checks, g_fail, g_worst);
    json_free(doc);
    return g_fail ? 1 : 0;
}
