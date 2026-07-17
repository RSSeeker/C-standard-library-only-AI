#include "AI.hpp"
#include <windows.h>
#include <commctrl.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <thread>
#include <mutex>

using namespace ai;

// ============ 控件 ID ============
#define IDC_BTN_REGRESSION    2001
#define IDC_BTN_CLASSIFY      2002
#define IDC_BTN_SAVELOAD      2003
#define IDC_BTN_GRADCLIP      2004
#define IDC_BTN_QUICKTRAIN    2005
#define IDC_BTN_CNN           2006
#define IDC_BTN_RUNALL        2007
#define IDC_BTN_CLEAR         2008
#define IDC_BTN_REG_PREDICT   2009
#define IDC_BTN_CLS_PREDICT   2010
#define IDC_OUTPUT            3001
#define IDC_EDIT_REG_INPUT    3002
#define IDC_EDIT_CLS_INPUT    3003
#define IDC_STATIC_REG        3004
#define IDC_STATIC_CLS        3005

// ============ 全局状态 ============
HWND g_hOutput   = nullptr;
HWND g_hStatus   = nullptr;
HINSTANCE g_hInst = nullptr;
HFONT   g_hFontOutput = nullptr;
HFONT   g_hFontUI     = nullptr;

// 保存训练好的模型供交互预测
Model g_reg_model;
Model g_cls_model;
bool  g_reg_trained = false;
bool  g_cls_trained = false;

std::ostringstream g_capture;
std::streambuf*    g_oldBuf = nullptr;

void BeginCapture() {
    g_capture.str("");
    g_capture.clear();
    g_oldBuf = std::cout.rdbuf(g_capture.rdbuf());
}
std::string EndCapture() {
    if (g_oldBuf) std::cout.rdbuf(g_oldBuf);
    return g_capture.str();
}

void AppendText(HWND hEdit, const std::string& text) {
    int len = GetWindowTextLengthA(hEdit);
    SendMessageA(hEdit, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageA(hEdit, EM_REPLACESEL, FALSE, (LPARAM)text.c_str());
}

void SetStatus(const std::string& s) {
    SetWindowTextA(g_hStatus, s.c_str());
}

// ============ Demo 实现 ============
static std::vector<Vec> g_x_reg = {
    {0.2, 0.4, 0.6, 0.8, 1.0}, {0.4, 0.6, 0.8, 1.0, 1.2},
    {0.6, 0.8, 1.0, 1.2, 1.4}, {0.1, 0.3, 0.5, 0.7, 0.9},
    {0.3, 0.5, 0.7, 0.9, 1.1},
};
static std::vector<Vec> g_y_reg = {
    {0.04, 0.08}, {0.16, 0.24}, {0.36, 0.48}, {0.01, 0.02}, {0.09, 0.18}
};
static std::vector<Vec> g_x_cls = {
    {0.1,0.2,0.3},{0.4,0.5,0.6},{0.7,0.8,0.9},
    {0.15,0.25,0.35},{0.45,0.55,0.65},{0.75,0.85,0.95},
};
static std::vector<int> g_y_cls = {0,1,2,0,1,2};

void DemoRegression() {
    seed(42);
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  Regression: y = x^2 / 5  (make_regressor + fit)\n";
    std::cout << "============================================================\n";
    g_reg_model = make_regressor(5, 2, {6,6,6,6,5,4,3}, "leaky_relu", 0.01);
    g_reg_model.summary();
    g_reg_model.fit(g_x_reg, g_y_reg, 2000, 2, 0.2, nullptr, 50);
    std::cout << "\n--- Regression Test ---\n";
    for (size_t i = 0; i < g_x_reg.size(); ++i) {
        Vec pred = g_reg_model.predict_one(g_x_reg[i]);
        std::cout << "  input: (" << g_x_reg[i][0];
        for (size_t j = 1; j < g_x_reg[i].size(); ++j)
            std::cout << ", " << g_x_reg[i][j];
        std::cout << ")  target: (" << g_y_reg[i][0] << ", " << g_y_reg[i][1]
                  << ")  pred: (" << pred[0] << ", " << pred[1] << ")\n";
    }
    double r2 = g_reg_model.score(g_x_reg, g_y_reg);
    std::cout << "  R^2 = " << r2 << "\n";
    std::string out = EndCapture();
    AppendText(g_hOutput, out);
    g_reg_trained = true;
    SetStatus("Regression completed, R^2 = " + std::to_string(r2));
}

void DemoClassification() {
    seed(42);
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  Classification: Softmax + CrossEntropy\n";
    std::cout << "============================================================\n";
    g_cls_model = make_classifier(3, 3, {8,8}, "leaky_relu", 0.01);
    g_cls_model.summary();
    g_cls_model.fit(g_x_cls, g_y_cls, 2000, 2, 0.2, 50);
    std::cout << "\n--- Classification Test ---\n";
    int ok = 0;
    for (size_t i = 0; i < g_x_cls.size(); ++i) {
        int pred = g_cls_model.predict_class(g_x_cls[i]);
        bool hit = (pred == g_y_cls[i]);
        if (hit) ++ok;
        std::cout << "  pred: " << pred << "  actual: " << g_y_cls[i]
                  << "  " << (hit ? "OK" : "FAIL") << "\n";
    }
    double acc = 100.0 * ok / g_x_cls.size();
    std::cout << "  Accuracy: " << acc << "%\n";
    std::string out = EndCapture();
    AppendText(g_hOutput, out);
    g_cls_trained = true;
    SetStatus("Classification completed, accuracy = " + std::to_string(acc) + "%");
}

void DemoSaveLoad() {
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  Save & Load\n============================================================\n";
    if (!g_cls_trained) {
        std::cout << "  Please run Classification first!\n";
        AppendText(g_hOutput, EndCapture());
        SetStatus("Save/Load failed: no trained model");
        return;
    }
    g_cls_model.save("cls_model.json");
    Model loaded = Model::load("cls_model.json", "cross_entropy", "adam");
    int ok = 0;
    for (size_t i = 0; i < g_x_cls.size(); ++i) {
        int pred = loaded.predict_class(g_x_cls[i]);
        bool hit = (pred == g_y_cls[i]);
        if (hit) ++ok;
        std::cout << "  pred: " << pred << "  actual: " << g_y_cls[i]
                  << "  " << (hit ? "OK" : "FAIL") << "\n";
    }
    std::cout << "  Accuracy (loaded): " << 100.0 * ok / g_x_cls.size() << "%\n";
    AppendText(g_hOutput, EndCapture());
    SetStatus("Save & Load verified");
}

void DemoGradClip() {
    seed(42);
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  Gradient Clipping + StepLR\n============================================================\n";
    Model model = make_regressor(5, 2, {8,8,4}, "leaky_relu", 0.01);
    StepLR scheduler(model.optimizer.get(), 100, 0.5);
    model.fit(g_x_reg, g_y_reg, 300, 2, 0.0, nullptr, 0, true, true, &scheduler, 1.0);
    std::cout << "  Final loss: " << model.train_history.back().second
              << "  |  LR: " << model.optimizer->lr << "\n";
    std::cout << "  R^2: " << model.score(g_x_reg, g_y_reg) << "\n";
    AppendText(g_hOutput, EndCapture());
    SetStatus("Gradient Clipping demo done");
}

void DemoQuickTrain() {
    seed(42);
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  quick_train: one-liner\n============================================================\n";
    Model qm_reg = quick_train(g_x_reg, g_y_reg, "regression", {16,8}, 300, 2);
    std::cout << "  Regression R^2 = " << qm_reg.score(g_x_reg, g_y_reg) << "\n";
    Model qm_cls = quick_train(g_x_cls, g_y_cls, {16}, 300, 2);
    std::cout << "  Classification acc = " << qm_cls.score(g_x_cls, g_y_cls) * 100 << "%\n";
    AppendText(g_hOutput, EndCapture());
    SetStatus("QuickTrain demo done");
}

void DemoCNN() {
    seed(42);
    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  CNN Demo\n============================================================\n";
    Conv2d conv(1, 2, 2, 1);
    MaxPool2d pool(2);
    Flatten flatten;
    MLP fc({2, 4, 2}, std::vector<std::string>{"leaky_relu", "softmax"});
    Vec3D img(1, Vec2D(4, Vec(4, 0.0)));
    std::cout << "  Input image (1x4x4):\n";
    for (int h = 0; h < 4; ++h) {
        std::cout << "    ";
        for (int w = 0; w < 4; ++w) {
            img[0][h][w] = randu();
            std::cout << std::fixed << std::setprecision(2) << img[0][h][w] << " ";
        }
        std::cout << "\n";
    }
    Vec3D c = conv.forward(img);
    Vec3D p = pool.forward(c);
    Vec flat = flatten.forward(p);
    Vec out = fc.forward(flat);
    std::cout << "  Image(1x4x4) -> Conv(2,2,2) -> Pool(2) -> Flatten -> MLP(2->4->2)\n";
    std::cout << "  Output: (" << out[0] << ", " << out[1] << ")\n";
    AppendText(g_hOutput, EndCapture());
    SetStatus("CNN demo done");
}

void DemoRunAll() {
    AppendText(g_hOutput, "\r\n========== RUN ALL DEMOS ==========\r\n\r\n");
    DemoRegression();
    DemoClassification();
    DemoSaveLoad();
    DemoGradClip();
    DemoQuickTrain();
    DemoCNN();
    AppendText(g_hOutput, "\r\n========== ALL DONE ==========\r\n\r\n");
    SetStatus("All demos completed");
}

void DemoClear() {
    SetWindowTextA(g_hOutput, "");
    SetStatus("Ready");
}

// ============ 交互预测 ============
void RegPredict() {
    if (!g_reg_trained) {
        AppendText(g_hOutput, "\r\n[!] Please run Regression demo first!\r\n");
        SetStatus("No trained regression model");
        return;
    }
    char buf[256] = {};
    GetWindowTextA(GetDlgItem(GetParent(g_hOutput), IDC_EDIT_REG_INPUT), buf, 256);
    std::string s(buf);
    if (s.empty()) {
        AppendText(g_hOutput, "\r\n[!] Enter 5 comma-separated values in the regression input box.\r\n");
        return;
    }
    Vec vals;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { vals.push_back(std::stod(token)); }
        catch (...) { break; }
    }
    if (vals.size() < 5) {
        AppendText(g_hOutput, "\r\n[!] Need exactly 5 numbers. Got " + std::to_string(vals.size()) + ".\r\n");
        return;
    }
    vals.resize(5);
    Vec pred = g_reg_model.predict_one(vals);
    std::ostringstream oss;
    oss << "\r\n[Reg Predict] input: (" << vals[0];
    for (size_t i = 1; i < 5; ++i) oss << ", " << vals[i];
    oss << ")  ->  (" << pred[0] << ", " << pred[1] << ")\r\n";
    AppendText(g_hOutput, oss.str());
    SetStatus("Prediction done");
}

void ClsPredict() {
    if (!g_cls_trained) {
        AppendText(g_hOutput, "\r\n[!] Please run Classification demo first!\r\n");
        SetStatus("No trained classification model");
        return;
    }
    char buf[256] = {};
    GetWindowTextA(GetDlgItem(GetParent(g_hOutput), IDC_EDIT_CLS_INPUT), buf, 256);
    std::string s(buf);
    if (s.empty()) {
        AppendText(g_hOutput, "\r\n[!] Enter 3 comma-separated values in the classification input box.\r\n");
        return;
    }
    Vec vals;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { vals.push_back(std::stod(token)); }
        catch (...) { break; }
    }
    if (vals.size() < 3) {
        AppendText(g_hOutput, "\r\n[!] Need exactly 3 numbers. Got " + std::to_string(vals.size()) + ".\r\n");
        return;
    }
    vals.resize(3);
    int cls = g_cls_model.predict_class(vals);
    std::ostringstream oss;
    oss << "\r\n[Cls Predict] input: (" << vals[0] << ", " << vals[1] << ", " << vals[2]
        << ")  ->  class " << cls << "\r\n";
    AppendText(g_hOutput, oss.str());
    SetStatus("Prediction done");
}

// ============ 窗口过程 ============
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        HINSTANCE hi = ((LPCREATESTRUCT)lp)->hInstance;

        // ---- 左侧按钮面板背景 ----
        // 按钮
        auto btn = [&](const char* text, int id, int y, int w, int h) {
            CreateWindowA("BUTTON", text,
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON | BS_FLAT,
                10, y, w, h, hwnd, (HMENU)(INT_PTR)id, hi, nullptr);
        };

        int xLeft = 10, btnW = 150, btnH = 32, gap = 6, y = 10;
        btn("Regression",       IDC_BTN_REGRESSION,  y, btnW, btnH); y += btnH + gap;
        btn("Classification",   IDC_BTN_CLASSIFY,     y, btnW, btnH); y += btnH + gap;
        btn("Save & Load",      IDC_BTN_SAVELOAD,     y, btnW, btnH); y += btnH + gap;
        btn("Gradient Clip",    IDC_BTN_GRADCLIP,     y, btnW, btnH); y += btnH + gap;
        btn("Quick Train",      IDC_BTN_QUICKTRAIN,   y, btnW, btnH); y += btnH + gap;
        btn("CNN",              IDC_BTN_CNN,          y, btnW, btnH); y += btnH + gap;
        y += 8;
        btn("RUN ALL",          IDC_BTN_RUNALL,       y, btnW, 40);  y += 40 + gap;
        btn("Clear Output",     IDC_BTN_CLEAR,        y, btnW, btnH); y += btnH + 16;

        // 分隔标签
        CreateWindowA("STATIC", "--- Predict ---",
            WS_CHILD | WS_VISIBLE | SS_CENTER,
            xLeft, y, btnW, 18, hwnd, nullptr, hi, nullptr);
        y += 22;

        CreateWindowA("STATIC", "Regression (5 vals):",
            WS_CHILD | WS_VISIBLE,
            xLeft, y, btnW, 16, hwnd, (HMENU)(INT_PTR)IDC_STATIC_REG, hi, nullptr);
        y += 18;
        CreateWindowA("EDIT", "0.2,0.4,0.6,0.8,1.0",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            xLeft, y, btnW, 22, hwnd, (HMENU)(INT_PTR)IDC_EDIT_REG_INPUT, hi, nullptr);
        y += 26;
        btn("Reg Predict",      IDC_BTN_REG_PREDICT,  y, btnW, btnH); y += btnH + 12;

        CreateWindowA("STATIC", "Classify (3 vals):",
            WS_CHILD | WS_VISIBLE,
            xLeft, y, btnW, 16, hwnd, (HMENU)(INT_PTR)IDC_STATIC_CLS, hi, nullptr);
        y += 18;
        CreateWindowA("EDIT", "0.1,0.2,0.3",
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            xLeft, y, btnW, 22, hwnd, (HMENU)(INT_PTR)IDC_EDIT_CLS_INPUT, hi, nullptr);
        y += 26;
        btn("Cls Predict",      IDC_BTN_CLS_PREDICT,  y, btnW, btnH);

        // ---- 输出区域 ----
        g_hOutput = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_READONLY |
            WS_VSCROLL | ES_AUTOVSCROLL,
            180, 10, 600, 440, hwnd, (HMENU)(INT_PTR)IDC_OUTPUT, hi, nullptr);

        // ---- 状态栏 ----
        g_hStatus = CreateWindowA("STATIC", "Ready. Click a button to run a demo.",
            WS_CHILD | WS_VISIBLE | SS_LEFT | SS_SUNKEN,
            180, 458, 600, 20, hwnd, nullptr, hi, nullptr);

        // 字体
        g_hFontOutput = CreateFontA(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, "Consolas");
        SendMessage(g_hOutput, WM_SETFONT, (WPARAM)g_hFontOutput, TRUE);

        g_hFontUI = CreateFontA(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
        // 应用到所有子控件
        for (HWND child = GetWindow(hwnd, GW_CHILD); child;
             child = GetWindow(child, GW_HWNDNEXT)) {
            char cls[32];
            GetClassNameA(child, cls, 32);
            if (strcmp(cls, "EDIT") != 0 || child != g_hOutput)
                SendMessage(child, WM_SETFONT, (WPARAM)g_hFontUI, TRUE);
        }

        break;
    }
    case WM_COMMAND: {
        int id = LOWORD(wp);
        switch (id) {
        case IDC_BTN_REGRESSION:   DemoRegression();    break;
        case IDC_BTN_CLASSIFY:     DemoClassification(); break;
        case IDC_BTN_SAVELOAD:     DemoSaveLoad();      break;
        case IDC_BTN_GRADCLIP:     DemoGradClip();      break;
        case IDC_BTN_QUICKTRAIN:   DemoQuickTrain();    break;
        case IDC_BTN_CNN:          DemoCNN();           break;
        case IDC_BTN_RUNALL:       DemoRunAll();        break;
        case IDC_BTN_CLEAR:        DemoClear();         break;
        case IDC_BTN_REG_PREDICT:  RegPredict();        break;
        case IDC_BTN_CLS_PREDICT:  ClsPredict();        break;
        }
        break;
    }
    case WM_SIZE: {
        int w = LOWORD(lp), h = HIWORD(lp);
        int outW = w - 195;
        int outH = h - 65;
        if (outW < 100) outW = 100;
        if (outH < 80) outH = 80;
        MoveWindow(g_hOutput, 180, 10, outW, outH, TRUE);
        MoveWindow(g_hStatus, 180, outH + 14, outW, 20, TRUE);
        break;
    }
    case WM_CTLCOLORSTATIC: {
        HDC hdc = (HDC)wp;
        if ((HWND)lp == g_hOutput) {
            SetBkColor(hdc, RGB(255,255,255));
            return (LRESULT)GetStockObject(WHITE_BRUSH);
        }
        return (LRESULT)GetStockObject(NULL_BRUSH);
    }
    case WM_CTLCOLOREDIT: {
        if ((HWND)lp == g_hOutput) {
            HDC hdc = (HDC)wp;
            SetBkColor(hdc, RGB(30,30,30));
            SetTextColor(hdc, RGB(220,220,220));
            return (LRESULT)GetStockObject(BLACK_BRUSH);
        }
        break;
    }
    case WM_DESTROY:
        if (g_hFontOutput) DeleteObject(g_hFontOutput);
        if (g_hFontUI)     DeleteObject(g_hFontUI);
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, msg, wp, lp);
    }
    return 0;
}

// ============ 入口 ============
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    seed(42);
    g_hInst = hInstance;

    // 注册窗口类
    WNDCLASSA wc = {};
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = CreateSolidBrush(RGB(240, 240, 245));
    wc.lpszClassName = "AIDemoWindow";
    RegisterClassA(&wc);

    // 创建窗口
    int winW = 820, winH = 540;
    HWND hwnd = CreateWindowExA(0, "AIDemoWindow",
        "AI Toolkit - Interactive Demo Panel",
        WS_OVERLAPPEDWINDOW & ~WS_MAXIMIZEBOX & ~WS_THICKFRAME,
        CW_USEDEFAULT, CW_USEDEFAULT, winW, winH,
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd) return 1;
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    // 消息循环
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}
