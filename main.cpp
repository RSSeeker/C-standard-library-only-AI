#include "AI.hpp"
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <map>
#include <ctime>

using namespace ai;

#ifdef _MSC_VER
#pragma comment(linker,"\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif

// ============ 颜色常量 ============
#define CLR_BG         RGB(255, 255, 255)
#define CLR_PANEL_BG   RGB(255, 255, 255)
#define CLR_TEXT       RGB(20, 20, 35)
#define CLR_TEXT_DIM   RGB(120, 122, 130)
#define CLR_ACCENT     RGB(59, 130, 246)
#define CLR_GREEN      RGB(5, 150, 105)
#define CLR_RED        RGB(220, 38, 38)
#define CLR_INPUT_BG   RGB(243, 244, 246)
#define CLR_OUTPUT_BG  RGB(248, 249, 250)
#define CLR_CHART_LINE  RGB(59, 130, 246)
#define CLR_CHART_VAL   RGB(220, 150, 50)
#define CLR_CHART_GRID  RGB(230, 232, 235)

// 控件 ID
#define IDC_OUTPUT          3001
#define IDC_STATUS          3002
#define IDC_BTN_TRAIN       3101
#define IDC_BTN_STOP        3102
#define IDC_BTN_SAVE        3103
#define IDC_BTN_LOAD        3104
#define IDC_BTN_CLEAR       3105
#define IDC_BTN_INFER       3106

// 示例按钮
#define IDC_EX_REGRESSION   3201
#define IDC_EX_CLASSIFY     3202
#define IDC_EX_GRADCLIP     3203
#define IDC_EX_QUICKTRAIN   3204
#define IDC_EX_CNN          3205
#define IDC_EX_RUNALL       3206

// 参数编辑
#define IDC_EDIT_LR         3301
#define IDC_EDIT_EPOCHS     3302
#define IDC_EDIT_BATCH      3303
#define IDC_EDIT_LAYERS     3304
#define IDC_EDIT_INFER      3305
#define IDC_CBO_OPTIMIZER   3306
#define IDC_CBO_ACTIVATION  3307
#define IDC_CBO_NETTYPE     3308
#define IDC_EDIT_SEED       3309
#define IDC_PROGRESS        3310
#define IDC_CBO_ARCH        3311
#define IDC_EDIT_HIDDEN     3312
#define IDC_EDIT_OUTPUT     3313
#define IDC_EDIT_INPUT      3314

// 架构面板子控件
#define IDC_EDIT_CNNKERNEL  3320
#define IDC_EDIT_CNNFILTERS 3321
#define IDC_EDIT_RNNHIDDEN  3322
#define IDC_EDIT_ATTNEMBED  3323
#define IDC_EDIT_ATTNHEADS  3324

// 早停控件
#define IDC_CHECK_EARLYSTOP 3330
#define IDC_EDIT_PATIENCE   3331
#define IDC_EDIT_VALSPLIT   3332

// Dropout / L2 / Test Split 控件
#define IDC_CHECK_DROPOUT   3340
#define IDC_EDIT_DROPOUT    3341
#define IDC_CHECK_L2        3342
#define IDC_EDIT_L2         3343
#define IDC_CHECK_TESTSPLIT 3344
#define IDC_EDIT_TESTSPLIT  3345

// 清除训练数据
#define IDC_BTN_CLEAR_DATA  3350

// 数据表格
#define IDC_LIST_DATA       3400
#define IDC_BTN_ADD_ROW     3401
#define IDC_BTN_DEL_ROW     3402
#define IDC_BTN_LOAD_DATA   3403

// 层编辑器
#define IDC_BTN_ADD_LAYER   3500
#define IDC_BTN_DEL_LAYER   3501
#define IDC_LIST_LAYERS     3510

// 训练线程
#define WM_USER_UPDATE_OUTPUT   (WM_USER + 100)
#define WM_USER_TRAINING_DONE   (WM_USER + 101)
#define WM_USER_UPDATE_PROGRESS (WM_USER + 102)
#define WM_USER_EDIT_CELL       (WM_USER + 103)

// ============ 全局状态 ============
HINSTANCE g_hInst   = nullptr;
HWND      g_hWnd    = nullptr;
HWND      g_hOutput = nullptr;
HWND      g_hStatus = nullptr;
HWND      g_hProgress = nullptr;
HFONT     g_hFontMono  = nullptr;
HFONT     g_hFontUI    = nullptr;
HFONT     g_hFontTitle = nullptr;
HFONT     g_hFontGrid  = nullptr;

// 训练状态
Model     g_model;
bool      g_trained  = false;
bool      g_training = false;
HANDLE    g_hThread  = nullptr;
std::string g_task = "regression";

// 输出捕获
std::ostringstream g_capture;
std::streambuf*    g_oldBuf = nullptr;

// 训练数据
std::vector<Vec> g_train_x, g_train_y_real;
std::vector<int> g_train_y_cls;
bool g_has_custom_data = false;

// 数据表格
HWND g_hDataList = nullptr;
HWND g_hDataEdit = nullptr;      // 子项编辑框
int  g_editRow = -1, g_editCol = -1;

// 层编辑器 - 使用 ListView 表格
HWND g_hLayerList = nullptr;        // 隐藏层 ListView
HWND g_hLayerEdit = nullptr;        // 层表格子项编辑框
int  g_layerEditRow = -1, g_layerEditCol = -1;
WNDPROC g_oldLayerEditProc = nullptr;
WNDPROC g_oldDataEditProc = nullptr;

// 输入输出层控件
HWND g_hEditInput = nullptr;
HWND g_hEditOutput = nullptr;
HWND g_hOutputLayerLabel = nullptr;
HWND g_hAddLayerBtn = nullptr;
HWND g_hKernelLabel = nullptr;
HWND g_hFilterLabel = nullptr;
HWND g_hDelLayerBtn = nullptr;

// 层配置区域布局
int g_leftX = 14;
int g_fieldW = 0;
int g_afterLayersBaseY = 0;
int g_afterLayersBaseH = 0;
std::vector<std::pair<HWND,int>> g_belowLayers; // HWND, 原始 y（未滚动时）
static void trackBelowLayer(HWND h, int y) { g_belowLayers.push_back({h, y}); }

// 滚动
int  g_scrollY = 0;
int  g_contentH = 0;


// 左侧面板
int g_leftPanelW = 440;
HWND g_hScrollPanel = nullptr;

// 折线图
RECT g_chartRect = {0,0,0,0};
std::vector<std::pair<int,double>> g_lossChartTrain;
std::vector<std::pair<int,double>> g_lossChartVal;

// ============ 默认示例数据 ============
static void loadDefaultRegData() {
    g_train_x = {
        {0.2,0.4,0.6,0.8,1.0}, {0.4,0.6,0.8,1.0,1.2},
        {0.6,0.8,1.0,1.4,1.6}, {0.1,0.3,0.5,0.7,0.9},
        {0.3,0.5,0.7,0.9,1.1},
    };
    g_train_y_real = {
        {0.04,0.08}, {0.16,0.24}, {0.36,0.48}, {0.01,0.02}, {0.09,0.18}
    };
    g_has_custom_data = true;
}
static void loadDefaultClsData() {
    g_train_x = {
        {0.1,0.2,0.3},{0.4,0.5,0.6},{0.7,0.8,0.9},
        {0.15,0.25,0.35},{0.45,0.55,0.65},{0.75,0.85,0.95},
    };
    g_train_y_cls = {0,1,2,0,1,2};
    g_has_custom_data = true;
}

// ============ 输出辅助 ============
void BeginCapture() { g_capture.str(""); g_capture.clear(); g_oldBuf = std::cout.rdbuf(g_capture.rdbuf()); }
std::string EndCapture() { if (g_oldBuf) std::cout.rdbuf(g_oldBuf); return g_capture.str(); }

void AppendText(HWND hList, const std::string& text) {
    std::string line;
    for (char c : text) {
        if (c == '\n') {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            SendMessageA(hList, LB_ADDSTRING, 0, (LPARAM)line.c_str());
            line.clear();
        } else {
            line.push_back(c);
        }
    }
    if (!line.empty()) {
        if (line.back() == '\r') line.pop_back();
        SendMessageA(hList, LB_ADDSTRING, 0, (LPARAM)line.c_str());
    }
    int count = (int)SendMessage(hList, LB_GETCOUNT, 0, 0);
    if (count > 0) {
        SendMessage(hList, LB_SETTOPINDEX, count - 1, 0);
    }
}

void ClearOutput(HWND hList) {
    SendMessage(hList, LB_RESETCONTENT, 0, 0);
}

void SetStatus(const std::string& s) {
    SetWindowTextA(g_hStatus, s.c_str());
}

// ============ 创建控件辅助 ============
HWND MakeStatic(HWND parent, const char* text, int x, int y, int w, int h, DWORD style = 0) {
    return CreateWindowA("STATIC", text,
        WS_CHILD | WS_VISIBLE | SS_LEFT | style,
        x, y, w, h, parent, nullptr, g_hInst, nullptr);
}
HWND MakeEdit(HWND parent, int id, const char* init, int x, int y, int w, int h, DWORD style = 0) {
    return CreateWindowExA(0, "EDIT", init,
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL | style,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, g_hInst, nullptr);
}
HWND MakeBtn(HWND parent, int id, const char* text, int x, int y, int w, int h) {
    return CreateWindowA("BUTTON", text,
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, g_hInst, nullptr);
}
HWND MakeCombo(HWND parent, int id, int x, int y, int w, int h) {
    return CreateWindowA("COMBOBOX", "",
        WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
        x, y, w, h, parent, (HMENU)(INT_PTR)id, g_hInst, nullptr);
}

void AddComboItem(HWND cb, const char* s, bool sel = false) {
    SendMessageA(cb, CB_ADDSTRING, 0, (LPARAM)s);
    if (sel) SendMessageA(cb, CB_SETCURSEL, SendMessageA(cb, CB_GETCOUNT, 0, 0) - 1, 0);
}

// ============ 读取编辑框 ============
static double readDouble(int id, double def = 0.0) {
    char buf[64];
    GetWindowTextA(GetDlgItem(g_hWnd, id), buf, 64);
    try { return std::stod(buf); } catch (...) { return def; }
}
static int readInt(int id, int def = 0) {
    char buf[64];
    GetWindowTextA(GetDlgItem(g_hWnd, id), buf, 64);
    try { return std::stoi(buf); } catch (...) { return def; }
}
static int readSeed() {
    char buf[64];
    GetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_SEED), buf, 64);
    if (buf[0] == 0) return rand();
    try { return std::stoi(buf); } catch (...) { return rand(); }
}
static std::string readStr(int id, const char* def = "") {
    char buf[512];
    GetWindowTextA(GetDlgItem(g_hWnd, id), buf, 512);
    return buf[0] ? std::string(buf) : std::string(def);
}
static int comboIdx(int id) {
    return (int)SendMessageA(GetDlgItem(g_hWnd, id), CB_GETCURSEL, 0, 0);
}

// ============ 架构可见性 ============
static void updateArchVisibility() {
    int arch = comboIdx(IDC_CBO_ARCH);
    int showCnn = (arch == 1) ? SW_SHOW : SW_HIDE;
    int showRnn = (arch == 2) ? SW_SHOW : SW_HIDE;
    int showAttn = (arch == 3) ? SW_SHOW : SW_HIDE;
    ShowWindow(GetDlgItem(g_hWnd, IDC_EDIT_CNNKERNEL), showCnn);
    ShowWindow(GetDlgItem(g_hWnd, IDC_EDIT_CNNFILTERS), showCnn);
    ShowWindow(g_hKernelLabel, showCnn);
    ShowWindow(g_hFilterLabel, showCnn);
    ShowWindow(GetDlgItem(g_hWnd, IDC_EDIT_RNNHIDDEN), showRnn);
    ShowWindow(GetDlgItem(g_hWnd, IDC_EDIT_ATTNEMBED), showAttn);
    ShowWindow(GetDlgItem(g_hWnd, IDC_EDIT_ATTNHEADS), showAttn);
}

// ============ 数据表格操作 ============
static int getDataColumns() {
    int inputDim = readInt(IDC_EDIT_INPUT, 1);
    int outputDim = readInt(IDC_EDIT_OUTPUT, 1);
    bool isCls = comboIdx(IDC_CBO_NETTYPE) == 1;
    return inputDim + (isCls ? 1 : outputDim);
}

static void rebuildDataColumns() {
    if (!g_hDataList) return;
    // 删除所有列
    while (ListView_DeleteColumn(g_hDataList, 0)) {}
    int inputDim = readInt(IDC_EDIT_INPUT, 1);
    int outputDim = readInt(IDC_EDIT_OUTPUT, 1);
    bool isCls = comboIdx(IDC_CBO_NETTYPE) == 1;

    LVCOLUMNA lvc = {};
    lvc.mask = LVCF_WIDTH | LVCF_TEXT | LVCF_FMT;
    lvc.fmt = LVCFMT_CENTER;
    lvc.cx = 62;

    char hdr[16];
    for (int i = 0; i < inputDim; ++i) {
        snprintf(hdr, 16, "X%d", i+1);
        lvc.pszText = hdr;
        ListView_InsertColumn(g_hDataList, i, &lvc);
    }
    if (isCls) {
        lvc.pszText = (char*)"Class";
        ListView_InsertColumn(g_hDataList, inputDim, &lvc);
    } else {
        for (int i = 0; i < outputDim; ++i) {
            snprintf(hdr, 16, "Y%d", i+1);
            lvc.pszText = hdr;
            ListView_InsertColumn(g_hDataList, inputDim + i, &lvc);
        }
    }
}

static void addDataRow() {
    if (!g_hDataList) return;
    int cols = getDataColumns();
    if (cols <= 0) return;

    LVITEMA lvi = {};
    lvi.mask = LVIF_TEXT;
    lvi.iItem = ListView_GetItemCount(g_hDataList);
    lvi.pszText = (char*)"0.0";
    lvi.iItem = ListView_InsertItem(g_hDataList, &lvi);

    char buf[32] = "0.0";
    for (int c = 1; c < cols; ++c) {
        ListView_SetItemText(g_hDataList, lvi.iItem, c, buf);
    }
}

static void delDataRow() {
    if (!g_hDataList) return;
    int sel = ListView_GetNextItem(g_hDataList, -1, LVNI_SELECTED);
    if (sel >= 0) {
        ListView_DeleteItem(g_hDataList, sel);
    } else {
        int cnt = ListView_GetItemCount(g_hDataList);
        if (cnt > 0) ListView_DeleteItem(g_hDataList, cnt - 1);
    }
}

static void readDataFromTable() {
    if (!g_hDataList) return;
    int inputDim = readInt(IDC_EDIT_INPUT, 1);
    int outputDim = readInt(IDC_EDIT_OUTPUT, 1);
    bool isCls = comboIdx(IDC_CBO_NETTYPE) == 1;

    int rows = ListView_GetItemCount(g_hDataList);
    int cols = getDataColumns();

    g_train_x.clear();
    g_train_y_real.clear();
    g_train_y_cls.clear();

    char buf[64];
    for (int r = 0; r < rows; ++r) {
        Vec x;
        for (int c = 0; c < inputDim && c < cols; ++c) {
            ListView_GetItemText(g_hDataList, r, c, buf, 64);
            try { x.push_back(std::stod(buf)); } catch (...) { x.push_back(0.0); }
        }
        if (x.empty()) continue;
        g_train_x.push_back(x);

        if (isCls) {
            ListView_GetItemText(g_hDataList, r, inputDim, buf, 64);
            try { g_train_y_cls.push_back(std::stoi(buf)); } catch (...) { g_train_y_cls.push_back(0); }
        } else {
            Vec y;
            for (int c = 0; c < outputDim; ++c) {
                int ci = inputDim + c;
                if (ci >= cols) break;
                ListView_GetItemText(g_hDataList, r, ci, buf, 64);
                try { y.push_back(std::stod(buf)); } catch (...) { y.push_back(0.0); }
            }
            if (!y.empty()) g_train_y_real.push_back(y);
        }
    }
    if (!g_train_x.empty()) g_has_custom_data = true;
}

static void updateInputOutputLock();

static void populateDataTable(const std::vector<Vec>& x, const std::vector<Vec>& yReal,
                               const std::vector<int>& yCls, bool isCls) {
    if (!g_hDataList) return;
    ListView_DeleteAllItems(g_hDataList);
    rebuildDataColumns();

    char buf[64];
    for (size_t r = 0; r < x.size(); ++r) {
        LVITEMA lvi = {};
        lvi.mask = LVIF_TEXT;
        lvi.iItem = (int)r;
        snprintf(buf, 64, "%.4g", x[r][0]);
        lvi.pszText = buf;
        ListView_InsertItem(g_hDataList, &lvi);

        for (size_t c = 1; c < x[r].size(); ++c) {
            snprintf(buf, 64, "%.4g", x[r][c]);
            ListView_SetItemText(g_hDataList, (int)r, (int)c, buf);
        }
        if (isCls) {
            snprintf(buf, 64, "%d", r < yCls.size() ? yCls[r] : 0);
            ListView_SetItemText(g_hDataList, (int)r, (int)x[r].size(), buf);
        } else {
            for (size_t c = 0; r < yReal.size() && c < yReal[r].size(); ++c) {
                snprintf(buf, 64, "%.4g", yReal[r][c]);
                ListView_SetItemText(g_hDataList, (int)r, (int)(x[r].size() + c), buf);
            }
        }
    }
    updateInputOutputLock();
}

// ============ ListView 子项编辑 ============

static void endCellEdit(bool save);
static void endLayerEdit(bool save);

// 子项编辑框子类化 — 处理 Enter/Escape 键
static LRESULT CALLBACK DataEditProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_KEYDOWN) {
        if (wp == VK_RETURN) { endCellEdit(true); return 0; }
        if (wp == VK_ESCAPE) { endCellEdit(false); return 0; }
    }
    return CallWindowProcA(g_oldDataEditProc, hwnd, msg, wp, lp);
}
static LRESULT CALLBACK LayerEditProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_KEYDOWN) {
        if (wp == VK_RETURN) { endLayerEdit(true); return 0; }
        if (wp == VK_ESCAPE) { endLayerEdit(false); return 0; }
    }
    return CallWindowProcA(g_oldLayerEditProc, hwnd, msg, wp, lp);
}

static void beginCellEdit(int row, int col) {
    if (!g_hDataList) return;
    RECT rc;
    ListView_GetSubItemRect(g_hDataList, row, col, LVIR_BOUNDS, &rc);
    if (g_hDataEdit) DestroyWindow(g_hDataEdit);
    g_hDataEdit = CreateWindowExA(0, "EDIT", "",
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
        rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top,
        g_hDataList, nullptr, g_hInst, nullptr);
    SendMessage(g_hDataEdit, WM_SETFONT, (WPARAM)g_hFontGrid, TRUE);
    g_oldDataEditProc = (WNDPROC)SetWindowLongPtrA(g_hDataEdit, GWLP_WNDPROC, (LONG_PTR)DataEditProc);

    char buf[256];
    ListView_GetItemText(g_hDataList, row, col, buf, 256);
    SetWindowTextA(g_hDataEdit, buf);
    SendMessage(g_hDataEdit, EM_SETSEL, 0, -1);
    SetFocus(g_hDataEdit);
    g_editRow = row; g_editCol = col;
}

static void endCellEdit(bool save) {
    if (!g_hDataEdit) return;
    if (save && g_editRow >= 0 && g_editCol >= 0) {
        char buf[256];
        GetWindowTextA(g_hDataEdit, buf, 256);
        ListView_SetItemText(g_hDataList, g_editRow, g_editCol, buf);
    }
    DestroyWindow(g_hDataEdit);
    g_hDataEdit = nullptr;
    g_editRow = g_editCol = -1;
}

// ============ 层编辑器（ListView 表格）============

// 隐藏层表格子项编辑
static void beginLayerEdit(int row, int col) {
    if (!g_hLayerList) return;
    RECT rc;
    ListView_GetSubItemRect(g_hLayerList, row, col, LVIR_BOUNDS, &rc);
    if (g_hLayerEdit) DestroyWindow(g_hLayerEdit);
    g_hLayerEdit = CreateWindowExA(0, "EDIT", "",
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL | ES_NUMBER,
        rc.left + 2, rc.top + 1, rc.right - rc.left - 2, rc.bottom - rc.top - 1,
        g_hLayerList, nullptr, g_hInst, nullptr);
    SendMessage(g_hLayerEdit, WM_SETFONT, (WPARAM)g_hFontUI, TRUE);
    g_oldLayerEditProc = (WNDPROC)SetWindowLongPtrA(g_hLayerEdit, GWLP_WNDPROC, (LONG_PTR)LayerEditProc);

    char buf[32];
    ListView_GetItemText(g_hLayerList, row, col, buf, 32);
    SetWindowTextA(g_hLayerEdit, buf);
    SendMessage(g_hLayerEdit, EM_SETSEL, 0, -1);
    SetFocus(g_hLayerEdit);
    g_layerEditRow = row; g_layerEditCol = col;
}

static void endLayerEdit(bool save) {
    if (!g_hLayerEdit) return;
    if (save && g_layerEditRow >= 0 && g_layerEditCol >= 0) {
        char buf[32];
        GetWindowTextA(g_hLayerEdit, buf, 32);
        ListView_SetItemText(g_hLayerList, g_layerEditRow, g_layerEditCol, buf);
    }
    DestroyWindow(g_hLayerEdit);
    g_hLayerEdit = nullptr;
    g_layerEditRow = g_layerEditCol = -1;
}

// 初始化/重建隐藏层 ListView 列
static void initLayerList() {
    if (!g_hLayerList) return;
    ListView_DeleteAllItems(g_hLayerList);
    while (ListView_DeleteColumn(g_hLayerList, 0)) {}

    LVCOLUMNA lvc = {};
    lvc.mask = LVCF_WIDTH | LVCF_TEXT | LVCF_FMT;
    lvc.fmt = LVCFMT_LEFT;
    lvc.cx = 75;
    lvc.pszText = (char*)"Layer";
    ListView_InsertColumn(g_hLayerList, 0, &lvc);
    lvc.cx = 65;
    lvc.pszText = (char*)"Neurons";
    lvc.fmt = LVCFMT_CENTER;
    ListView_InsertColumn(g_hLayerList, 1, &lvc);
}

// 添加隐藏层行
static void addLayer(int dim = 4) {
    if (!g_hLayerList) return;
    int cnt = ListView_GetItemCount(g_hLayerList);
    if (cnt >= 50) return;

    char lbl[32];
    snprintf(lbl, 32, "Hidden %d", cnt + 1);

    LVITEMA lvi = {};
    lvi.mask = LVIF_TEXT;
    lvi.iItem = cnt;
    lvi.pszText = lbl;
    ListView_InsertItem(g_hLayerList, &lvi);

    char buf[16];
    snprintf(buf, 16, "%d", dim);
    ListView_SetItemText(g_hLayerList, cnt, 1, buf);
}

// 删除隐藏层行
static void delLayer(int index) {
    if (!g_hLayerList) return;
    int cnt = ListView_GetItemCount(g_hLayerList);
    if (cnt <= 1) return;
    if (index < 0 || index >= cnt) index = cnt - 1;

    endLayerEdit(false);
    ListView_DeleteItem(g_hLayerList, index);

    // 重新编号：更新 Layer 列文字
    for (int i = 0; i < ListView_GetItemCount(g_hLayerList); ++i) {
        char lbl[32];
        snprintf(lbl, 32, "Hidden %d", i + 1);
        ListView_SetItemText(g_hLayerList, i, 0, lbl);
    }
}

// 从表格读取隐藏层维度
static std::vector<int> getLayerDims() {
    std::vector<int> dims;
    if (!g_hLayerList) return dims;
    char buf[32];
    for (int i = 0; i < ListView_GetItemCount(g_hLayerList); ++i) {
        ListView_GetItemText(g_hLayerList, i, 1, buf, 32);
        try { dims.push_back(std::stoi(buf)); } catch (...) { dims.push_back(16); }
    }
    if (dims.empty()) dims.push_back(16);
    return dims;
}

// 重新计算内容总高度（固定布局，不再随层数变化）
static void recalcContentHeight() {
    if (!g_hWnd) return;
    // g_afterLayersBaseY + baseH 已在 WM_CREATE 末尾计算
    g_contentH = g_afterLayersBaseY + g_afterLayersBaseH;
    if (g_contentH < 100) g_contentH = 100;
}

// 绘制 loss 折线图
static void drawLossChart(HDC hdc, RECT& cr) {
    if (cr.right - cr.left < 50 || cr.bottom - cr.top < 30) return;
    int x0 = cr.left, y0 = cr.top, cw = cr.right - cr.left, ch = cr.bottom - cr.top;

    // 保存 DC 原始状态
    int savedDc = SaveDC(hdc);

    // 背景 + 边框
    HBRUSH whiteBr = (HBRUSH)GetStockObject(WHITE_BRUSH);
    FillRect(hdc, &cr, whiteBr);
    {
        HPEN borderPen = CreatePen(PS_SOLID, 1, CLR_CHART_GRID);
        HGDIOBJ oldPen = SelectObject(hdc, borderPen);
        HGDIOBJ oldBr2 = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Rectangle(hdc, x0, y0, cr.right, cr.bottom);
        SelectObject(hdc, oldBr2);
        SelectObject(hdc, oldPen);
        DeleteObject(borderPen);
    }

    // 标题
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, CLR_TEXT_DIM);
    TextOutA(hdc, x0 + 8, y0 + 4, "Loss Curve", 10);

    // 合并 train + val 数据找范围
    struct DataPt { int epoch; double loss; };
    std::vector<DataPt> trainPts, valPts;
    for (auto& p : g_lossChartTrain) trainPts.push_back({p.first, p.second});
    for (auto& p : g_lossChartVal)   valPts.push_back({p.first, p.second});

    int totalPts = (int)(trainPts.size() + valPts.size());
    if (totalPts == 0) {
        SetTextColor(hdc, CLR_TEXT_DIM);
        TextOutA(hdc, x0 + 20, y0 + ch/2 - 8, "(loss data after training)", 26);
        RestoreDC(hdc, savedDc);
        return;
    }

    int minEp = 999999, maxEp = -1;
    double minLoss = 1e18, maxLoss = -1e18;
    for (auto& d : trainPts) { if (d.epoch < minEp) minEp = d.epoch; if (d.epoch > maxEp) maxEp = d.epoch; if (d.loss < minLoss) minLoss = d.loss; if (d.loss > maxLoss) maxLoss = d.loss; }
    for (auto& d : valPts)   { if (d.epoch < minEp) minEp = d.epoch; if (d.epoch > maxEp) maxEp = d.epoch; if (d.loss < minLoss) minLoss = d.loss; if (d.loss > maxLoss) maxLoss = d.loss; }
    if (maxEp == minEp) maxEp = minEp + 1;
    double lossRange = maxLoss - minLoss;
    if (lossRange < 1e-8) lossRange = 1.0;
    minLoss -= lossRange * 0.1;
    maxLoss += lossRange * 0.1;
    lossRange = maxLoss - minLoss;

    // 绘图区域边距
    int marginL = 50, marginR = 24, marginT = 22, marginB = 32;
    int px0 = x0 + marginL, py0 = y0 + marginT;
    int pw = cw - marginL - marginR, ph = ch - marginT - marginB;
    if (pw < 10 || ph < 10) { RestoreDC(hdc, savedDc); return; }

    // 网格线
    {
        HPEN gridPen = CreatePen(PS_SOLID, 1, CLR_CHART_GRID);
        HGDIOBJ oldPen = SelectObject(hdc, gridPen);
        int gridY = 4;
        for (int i = 0; i <= gridY; i++) {
            int y = py0 + ph * i / gridY;
            MoveToEx(hdc, px0, y, nullptr); LineTo(hdc, px0 + pw, y);
        }
        int gridX = 6;
        for (int i = 0; i <= gridX; i++) {
            int x = px0 + pw * i / gridX;
            MoveToEx(hdc, x, py0, nullptr); LineTo(hdc, x, py0 + ph);
        }
        SelectObject(hdc, oldPen);
        DeleteObject(gridPen);
    }

    // Y 轴标签
    SetTextColor(hdc, CLR_TEXT_DIM);
    char buf[32];
    for (int i = 0; i <= 4; i++) {
        double v = maxLoss - (maxLoss - minLoss) * i / 4.0;
        snprintf(buf, sizeof(buf), "%.3f", v);
        int tw = (int)strlen(buf) * 7;
        TextOutA(hdc, px0 - tw - 4, py0 + ph * i / 4 - 7, buf, (int)strlen(buf));
    }
    // X 轴标签
    for (int i = 0; i <= 6; i++) {
        int ep = minEp + (maxEp - minEp) * i / 6;
        snprintf(buf, sizeof(buf), "%d", ep);
        int tw = (int)strlen(buf) * 7;
        TextOutA(hdc, px0 + pw * i / 6 - tw/2, py0 + ph + 5, buf, (int)strlen(buf));
    }

    auto toScreen = [&](double ep, double loss) -> POINT {
        POINT pt;
        pt.x = px0 + (int)((ep - minEp) / (double)(maxEp - minEp) * pw);
        pt.y = py0 + (int)((maxLoss - loss) / lossRange * ph);
        return pt;
    };

    // 绘制 train loss 折线
    if (!trainPts.empty()) {
        HPEN trainPen = CreatePen(PS_SOLID, 2, CLR_CHART_LINE);
        HGDIOBJ oldPen = SelectObject(hdc, trainPen);
        POINT prev = toScreen((double)trainPts[0].epoch, trainPts[0].loss);
        MoveToEx(hdc, prev.x, prev.y, nullptr);
        for (size_t i = 1; i < trainPts.size(); i++) {
            POINT pt = toScreen((double)trainPts[i].epoch, trainPts[i].loss);
            LineTo(hdc, pt.x, pt.y);
        }
        SelectObject(hdc, oldPen);
        DeleteObject(trainPen);

        // 数据点小圆
        if (trainPts.size() <= 100) {
            HBRUSH ptBr = CreateSolidBrush(CLR_CHART_LINE);
            HGDIOBJ oldBr = SelectObject(hdc, ptBr);
            HPEN nullPen = (HPEN)GetStockObject(NULL_PEN);
            HGDIOBJ oldPen2 = SelectObject(hdc, nullPen);
            for (auto& d : trainPts) {
                POINT pt = toScreen((double)d.epoch, d.loss);
                Ellipse(hdc, pt.x - 2, pt.y - 2, pt.x + 3, pt.y + 3);
            }
            SelectObject(hdc, oldPen2);
            SelectObject(hdc, oldBr);
            DeleteObject(ptBr);
        }
    }

    // 绘制 val loss 折线
    if (!valPts.empty()) {
        HPEN valPen = CreatePen(PS_DASH, 2, CLR_CHART_VAL);
        HGDIOBJ oldPen = SelectObject(hdc, valPen);
        POINT prev = toScreen((double)valPts[0].epoch, valPts[0].loss);
        MoveToEx(hdc, prev.x, prev.y, nullptr);
        for (size_t i = 1; i < valPts.size(); i++) {
            POINT pt = toScreen((double)valPts[i].epoch, valPts[i].loss);
            LineTo(hdc, pt.x, pt.y);
        }
        SelectObject(hdc, oldPen);
        DeleteObject(valPen);
    }

    // 图例
    int legendY = y0 + ch - 16;
    {
        HPEN legTrain = CreatePen(PS_SOLID, 2, CLR_CHART_LINE);
        HGDIOBJ oldPen = SelectObject(hdc, legTrain);
        MoveToEx(hdc, x0 + 10, legendY, nullptr); LineTo(hdc, x0 + 30, legendY);
        SelectObject(hdc, oldPen);
        DeleteObject(legTrain);
    }
    SetTextColor(hdc, CLR_TEXT);
    TextOutA(hdc, x0 + 34, legendY - 7, "train", 5);

    if (!valPts.empty()) {
        HPEN legVal = CreatePen(PS_DASH, 2, CLR_CHART_VAL);
        HGDIOBJ oldPen = SelectObject(hdc, legVal);
        MoveToEx(hdc, x0 + 75, legendY, nullptr); LineTo(hdc, x0 + 95, legendY);
        SelectObject(hdc, oldPen);
        DeleteObject(legVal);
        TextOutA(hdc, x0 + 99, legendY - 7, "val", 3);
    }

    RestoreDC(hdc, savedDc);
}

// 重新定位右侧输出/状态/图表区域（不参与左侧垂直滚动）
static void layoutRightPanel() {
    if (!g_hWnd || !g_hOutput) return;
    RECT cr;
    GetClientRect(g_hWnd, &cr);
    int w = cr.right;
    int h = cr.bottom;
    int rightX = g_leftPanelW + 20;
    int outW = w - rightX - 16;
    if (outW < 200) outW = 200;

    // 输出框占 35%，图表占剩余空间
    int availH = h - 36 - 24;  // 顶部 36，底部 24 留给 status
    if (availH < 120) availH = 120;
    int outH = std::max(80, availH * 33 / 100);
    int chartTop = 36 + outH + 8;
    int chartBottom = h - 24;
    g_chartRect = {rightX, chartTop, rightX + outW, chartBottom};

    MoveWindow(g_hOutput, rightX, 36, outW, outH, TRUE);
    MoveWindow(g_hStatus, rightX, h - 19, outW, 17, TRUE);
}
static void updateInputOutputLock() {
    if (!g_hEditInput || !g_hEditOutput) return;
    int rows = g_hDataList ? (int)ListView_GetItemCount(g_hDataList) : 0;
    bool hasData = g_has_custom_data || (rows > 0);
    EnableWindow(g_hEditInput,  !hasData);
    EnableWindow(g_hEditOutput, !hasData);
}

// ============ 训练线程 ============
DWORD WINAPI TrainThread(LPVOID) {
    g_training = true;
    seed(readSeed());
    BeginCapture();

    std::string netType = "regression";
    {
        int idx = comboIdx(IDC_CBO_NETTYPE);
        if (idx == 1) netType = "classification";
    }

    double lr = readDouble(IDC_EDIT_LR, 0.01);
    int epochs = readInt(IDC_EDIT_EPOCHS, 500);
    int batch = readInt(IDC_EDIT_BATCH, 16);
    int inputDim = readInt(IDC_EDIT_INPUT, 1);
    int outputDim = readInt(IDC_EDIT_OUTPUT, 1);

    // 早停参数
    bool useEarlyStop = (SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_EARLYSTOP), BM_GETCHECK, 0, 0) == BST_CHECKED);
    int patience = useEarlyStop ? readInt(IDC_EDIT_PATIENCE, 50) : 0;
    double valSplit = readDouble(IDC_EDIT_VALSPLIT, 0.1);
    if (valSplit <= 0) valSplit = 0.1;

    // Dropout
    bool useDropout = (SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_DROPOUT), BM_GETCHECK, 0, 0) == BST_CHECKED);
    double dropoutP = useDropout ? readDouble(IDC_EDIT_DROPOUT, 0.3) : 0.0;
    if (dropoutP < 0) dropoutP = 0.0;
    if (dropoutP > 0.9) dropoutP = 0.9;

    // L2 正则化
    bool useL2 = (SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_L2), BM_GETCHECK, 0, 0) == BST_CHECKED);
    double l2Lambda = useL2 ? readDouble(IDC_EDIT_L2, 0.0001) : 0.0;

    // 测试集
    bool useTestSplit = (SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_TESTSPLIT), BM_GETCHECK, 0, 0) == BST_CHECKED);
    double testSplit = useTestSplit ? readDouble(IDC_EDIT_TESTSPLIT, 0.1) : 0.0;
    if (testSplit < 0) testSplit = 0.0;
    if (testSplit > 0.5) testSplit = 0.5;

    std::string optName = "adam";
    {
        int idx = comboIdx(IDC_CBO_OPTIMIZER);
        if (idx == 1) optName = "sgd";
        else if (idx == 2) optName = "sgd_momentum";
        else if (idx == 3) optName = "rmsprop";
    }

    std::string actName = "leaky_relu";
    {
        int idx = comboIdx(IDC_CBO_ACTIVATION);
        if (idx == 1) actName = "relu";
    }

    // 从表格读取数据
    readDataFromTable();
    auto hidden = getLayerDims();

    if (!g_has_custom_data || g_train_x.empty()) {
        loadDefaultRegData();
        if (netType == "classification") {
            loadDefaultClsData();
            inputDim = 3; outputDim = 3;
            hidden = {16};
        }
    }

    if (netType == "regression") {
        g_model = make_regressor(inputDim, outputDim, hidden, actName, lr, optName);
    } else {
        g_model = make_classifier(inputDim, outputDim, hidden, actName, lr, optName);
    }
    g_model.dropout_rate = dropoutP;
    g_model.l2_lambda = l2Lambda;
    g_model.test_split = testSplit;

    std::cout << "============================================================\n";
    std::cout << "  " << ((netType=="regression")?"Regression":"Classification")
              << "  |  Optimizer: " << optName
              << "  |  LR: " << lr
              << "  |  Epochs: " << epochs
              << "  |  Batch: " << batch << "\n";
    std::cout << "============================================================\n";
    g_model.summary();

    if (netType == "regression") {
        g_model.fit(g_train_x, g_train_y_real, epochs, batch, valSplit, nullptr, patience, true, true, nullptr, 0.0);
    } else {
        g_model.fit(g_train_x, g_train_y_cls, epochs, batch, valSplit, patience, true, nullptr, 0.0);
    }

    if (useEarlyStop) {
        int stoppedAt = (int)g_model.train_history.size();
        std::cout << "\n  Training stopped at epoch " << stoppedAt;
        if (!g_model.val_history.empty())
            std::cout << " (best val_loss=" << g_model.val_history.back().second << ")";
        std::cout << "\n";
    }

    std::cout << "\n--- Results ---\n";
    if (netType == "regression") {
        double r2 = g_model.score(g_train_x, g_train_y_real);
        std::cout << "  R^2 = " << r2 << "\n";
        for (size_t i = 0; i < g_train_x.size(); ++i) {
            Vec pred = g_model.predict_one(g_train_x[i]);
            std::cout << "  pred: (" << pred[0] << ", " << pred[1]
                      << ")  target: (" << g_train_y_real[i][0] << ", " << g_train_y_real[i][1] << ")\n";
        }
    } else {
        int ok = 0;
        for (size_t i = 0; i < g_train_x.size(); ++i) {
            int pred = g_model.predict_class(g_train_x[i]);
            if (pred == g_train_y_cls[i]) ++ok;
            std::cout << "  pred: " << pred << "  actual: " << g_train_y_cls[i]
                      << "  " << (pred == g_train_y_cls[i] ? "OK" : "FAIL") << "\n";
        }
        std::cout << "  Accuracy: " << 100.0*ok/g_train_x.size() << "%\n";
    }

    std::string out = EndCapture();
    PostMessage(g_hWnd, WM_USER_UPDATE_OUTPUT, 0, (LPARAM)new std::string(out));
    PostMessage(g_hWnd, WM_USER_TRAINING_DONE, 0, 0);
    return 0;
}

// ============ 快速 Demo ============

// 将示例配置同步到 UI 控件
static void syncUIConfig(int netType, int arch, int inputDim, int outputDim,
                         const std::vector<int>& hiddenDims,
                         int optimizer, int activation,
                         double lr, int epochs, int batch,
                         bool earlyStop = true, int patience = 50, double valSplit = 0.1,
                         bool useDropout = false, double dropoutP = 0.3,
                         bool useL2 = false, double l2Lambda = 0.0001,
                         bool useTestSplit = false, double testSplit = 0.1) {
    // 网络类型
    SendMessageA(GetDlgItem(g_hWnd, IDC_CBO_NETTYPE), CB_SETCURSEL, netType, 0);
    // 架构
    SendMessageA(GetDlgItem(g_hWnd, IDC_CBO_ARCH), CB_SETCURSEL, arch, 0);
    updateArchVisibility();
    // 优化器 / 激活函数
    SendMessageA(GetDlgItem(g_hWnd, IDC_CBO_OPTIMIZER), CB_SETCURSEL, optimizer, 0);
    SendMessageA(GetDlgItem(g_hWnd, IDC_CBO_ACTIVATION), CB_SETCURSEL, activation, 0);
    // 输入/输出维度
    char b1[16], b2[16];
    snprintf(b1, 16, "%d", inputDim);
    snprintf(b2, 16, "%d", outputDim);
    SetWindowTextA(g_hEditInput, b1);
    SetWindowTextA(g_hEditOutput, b2);
    // 隐藏层
    endLayerEdit(false);
    ListView_DeleteAllItems(g_hLayerList);
    for (int d : hiddenDims) addLayer(d);
    if (hiddenDims.empty()) addLayer(4);
    // 超参数
    snprintf(b1, 16, "%g", lr);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_LR), b1);
    snprintf(b1, 16, "%d", epochs);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_EPOCHS), b1);
    snprintf(b1, 16, "%d", batch);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_BATCH), b1);
    // 早停
    SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_EARLYSTOP), BM_SETCHECK,
                 earlyStop ? BST_CHECKED : BST_UNCHECKED, 0);
    snprintf(b1, 16, "%d", patience);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_PATIENCE), b1);
    snprintf(b1, 16, "%g", valSplit);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_VALSPLIT), b1);
    // Dropout
    SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_DROPOUT), BM_SETCHECK,
                 useDropout ? BST_CHECKED : BST_UNCHECKED, 0);
    snprintf(b1, 16, "%g", dropoutP);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_DROPOUT), b1);
    // L2
    SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_L2), BM_SETCHECK,
                 useL2 ? BST_CHECKED : BST_UNCHECKED, 0);
    snprintf(b1, 16, "%g", l2Lambda);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_L2), b1);
    // Test Split
    SendMessageA(GetDlgItem(g_hWnd, IDC_CHECK_TESTSPLIT), BM_SETCHECK,
                 useTestSplit ? BST_CHECKED : BST_UNCHECKED, 0);
    snprintf(b1, 16, "%g", testSplit);
    SetWindowTextA(GetDlgItem(g_hWnd, IDC_EDIT_TESTSPLIT), b1);
    // 更新数据表列
    rebuildDataColumns();
}

static void runDemo(const std::string& name) {
    if (g_training) return;

    // 从表格加载数据
    readDataFromTable();

    if (name != "Run All") {
        g_has_custom_data = (g_train_x.size() >= 2);
    }
    seed(readSeed());

    if (name != "Run All") {
        ClearOutput(g_hOutput);
        g_capture.str(""); g_capture.clear();
    }
    SendMessage(g_hProgress, PBM_SETPOS, 0, 0);

    BeginCapture();
    std::cout << "============================================================\n";
    std::cout << "  " << name << "\n";
    std::cout << "============================================================\n";

    if (name == "Regression") {
        syncUIConfig(0, 0, 5, 2, {6,6,6,6,5,4,3}, 0, 0, 0.01, 2000, 2);
        loadDefaultRegData();
        g_model = make_regressor(5, 2, {6,6,6,6,5,4,3}, "leaky_relu", 0.01);
        g_model.summary();
        g_model.fit(g_train_x, g_train_y_real, 2000, 2, 0.2, nullptr, 50);
        double r2 = g_model.score(g_train_x, g_train_y_real);
        std::cout << "\n  R^2 = " << r2 << "\n";
        for (size_t i = 0; i < g_train_x.size(); ++i) {
            Vec p = g_model.predict_one(g_train_x[i]);
            std::cout << "  pred: (" << p[0] << ", " << p[1]
                      << ")  target: (" << g_train_y_real[i][0] << ", " << g_train_y_real[i][1] << ")\n";
        }
        g_task = "regression";
        populateDataTable(g_train_x, g_train_y_real, {}, false);
    } else if (name == "Classification") {
        syncUIConfig(1, 0, 3, 3, {8,8}, 0, 0, 0.01, 2000, 2);
        loadDefaultClsData();
        g_model = make_classifier(3, 3, {8,8}, "leaky_relu", 0.01);
        g_model.summary();
        g_model.fit(g_train_x, g_train_y_cls, 2000, 2, 0.2, 50);
        int ok = 0;
        for (size_t i = 0; i < g_train_x.size(); ++i) {
            int p = g_model.predict_class(g_train_x[i]);
            if (p == g_train_y_cls[i]) ++ok;
            std::cout << "  pred: " << p << "  actual: " << g_train_y_cls[i]
                      << "  " << (p == g_train_y_cls[i] ? "OK" : "FAIL") << "\n";
        }
        std::cout << "  Accuracy: " << 100.0*ok/g_train_x.size() << "%\n";
        g_task = "classification";
        populateDataTable(g_train_x, {}, g_train_y_cls, true);
    } else if (name == "Gradient Clip") {
        syncUIConfig(0, 0, 5, 2, {8,8,4}, 0, 0, 0.01, 300, 2);
        loadDefaultRegData();
        Model m = make_regressor(5, 2, {8,8,4}, "leaky_relu", 0.01);
        StepLR sch(m.optimizer.get(), 100, 0.5);
        m.fit(g_train_x, g_train_y_real, 300, 2, 0.0, nullptr, 0, true, true, &sch, 1.0);
        std::cout << "  Final loss: " << m.train_history.back().second
                  << "  |  LR: " << m.optimizer->lr << "\n";
        std::cout << "  R^2: " << m.score(g_train_x, g_train_y_real) << "\n";
        g_model = std::move(m);
        g_task = "regression";
    } else if (name == "Quick Train") {
        syncUIConfig(0, 0, 5, 2, {16,8}, 0, 0, 0.01, 300, 2);
        loadDefaultRegData();
        Model qr = quick_train(g_train_x, g_train_y_real, "regression", {16,8}, 300, 2);
        std::cout << "  Regression R^2 = " << qr.score(g_train_x, g_train_y_real) << "\n";
        g_model = std::move(qr);
        g_task = "regression";
        loadDefaultClsData();
        Model qc = quick_train(g_train_x, g_train_y_cls, {16}, 300, 2);
        std::cout << "  Classification acc = " << qc.score(g_train_x, g_train_y_cls) * 100 << "%\n";
    } else if (name == "CNN") {
        syncUIConfig(1, 1, 4, 2, {}, 0, 0, 0.01, 500, 2);
        g_model = Model();
        g_task = "classification";
        Conv2d conv(1, 2, 2, 1);
        MaxPool2d pool(2);
        Flatten flatten;
        MLP fc({2, 4, 2}, std::vector<std::string>{"leaky_relu", "softmax"});
        Vec3D img(1, Vec2D(4, Vec(4, 0.0)));
        std::cout << "  Input (1x4x4):\n";
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
        std::cout << "  Image -> Conv(2,2,2) -> Pool(2) -> Flatten -> MLP\n";
        std::cout << "  Output: (" << out[0] << ", " << out[1] << ")\n";
    } else if (name == "Run All") {
        syncUIConfig(0, 0, 5, 2, {6,6,6,6,5,4,3}, 0, 0, 0.01, 2000, 2);
        EndCapture();
        AppendText(g_hOutput, "\r\n========== RUN ALL ==========\r\n\r\n");

        auto runSubDemo = [&](const std::string& subName) {
            g_has_custom_data = false;
            seed(readSeed());
            BeginCapture();
            std::cout << "============================================================\n";
            std::cout << "  " << subName << "\n";
            std::cout << "============================================================\n";
            if (subName == "Regression") {
                loadDefaultRegData();
                g_model = make_regressor(5, 2, {6,6,6,6,5,4,3}, "leaky_relu", 0.01);
                g_model.summary();
                g_model.fit(g_train_x, g_train_y_real, 2000, 2, 0.2, nullptr, 50);
                double r2 = g_model.score(g_train_x, g_train_y_real);
                std::cout << "\n  R^2 = " << r2 << "\n";
                for (size_t i = 0; i < g_train_x.size(); ++i) {
                    Vec p = g_model.predict_one(g_train_x[i]);
                    std::cout << "  pred: (" << p[0] << ", " << p[1]
                              << ")  target: (" << g_train_y_real[i][0] << ", " << g_train_y_real[i][1] << ")\n";
                }
                g_task = "regression";
            } else if (subName == "Classification") {
                loadDefaultClsData();
                g_model = make_classifier(3, 3, {8,8}, "leaky_relu", 0.01);
                g_model.summary();
                g_model.fit(g_train_x, g_train_y_cls, 2000, 2, 0.2, 50);
                int ok = 0;
                for (size_t i = 0; i < g_train_x.size(); ++i) {
                    int p = g_model.predict_class(g_train_x[i]);
                    if (p == g_train_y_cls[i]) ++ok;
                    std::cout << "  pred: " << p << "  actual: " << g_train_y_cls[i]
                              << "  " << (p == g_train_y_cls[i] ? "OK" : "FAIL") << "\n";
                }
                std::cout << "  Accuracy: " << 100.0*ok/g_train_x.size() << "%\n";
                g_task = "classification";
            } else if (subName == "Gradient Clip") {
                loadDefaultRegData();
                Model m = make_regressor(5, 2, {8,8,4}, "leaky_relu", 0.01);
                StepLR sch(m.optimizer.get(), 100, 0.5);
                m.fit(g_train_x, g_train_y_real, 300, 2, 0.0, nullptr, 0, true, true, &sch, 1.0);
                std::cout << "  Final loss: " << m.train_history.back().second
                          << "  |  LR: " << m.optimizer->lr << "\n";
                std::cout << "  R^2: " << m.score(g_train_x, g_train_y_real) << "\n";
                g_model = std::move(m);
                g_task = "regression";
            } else if (subName == "Quick Train") {
                loadDefaultRegData();
                Model qr = quick_train(g_train_x, g_train_y_real, "regression", {16,8}, 300, 2);
                std::cout << "  Regression R^2 = " << qr.score(g_train_x, g_train_y_real) << "\n";
                g_model = std::move(qr);
                g_task = "regression";
                loadDefaultClsData();
                Model qc = quick_train(g_train_x, g_train_y_cls, {16}, 300, 2);
                std::cout << "  Classification acc = " << qc.score(g_train_x, g_train_y_cls) * 100 << "%\n";
            } else if (subName == "CNN") {
                g_model = Model();
                g_task = "classification";
                Conv2d conv(1, 2, 2, 1);
                MaxPool2d pool(2);
                Flatten flatten;
                MLP fc({2, 4, 2}, std::vector<std::string>{"leaky_relu", "softmax"});
                Vec3D img(1, Vec2D(4, Vec(4, 0.0)));
                std::cout << "  Input (1x4x4):\n";
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
                std::cout << "  Image -> Conv(2,2,2) -> Pool(2) -> Flatten -> MLP\n";
                std::cout << "  Output: (" << out[0] << ", " << out[1] << ")\n";
            }
            std::string s = EndCapture();
            AppendText(g_hOutput, s);
            g_trained = true;
        };

        runSubDemo("Regression");
        runSubDemo("Classification");
        runSubDemo("Gradient Clip");
        runSubDemo("Quick Train");
        runSubDemo("CNN");
        AppendText(g_hOutput, "\r\n========== ALL DONE ==========\r\n\r\n");
        SendMessage(g_hProgress, PBM_SETPOS, 100, 0);
        SetStatus("All demos completed");
        // 同步图表
        g_lossChartTrain = g_model.train_history;
        g_lossChartVal = g_model.val_history;
        InvalidateRect(g_hWnd, &g_chartRect, FALSE);
        return;
    }

    std::string out = EndCapture();
    AppendText(g_hOutput, out);
    g_trained = true;
    SendMessage(g_hProgress, PBM_SETPOS, 100, 0);
    SetStatus(name + " completed");
    // 同步图表（空历史也会刷新，显示占位文字）
    g_lossChartTrain = g_model.train_history;
    g_lossChartVal = g_model.val_history;
    InvalidateRect(g_hWnd, &g_chartRect, FALSE);
}

// ============ 自定义绘制颜色 ============
static HBRUSH g_hBrushPanel    = nullptr;
static HBRUSH g_hBrushInput    = nullptr;
static HBRUSH g_hBrushWhite    = nullptr;

static void initBrushes() {
    g_hBrushPanel = CreateSolidBrush(CLR_PANEL_BG);
    g_hBrushInput = CreateSolidBrush(CLR_INPUT_BG);
    g_hBrushWhite = CreateSolidBrush(CLR_BG);
}

// ============ 窗口过程 ============
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE: {
        g_hWnd = hwnd;
        initBrushes();

        // 字体
        g_hFontMono = CreateFontA(17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, "Consolas");
        g_hFontUI = CreateFontA(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
        g_hFontTitle = CreateFontA(19, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
        g_hFontGrid = CreateFontA(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            DEFAULT_QUALITY, FIXED_PITCH | FF_MODERN, "Consolas");

        g_leftPanelW = 440;
        int leftX = 14, fieldW = g_leftPanelW - 24;
        int y = 56, lineH = 22, gap = 3;

        // ======== 标题 ========
        HWND hTitle = MakeStatic(hwnd, "AI Toolkit - Neural Network Playground", 18, 14, 400, 28);
        SendMessage(hTitle, WM_SETFONT, (WPARAM)g_hFontTitle, TRUE);

        // ======== 网络类型 ========
        MakeStatic(hwnd, "Network Type", leftX, y, fieldW, 14); y += 15;
        HWND cbType = MakeCombo(hwnd, IDC_CBO_NETTYPE, leftX, y, fieldW, 150);
        AddComboItem(cbType, "Regression (MSE)", true);
        AddComboItem(cbType, "Classification (CrossEntropy+Softmax)");
        y += lineH + gap + 2;

        // 架构
        MakeStatic(hwnd, "Architecture", leftX, y, fieldW, 14); y += 15;
        HWND cbArch = MakeCombo(hwnd, IDC_CBO_ARCH, leftX, y, fieldW, 150);
        AddComboItem(cbArch, "MLP (Dense)", true);
        AddComboItem(cbArch, "CNN (Conv1D)");
        AddComboItem(cbArch, "RNN");
        AddComboItem(cbArch, "Attention");
        y += lineH + gap;

        // 架构子参数
        g_hKernelLabel = MakeStatic(hwnd, "Kernel", leftX, y, 40, 15);
        ShowWindow(g_hKernelLabel, SW_HIDE);
        HWND ek = MakeEdit(hwnd, IDC_EDIT_CNNKERNEL, "3", leftX+42, y-2, 50, 20, ES_NUMBER);
        ShowWindow(ek, SW_HIDE);
        g_hFilterLabel = MakeStatic(hwnd, "Filters", leftX+100, y, 40, 15);
        ShowWindow(g_hFilterLabel, SW_HIDE);
        HWND ef = MakeEdit(hwnd, IDC_EDIT_CNNFILTERS, "8", leftX+142, y-2, 50, 20, ES_NUMBER);
        ShowWindow(ef, SW_HIDE);
        HWND er = MakeEdit(hwnd, IDC_EDIT_RNNHIDDEN, "8", leftX+42, y-2, 50, 20, ES_NUMBER);
        ShowWindow(er, SW_HIDE);
        HWND ea = MakeEdit(hwnd, IDC_EDIT_ATTNEMBED, "16", leftX+42, y-2, 50, 20, ES_NUMBER);
        ShowWindow(ea, SW_HIDE);
        HWND eh = MakeEdit(hwnd, IDC_EDIT_ATTNHEADS, "2", leftX+142, y-2, 50, 20, ES_NUMBER);
        ShowWindow(eh, SW_HIDE);
        y += lineH + gap + 2;

        // ======== 层配置 ========
        MakeStatic(hwnd, "Layer Configuration", leftX, y, fieldW, 14); y += 15;

        // 输入层
        MakeStatic(hwnd, "Input Layer", leftX+4, y+2, 78, 16);
        g_hEditInput = MakeEdit(hwnd, IDC_EDIT_INPUT, "1", leftX+85, y-2, 55, 20, ES_NUMBER);
        y += lineH + gap + 2;

        // 隐藏层 ListView 表格（固定高度，内部自带滚动）
        int layerListH = 140;  // 约显示 7 行
        g_hLayerList = CreateWindowA(WC_LISTVIEWA, "",
            WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_REPORT |
            LVS_SINGLESEL | LVS_SHOWSELALWAYS,
            leftX+8, y, fieldW - 16, layerListH, hwnd, (HMENU)(INT_PTR)IDC_LIST_LAYERS, g_hInst, nullptr);
        SendMessage(g_hLayerList, WM_SETFONT, (WPARAM)g_hFontUI, TRUE);
        ListView_SetExtendedListViewStyle(g_hLayerList,
            LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
        initLayerList();
        addLayer();  // 默认添加一层隐藏层
        y += layerListH + 4;

        // 输出层
        g_hOutputLayerLabel = MakeStatic(hwnd, "Output Layer", leftX+4, y+2, 78, 16);
        g_hEditOutput = MakeEdit(hwnd, IDC_EDIT_OUTPUT, "1", leftX+85, y-2, 55, 20, ES_NUMBER);
        y += lineH + gap;

        // 添加/删除隐藏层按钮
        int btnHalfW = (fieldW - 24) / 2;
        g_hAddLayerBtn = MakeBtn(hwnd, IDC_BTN_ADD_LAYER, "+ Add", leftX+8, y, btnHalfW, 22);
        g_hDelLayerBtn = MakeBtn(hwnd, IDC_BTN_DEL_LAYER, "- Remove", leftX + 10 + btnHalfW, y, btnHalfW, 22);
        y += 22 + 6;

        // 记录 AddLayer 按钮底部 + 6 作为下方控件的基准 y
        g_afterLayersBaseY = y;
        g_leftX = leftX;
        g_fieldW = fieldW;

        // Optimizer + Activation
        trackBelowLayer(MakeStatic(hwnd, "Optimizer", leftX, y, fieldW, 14), y); y += 15;
        HWND cbOpt = MakeCombo(hwnd, IDC_CBO_OPTIMIZER, leftX, y, fieldW/2 - 4, 150);
        AddComboItem(cbOpt, "Adam", true);
        AddComboItem(cbOpt, "SGD");
        AddComboItem(cbOpt, "Momentum");
        AddComboItem(cbOpt, "RMSprop");
        trackBelowLayer(cbOpt, y);
        HWND cbAct = MakeCombo(hwnd, IDC_CBO_ACTIVATION, leftX + fieldW/2 + 4, y, fieldW/2 - 4, 150);
        AddComboItem(cbAct, "LeakyReLU", true);
        AddComboItem(cbAct, "ReLU");
        trackBelowLayer(cbAct, y);
        y += lineH + gap;

        // LR / Seed / Epochs / Batch  (Seed empty = random)
        trackBelowLayer(MakeStatic(hwnd, "LR", leftX, y, 18, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_LR, "0.01", leftX+20, y-2, 58, 20), y-2);
        trackBelowLayer(MakeStatic(hwnd, "Seed", leftX+82, y, 30, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_SEED, "", leftX+112, y-2, 42, 20), y-2);
        trackBelowLayer(MakeStatic(hwnd, "Epochs", leftX+160, y, 42, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_EPOCHS, "500", leftX+202, y-2, 52, 20, ES_NUMBER), y-2);
        trackBelowLayer(MakeStatic(hwnd, "Batch", leftX+260, y, 35, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_BATCH, "16", leftX+295, y-2, 48, 20, ES_NUMBER), y-2);
        y += lineH + gap + 4;

        // ======== 早停 (Early Stopping) ========
        {
            HWND cbEs = CreateWindowA("BUTTON", "Early Stop",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                leftX, y-2, 80, 20, hwnd, (HMENU)(INT_PTR)IDC_CHECK_EARLYSTOP, g_hInst, nullptr);
            trackBelowLayer(cbEs, y-2);
            SendMessageA(cbEs, BM_SETCHECK, BST_CHECKED, 0);
            EnableWindow(cbEs, TRUE);
        }
        trackBelowLayer(MakeStatic(hwnd, "Val%", leftX+84, y, 30, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_VALSPLIT, "0.1", leftX+110, y-2, 42, 20), y-2);
        trackBelowLayer(MakeStatic(hwnd, "Patience", leftX+160, y, 50, 15), y);
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_PATIENCE, "50", leftX+212, y-2, 48, 20, ES_NUMBER), y-2);
        y += lineH + gap + 4;

        // ======== Dropout + L2 + Test Split ========
        {
            HWND cbDo = CreateWindowA("BUTTON", "Dropout",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                leftX, y-2, 68, 20, hwnd, (HMENU)(INT_PTR)IDC_CHECK_DROPOUT, g_hInst, nullptr);
            trackBelowLayer(cbDo, y-2);
        }
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_DROPOUT, "0.3", leftX+72, y-2, 42, 20), y-2);
        {
            HWND cbL2 = CreateWindowA("BUTTON", "L2",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                leftX+120, y-2, 38, 20, hwnd, (HMENU)(INT_PTR)IDC_CHECK_L2, g_hInst, nullptr);
            trackBelowLayer(cbL2, y-2);
        }
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_L2, "0.0001", leftX+158, y-2, 54, 20), y-2);
        {
            HWND cbTs = CreateWindowA("BUTTON", "Test%",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                leftX+218, y-2, 52, 20, hwnd, (HMENU)(INT_PTR)IDC_CHECK_TESTSPLIT, g_hInst, nullptr);
            trackBelowLayer(cbTs, y-2);
        }
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_TESTSPLIT, "0.1", leftX+274, y-2, 42, 20), y-2);
        y += lineH + gap + 4;

        // ======== 训练数据按钮 ========
        trackBelowLayer(MakeStatic(hwnd, "Training Data", leftX, y, fieldW/2, 15), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_ADD_ROW,    "+Row", leftX + fieldW - 170, y-2, 48, 20), y-2);
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_DEL_ROW,    "-Row", leftX + fieldW - 118, y-2, 48, 20), y-2);
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_CLEAR_DATA, "Clear", leftX + fieldW - 66, y-2, 48, 20), y-2);
        y += 18;

        // 数据表格
        int tableH = 180;
        g_hDataList = CreateWindowA(WC_LISTVIEWA, "",
            WS_CHILD | WS_VISIBLE | WS_BORDER | LVS_REPORT |
            LVS_EDITLABELS | LVS_SINGLESEL,
            leftX, y, fieldW, tableH, hwnd, (HMENU)(INT_PTR)IDC_LIST_DATA, g_hInst, nullptr);
        SendMessage(g_hDataList, WM_SETFONT, (WPARAM)g_hFontGrid, TRUE);
        ListView_SetExtendedListViewStyle(g_hDataList,
            LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
        rebuildDataColumns();
        updateInputOutputLock();
        trackBelowLayer(g_hDataList, y);
        y += tableH + 4;

        // ======== 训练按钮 + 进度条 ========
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_TRAIN, "Train", leftX, y, fieldW/2 - 4, 26), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_STOP, "Stop", leftX + fieldW/2 + 4, y, fieldW/2 - 4, 26), y);
        y += 30;

        g_hProgress = CreateWindowA(PROGRESS_CLASSA, "",
            WS_CHILD | WS_VISIBLE | PBS_SMOOTH | PBS_MARQUEE,
            leftX, y, fieldW, 10, hwnd, (HMENU)(INT_PTR)IDC_PROGRESS, g_hInst, nullptr);
        SendMessage(g_hProgress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        trackBelowLayer(g_hProgress, y);
        y += 14;

        // ======== Save/Load ========
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_SAVE, "Save Model", leftX, y, fieldW/2 - 4, 24), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_LOAD, "Load Model", leftX + fieldW/2 + 4, y, fieldW/2 - 4, 24), y);
        y += 28;

        // ======== 示例按钮 ========
        trackBelowLayer(MakeStatic(hwnd, "-- Examples --", leftX, y, fieldW, 14), y);
        y += 16;
        int btnH = 22, exW = fieldW/2 - 4;
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_REGRESSION, "Regression", leftX, y, exW, btnH), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_CLASSIFY, "Classification", leftX+exW+8, y, exW, btnH), y); y += btnH+2;
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_GRADCLIP, "Gradient Clip", leftX, y, exW, btnH), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_QUICKTRAIN, "Quick Train", leftX+exW+8, y, exW, btnH), y); y += btnH+2;
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_CNN, "CNN", leftX, y, exW, btnH), y);
        trackBelowLayer(MakeBtn(hwnd, IDC_EX_RUNALL, "RUN ALL", leftX+exW+8, y, exW, btnH), y); y += btnH+2;
        y += 32;

        // ======== 推理 ========
        trackBelowLayer(MakeStatic(hwnd, "-- Inference --", leftX, y, fieldW, 14), y);
        y += 16;
        trackBelowLayer(MakeEdit(hwnd, IDC_EDIT_INFER, "", leftX, y-2, fieldW, 22), y-2);
        y += 24;
        trackBelowLayer(MakeBtn(hwnd, IDC_BTN_INFER, "Predict", leftX, y, fieldW, 24), y);
        y += 30;

        // 记录层配置下方固定区域的总高度（用于后续计算 g_contentH）
        g_afterLayersBaseH = y - g_afterLayersBaseY;

        // 计算内容总高度（固定布局，不再随层数变化）
        recalcContentHeight();

        // ======== 右侧: 输出区域 ========
        int rightX = g_leftPanelW + 20;
        MakeStatic(hwnd, "Output", rightX, 14, 100, 17);
        MakeBtn(hwnd, IDC_BTN_CLEAR, "Clear", rightX + 120, 10, 50, 22);

        g_hOutput = CreateWindowExA(WS_EX_CLIENTEDGE, "LISTBOX", "",
            WS_CHILD | WS_VISIBLE | LBS_NOTIFY | LBS_HASSTRINGS |
            LBS_DISABLENOSCROLL | WS_VSCROLL | WS_HSCROLL |
            LBS_NOINTEGRALHEIGHT | LBS_NOSEL,
            rightX, 36, 460, 480, hwnd, (HMENU)(INT_PTR)IDC_OUTPUT, g_hInst, nullptr);
        SendMessage(g_hOutput, LB_SETHORIZONTALEXTENT, 2000, 0);

        g_hStatus = MakeStatic(hwnd, "Ready. Edit data in the table, configure layers, and click Train.",
            rightX, 520, 460, 17, SS_SUNKEN);

        // 应用字体
        SendMessage(g_hOutput, WM_SETFONT, (WPARAM)g_hFontMono, TRUE);
        for (HWND child = GetWindow(hwnd, GW_CHILD); child;
             child = GetWindow(child, GW_HWNDNEXT)) {
            char cls[32]; GetClassNameA(child, cls, 32);
            if (strcmp(cls, "EDIT") == 0 && child != g_hOutput) {
                // 数据表格内的 Edit 不加字体(由 g_hFontGrid 单独处理)
            } else if (child != g_hOutput) {
                SendMessage(child, WM_SETFONT, (WPARAM)g_hFontUI, TRUE);
            }
        }
        break;
    }
    case WM_NOTIFY: {
        NMHDR* nm = (NMHDR*)lp;
        // 数据表格的双击编辑
        if (nm->idFrom == IDC_LIST_DATA && nm->hwndFrom == g_hDataList) {
            switch (nm->code) {
            case NM_DBLCLK: {
                NMITEMACTIVATE* nia = (NMITEMACTIVATE*)lp;
                if (nia->iItem >= 0 && nia->iSubItem >= 0) {
                    endCellEdit(true);
                    beginCellEdit(nia->iItem, nia->iSubItem);
                }
                break;
            }
            case LVN_ENDLABELEDITA: {
                NMLVDISPINFOA* di = (NMLVDISPINFOA*)lp;
                if (di->item.pszText) {
                    ListView_SetItemText(g_hDataList, di->item.iItem, 0, di->item.pszText);
                }
                return TRUE;
            }
            }
        }
        // 层表格的双击编辑
        if (nm->idFrom == IDC_LIST_LAYERS && nm->hwndFrom == g_hLayerList) {
            if (nm->code == NM_DBLCLK) {
                NMITEMACTIVATE* nia = (NMITEMACTIVATE*)lp;
                if (nia->iItem >= 0 && nia->iSubItem == 1) {
                    endLayerEdit(true);
                    beginLayerEdit(nia->iItem, nia->iSubItem);
                }
            }
        }
        break;
    }
    case WM_COMMAND: {
        // 表格子项编辑框失去焦点时保存
        if (HIWORD(wp) == EN_KILLFOCUS && g_hDataEdit) {
            endCellEdit(true);
        }
        if (HIWORD(wp) == EN_KILLFOCUS && g_hLayerEdit) {
            endLayerEdit(true);
        }
        int id = LOWORD(wp);
        switch (id) {
        case IDC_CBO_ARCH:
            updateArchVisibility();
            break;
        case IDC_CBO_NETTYPE:
            rebuildDataColumns();
            // 重建层列表
            if (g_hLayerList) {
                endLayerEdit(false);
                ListView_DeleteAllItems(g_hLayerList);
                initLayerList();
                addLayer();  // 默认添加一层隐藏层
            }
            break;
        case IDC_EDIT_INPUT:
        case IDC_EDIT_OUTPUT:
            if (HIWORD(wp) == EN_CHANGE) {
                rebuildDataColumns();
            }
            break;

        case IDC_BTN_ADD_ROW:
            addDataRow();
            updateInputOutputLock();
            break;
        case IDC_BTN_DEL_ROW:
            delDataRow();
            updateInputOutputLock();
            break;
        case IDC_BTN_CLEAR_DATA:
            if (g_hDataList) {
                endCellEdit(false);
                ListView_DeleteAllItems(g_hDataList);
                g_has_custom_data = false;
                g_train_x.clear();
                g_train_y_real.clear();
                g_train_y_cls.clear();
                updateInputOutputLock();
                SetStatus("Training data cleared");
            }
            break;
        case IDC_BTN_ADD_LAYER:
            endLayerEdit(true);
            addLayer();
            break;
        case IDC_BTN_DEL_LAYER: {
            endLayerEdit(true);
            // 删除选中行，无选中时删最后一行
            int sel = ListView_GetNextItem(g_hLayerList, -1, LVNI_SELECTED);
            delLayer(sel);
            break;
        }

        case IDC_BTN_TRAIN:
            if (!g_training) {
                endCellEdit(true);
                ClearOutput(g_hOutput);
                g_lossChartTrain.clear();
                g_lossChartVal.clear();
                InvalidateRect(hwnd, &g_chartRect, FALSE);
                g_capture.str(""); g_capture.clear();
                SendMessage(g_hProgress, PBM_SETMARQUEE, TRUE, 30);
                SendMessage(g_hProgress, PBM_SETPOS, 0, 0);
                g_hThread = CreateThread(nullptr, 0, TrainThread, nullptr, 0, nullptr);
                SetStatus("Training started...");
            }
            break;
        case IDC_BTN_STOP:
            SetStatus("Training in progress (stop not supported in current version)");
            break;

        case IDC_BTN_SAVE:
            if (!g_trained) {
                AppendText(g_hOutput, "\r\n[!] Train a model first!\r\n");
                break;
            }
            {
                char fname[MAX_PATH] = "model.aimodel";
                OPENFILENAMEA ofn = {};
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = g_hWnd;
                ofn.lpstrFilter = "AI Model Files (*.aimodel)\0*.aimodel\0All Files (*.*)\0*.*\0\0";
                ofn.lpstrFile = fname;
                ofn.nMaxFile = MAX_PATH;
                ofn.lpstrDefExt = "aimodel";
                ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
                if (GetSaveFileNameA(&ofn)) {
                    int ep = readInt(IDC_EDIT_EPOCHS, 500);
                    int bt = readInt(IDC_EDIT_BATCH, 16);
                    g_model.save(fname, ep, bt);
                    char msg[512];
                    snprintf(msg, 512, "\r\n[Saved] Model saved to %s\r\n", fname);
                    AppendText(g_hOutput, msg);
                    SetStatus("Model saved");
                }
            }
            break;
        case IDC_BTN_LOAD:
            {
                char fname[MAX_PATH] = "";
                OPENFILENAMEA ofn = {};
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = g_hWnd;
                ofn.lpstrFilter = "AI Model Files (*.aimodel)\0*.aimodel\0All Files (*.*)\0*.*\0\0";
                ofn.lpstrFile = fname;
                ofn.nMaxFile = MAX_PATH;
                ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
                if (GetOpenFileNameA(&ofn)) {
                    try {
                        int ep = 500, bt = 16;
                        g_model = Model::load(fname, "mse", "adam", 0.01, &ep, &bt);
                        g_trained = true;
                        char msg[512];
                        snprintf(msg, 512, "\r\n[Loaded] Model loaded from %s\r\n", fname);
                        AppendText(g_hOutput, msg);
                        SetStatus("Model loaded");

                        // 从加载的模型同步 UI 配置
                        if (!g_model.network.layers.empty()) {
                            int netType = (g_model.task == "classification") ? 1 : 0;
                            int arch = 0;  // MLP

                            auto& layers = g_model.network.layers;
                            int inputDim = layers[0].fan_in;
                            int outputDim = layers.back().fan_out;
                            std::vector<int> hiddenDims;
                            for (size_t i = 0; i + 1 < layers.size(); ++i)
                                hiddenDims.push_back(layers[i].fan_out);

                            int optIdx = 0;
                            if (g_model.opt_name == "sgd") optIdx = 1;
                            else if (g_model.opt_name == "sgd_momentum") optIdx = 2;
                            else if (g_model.opt_name == "rmsprop") optIdx = 3;

                            int actIdx = 0;
                            if (!layers.empty() && layers[0].act_type == Layer::Activation::RELU) actIdx = 1;

                            syncUIConfig(netType, arch, inputDim, outputDim, hiddenDims,
                                         optIdx, actIdx, g_model.lr, ep, bt,
                                         true, 50, 0.1,
                                         g_model.dropout_rate > 0, g_model.dropout_rate,
                                         g_model.l2_lambda > 0, g_model.l2_lambda,
                                         g_model.test_split > 0, g_model.test_split);
                        }
                    } catch (...) {
                        char msg[512];
                        snprintf(msg, 512, "\r\n[!] Load failed: %s not found or invalid format\r\n", fname);
                        AppendText(g_hOutput, msg);
                    }
                }
            }
            break;
        case IDC_BTN_CLEAR:
            ClearOutput(g_hOutput);
            g_capture.str(""); g_capture.clear();
            SendMessage(g_hProgress, PBM_SETPOS, 0, 0);
            SetStatus("Output cleared");
            break;
        case IDC_BTN_INFER: {
            if (!g_trained) {
                AppendText(g_hOutput, "\r\n[!] Train a model first!\r\n");
                SetStatus("No trained model");
                break;
            }
            std::string s = readStr(IDC_EDIT_INFER);
            Vec vals;
            std::stringstream ss(s); std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { vals.push_back(std::stod(tok)); }
                catch (...) { break; }
            }
            if (vals.empty()) { SetStatus("Invalid input"); break; }

            // 检查输入维度是否匹配
            int inputDim = readInt(IDC_EDIT_INPUT, 1);
            if ((int)vals.size() != inputDim) {
                char buf[256];
                snprintf(buf, 256, "Input dimension mismatch! Expected %d values, got %d", inputDim, (int)vals.size());
                SetStatus(buf);
                std::string errMsg = "\r\n[!] ";
                errMsg += buf;
                errMsg += "\r\n";
                AppendText(g_hOutput, errMsg);
                break;
            }

            if (g_task == "classification") {
                int cls = g_model.predict_class(vals);
                char buf[256];
                snprintf(buf, 256, "Predicted class: %d", cls);
                SetStatus(buf);
                std::ostringstream oss;
                oss << "\r\n[Infer] input: (" << vals[0];
                for (size_t i=1;i<vals.size();++i) oss<<", "<<vals[i];
                oss << ") -> class " << cls << "\r\n";
                AppendText(g_hOutput, oss.str());
            } else {
                Vec pred = g_model.predict_one(vals);
                std::ostringstream oss;
                oss << "\r\n[Infer] input: (" << vals[0];
                for (size_t i=1;i<vals.size();++i) oss<<", "<<vals[i];
                oss << ") -> (";
                for (size_t i=0;i<pred.size();++i) {
                    if (i) oss << ", ";
                    oss << pred[i];
                }
                oss << ")\r\n";
                AppendText(g_hOutput, oss.str());
                char buf[256];
                snprintf(buf, 256, "Prediction: (%.4f, %.4f)", pred[0], pred.size()>1?pred[1]:0.0);
                SetStatus(buf);
            }
            break;
        }
        // 示例
        case IDC_EX_REGRESSION: runDemo("Regression"); break;
        case IDC_EX_CLASSIFY:   runDemo("Classification"); break;
        case IDC_EX_GRADCLIP:   runDemo("Gradient Clip"); break;
        case IDC_EX_QUICKTRAIN: runDemo("Quick Train"); break;
        case IDC_EX_CNN:        runDemo("CNN"); break;
        case IDC_EX_RUNALL:     runDemo("Run All"); break;
        }
        break;
    }
    case WM_USER_UPDATE_OUTPUT: {
        std::string* s = (std::string*)lp;
        if (s) { AppendText(g_hOutput, *s); delete s; }
        break;
    }
    case WM_USER_TRAINING_DONE:
        g_training = false;
        g_trained = true;
        if (g_hThread) { CloseHandle(g_hThread); g_hThread = nullptr; }
        SendMessage(g_hProgress, PBM_SETMARQUEE, FALSE, 0);
        SendMessage(g_hProgress, PBM_SETPOS, 100, 0);
        SetStatus("Training complete");
        // 同步 loss 历史到图表数据
        g_lossChartTrain = g_model.train_history;
        g_lossChartVal   = g_model.val_history;
        InvalidateRect(hwnd, &g_chartRect, FALSE);
        break;
    case WM_USER_UPDATE_PROGRESS:
        SendMessage(g_hProgress, PBM_SETPOS, (WPARAM)wp, 0);
        break;

    case WM_CTLCOLORSTATIC: {
        HDC hdc = (HDC)wp;
        HWND ctrl = (HWND)lp;
        SetBkMode(hdc, TRANSPARENT);
        if (ctrl == g_hStatus) {
            SetBkColor(hdc, CLR_PANEL_BG);
            SetTextColor(hdc, CLR_TEXT_DIM);
            return (LRESULT)g_hBrushPanel;
        }
        char cls[32]; GetClassNameA(ctrl, cls, 32);
        if (strcmp(cls, "STATIC") == 0) {
            RECT r; GetWindowRect(ctrl, &r);
            if (r.bottom - r.top <= 18) SetTextColor(hdc, CLR_TEXT_DIM);
            else SetTextColor(hdc, CLR_TEXT);
        }
        // 给 STATIC 标签一个白色背景，避免透明背景在移动时留下旧文字残影
        SetBkColor(hdc, CLR_BG);
        return (LRESULT)g_hBrushWhite;
    }
    case WM_CTLCOLOREDIT: {
        HDC hdc = (HDC)wp;
        SetBkColor(hdc, CLR_INPUT_BG);
        SetTextColor(hdc, CLR_TEXT);
        return (LRESULT)g_hBrushInput;
    }
    case WM_CTLCOLORBTN: {
        HDC hdc = (HDC)wp;
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, CLR_TEXT);
        return (LRESULT)GetStockObject(NULL_BRUSH);
    }
    case WM_CTLCOLORLISTBOX: {
        HDC hdc = (HDC)wp;
        HWND ctrl = (HWND)lp;
        if (ctrl == g_hOutput) {
            SetBkColor(hdc, CLR_OUTPUT_BG);
            SetTextColor(hdc, CLR_TEXT);
            static HBRUSH hOutBr = CreateSolidBrush(CLR_OUTPUT_BG);
            return (LRESULT)hOutBr;
        }
        SetBkColor(hdc, CLR_INPUT_BG);
        SetTextColor(hdc, CLR_TEXT);
        return (LRESULT)g_hBrushInput;
    }
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        if (g_chartRect.right > g_chartRect.left && g_chartRect.bottom > g_chartRect.top) {
            drawLossChart(hdc, g_chartRect);
        }
        EndPaint(hwnd, &ps);
        break;
    }
    case WM_ERASEBKGND: {
        HDC hdc = (HDC)wp;
        RECT rc;
        if (GetClipBox(hdc, &rc) == NULLREGION) return 1;
        HBRUSH whiteBr = (HBRUSH)GetStockObject(WHITE_BRUSH);
        FillRect(hdc, &rc, whiteBr);
        return 1;
    }
    case WM_GETMINMAXINFO: {
        // 左侧面板 440 + 间距 20 + 右侧输出最低 200 + 右侧边距 16 = 676
        // 高度：output 区域最低 100 + 顶部 36 + status 区 ≈ 200
        MINMAXINFO* mmi = (MINMAXINFO*)lp;
        mmi->ptMinTrackSize.x = 700;
        mmi->ptMinTrackSize.y = 400;
        break;
    }
    case WM_SIZE: {
        int h = HIWORD(lp);
        // 设置垂直滚动条
        int maxPos = g_contentH - h + 20;
        if (maxPos < 0) maxPos = 0;
        SCROLLINFO si = {sizeof(SCROLLINFO), SIF_RANGE | SIF_PAGE};
        si.nMin = 0;
        si.nMax = g_contentH;
        si.nPage = h;
        SetScrollInfo(hwnd, SB_VERT, &si, TRUE);
        if (g_scrollY > maxPos) g_scrollY = maxPos;

        layoutRightPanel();
        InvalidateRect(hwnd, nullptr, TRUE);
        break;
    }
    case WM_VSCROLL: {
        SCROLLINFO si = {sizeof(SCROLLINFO), SIF_ALL};
        GetScrollInfo(hwnd, SB_VERT, &si);
        int newPos = si.nPos;
        int lineDelta = 20;
        switch (LOWORD(wp)) {
            case SB_LINEUP:       newPos -= lineDelta; break;
            case SB_LINEDOWN:     newPos += lineDelta; break;
            case SB_PAGEUP:       newPos -= (int)si.nPage; break;
            case SB_PAGEDOWN:     newPos += (int)si.nPage; break;
            case SB_THUMBTRACK:   newPos = HIWORD(wp); break;
            default: break;
        }
        int maxPos = si.nMax - (int)si.nPage + 1;
        if (newPos < 0) newPos = 0;
        if (newPos > maxPos) newPos = maxPos;
        if (newPos != (int)si.nPos) {
            int delta = si.nPos - newPos;
            // 批量移动左侧面板子控件：先禁用重绘，移动后统一刷新，避免快速滚动错位
            RECT rcClient;
            GetClientRect(hwnd, &rcClient);
            RECT rcLeft = {0, 0, g_leftPanelW + 18, rcClient.bottom};

            std::vector<HWND> leftChildren;
            for (HWND child = GetWindow(hwnd, GW_CHILD); child;
                 child = GetWindow(child, GW_HWNDNEXT)) {
                if (child == g_hOutput || child == g_hStatus) continue;
                RECT rc;
                GetWindowRect(child, &rc);
                MapWindowPoints(HWND_DESKTOP, hwnd, (POINT*)&rc, 2);
                if (rc.left < g_leftPanelW + 20) {
                    leftChildren.push_back(child);
                    SendMessage(child, WM_SETREDRAW, FALSE, 0);
                }
            }

            for (HWND child : leftChildren) {
                RECT rc;
                GetWindowRect(child, &rc);
                MapWindowPoints(HWND_DESKTOP, hwnd, (POINT*)&rc, 2);
                SetWindowPos(child, NULL, rc.left, rc.top + delta, 0, 0,
                    SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOREDRAW);
            }

            for (HWND child : leftChildren) {
                SendMessage(child, WM_SETREDRAW, TRUE, 0);
            }
            RedrawWindow(hwnd, &rcLeft, NULL,
                RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);

            si.nPos = newPos;
            SetScrollInfo(hwnd, SB_VERT, &si, TRUE);
            g_scrollY = newPos;
        }
        break;
    }
    case WM_MOUSEWHEEL: {
        int zDelta = GET_WHEEL_DELTA_WPARAM(wp);
        // 转发为 SB_LINEUP/SB_LINEDOWN (每格 40 像素)
        for (int i = 0; i < 3; i++) {
            SendMessage(hwnd, WM_VSCROLL,
                zDelta > 0 ? SB_LINEUP : SB_LINEDOWN, 0);
        }
        break;
    }
    case WM_DESTROY:
        endCellEdit(false);
        if (g_hFontMono)  DeleteObject(g_hFontMono);
        if (g_hFontUI)    DeleteObject(g_hFontUI);
        if (g_hFontTitle) DeleteObject(g_hFontTitle);
        if (g_hFontGrid)  DeleteObject(g_hFontGrid);
        if (g_hBrushPanel) DeleteObject(g_hBrushPanel);
        if (g_hBrushInput) DeleteObject(g_hBrushInput);
        if (g_hBrushWhite) DeleteObject(g_hBrushWhite);
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hwnd, msg, wp, lp);
    }
    return 0;
}

// ListView 子类处理（处理子项编辑框的键盘消息）
LRESULT CALLBACK DataListProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp,
                               UINT_PTR, DWORD_PTR) {
    switch (msg) {
    case WM_VSCROLL:
    case WM_HSCROLL:
    case WM_MOUSEWHEEL:
        // 滚动时结束编辑
        endCellEdit(true);
        break;
    case WM_KEYDOWN:
        if (wp == VK_RETURN) {
            endCellEdit(true);
            return 0;
        }
        if (wp == VK_ESCAPE) {
            endCellEdit(false);
            return 0;
        }
        if (wp == VK_TAB) {
            endCellEdit(true);
            // 移到下一列
            if (g_editCol >= 0) {
                int newRow = g_editRow;
                int newCol = g_editCol + 1;
                int totalCols = getDataColumns();
                if (newCol >= totalCols) {
                    newCol = 0;
                    newRow++;
                }
                if (newRow < ListView_GetItemCount(g_hDataList)) {
                    beginCellEdit(newRow, newCol);
                }
            }
            return 0;
        }
        break;
    case WM_DESTROY:
        RemoveWindowSubclass(hwnd, DataListProc, 0);
        break;
    }
    return DefSubclassProc(hwnd, msg, wp, lp);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    g_hInst = hInstance;
    srand((unsigned)time(nullptr));
    INITCOMMONCONTROLSEX icc = {sizeof(icc), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES | ICC_LISTVIEW_CLASSES};
    InitCommonControlsEx(&icc);

    WNDCLASSA wc = {};
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "AIDemoLight";
    RegisterClassA(&wc);

    int ww = 830, wh = 640;
    HWND hwnd = CreateWindowExA(0, "AIDemoLight",
        "AI Toolkit - Neural Network Playground",
        WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, ww, wh,
        nullptr, nullptr, hInstance, nullptr);

    if (!hwnd) return 1;

    // hook: 给 ListView 加子类处理键盘事件
    // 在 Create 之后、ShowWindow 之前设置
    // 实际通过 WM_CREATE 中 g_hDataList 创建后子类化
    // 这里用 PostMessage 延迟设置
    PostMessage(hwnd, WM_USER + 200, 0, 0);

    ShowWindow(hwnd, SW_SHOWMAXIMIZED);
    UpdateWindow(hwnd);

    // 处理额外的 ListView 子类设置
    // 用 PeekMessage 确保在消息循环中处理
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        // 在消息循环中首次发现 ListView 存在时子类化
        static bool subclassed = false;
        if (!subclassed && g_hDataList) {
            SetWindowSubclass(g_hDataList, DataListProc, 0, 0);
            subclassed = true;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}
