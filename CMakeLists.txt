cmake_minimum_required(VERSION 3.10)
# 注意：必须显式启用 C 语言，否则 CMake 会静默忽略所有 .c 源文件
# （只写 CXX 时，c/ 下的引擎源码不会被编译，链接期才会报 undefined reference）
project(AI CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ============================================================
#  纯 C 引擎（c/ 目录）
#  只取库源码：排除各程序入口（demo.c / bench.c / compare_python.c /
#  check_noalloc.c）与 tests/ 下的自测，它们各自有 main()。
#  数值上与 Python 版逐位一致，详见 c/README.md。
# ============================================================
set(C_ENGINE_SOURCES
    c/ai.c
    c/tensor.c
    c/layers_conv.c
    c/layers_norm.c
    c/rnn.c
    c/attention.c
    c/containers.c
    c/json.c
    c/model_io.c
    c/model_high.c
    c/text.c
    c/sched.c
)

# c_engine.cpp：把 C 引擎桥接给 Win32 界面（唯一包含 c/*.h 的 C++ 编译单元）
# c_model.cpp ：兼容层，用 C 引擎实现界面原本调用的 Model / make_regressor 等 API
#
# 注意：原 C++ 引擎 AI.cpp / AI.hpp 已从构建中移除（文件保留在仓库里作参考），
# 现在 ai.exe 完全跑在 c/ 下的纯 C 引擎上。
# c_engine.cpp（面板桥接层）已不再被引用：引擎统一后那个面板是冗余的，
# 该文件保留在仓库里，需要时把它加回这一行即可。
add_executable(ai WIN32 main.cpp c_model.cpp ${C_ENGINE_SOURCES})

target_link_libraries(ai PRIVATE comctl32)

if (MSVC)
    # C 源码里也有 UTF-8 中文注释，同样需要 /utf-8
    target_compile_options(ai PRIVATE /utf-8)
endif()

target_include_directories(ai PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/c)

# ============================================================
#  纯命令行版：与控制台/界面版共用同一个纯 C 引擎与兼容层
#  （不含任何 Win32 界面代码，适合脚本化 / 服务器 / CI）
# ============================================================
add_executable(ai_cli cli_main.cpp c_model.cpp ${C_ENGINE_SOURCES})
target_include_directories(ai_cli PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/c)
if (MSVC)
    target_compile_options(ai_cli PRIVATE /utf-8)
endif()
# serve 子命令用 WinSock
if (WIN32)
    target_link_libraries(ai_cli PRIVATE ws2_32)
endif()

