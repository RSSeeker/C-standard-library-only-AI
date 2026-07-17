cmake_minimum_required(VERSION 3.10)
project(AI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ai WIN32 main.cpp AI.cpp)

if (MSVC)
    target_compile_options(ai PRIVATE /utf-8)
else()
    target_link_libraries(ai comctl32)
endif()

target_include_directories(ai PRIVATE ${CMAKE_SOURCE_DIR})
