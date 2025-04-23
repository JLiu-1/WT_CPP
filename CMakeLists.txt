cmake_minimum_required(VERSION 3.10)
project(PyWaveletsCpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 定义一个静态库来编译 C API
add_library(pywt_c_api STATIC
    include/pywt_c/wavelets.c
)
# 告诉它头文件在哪里
target_include_directories(pywt_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pywt_c
)

# 主程序
add_executable(wt main.cpp)

# 链接上刚刚编译的 C API 库
target_link_libraries(wt PRIVATE pywt_c_api)