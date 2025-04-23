cmake_minimum_required(VERSION 3.10)
project(PyWaveletsCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(wt main.cpp)

# No external libraries required (header-only)
