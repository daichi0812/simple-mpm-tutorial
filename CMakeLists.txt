cmake_minimum_required(VERSION 3.31)
project(simple-mpm-cpp)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

add_executable(simple-mpm-cpp main.cpp)

# Eigen を実行ファイルにリンクする
target_include_directories(simple-mpm-cpp PRIVATE ${EIGEN3_INCLUDE_DIR})

message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION_STRING}")