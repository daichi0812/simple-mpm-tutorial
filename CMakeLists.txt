cmake_minimum_required(VERSION 3.21)
project(simple-mpm-cpp)

set(CMAKE_CXX_STANDARD 20)

# Eigen3 を検索
find_package(Eigen3 REQUIRED)

# SFML 3.0以上を検索
find_package(SFML 3.0 REQUIRED COMPONENTS Graphics Window System)

add_executable(${PROJECT_NAME} main.cpp)

# ライブラリをリンク
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Eigen3::Eigen
            SFML::Graphics
)

message(STATUS "Found SFML version: ${SFML_VERSION}")