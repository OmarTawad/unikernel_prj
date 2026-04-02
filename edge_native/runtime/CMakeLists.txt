cmake_minimum_required(VERSION 3.16)
project(unisplit_edge_native_runtime C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(unisplit_edge_runtime STATIC
    src/tensor.c
    src/layers_conv1d.c
    src/layers_batchnorm.c
    src/layers_relu.c
    src/layers_pool.c
    src/model_k7.c
    src/quantize.c
    src/base64.c
    src/transport_posix_http.c
    src/cloud_client.c
)

target_include_directories(unisplit_edge_runtime PUBLIC include)
target_link_libraries(unisplit_edge_runtime PUBLIC m)
target_compile_options(unisplit_edge_runtime PRIVATE -Wall -Wextra)
target_compile_definitions(unisplit_edge_runtime PRIVATE _POSIX_C_SOURCE=200809L)

add_executable(unisplit_edge_k7_cli src/main_edge_k7_cli.c)
target_link_libraries(unisplit_edge_k7_cli PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_k7_forward_testbin src/forward_testbin.c)
target_link_libraries(unisplit_edge_k7_forward_testbin PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_k7_quant_testbin src/quant_testbin.c)
target_link_libraries(unisplit_edge_k7_quant_testbin PRIVATE unisplit_edge_runtime)
