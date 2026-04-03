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
    src/layers_linear.c
    src/edge_model.c
    src/edge_runtime.c
    src/model_k7.c
    src/quantize.c
    src/base64.c
    src/transport_common.c
    src/transport_backend_factory.c
    src/transport_lwip_backend_stub.c
    src/transport_posix_backend.c
    src/transport_ukstub_backend.c
    src/cloud_client.c
    src/controller.c
    src/controller_linucb.c
    src/controller_reward.c
)

target_include_directories(unisplit_edge_runtime PUBLIC include)
target_link_libraries(unisplit_edge_runtime PUBLIC m)
target_compile_options(unisplit_edge_runtime PRIVATE -Wall -Wextra)
target_compile_definitions(unisplit_edge_runtime PRIVATE _POSIX_C_SOURCE=200809L)

add_executable(unisplit_edge_k7_cli src/main_edge_k7_cli.c)
target_link_libraries(unisplit_edge_k7_cli PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_cli src/main_edge_cli.c)
target_link_libraries(unisplit_edge_cli PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_k7_forward_testbin src/forward_testbin.c)
target_link_libraries(unisplit_edge_k7_forward_testbin PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_forward_testbin src/forward_split_testbin.c)
target_link_libraries(unisplit_edge_forward_testbin PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_k7_quant_testbin src/quant_testbin.c)
target_link_libraries(unisplit_edge_k7_quant_testbin PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_controller_testbin src/controller_testbin.c)
target_link_libraries(unisplit_edge_controller_testbin PRIVATE unisplit_edge_runtime)

add_executable(unisplit_edge_failure_testbin src/failure_testbin.c)
target_link_libraries(unisplit_edge_failure_testbin PRIVATE unisplit_edge_runtime)
