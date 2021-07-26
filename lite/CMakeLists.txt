include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build")

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/framework_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/model_parser/flatbuffers/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/param_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/model_parser/flatbuffers/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/cache_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/backends/opencl/utils/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/tune_cache_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/backends/opencl/utils/")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third-party/flatbuffers/pre-build/)

add_subdirectory(core)
add_subdirectory(model_parser)
add_subdirectory(api)
add_subdirectory(utils)
add_subdirectory(operators)
add_subdirectory(kernels)
add_subdirectory(backends)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lite/api/)
# tiny_publish results
add_library(paddle_light_api_shared SHARED $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:model_parser> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:operators> $<TARGET_OBJECTS:kernels> $<TARGET_OBJECTS:host_math> $<TARGET_OBJECTS:arm_math>  $<TARGET_OBJECTS:api>)
add_library(paddle_api_light_bundled STATIC $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:model_parser> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:operators> $<TARGET_OBJECTS:kernels> $<TARGET_OBJECTS:host_math> $<TARGET_OBJECTS:arm_math> $<TARGET_OBJECTS:api>)


# full_publish results
add_library(paddle_full_api_shared $<TARGET_OBJECTS:full_api> $<TARGET_OBJECTS:model_parser_pb>)
target_link_libraries(paddle_full_api_shared paddle_light_api_shared)


add_library(paddle_api_full_bundled STATIC $<TARGET_OBJECTS:full_api>  $<TARGET_OBJECTS:model_parser_pb>)

target_link_libraries(paddle_api_full_bundled paddle_api_light_bundled)

add_executable(test_model_bin  api/tools/model_test.cc)
add_dependencies(test_model_bin paddle_full_api_shared)
target_link_libraries(test_model_bin paddle_full_api_shared)
add_executable(benchmark_bin api/tools/benchmark.cc)
add_dependencies(benchmark_bin paddle_full_api_shared)
target_link_libraries(benchmark_bin paddle_full_api_shared)
