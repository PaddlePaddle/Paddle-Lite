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
add_library(paddle_light_api_shared SHARED $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:model_parser> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:operators> $<TARGET_OBJECTS:kernels> $<TARGET_OBJECTS:host_math> $<TARGET_OBJECTS:arm_math>  $<TARGET_OBJECTS:api>)
add_library(paddle_api_light_bundled STATIC $<TARGET_OBJECTS:core> $<TARGET_OBJECTS:model_parser> $<TARGET_OBJECTS:utils> $<TARGET_OBJECTS:operators> $<TARGET_OBJECTS:kernels> $<TARGET_OBJECTS:host_math> $<TARGET_OBJECTS:arm_math> $<TARGET_OBJECTS:api>)
