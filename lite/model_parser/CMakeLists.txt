FILE(GLOB MODEL_BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
FILE(GLOB BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc)
FILE(GLOB FLATBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers//*.cc)
FILE(GLOB GENERAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/general/*.cc)
FILE(GLOB NAIVEBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/naive_buffer/*.cc)
FILE(GLOB PBBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pb/*.cc)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build")
add_custom_target(fbs_headers ${TARGET}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/framework_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/model_parser/flatbuffers/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/param_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/model_parser/flatbuffers/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/cache_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/backends/opencl/utils/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/flatbuffers/pre-build/tune_cache_generated.h" "${CMAKE_CURRENT_SOURCE_DIR}/../../lite/backends/opencl/utils/"
        )
add_library(model_parser OBJECT ${BASE_SRC} ${FLATBUF_SRC})
add_dependencies(model_parser fbs_headers)
