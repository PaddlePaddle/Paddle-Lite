FILE(GLOB MODEL_BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
FILE(GLOB BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc)
FILE(GLOB FLATBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers//*.cc)
FILE(GLOB GENERAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/general/*.cc)
FILE(GLOB NAIVEBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/naive_buffer/*.cc)
FILE(GLOB PBBUF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/pb/*.cc)



add_library(model_parser OBJECT ${BASE_SRC} ${GENERAL_SRC} ${FLATBUF_SRC} ${MODEL_BASE_SRC})

if(NOT LITE_ON_TINY_PUBLISH)
  add_library(model_parser_pb OBJECT ${PBBUF_SRC} ${NAIVEBUF_SRC})
  add_dependencies(model_parser_pb framework_proto)
  add_dependencies(model_parser framework_proto)
endif()
