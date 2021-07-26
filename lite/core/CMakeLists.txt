FILE(GLOB BASE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
FILE(GLOB PROFILE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/profile/*.cc)
FILE(GLOB SUBGRAPH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/subgraph/*.cc)
FILE(GLOB OPTIMIZER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/mir/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/mir/fusion/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/mir/elimination/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/mir/subgraph/*.cc)
FILE(GLOB TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/test/arena/*.cc)
configure_file(version.h.in version.h)


add_library(core OBJECT ${BASE_SRC})
if(NOT LITE_ON_TINY_PUBLISH)
  target_sources(core PRIVATE ${OPTIMIZER_SRC})
endif()
if(LITE_WITH_NPU OR LITE_WITH_XTCL OR LITE_WITH_BM OR LITE_WITH_RKNPU OR LITE_WITH_MLU OR LITE_WITH_APU OR LITE_WITH_HUAWEI_ASCEND_NPU OR LITE_WITH_IMAGINATION_NNA)
  target_sources(core PRIVATE ${SUBGRAPH_SRC})
endif()
if(LITE_WITH_PRECISION_PROFILE)
  target_sources(core PRIVATE ${PROFILE_SRC})
endif()


if(NOT LITE_ON_TINY_PUBLISH)
  proto_library(framework_proto SRCS framework.proto)
  add_dependencies(core framework_proto)
endif()
