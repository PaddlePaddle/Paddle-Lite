if(NOT LITE_WITH_NPU)
  return()
endif()

add_subdirectory(bridges)

lite_cc_library(npu_engine SRCS engine.cc DEPS ${npu_builder_libs} ${npu_runtime_libs} ${npu_bridges} tensor op scope program)
