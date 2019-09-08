if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(npu_runtime SRCS npu_runtime.cc DEPS npu_ddk_hiai)

add_subdirectory(bridge)
