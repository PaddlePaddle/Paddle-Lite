if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(npu_runtime SRCS runtime.cc DEPS npu_ddk_hiai)

if(NOT LITE_ON_TINY_PUBLISH)
    add_subdirectory(bridge)
endif()
