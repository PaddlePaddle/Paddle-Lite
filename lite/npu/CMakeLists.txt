if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(npu_helper SRCS npu_helper.cc DEPS ${npu_ddk_libs})
add_subdirectory(bridge)
