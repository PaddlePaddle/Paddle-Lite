if(NOT LITE_WITH_NPU)
  return()
endif()

lite_cc_library(device_npu SRCS device.cc DEPS ${npu_builder_libs} ${npu_runtime_libs})
