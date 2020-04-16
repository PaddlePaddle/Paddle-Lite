if(NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

lite_cc_library(target_wrapper_hw_ascend_npu SRCS target_wrapper.cc DEPS
  ${hw_ascend_npu_runtime_libs})
lite_cc_library(runtime_hw_ascend_npu SRCS runtime.cc DEPS
  ${hw_ascend_npu_runtime_libs}
  target_wrapper_hw_ascend_npu)
lite_cc_library(device_hw_ascend_npu SRCS device.cc DEPS
  ${hw_ascend_npu_runtime_libs}
  target_wrapper_hw_ascend_npu
  runtime_hw_ascend_npu)
