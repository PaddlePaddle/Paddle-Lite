if(NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

lite_cc_library(build_hw_ascend_npu SRCS build.cc DEPS
  hw_ascend_npu_libs)

lite_cc_library(device_hw_ascend_npu SRCS device.cc DEPS
  hw_ascend_npu_libs
  build_hw_ascend_npu
  )
lite_cc_library(target_wrapper_hw_ascend_npu SRCS target_wrapper.cc DEPS
  hw_ascend_npu_libs
  build_hw_ascend_npu
  )
lite_cc_library(runtime_hw_ascend_npu SRCS runtime.cc DEPS
  hw_ascend_npu_libs
  target_wrapper_hw_ascend_npu
  device_hw_ascend_npu
  build_hw_ascend_npu
  )

add_executable(test_build test_build.cc)

target_link_libraries(test_build build_hw_ascend_npu)
