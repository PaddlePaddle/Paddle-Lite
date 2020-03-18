if(NOT LITE_WITH_XPU)
  return()
endif()

if(LITE_WITH_XTCL)
  lite_cc_library(device_xpu SRCS device.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs})
endif()
lite_cc_library(target_wrapper_xpu SRCS target_wrapper.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs})
