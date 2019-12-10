if(NOT LITE_WITH_XPU)
  return()
endif()

lite_cc_library(device_xpu SRCS device.cc DEPS ${xpu_builder_libs} ${xpu_runtime_libs})
