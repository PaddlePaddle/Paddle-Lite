if(NOT LITE_WITH_RKNPU)
  return()
endif()

lite_cc_library(device_rknpu SRCS device.cc DEPS ${rknpu_builder_libs} ${rknpu_runtime_libs})
