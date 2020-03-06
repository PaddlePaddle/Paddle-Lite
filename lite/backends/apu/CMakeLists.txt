if(NOT LITE_WITH_APU)
  return()
endif()

lite_cc_library(device_apu SRCS device.cc DEPS ${apu_runtime_libs})
