if(NOT LITE_WITH_APU)
  return()
endif()

lite_cc_library(device_apu SRCS device.cc)
