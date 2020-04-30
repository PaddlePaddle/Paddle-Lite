if(NOT LITE_WITH_APU)
  return()
endif()

lite_cc_library(neuron_adapter SRCS neuron_adapter.cc)
lite_cc_library(device_apu SRCS device.cc DEPS neuron_adapter)
