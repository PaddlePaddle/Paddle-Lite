if(NOT LITE_WITH_NNA)
  return()
endif()

lite_cc_library(device_nna SRCS imgdnn_manager.cc DEPS ${nna_builder_libs} ${nna_runtime_libs})
