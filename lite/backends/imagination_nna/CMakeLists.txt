if(NOT LITE_WITH_IMAGINATION_NNA)
  return()
endif()

lite_cc_library(device_imagination_nna SRCS imgdnn_manager.cc DEPS ${imagination_nna_builder_libs} ${imagination_nna_runtime_libs})
