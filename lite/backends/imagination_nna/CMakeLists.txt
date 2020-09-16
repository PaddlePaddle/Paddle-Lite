if(NOT LITE_WITH_IMAGINATION_NNA)
  return()
endif()

configure_file(ImgdnnManagerConfig.h.in ImgdnnManagerConfig.h)

lite_cc_library(device_imagination_nna SRCS imgdnn_manager.cc DEPS ${imagination_nna_builder_libs} ${imagination_nna_runtime_libs})
target_include_directories(device_imagination_nna PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
