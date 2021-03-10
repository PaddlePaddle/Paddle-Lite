if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

add_subdirectory(drivers)

lite_cc_library(nnadapter SRCS nnadapter_api.cc nnadapter_runtime.cc DEPS ${nnadapter_drivers})
