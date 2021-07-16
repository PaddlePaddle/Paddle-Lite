if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

add_subdirectory(nnadapter)

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK OR (NOT LITE_WITH_LOG))
  set(log_lib logging)
else()
  set(log_lib glog)
endif()

lite_cc_library(nnadapter_wrapper SRCS nnadapter_wrapper.cc DEPS ${log_lib})
