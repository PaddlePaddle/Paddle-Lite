if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

add_subdirectory(nnadapter)

lite_cc_library(nnadapter_wrapper SRCS nnadapter_wrapper.cc DEPS utils)
add_dependencies(nnadapter_wrapper nnadapter ${NNADAPTER_DEVICES})
