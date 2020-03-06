if(NOT LITE_WITH_MLU)
    return()
endif()

message (STATUS "Lite with mlu backend")

lite_cc_library(target_wrapper_mlu SRCS target_wrapper.cc DEPS cnml_lib cnrt_lib)
