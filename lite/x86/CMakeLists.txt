if (NOT LITE_WITH_X86)
    return()
endif()

configure_file(cupti_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/cupti_lib_path.h)
configure_file(warpctc_lib_path.h.in ${CMAKE_CURRENT_BINARY_DIR}/warpctc_lib_path.h)

lite_cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags)
#lite_cc_library(dynload_mklml SRCS mklml.cc DEPS dynamic_loader mklml)
lite_cc_library(target_wrapper_x86 SRCS target_wrapper.cc)
lite_cc_library(x86_cpu_info SRCS cpu_info.cc DEPS xbyak)

add_subdirectory(jit)
add_subdirectory(math)
