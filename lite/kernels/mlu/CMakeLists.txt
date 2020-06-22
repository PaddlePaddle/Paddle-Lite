if(NOT LITE_WITH_MLU)
    return()
endif()

add_subdirectory(bridges)
add_kernel(subgraph_compute_mlu MLU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} ${mlu_subgraph_bridges})
add_kernel(io_copy_compute_mlu MLU basic SRCS io_copy_compute.cc DEPS ${lite_kernel_deps} ${target_wrapper_mlu})
add_kernel(calib_compute_mlu MLU basic SRCS calib_compute.cc DEPS ${lite_kernel_deps})
# depend on transpose function in backend/x86/math/math_function
add_kernel(roi_align_compute_mlu MLU extra SRCS roi_align_compute.cc mlu_kernel/roi_align_kernel.o DEPS ${lite_kernel_deps})

if(LITE_BUILD_EXTRA)
    lite_cc_test(test_roi_align_compute_mlu SRCS roi_align_compute_test.cc DEPS roi_align_compute_mlu)
endif()

add_kernel(layout_compute_mlu MLU basic SRCS layout_compute.cc DEPS ${lite_kernel_deps} ${math_function} ${target_wrapper_mlu})
add_kernel(cast_compute_mlu MLU basic SRCS cast_compute.cc DEPS ${lite_kernel_deps} ${target_wrapper_mlu})
