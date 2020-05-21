if(NOT LITE_WITH_MLU)
    return()
endif()

add_subdirectory(bridges)
add_kernel(subgraph_compute_mlu MLU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} ${mlu_subgraph_bridges})
add_kernel(io_copy_compute_mlu MLU basic SRCS io_copy_compute.cc DEPS ${lite_kernel_deps} ${math_mlu})
add_kernel(calib_compute_mlu MLU basic SRCS calib_compute.cc DEPS ${lite_kernel_deps} ${math_mlu})
add_kernel(layout_compute_mlu MLU basic SRCS layout_compute.cc DEPS ${lite_kernel_deps} ${math_mlu})
