if (NOT LITE_WITH_FPGA)
    return()
endif()

set(fpga_deps fpga_target_wrapper kernel_fpga)

lite_cc_library(elementwise_compute_fpga SRCS elementwise_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_elementwise_fpga SRCS elementwise_compute_test.cc DEPS ${lite_kernel_deps} elementwise_compute_fpga ${fpga_deps})
