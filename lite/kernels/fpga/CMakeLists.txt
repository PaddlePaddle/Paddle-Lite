if (NOT LITE_WITH_FPGA)
    return()
endif()

set(fpga_deps fpga_target_wrapper kernel_fpga)

add_kernel(activation_compute_fpga FPGA basic SRCS activation_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_acivation_fpga SRCS activation_compute_test.cc DEPS ${lite_kernel_deps} activation_compute_fpga ${fpga_deps})

add_kernel(conv_compute_fpga FPGA basic SRCS conv_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_conv_fpga SRCS conv_compute_test.cc DEPS ${lite_kernel_deps} conv_compute_fpga ${fpga_deps})

add_kernel(elementwise_compute_fpga FPGA basic SRCS elementwise_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_elementwise_fpga SRCS elementwise_compute_test.cc DEPS ${lite_kernel_deps} elementwise_compute_fpga ${fpga_deps})


add_kernel(pooling_compute_fpga FPGA basic SRCS pooling_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_pooling_compute_fpga SRCS pooling_compute_test.cc DEPS ${lite_kernel_deps} pooling_compute_fpga ${fpga_deps})

add_kernel(scale_compute_fpga FPGA basic SRCS scale_compute.cc DEPS ${fpga_deps})

add_kernel(softmax_compute_fpga FPGA basic SRCS softmax_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_softmax_compute_fpga SRCS softmax_compute_test.cc DEPS ${lite_kernel_deps} softmax_compute_fpga ${fpga_deps})

add_kernel(fc_compute_fpga FPGA basic SRCS fc_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_fc_compute_fpga SRCS fc_compute_test.cc DEPS ${lite_kernel_deps} fc_compute_fpga ${fpga_deps})

add_kernel(io_copy_compute_fpga FPGA basic SRCS io_copy_compute.cc DEPS ${fpga_deps})
add_kernel(calib_compute_fpga FPGA basic SRCS calib_compute.cc DEPS ${fpga_deps})
add_kernel(layout_compute_fpga FPGA basic SRCS layout_compute.cc DEPS ${fpga_deps})
add_kernel(feed_compute_fpga FPGA basic SRCS feed_compute.cc DEPS ${fpga_deps})
add_kernel(fetch_compute_fpga FPGA basic SRCS fetch_compute.cc DEPS ${fpga_deps})
