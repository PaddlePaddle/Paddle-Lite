if (NOT LITE_WITH_FPGA)
    return()
endif()
message("fpga : ${lite_kernel_deps}")

set(fpga_deps fpga_target_wrapper kernel_fpga)

lite_cc_library(activation_compute_fpga SRCS activation_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_acivation_fpga SRCS activation_compute_test.cc DEPS ${lite_kernel_deps} activation_compute_fpga ${fpga_deps})

lite_cc_library(conv_compute_fpga SRCS conv_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_conv_fpga SRCS conv_compute_test.cc DEPS ${lite_kernel_deps} conv_compute_fpga ${fpga_deps})

lite_cc_library(elementwise_compute_fpga SRCS elementwise_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_elementwise_fpga SRCS elementwise_compute_test.cc DEPS ${lite_kernel_deps} elementwise_compute_fpga ${fpga_deps})


lite_cc_library(pooling_compute_fpga SRCS pooling_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_pooling_compute_fpga SRCS pooling_compute_test.cc DEPS ${lite_kernel_deps} pooling_compute_fpga ${fpga_deps})

lite_cc_library(scale_compute_fpga SRCS scale_compute.cc DEPS ${fpga_deps})

lite_cc_library(softmax_compute_fpga SRCS softmax_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_softmax_compute_fpga SRCS softmax_compute_test.cc DEPS ${lite_kernel_deps} softmax_compute_fpga ${fpga_deps})

lite_cc_library(fc_compute_fpga SRCS fc_compute.cc DEPS ${fpga_deps})
lite_cc_test(test_fc_compute_fpga SRCS fc_compute_test.cc DEPS ${lite_kernel_deps} fc_compute_fpga ${fpga_deps})

lite_cc_library(io_copy_compute_fpga SRCS io_copy_compute.cc DEPS ${fpga_deps})
lite_cc_library(calib_compute_fpga SRCS calib_compute.cc DEPS ${fpga_deps})
lite_cc_library(layout_compute_fpga SRCS layout_compute.cc DEPS ${fpga_deps})
lite_cc_library(feed_compute_fpga SRCS feed_compute.cc DEPS ${fpga_deps})
#lite_cc_library(fetch_compute_fpga SRCS fetch_compute.cc DEPS ${fpga_deps})

set (fpga_kernels 
     activation_compute_fpga
     conv_compute_fpga
     elementwise_compute_fpga
     pooling_compute_fpga
     scale_compute_fpga
     softmax_compute_fpga
     fc_compute_fpga
     io_copy_compute_fpga 
     calib_compute_fpga
     layout_compute_fpga
     feed_compute_fpga
 #    fetch_compute_fpga
)

set(fpga_kernels "${fpga_kernels}" CACHE INTERNAL "fpga kernels")
