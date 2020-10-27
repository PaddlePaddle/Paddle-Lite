if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT LITE_WITH_FPGA))
    return()
endif()

set(fpga_deps fpga_target_wrapper kernel_fpga fbs_headers)


add_kernel(activation_compute_fpga FPGA basic SRCS activation_compute.cc DEPS ${fpga_deps})
# add_kernel(box_coder_compute_fpga FPGA basic SRCS box_coder_compute.cc DEPS ${fpga_deps})


add_kernel(cast_compute_fpga FPGA basic SRCS cast_compute.cc DEPS ${fpga_deps})
add_kernel(concat_compute_fpga FPGA basic SRCS concat_compute.cc DEPS ${fpga_deps})

add_kernel(conv_compute_fpga FPGA basic SRCS conv_compute.cc DEPS ${fpga_deps})
# add_kernel(density_prior_box_compute_fpga FPGA basic SRCS density_prior_box_compute.cc DEPS ${fpga_deps})
add_kernel(dropout_compute_fpga FPGA basic SRCS dropout_compute.cc DEPS ${fpga_deps})
add_kernel(elementwise_compute_fpga FPGA basic SRCS elementwise_compute.cc DEPS ${fpga_deps})

add_kernel(interpolate_compute_fpga FPGA basic SRCS interpolate_compute.cc DEPS ${fpga_deps})

add_kernel(fc_compute_fpga FPGA basic SRCS fc_compute.cc DEPS ${fpga_deps})
add_kernel(gru_compute_fpga FPGA extra SRCS gru_compute.cc DEPS ${fpga_deps})

# add_kernel(mul_compute_fpga FPGA basic SRCS mul_compute.cc DEPS ${fpga_deps})
add_kernel(multiclass_nms_compute_fpga FPGA basic SRCS multiclass_nms_compute.cc DEPS ${fpga_deps})
add_kernel(norm_compute_fpga FPGA basic SRCS norm_compute.cc DEPS ${fpga_deps})

# add_kernel(im2sequence_compute_fpga FPGA basic SRCS im2sequence_compute.cc DEPS ${fpga_deps})
add_kernel(pooling_compute_fpga FPGA basic SRCS pooling_compute.cc DEPS ${fpga_deps})
add_kernel(prior_box_compute_fpga FPGA basic SRCS prior_box_compute.cc DEPS ${fpga_deps})
add_kernel(reshape_compute_fpga FPGA basic SRCS reshape_compute.cc DEPS ${fpga_deps} reshape_op)
# add_kernel(sequence_pool_compute_fpga FPGA basic SRCS sequence_pool_compute.cc DEPS ${fpga_deps})
add_kernel(scale_compute_fpga FPGA basic SRCS scale_compute.cc DEPS ${fpga_deps})
add_kernel(softmax_compute_fpga FPGA basic SRCS softmax_compute.cc DEPS ${fpga_deps})
add_kernel(split_compute_fpga FPGA basic SRCS split_compute.cc DEPS ${fpga_deps})
add_kernel(transpose_compute_fpga FPGA basic SRCS transpose_compute.cc DEPS ${fpga_deps})

add_kernel(io_copy_compute_fpga FPGA basic SRCS io_copy_compute.cc DEPS ${fpga_deps})
add_kernel(calib_compute_fpga FPGA basic SRCS calib_compute.cc DEPS ${fpga_deps})
add_kernel(layout_compute_fpga FPGA basic SRCS layout_compute.cc DEPS ${fpga_deps})
add_kernel(feed_compute_fpga FPGA basic SRCS feed_compute.cc DEPS ${fpga_deps})
add_kernel(fetch_compute_fpga FPGA basic SRCS fetch_compute.cc DEPS ${fpga_deps})

add_kernel(yolo_box_compute_fpga FPGA basic SRCS yolo_box_compute.cc DEPS ${fpga_deps})

# add_kernel(while_compute_fpga FPGA extra SRCS while_compute.cc DEPS ${fpga_deps})
# add_kernel(write_to_array_compute_fpga FPGA extra SRCS write_to_array_compute.cc DEPS ${fpga_deps})

# lite_cc_test(test_acivation_fpga SRCS activation_compute_test.cc DEPS ${lite_kernel_deps} activation_compute_fpga ${fpga_deps})
lite_cc_test(test_conv_fpga SRCS conv_compute_test.cc DEPS ${lite_kernel_deps} conv_compute_fpga ${fpga_deps})
lite_cc_test(test_elementwise_fpga SRCS elementwise_compute_test.cc DEPS ${lite_kernel_deps} elementwise_compute_fpga ${fpga_deps})
lite_cc_test(test_fc_compute_fpga SRCS fc_compute_test.cc DEPS ${lite_kernel_deps} fc_compute_fpga ${fpga_deps})
lite_cc_test(test_pooling_compute_fpga SRCS pooling_compute_test.cc DEPS ${lite_kernel_deps} pooling_compute_fpga ${fpga_deps})
# lite_cc_test(test_softmax_compute_fpga SRCS softmax_compute_test.cc DEPS ${lite_kernel_deps} softmax_compute_fpga ${fpga_deps})
