if(LITE_WITH_FPGA)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  set(lite_kernel_deps ${lite_kernel_deps} ${fpga_deps} CACHE INTERNAL "")
  set(fpga_deps fpga_target_wrapper kernel_fpga)
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

add_kernel(activation_compute_fpga FPGA basic SRCS activation_compute.cc)
add_kernel(cast_compute_fpga FPGA basic SRCS cast_compute.cc)
add_kernel(concat_compute_fpga FPGA basic SRCS concat_compute.cc)

add_kernel(conv_compute_fpga FPGA basic SRCS conv_compute.cc)
add_kernel(conv_transpose_compute_fpga FPGA basic SRCS conv_transpose_compute.cc)
add_kernel(dropout_compute_fpga FPGA basic SRCS dropout_compute.cc)
add_kernel(elementwise_compute_fpga FPGA basic SRCS elementwise_compute.cc)

add_kernel(interpolate_compute_fpga FPGA basic SRCS interpolate_compute.cc)

add_kernel(fc_compute_fpga FPGA basic SRCS fc_compute.cc)
add_kernel(gru_compute_fpga FPGA extra SRCS gru_compute.cc)

# add_kernel(mul_compute_fpga FPGA basic SRCS mul_compute.cc)
add_kernel(multiclass_nms_compute_fpga FPGA basic SRCS multiclass_nms_compute.cc)
add_kernel(norm_compute_fpga FPGA basic SRCS norm_compute.cc)

add_kernel(pooling_compute_fpga FPGA basic SRCS pooling_compute.cc)
add_kernel(subgraph_compute_fpga FPGA basic SRCS subgraph_compute.cc)
add_kernel(prior_box_compute_fpga FPGA basic SRCS prior_box_compute.cc)
add_kernel(reduce_mean_compute_fpga FPGA basic SRCS reduce_mean_compute.cc)
add_kernel(reshape_compute_fpga FPGA basic SRCS reshape_compute.cc)
add_kernel(scale_compute_fpga FPGA basic SRCS scale_compute.cc)
# add_kernel(softmax_compute_fpga FPGA basic SRCS softmax_compute.cc)
add_kernel(split_compute_fpga FPGA basic SRCS split_compute.cc)
add_kernel(transpose_compute_fpga FPGA basic SRCS transpose_compute.cc)

add_kernel(io_copy_compute_fpga FPGA basic SRCS io_copy_compute.cc)
add_kernel(calib_compute_fpga FPGA basic SRCS calib_compute.cc)
add_kernel(layout_compute_fpga FPGA basic SRCS layout_compute.cc)
add_kernel(feed_compute_fpga FPGA basic SRCS feed_compute.cc)
add_kernel(fetch_compute_fpga FPGA basic SRCS fetch_compute.cc)

add_kernel(yolo_box_compute_fpga FPGA basic SRCS yolo_box_compute.cc)
add_kernel(batch_norm_compute_fpga FPGA basic SRCS batch_norm_compute.cc)
add_kernel(shape_compute_fpga FPGA basic SRCS shape_compute.cc)
add_kernel(slice_compute_fpga FPGA basic SRCS slice_compute.cc)


add_kernel(unsqueeze_compute_fpga FPGA basic SRCS unsqueeze_compute.cc)


# lite_cc_test(test_acivation_fpga SRCS activation_compute_test.cc DEPS kernels)
lite_cc_test(test_conv_fpga SRCS conv_compute_test.cc DEPS kernels)
lite_cc_test(test_elementwise_fpga SRCS elementwise_compute_test.cc DEPS kernels)
lite_cc_test(test_fc_compute_fpga SRCS fc_compute_test.cc DEPS kernels)
lite_cc_test(test_pooling_compute_fpga SRCS pooling_compute_test.cc DEPS kernels)
# lite_cc_test(test_softmax_compute_fpga SRCS softmax_compute_test.cc DEPS kernels)
