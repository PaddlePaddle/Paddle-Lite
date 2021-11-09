if(LITE_WITH_METAL)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  set(metal_kernel_deps ops metal_target_wrapper)
  find_library(METAL_LIBRARY Metal REQUIRED)
  set(lite_kernel_deps ${lite_kernel_deps} metal_target_wrapper CACHE INTERNAL "")
  add_library(METAL_BASE STATIC IMPORTED )
  set_target_properties(METAL_BASE PROPERTIES IMPORTED_LOCATION ${METAL_LIBRARY})
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()

message(STATUS "compile with lite Metal kernels")

#####################
# image kernel      #
#####################
add_kernel(activation_metal_image METAL basic SRCS image_op/activation_image_compute.mm)
add_kernel(argmax_metal_image METAL basic SRCS image_op/argmax_image_compute.mm)
add_kernel(batch_norm_metal_image METAL basic SRCS image_op/batch_norm_image_compute.mm)
add_kernel(box_coder_metal_image METAL basic SRCS image_op/box_coder_image_compute.mm)
add_kernel(cast_metal_image METAL basic SRCS image_op/cast_image_compute.mm)
add_kernel(compare_metal_image METAL basic SRCS image_op/compare_image_compute.mm)
add_kernel(concat_metal_image METAL basic SRCS image_op/concat_image_compute.mm)
add_kernel(conv2d_metal_image METAL basic SRCS image_op/conv2d_image_compute.mm)
add_kernel(conv2d_transpose_metal_image METAL basic SRCS image_op/conv2d_transpose_image_compute.mm)
add_kernel(depthwise_conv2d_metal_image METAL basic SRCS image_op/depthwise_conv2d_image_compute.mm)
add_kernel(dropout_metal_image METAL basic SRCS image_op/dropout_image_compute.mm)
add_kernel(elementwise_metal_image METAL basic SRCS image_op/elementwise_image_compute.mm)
add_kernel(exp_metal_image METAL basic SRCS image_op/exp_image_compute.mm)
add_kernel(fc_metal_image METAL basic SRCS image_op/fc_image_compute.mm)
add_kernel(feed_metal_image METAL basic SRCS image_op/feed_image_compute.mm)
add_kernel(fetch_metal_image METAL basic SRCS image_op/fetch_image_compute.mm)
add_kernel(interp_metal_image METAL basic SRCS image_op/interp_image_compute.mm)
add_kernel(io_copy_metal_image METAL basic SRCS image_op/io_copy_image_compute.mm)
add_kernel(matmul_metal_compute METAL basic SRCS image_op/matmul_image_compute.mm)
add_kernel(pad2d_metal_compute METAL basic SRCS image_op/pad2d_image_compute.mm)
add_kernel(pool_metal_image METAL basic SRCS image_op/pool_image_compute.mm)
add_kernel(reduce_metal_image METAL basic SRCS image_op/reduce_image_compute.mm)
add_kernel(reshape_metal_image METAL basic SRCS image_op/reshape_image_compute.mm)
add_kernel(scale_metal_image METAL basic SRCS image_op/scale_image_compute.mm)
add_kernel(slice_metal_compute METAL basic SRCS image_op/slice_image_compute.mm)
add_kernel(softmax_metal_image METAL basic SRCS image_op/softmax_image_compute.mm)
add_kernel(split_metal_image METAL basic SRCS image_op/split_image_compute.mm)
add_kernel(shuffle_channel_metal_compute METAL basic SRCS image_op/shuffle_channel_image_compute.mm)
add_kernel(transpose_metal_image METAL basic SRCS image_op/transpose_image_compute.mm)
add_kernel(yolo_box_metal_image METAL basic SRCS image_op/yolo_box_image_compute.mm)

# add_kernel(mul_metal_image METAL basic SRCS image_op/mul_image_compute.mm)
# add_kernel(elementwise_max_metal_image METAL basic SRCS image_op/elementwise_max_image_compute.mm)
# add_kernel(expand_metal_image METAL basic SRCS image_op/expand_image_compute.mm)
# add_kernel(grid_sampler_metal_image METAL basic SRCS image_op/grid_sampler_image_compute.mm)
# add_kernel(instance_norm_metal_image METAL basic SRCS image_op/instance_norm_image_compute.mm)
# add_kernel(lrn_metal_image METAL basic SRCS image_op/lrn_image_compute.mm)
# add_kernel(pixel_shuffle_metal_image METAL basic SRCS image_op/pixel_shuffle_image_compute.mm)
# add_kernel(rsqrt_metal_image METAL basic SRCS image_op/rsqrt_image_compute.mm)
# add_kernel(tanh_metal_image METAL basic SRCS image_op/tanh_image_compute.mm)
# add_kernel(prior_box_metal_image METAL basic SRCS image_op/prior_box_image_compute.mm)


if(LITE_WITH_METAL)
  lite_cc_test(test_batch_norm_metal_image_metal SRCS image_op/batch_norm_image_compute_test.mm
          DEPS kernels ${METAL_BASE} ${tensor_lite} core)

  lite_cc_test(test_activation_metal_image_metal SRCS image_op/activation_image_compute_test.mm
          DEPS kernels ${METAL_BASE} ${tensor_lite} core)

  lite_cc_test(test_pool_metal_image_metal SRCS image_op/pool_image_compute_test.mm
          DEPS kernels ${METAL_BASE} ${tensor_lite} core)

  lite_cc_test(test_dw_metal_image_metal SRCS image_op/conv2d_image_compute_test.mm
          DEPS kernels ${METAL_BASE} ${tensor_lite} core)
endif()
