if ((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT LITE_WITH_METAL))
    return ()
endif()

set(metal_kernel_deps op_params metal_target_wrapper)
find_library(METAL_LIBRARY Metal REQUIRED)
#####################
# image kernel      #
#####################
# add_kernel(batch_norm_metal_image APPLE_METAL basic SRCS image_op/batch_norm_image_compute.mm METAL_DEPS metal_target_wrapper)
add_kernel(relu_metal_image APPLE_METAL basic SRCS image_op/relu_image_compute.mm DEPS metal_target_wrapper)
add_kernel(pool_metal_image APPLE_METAL basic SRCS image_op/pool_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(mul_metal_image APPLE_METAL basic SRCS image_op/mul_image_compute.mm DEPS metal_target_wrapper)
add_kernel(reshape_metal_image APPLE_METAL basic SRCS image_op/reshape_image_compute.mm DEPS metal_target_wrapper)
add_kernel(elementwise_add_metal_image APPLE_METAL basic SRCS image_op/elementwise_add_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(elementwise_max_metal_image APPLE_METAL basic SRCS image_op/elementwise_max_image_compute.mm DEPS metal_target_wrapper)
add_kernel(elementwise_mul_metal_image APPLE_METAL basic SRCS image_op/elementwise_mul_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(elementwise_sub_metal_image APPLE_METAL basic SRCS image_op/elementwise_sub_image_compute.mm DEPS metal_target_wrapper)
add_kernel(conv2d_metal_image APPLE_METAL basic SRCS image_op/conv2d_image_compute.mm DEPS metal_target_wrapper)
add_kernel(depthwise_conv2d_metal_image APPLE_METAL basic SRCS image_op/depthwise_conv2d_image_compute.mm DEPS metal_target_wrapper)
add_kernel(softmax_metal_image APPLE_METAL basic SRCS image_op/softmax_image_compute.mm DEPS metal_target_wrapper)
add_kernel(concat_metal_image APPLE_METAL basic SRCS image_op/concat_image_compute.mm DEPS metal_target_wrapper)
add_kernel(bilinear_interp_metal_image APPLE_METAL basic SRCS image_op/bilinear_interp_image_compute.mm DEPS metal_target_wrapper)
add_kernel(scale_metal_image APPLE_METAL basic SRCS image_op/scale_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(dropout_metal_image APPLE_METAL basic SRCS image_op/dropout_image_compute.mm DEPS metal_target_wrapper)
add_kernel(fc_metal_image APPLE_METAL basic SRCS image_op/fc_image_compute.mm DEPS metal_target_wrapper)
add_kernel(feed_metal_image APPLE_METAL basic SRCS image_op/feed_image_compute.mm DEPS metal_target_wrapper)
add_kernel(sigmoid_metal_image APPLE_METAL basic SRCS image_op/sigmoid_image_compute.mm DEPS metal_target_wrapper)
add_kernel(nearest_interp_metal_image APPLE_METAL basic SRCS image_op/nearest_interp_image_compute.mm DEPS metal_target_wrapper)
add_kernel(io_copy_metal_image APPLE_METAL basic SRCS image_op/io_copy_image_compute.mm DEPS metal_target_wrapper)
add_kernel(exp_metal_image APPLE_METAL basic SRCS image_op/exp_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(expand_metal_image APPLE_METAL basic SRCS image_op/expand_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(grid_sampler_metal_image APPLE_METAL basic SRCS image_op/grid_sampler_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(hard_swish_metal_image APPLE_METAL basic SRCS image_op/hard_swish_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(instance_norm_metal_image APPLE_METAL basic SRCS image_op/instance_norm_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(lrn_metal_image APPLE_METAL basic SRCS image_op/lrn_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(pixel_shuffle_metal_image APPLE_METAL basic SRCS image_op/pixel_shuffle_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(rsqrt_metal_image APPLE_METAL basic SRCS image_op/rsqrt_image_compute.mm METAL_DEPS metal_target_wrapper)
# add_kernel(tanh_metal_image APPLE_METAL basic SRCS image_op/tanh_image_compute.mm DEPS metal_target_wrapper)
add_kernel(transpose_metal_image APPLE_METAL basic SRCS image_op/transpose_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(prior_box_metal_image APPLE_METAL basic SRCS image_op/prior_box_image_compute.mm DEPS metal_target_wrapper)
add_kernel(split_metal_image APPLE_METAL basic SRCS image_op/split_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(conv2d_transpose_metal_image APPLE_METAL basic SRCS image_op/conv2d_transpose_image_compute.mm DEPS metal_target_wrapper)
# add_kernel(box_coder_metal_image APPLE_METAL basic SRCS image_op/box_coder_image_compute.mm DEPS metal_target_wrapper)
add_kernel(fetch_metal_image APPLE_METAL basic SRCS image_op/fetch_image_compute.mm DEPS metal_target_wrapper)
add_kernel(slice_metal_compute APPLE_METAL basic SRCS image_op/slice_image_compute.mm DEPS metal_target_wrapper)
add_kernel(matmul_metal_compute APPLE_METAL basic SRCS image_op/matmul_image_compute.mm DEPS metal_target_wrapper)
add_kernel(pad2d_metal_compute APPLE_METAL basic SRCS image_op/pad2d_image_compute.mm DEPS metal_target_wrapper)


add_library(METAL_BASE STATIC IMPORTED )
set_target_properties(METAL_BASE PROPERTIES IMPORTED_LOCATION ${METAL_LIBRARY})

lite_cc_test(test_batch_norm_metal_image_metal SRCS image_op/batch_norm_image_compute_test.mm
        DEPS batch_norm_metal_image io_copy_metal_image op_registry program metal_target_wrapper ${METAL_BASE} ${tensor_lite} op_registry program context )

lite_cc_test(test_relu_metal_image_metal SRCS image_op/relu_image_compute_test.mm
        DEPS relu_metal_image io_copy_metal_image op_registry program metal_target_wrapper ${METAL_BASE} ${tensor_lite} op_registry program context )

lite_cc_test(test_pool_metal_image_metal SRCS image_op/pool_image_compute_test.mm
        DEPS pool_metal_image io_copy_metal_image op_registry program metal_target_wrapper ${METAL_BASE} ${tensor_lite} op_registry program context )

lite_cc_test(test_dw_metal_image_metal SRCS image_op/conv2d_image_compute_test.mm
        DEPS conv2d_metal_image io_copy_metal_image op_registry program metal_target_wrapper ${METAL_BASE} ${tensor_lite} op_registry program context )
