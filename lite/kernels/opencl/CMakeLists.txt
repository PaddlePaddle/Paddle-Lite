
if(LITE_WITH_OPENCL)
  set(IS_FAKED_KERNEL false CACHE INTERNAL "")
  set(cl_kernel_deps ops cl_runtime cl_context cl_wrapper cl_target_wrapper cl_image_converter)
  set(lite_kernel_deps ${lite_kernel_deps} ${cl_kernel_deps} CACHE INTERNAL "")
elseif(LITE_ON_MODEL_OPTIMIZE_TOOL OR LITE_WITH_PYTHON)
  set(IS_FAKED_KERNEL true CACHE INTERNAL "")
else()
  return()
endif()


#####################
# image kernel      #
#####################
# basic
add_kernel(gather_opencl_image OPENCL basic SRCS gather_image_compute.cpp)
#add_kernel(elementwise_add_opencl_image OPENCL basic SRCS elementwise_add_image_compute.cc)
#add_kernel(elementwise_sub_opencl_image OPENCL basic SRCS elementwise_sub_image_compute.cc)
#add_kernel(elementwise_mul_opencl_image OPENCL basic SRCS elementwise_mul_image_compute.cc)
#add_kernel(fusion_elementwise_sub_activation_opencl_image
#           OPENCL basic SRCS fusion_elementwise_sub_activation_image_compute.cc)
add_kernel(elementwise_opencl_image OPENCL basic SRCS elementwise_image_compute.cc)

add_kernel(pool_opencl_image OPENCL basic SRCS pool_image_compute.cc)
add_kernel(activation_opencl_image OPENCL basic SRCS activation_image_compute.cc)
add_kernel(reshape_opencl_image OPENCL basic SRCS reshape_image_compute.cc)
add_kernel(transpose_opencl_image OPENCL basic SRCS transpose_image_compute.cc)
add_kernel(conv_opencl_image OPENCL basic SRCS conv_image_compute.cc)
add_kernel(conv_transpose_opencl_image OPENCL basic SRCS conv_transpose_image_compute.cc)
add_kernel(layout_opencl_image OPENCL basic SRCS layout_image_compute.cc)
add_kernel(concat_opencl_image OPENCL basic SRCS concat_image_compute.cc)
add_kernel(split_opencl_image OPENCL basic SRCS split_image_compute.cc)
add_kernel(nearest_interp_opencl_image OPENCL basic SRCS nearest_interp_image_compute.cc)
add_kernel(scale_opencl_image OPENCL basic SRCS scale_image_compute.cc)
add_kernel(grid_sampler_opencl_image OPENCL basic SRCS grid_sampler_image_compute.cc)
add_kernel(lrn_opencl_image OPENCL basic SRCS lrn_image_compute.cc)
add_kernel(bilinear_interp_opencl_image OPENCL basic SRCS bilinear_interp_image_compute.cc)
# add_kernel(slice_opencl_image OPENCL basic SRCS slice_image_compute.cc)
add_kernel(instance_norm_opencl_image OPENCL basic SRCS instance_norm_image_compute.cc)
add_kernel(dropout_opencl_image OPENCL basic SRCS dropout_image_compute.cc)
add_kernel(pad2d_opencl_image OPENCL basic SRCS pad2d_image_compute.cc)
add_kernel(box_coder_opencl_image OPENCL basic SRCS box_coder_image_compute.cc)
add_kernel(pixel_shuffle_opencl_image OPENCL basic SRCS pixel_shuffle_image_compute.cc)
add_kernel(expand_opencl_image OPENCL basic SRCS expand_image_compute.cc)
add_kernel(shuffle_channel_opencl_image OPENCL basic SRCS shuffle_channel_image_compute.cc)
add_kernel(trigonometric_opencl_image OPENCL basic SRCS trigonometric_image_compute.cc)
add_kernel(batch_norm_opencl_image OPENCL basic SRCS batch_norm_image_compute.cc)
add_kernel(reduce_mean_opencl_image OPENCL basic SRCS reduce_mean_image_compute.cc)
add_kernel(clip_opencl_image OPENCL basic SRCS clip_image_compute.cc)
add_kernel(softmax_opencl_image OPENCL basic SRCS softmax_image_compute.cc)
add_kernel(greater_than_opencl_image OPENCL basic SRCS greater_than_image_compute.cc)
add_kernel(fc_opencl_image OPENCL basic SRCS fc_image_compute.cc)
add_kernel(argmax_opencl_image OPENCL basic SRCS argmax_image_compute.cc)
add_kernel(max_opencl_image OPENCL basic SRCS max_image_compute.cc)
add_kernel(matmul_opencl_image OPENCL basic SRCS matmul_image_compute.cc)
add_kernel(layer_norm_opencl_image OPENCL basic SRCS layer_norm_image_compute.cc)
# extra
# wait to add ...




######################
# image kernel test  #
######################
#lite_cc_test(test_gather_image_opencl SRCS gather_image_compute_test.cpp
#             DEPS kernels core)

lite_cc_test(test_activation_image_opencl SRCS activation_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_conv_image_opencl SRCS conv_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_depthwise_conv2d_image_opencl SRCS depthwise_conv2d_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_conv_transpose_image_opencl SRCS conv_transpose_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_depthwise_conv2d_transpose_image_opencl SRCS depthwise_conv2d_transpose_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_nearest_interp_image_opencl SRCS nearest_interp_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_pool_image_opencl SRCS pool_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_reshape_image_opencl SRCS reshape_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_transpose_image_opencl SRCS transpose_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_concat_image_opencl SRCS concat_image_compute_test.cc
             DEPS kernels core)

#lite_cc_test(test_elementwise_mul_image_opencl SRCS elementwise_mul_image_compute_test.cc
#             DEPS kernels core)

lite_cc_test(test_layout_image_opencl SRCS layout_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_pixel_shuffle_image_opencl SRCS pixel_shuffle_image_compute_test.cc
             DEPS kernels core)

#lite_cc_test(test_expand_image_opencl SRCS expand_image_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_elementwise_add_image_opencl SRCS elementwise_add_image_compute_test.cc
#             DEPS kernels core)
#lite_cc_test(test_elementwise_sub_image_opencl SRCS elementwise_sub_image_compute_test.cc
#             DEPS kernels core)

lite_cc_test(test_elementwise_image_opencl SRCS elementwise_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_grid_sampler_image_opencl SRCS grid_sampler_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_lrn_image_opencl SRCS lrn_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_bilinear_interp_image_opencl SRCS bilinear_interp_image_compute_test.cc
             DEPS kernels core)
#lite_cc_test(test_slice_image_opencl SRCS slice_image_compute_test.cc
#             DEPS kernels core)

lite_cc_test(test_instance_norm_image_opencl SRCS instance_norm_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_dropout_image_opencl SRCS dropout_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_pad2d_image_opencl SRCS pad2d_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_box_coder_image_opencl SRCS box_coder_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_trigonometric_image_opencl SRCS trigonometric_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_fc_image_opencl SRCS fc_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_softmax_image_opencl SRCS softmax_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_argmax_image_opencl SRCS argmax_image_compute_test.cc
             DEPS kernels core)

lite_cc_test(test_max_image_opencl SRCS max_image_compute_test.cc
             DEPS kernels core)

#lite_cc_test(test_matmul_image_opencl SRCS matmul_image_compute_test.cc
#             DEPS kernels core)
######################
# buffer kernel      #
######################
# basic
add_kernel(activation_opencl_buffer OPENCL basic SRCS activation_buffer_compute.cc)
#add_kernel(conv_opencl_buffer OPENCL basic SRCS conv_buffer_compute.cc)
#add_kernel(depthwise_conv2d_opencl OPENCL basic SRCS depthwise_conv2d_buffer_compute.cc)
#add_kernel(pool_opencl_buffer OPENCL basic SRCS pool_buffer_compute.cc)
#add_kernel(concat_opencl_buffer OPENCL basic SRCS concat_buffer_compute.cc)
add_kernel(fc_opencl_buffer OPENCL basic SRCS fc_buffer_compute.cc)
# NOTE(ysh329): use fc as `mul`, and mul is not used.
#add_kernel(mul_opencl_buffer OPENCL basic SRCS mul_buffer_compute.cc)
#add_kernel(elementwise_add_opencl_buffer OPENCL basic SRCS elementwise_add_buffer_compute.cc)
#add_kernel(fusion_elementwise_add_activation_opencl_buffer
#           OPENCL basic SRCS fusion_elementwise_add_activation_buffer_compute.cc)
add_kernel(io_copy_opencl_buffer OPENCL basic SRCS io_copy_buffer_compute.cc)
add_kernel(shape_opencl_buffer OPENCL basic SRCS shape_buffer_compute.cc)
add_kernel(slice_opencl_buffer OPENCL basic SRCS slice_buffer_compute.cc)
add_kernel(yolo_box_opencl_buffer OPENCL basic SRCS yolo_box_buffer_compute.cc)
add_kernel(squeeze_unsqueeze_opencl_buffer OPENCL basic SRCS squeeze_unsqueeze_buffer_compute.cc)
add_kernel(matmul_opencl_buffer OPENCL basic SRCS matmul_buffer_compute.cc)
add_kernel(reshape_opencl_buffer OPENCL basic SRCS reshape_buffer_compute.cc)
add_kernel(transpose_opencl_buffer OPENCL basic SRCS transpose_buffer_compute.cc)
add_kernel(pow_opencl_buffer OPENCL basic SRCS pow_buffer_compute.cc)
add_kernel(scale_opencl_buffer OPENCL basic SRCS scale_buffer_compute.cc)
# extra
# wait to add ...



######################
# buffer kernel test #
######################
#lite_cc_test(test_activation_buffer_opencl SRCS activation_buffer_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_conv_buffer_opencl SRCS conv_buffer_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_im2col_buffer_opencl SRCS im2col_buffer_test.cc
#             DEPS kernels core)

#lite_cc_test(test_depthwise_conv2d_buffer_opencl SRCS depthwise_conv2d_buffer_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_pool_buffer_opencl SRCS pool_buffer_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_concat_buffer_opencl SRCS concat_buffer_compute_test.cc
#             DEPS kernels core)

lite_cc_test(test_fc_buffer_opencl SRCS fc_buffer_compute_test.cc
             DEPS kernels core)

#lite_cc_test(test_mul_buffer_opencl SRCS mul_buffer_compute_test.cc
#             DEPS kernels core)

#lite_cc_test(test_elementwise_add_buffer_opencl SRCS elementwise_add__buffer_compute_test.cc
#             DEPS kernels core)

lite_cc_test(test_io_copy_buffer_opencl SRCS io_copy_buffer_compute_test.cc
             DEPS kernels core)
