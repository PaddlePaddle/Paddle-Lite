# NOTE we leave the add_kernel not protected by LITE_WITH_LIGHT_WEIGHT_FRAMEWORK so that all the kernels will be registered
# to the model_optimize_tool.
if((NOT LITE_ON_MODEL_OPTIMIZE_TOOL) AND (NOT LITE_WITH_PYTHON) AND (NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM)))
    return()
endif()

message(STATUS "compile with lite ARM kernels")

# 1. basic kernels for basic models
add_kernel(arm_basic_kernels ARM basic 
    SRCS
     conv_depthwise.cc
     conv_direct.cc
     conv_gemmlike.cc
     conv_winograd.cc 
     conv_compute.cc
     fc_compute.cc
     activation_compute.cc
     mul_compute.cc
     matmul_compute.cc
     scale_compute.cc
     softmax_compute.cc
     batch_norm_compute.cc
     elementwise_compute.cc
     pool_compute.cc
     split_compute.cc
     concat_compute.cc
     pad2d_compute.cc
     prior_box_compute.cc
     calib_compute.cc
     transpose_compute.cc
     yolo_box_compute.cc
     shuffle_channel_compute.cc
     argmax_compute.cc
     conv_transpose_compute.cc
     interpolate_compute.cc
     box_coder_compute.cc
     slice_compute.cc
     cast_compute.cc
     reduce_mean_compute.cc
     stack_compute.cc
     affine_grid_compute.cc
     affine_channel_compute.cc
     range_compute.cc
     dropout_compute.cc
     layout_compute.cc
     instance_norm_compute.cc
     grid_sampler_compute.cc
     density_prior_box_compute.cc
     assign_value_compute.cc
DEPS ${lite_kernel_deps} math_arm)


# 2. extra kernels
add_kernel(arm_extra_kernels ARM extra 
    SRCS
        negative_compute.cc
        crop_compute.cc
        pow_compute.cc
        norm_compute.cc
        group_norm_compute.cc
        lrn_compute.cc
        decode_bboxes_compute.cc
        axpy_compute.cc
        reduce_max_compute.cc
        sequence_expand_compute.cc
        im2sequence_compute.cc
        sequence_pool_compute.cc
        sequence_conv_compute.cc
        layer_norm_compute.cc 
        gather_compute.cc
        reduce_prod_compute.cc
        reduce_sum_compute.cc
        split_lod_tensor_compute.cc
        merge_lod_tensor_compute.cc
        anchor_generator_compute.cc
        generate_proposals_compute.cc
        roi_align_compute.cc
        box_clip_compute.cc
        collect_fpn_proposals_compute.cc
        distribute_fpn_proposals_compute.cc
        clip_compute.cc
        pixel_shuffle_compute.cc
        scatter_compute.cc
        sequence_expand_as_compute.cc
        gru_unit_compute.cc
        gru_compute.cc
        beam_search_decode_compute.cc
        lookup_table_compute.cc
        lookup_table_dequant_compute.cc
        sequence_softmax_compute.cc
        topk_compute.cc
        increment_compute.cc
        beam_search_compute.cc
        lod_reset_compute.cc
        lstm_compute.cc
        deformable_conv_compute.cc
        mean_compute.cc
DEPS ${lite_kernel_deps} math_arm)

# 2. extra kernels
add_kernel(arm_train_kernels ARM extra 
    SRCS
        mean_grad_compute.cc
        elementwise_grad_compute.cc
        mul_grad_compute.cc
        sgd_compute.cc
        sequence_pool_grad_compute.cc
    DEPS ${lite_kernel_deps} math_arm)

 lite_cc_test(test_scale_compute_arm SRCS scale_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_softmax_compute_arm SRCS softmax_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_batch_norm_compute_arm SRCS batch_norm_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_elementwise_compute_arm SRCS elementwise_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_pool_compute_arm SRCS pool_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_mul_compute_arm SRCS mul_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_split_compute_arm SRCS split_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_concat_compute_arm SRCS concat_compute_test.cc DEPS arm_basic_kernels)
 lite_cc_test(test_transpose_compute_arm SRCS transpose_compute_test.cc DEPS arm_basic_kernels COMPILE_LEVEL extra)
 lite_cc_test(test_dropout_compute_arm SRCS dropout_compute_test.cc DEPS arm_basic_kernels) 
 if(LITE_BUILD_EXTRA)
     lite_cc_test(test_split_lod_tensor_compute_arm SRCS split_lod_tensor_compute_test.cc DEPS arm_extra_kernels)
     lite_cc_test(test_merge_lod_tensor_compute_arm SRCS merge_lod_tensor_compute_test.cc DEPS arm_extra_kernels)
     lite_cc_test(test_lrn_compute_arm SRCS lrn_compute_test.cc DEPS arm_extra_kernels)
     lite_cc_test(test_decode_bboxes_compute_arm SRCS decode_bboxes_compute_test.cc DEPS arm_extra_kernels)
     lite_cc_test(test_axpy_compute_arm SRCS axpy_compute_test.cc DEPS arm_extra_kernels)
     lite_cc_test(test_layer_norm_compute_arm SRCS layer_norm_compute_test.cc DEPS arm_extra_kernels)
 endif()

