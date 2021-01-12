set(op_DEPS tensor op op_params scope memory)


lite_cc_library(op_params SRCS op_params.cc DEPS tensor any)

# 1.baisc ops used in basic models
add_operator(basic_ops basic 
     SRCS 
         conv_op.cc
         pool_op.cc
         fc_op.cc
         mul_op.cc
         matmul_op.cc
         scale_op.cc
         softmax_op.cc
         reshape_op.cc
         batch_norm_op.cc
         feed_op.cc
         fetch_op.cc
         activation_ops.cc
         elementwise_ops.cc
         box_coder_op.cc
         multiclass_nms_op.cc
         fill_constant_op.cc 
         fill_constant_batch_size_like_op.cc
         shuffle_channel_op.cc
         yolo_box_op.cc
         interpolate_op.cc
         interpolate_v2_op.cc
         argmax_op.cc
         prior_box_op.cc
         concat_op.cc
         pad2d_op.cc
         calib_op.cc
         split_op.cc
         transpose_op.cc
         conv_transpose_op.cc
         expand_op.cc
         expand_as_op.cc
         squeeze_op.cc
         unsqueeze_op.cc
         stack_op.cc
         cast_op.cc
         affine_channel_op.cc
         affine_grid_op.cc
         range_op.cc
         reduce_mean_op.cc
         relu_op.cc
         io_copy_op.cc
         fusion_elementwise_activation_ops.cc
         io_copy_once_op.cc
         dropout_op.cc 
         layout_op.cc
         layout_once_op.cc
         instance_norm_op.cc
         subgraph_op.cc
         grid_sampler_op.cc
         flatten_op.cc
         density_prior_box_op.cc
         assign_value_op.cc
         slice_op.cc
         sequence_topk_avg_pooling_op.cc
         search_fc_op.cc
     DEPS
         ${op_DEPS})



# 2.extra ops
add_operator(extra_ops extra 
     SRCS 
         fake_quantize_moving_avg_max_abs.cc
         fake_dequantize_max_abs.cc
         pow_op.cc
         sign_op.cc
         negative_op.cc
         crop_op.cc
         crop_tensor_op.cc
         assign_op.cc
         group_norm_op.cc
         norm_op.cc
         activation_extra_ops.cc
         search_group_padding_op.cc
         lrn_op.cc
         decode_bboxes_op.cc
         uniform_random_op.cc
         axpy_op.cc
         gru_unit_op.cc
         gru_op.cc
         calib_once_op.cc
         reduce_max_op.cc
         shape_op.cc
         sequence_expand_op.cc
         sequence_unpad_op.cc
         sequence_pad_op.cc
         sequence_mask_op.cc
         im2sequence_op.cc
         gather_op.cc
         anchor_generator_op.cc
         generate_proposals_op.cc
         roi_align_op.cc
         box_clip_op.cc
         fake_quantize_range_abs_max.cc
         sequence_expand_as_op.cc
         fake_quantize_dequantize_moving_avg_max_abs.cc
         fake_quantize_dequantize_abs_max.cc
         fake_channel_wise_dequantize_max_abs.cc
         split_lod_tensor_op.cc
         merge_lod_tensor_op.cc
         reduce_prod_op.cc
         sequence_reshape_op.cc
         sequence_reverse_op.cc
         sequence_pool_op.cc
         sequence_conv_op.cc
         sequence_pool_concat_op.cc
         sequence_reverse_embedding_op.cc
         reduce_ops.cc
         match_matrix_tensor_op.cc
         search_seq_depadding_op.cc
         search_grnn_op.cc
         search_seq_softmax_op.cc
         sequence_concat_op.cc
         var_conv_2d_op.cc
         attention_padding_mask_op.cc
         sequence_arithmetic_op.cc
         conditional_block_op.cc
         collect_fpn_proposals_op.cc
         distribute_fpn_proposals_op.cc
         crf_decoding_op.cc
         ctc_align_op.cc
         max_pool_with_index_op.cc
         pixel_shuffle_op.cc
         clip_op.cc
         print_op.cc
         scatter_op.cc
         matrix_nms_op.cc
         sin_op.cc
         cos_op.cc
         unstack_op.cc
         pad3d_op.cc
         select_input_op.cc
         tensor_array_to_tensor_op.cc
         expand_v2_op.cc
         while_op.cc
         lookup_table_op.cc
         lookup_table_dequant_op.cc
         lookup_table_v2_op.cc
         beam_search_decode_op.cc
         logical_op.cc
         compare_op.cc
         read_from_array_op.cc
         beam_search_op.cc
         lod_reset_op.cc
         is_empty_op.cc
         write_to_array_op.cc
         topk_op.cc
         increment_op.cc
         layer_norm_op.cc
         sequence_softmax_op.cc
         retinanet_detection_output_op.cc
         where_index_op.cc
         one_hot_op.cc
         search_aligned_mat_mul_op.cc
         search_seq_fc_op.cc
         lstm_op.cc
         topk_pooling_op.cc
         deformable_conv_op.cc
         mean_op.cc
         __xpu__resnet50_op.cc
         __xpu__softmax_topk_op.cc
         __xpu__multi_encoder_op.cc
         __xpu__embedding_with_eltwise_add_op.cc
         __xpu__fc_op.cc
         __xpu__resnet_cbam_op.cc
         __xpu__search_attention_op.cc
         __xpu__mmdnn_op.cc
         __xpu__conv2d_op.cc
         __xpu__sfa_head_op.cc
     DEPS
         ${op_DEPS})
# 3.train ops
 add_operator(train_ops extra 
     SRCS 
         mean_grad_op.cc
         activation_grad_ops.cc
         elementwise_grad_ops.cc
         mul_grad_op.cc
         sgd_op.cc
         sequence_pool_grad_op.cc
     DEPS
         ${op_DEPS})

 
 
 if (NOT LITE_WITH_X86)
     if(NOT LITE_WITH_ARM)
         lite_cc_test(test_one_hot_op SRCS one_hot_op_test.cc DEPS one_hot_op memory scope ${op_deps} extra_ops)
     endif()
     lite_cc_test(test_fc_op SRCS fc_op_test.cc
                 DEPS basic_ops memory
                 X86_DEPS fc_compute_x86
                 ARM_DEPS arm_basic_kernels)
     lite_cc_test(test_pool_op SRCS pool_op_test.cc
                 DEPS basic_ops memory
                 ARM_DEPS arm_basic_kernels)
     lite_cc_test(test_scale_op SRCS scale_op_test.cc DEPS basic_ops memory)
     lite_cc_test(test_softmax_op SRCS softmax_op_test.cc DEPS basic_ops memory)
     #lite_cc_test(test_reshape_op SRCS reshape_op_test.cc DEPS reshape_op memory)
     lite_cc_test(test_batch_norm_op SRCS batch_norm_op_test.cc DEPS basic_ops memory)
     lite_cc_test(test_concat_op SRCS concat_op_test.cc DEPS basic_ops memory scope)
     lite_cc_test(test_calib_op SRCS calib_op_test.cc DEPS basic_ops memory ARM_DEPS arm_basic_kernels)
     lite_cc_test(test_transpose_op SRCS transpose_op_test.cc DEPS basic_ops memory)
     lite_cc_test(test_fusion_elementwise_activation_ops
                  SRCS fusion_elementwise_activation_ops_test.cc
                  DEPS basic_ops memory)
 endif()
