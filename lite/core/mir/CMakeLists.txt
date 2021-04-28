lite_cc_library(mir_node SRCS node.cc DEPS core)
lite_cc_library(mir_ssa_graph SRCS ssa_graph.cc DEPS mir_node core)
lite_cc_library(mir_pass SRCS pass.cc pass_utils.cc DEPS mir_ssa_graph)
lite_cc_library(mir_pass_manager SRCS pass_manager.cc DEPS mir_pass mir_ssa_graph mir_passes)
lite_cc_library(mir_pass_registry SRCS pass_registry.cc DEPS mir_pass_manager)

add_subdirectory(fusion)
add_subdirectory(elimination)
add_subdirectory(subgraph)

lite_cc_library(mir_passes
  SRCS
      fusion/fc_fuse_pass.cc
      fusion/matmul_fuse_pass.cc
      fusion/matmul_elementwise_add_fuse_pass.cc
      fusion/reshape2_matmul_fuse_pass.cc
      fusion/squeeze2_matmul_fuse_pass.cc
      fusion/shuffle_channel_fuse_pass.cc
      fusion/transpose_softmax_transpose_fuse_pass.cc
      fusion/interpolate_fuse_pass.cc
      fusion/conv_elementwise_fuse_pass.cc
      fusion/conv_activation_fuse_pass.cc
      fusion/var_conv_2d_activation_fuse_pass.cc
      fusion/conv_bn_fuse_pass.cc
      fusion/conv_conv_fuse_pass.cc
      fusion/conv_scale_fuse_pass.cc
      fusion/elementwise_add_activation_fuse_pass.cc
      fusion/quant_dequant_fuse_pass.cc
      fusion/sequence_pool_concat_fuse_pass.cc
      fusion/scale_activation_fuse_pass.cc
      fusion/inplace_fuse_pass.cc
      fusion/__xpu__resblock_reduction_fuse_pass.cc
      fusion/__xpu__concat_conv2d_fuse_pass.cc
      fusion/__xpu__resblock_normal_fuse_pass.cc
      fusion/__xpu__resnet_fuse_pass.cc
      fusion/__xpu__resnet_cbam_fuse_pass.cc
      fusion/__xpu__conv2d_pool2d_fuse_pass.cc
      fusion/__xpu__conv2d_concat_pool2d_fuse_pass.cc
      fusion/__xpu__consecutive_conv2d_fuse_pass.cc
      fusion/__xpu__consecutive_block_fuse_pass.cc
      fusion/__xpu__multi_encoder_fuse_pass.cc
      fusion/__xpu__embedding_with_eltwise_add_fuse_pass.cc
      fusion/__xpu__fc_fuse_pass.cc
      fusion/__xpu__mmdnn_fuse_pass.cc
      fusion/__xpu__conv2d_fuse_pass.cc
      fusion/__xpu__link_previous_out_max_pass.cc
      fusion/__xpu__sfa_head_meanstd_fuse_pass.cc
      fusion/__xpu__sfa_head_moment_fuse_pass.cc
      fusion/__xpu__softmax_topk_fuse_pass.cc
      fusion/__xpu__multi_encoder_slice_link_fuse_pass.cc
      fusion/__xpu__multi_encoder_adaptive_seqlen_fuse_pass.cc
      fusion/__xpu__generate_sequence_fuse_pass.cc
      fusion/__xpu__logit_fuse_pass.cc
      fusion/__xpu__squeeze_excitation_fuse_pass.cc
      fusion/__xpu__graph_dedup_pass.cc
      fusion/match_matrix_activation_fuse_pass.cc
      fusion/scales_fuse_pass.cc
      fusion/scaleacts_fuse_pass.cc
      fusion/sequence_reverse_embedding_fuse_pass.cc
      fusion/instance_norm_activation_fuse_pass.cc
      fusion/elementwise_add_scale_fuse_pass.cc
      fusion/fc_prelu_fuse_pass.cc
      elimination/identity_scale_eliminate_pass.cc
      elimination/identity_dropout_eliminate_pass.cc
      elimination/elementwise_mul_constant_eliminate_pass.cc
      elimination/remove_tf_redundant_ops_pass.cc
      elimination/remove_scale1_pass.cc
      elimination/ssd_boxes_calc_offline_pass.cc
      adaptive_1x1_pool2d_convert_global_pass.cc
      elimination/control_flow_op_unused_inputs_and_outputs_eliminate_pass.cc
      control_flow_op_shared_inputs_and_outputs_place_sync_pass.cc
      static_kernel_pick_pass.cc
      variable_place_inference_pass.cc
      fpga_kernel_place_correct_pass.cc
      type_target_cast_pass.cc
      type_layout_cast_pass.cc
      type_precision_cast_pass.cc
      io_copy_kernel_pick_pass.cc
      graph_visualize_pass.cc
      generate_program_pass.cc
      argument_type_display_pass.cc
      demo_pass.cc
      runtime_context_assign_pass.cc
      memory_optimize_pass.cc
      multi_stream_analysis_pass.cc
      mlu_postprocess_pass.cc
      weight_quantization_preprocess_pass.cc
      quantized_op_attributes_inference_pass.cc
      restrict_quantized_op_with_same_input_output_scale_pass.cc
      post_quant_dynamic_pass.cc
      fp16_attribute_pass.cc
  DEPS mir_pass core ${mir_fusers} ${mir_subgraphs})

set(pattern_deps mir_node mir_ssa_graph core)
if (WITH_TESTING)
  list(APPEND pattern_deps gtest)
endif()
lite_cc_library(pattern_matcher SRCS pattern_matcher.cc xpu_pattern_matcher.cc DEPS ${pattern_deps})
lite_cc_test(test_pattern_matcher SRCS pattern_matcher_test.cc DEPS pattern_matcher ${UNIT_TEST_DEPS})

lite_cc_library(pattern_matcher_high_api SRCS pattern_matcher_high_api.cc xpu_pattern_matcher_high_api.cc DEPS pattern_matcher)


# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()
lite_cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_passes ${UNIT_TEST_DEPS})
