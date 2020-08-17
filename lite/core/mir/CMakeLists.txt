lite_cc_library(mir_node SRCS node.cc DEPS kernel)
lite_cc_library(mir_ssa_graph SRCS ssa_graph.cc DEPS mir_node program)
lite_cc_library(mir_pass SRCS pass.cc pass_utils.cc DEPS mir_ssa_graph)
lite_cc_library(mir_pass_manager SRCS pass_manager.cc DEPS mir_pass mir_ssa_graph mir_passes)
lite_cc_library(mir_pass_registry SRCS pass_registry.cc DEPS mir_pass_manager)

add_subdirectory(fusion)
add_subdirectory(elimination)
add_subdirectory(subgraph)

lite_cc_library(mir_passes
  SRCS
      fusion/fc_fuse_pass.cc
      fusion/shuffle_channel_fuse_pass.cc
      fusion/transpose_softmax_transpose_fuse_pass.cc
      fusion/interpolate_fuse_pass.cc
      fusion/conv_elementwise_fuse_pass.cc
      fusion/conv_activation_fuse_pass.cc
      fusion/var_conv_2d_activation_fuse_pass.cc
      fusion/conv_bn_fuse_pass.cc
      fusion/conv_conv_fuse_pass.cc
      fusion/elementwise_add_activation_fuse_pass.cc
      fusion/quant_dequant_fuse_pass.cc
      fusion/sequence_pool_concat_fuse_pass.cc
      fusion/scale_activation_fuse_pass.cc
      fusion/reshape_fuse_pass.cc
      fusion/__xpu__resnet_fuse_pass.cc
      fusion/__xpu__resnet_cbam_fuse_pass.cc
      fusion/__xpu__multi_encoder_fuse_pass.cc
      fusion/__xpu__embedding_with_eltwise_add_fuse_pass.cc
      fusion/__xpu__fc_fuse_pass.cc
      fusion/__xpu__mmdnn_fuse_pass.cc
      fusion/match_matrix_activation_fuse_pass.cc
      fusion/scales_fuse_pass.cc
      fusion/sequence_reverse_embedding_fuse_pass.cc
      elimination/identity_scale_eliminate_pass.cc
      elimination/identity_dropout_eliminate_pass.cc
      elimination/elementwise_mul_constant_eliminate_pass.cc
      elimination/remove_tf_redundant_ops_pass.cc
      elimination/control_flow_op_unused_inputs_and_outputs_eliminate_pass.cc
      static_kernel_pick_pass.cc
      variable_place_inference_pass.cc
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
  DEPS mir_pass types context ${mir_fusers} ${mir_subgraphs})

# lite_cc_test(test_ssa_graph SRCS ssa_graph_test.cc DEPS
        #mir_ssa_graph scope op
        #fc_op
        #${host_kernels}
        #mir_passes
        #mir_pass_manager
        #program_fake_utils
        #)
# lite_cc_test(test_variable_place_infrence_pass SRCS variable_place_inference_pass_test.cc
#   DEPS
#       mul_op
#       feed_op
#       fetch_op
#       io_copy_op
#       ${host_kernels}
#       mir_passes
#       mir_pass_manager
#       optimizer
#       program_fake_utils
#       target_wrapper_host
#   PROFILE_DEPS basic_profiler
#   CUDA_DEPS target_wrapper_cuda kernels_cuda
#   ARM_DEPS mul_compute_arm
#   X86_DEPS mul_compute_x86
# )

set(pattern_deps mir_node mir_ssa_graph op)
if (WITH_TESTING)
  list(APPEND pattern_deps gtest)
endif()
lite_cc_library(pattern_matcher SRCS pattern_matcher.cc xpu_pattern_matcher.cc DEPS ${pattern_deps})
lite_cc_test(test_pattern_matcher SRCS pattern_matcher_test.cc DEPS pattern_matcher)

lite_cc_library(pattern_matcher_high_api SRCS pattern_matcher_high_api.cc xpu_pattern_matcher_high_api.cc DEPS pattern_matcher)


# for mobile, unnecessary to compile the following testings.
if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()
lite_cc_test(test_mir_pass_manager SRCS pass_manager_test.cc DEPS mir_pass_manager mir_passes)


# TODO(wz) replace framework/proto to lite proto.
if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    # it depends on the fluid/framework/proto, that is too heavy for mobile execution.
    # TODO(wz) enable it latter.
    # lite_cc_test(test_pattern_matcher_high_api SRCS pattern_matcher_high_api_test.cc DEPS
    #     pattern_matcher_high_api proto_desc mir_pass_manager fc_op mul_op elementwise_ops
    #     mir_passes compatible_pb program ${ops})
endif()

message(STATUS "----> Ops lite: ${ops}")
message(STATUS "----> Host kernels: ${host_kernels}")
message(STATUS "----> X86 kernels: ${x86_kernels}")

# lite_cc_test(test_lite_fc_fuse SRCS fusion/fc_fuse_pass_test.cc
#    DEPS cxx_api mir_passes
#    ${ops} ${host_kernels} ${x86_kernels} ${arm_kernels}
#    ARGS --model_dir=${LITE_MODEL_DIR}/lite_fc_model
#         --optimized_model=${LITE_MODEL_DIR}/lite_fc_model_opt SERIAL)

# lite_download_and_uncompress(${LITE_MODEL_DIR} ${LITE_URL} "lite_fc_model.tar.gz")
# add_dependencies(test_lite_fc_fuse extern_lite_download_lite_fc_model_tar_gz)


# lite_cc_test(test_lite_conv_elementwise_add_activation_fuse
#              SRCS fusion/conv_elementwise_add_activation_fuse_pass_test.cc
#              DEPS cxx_api mir_passes
#              ${ops} ${host_kernels} ${x86_kernels})
# lite_cc_test(test_lite_elementwise_add_activation_fuse
#              SRCS fusion/elementwise_add_activation_fuse_pass_test.cc
#              DEPS cxx_api mir_passes
#              ${ops} ${host_kernels} ${x86_kernels})
