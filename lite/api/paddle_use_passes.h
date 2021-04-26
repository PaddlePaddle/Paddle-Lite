// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle_lite_factory_helper.h"  // NOLINT

USE_MIR_PASS(demo);
USE_MIR_PASS(static_kernel_pick_pass);
USE_MIR_PASS(variable_place_inference_pass);
USE_MIR_PASS(type_target_cast_pass);
USE_MIR_PASS(__fpga_kernel_place_correct_pass);
USE_MIR_PASS(generate_program_pass);

USE_MIR_PASS(io_copy_kernel_pick_pass);
USE_MIR_PASS(argument_type_display_pass);
USE_MIR_PASS(runtime_context_assign_pass);
USE_MIR_PASS(graph_visualize_pass);

USE_MIR_PASS(adaptive_1x1_pool2d_convert_global_pass);
USE_MIR_PASS(remove_scale1_pass);
USE_MIR_PASS(remove_tf_redundant_ops_pass);
USE_MIR_PASS(lite_conv_bn_fuse_pass);
USE_MIR_PASS(lite_conv_conv_fuse_pass);
USE_MIR_PASS(lite_squeeze2_matmul_fuse_pass);
USE_MIR_PASS(lite_reshape2_matmul_fuse_pass);
USE_MIR_PASS(lite_matmul_fuse_pass);
USE_MIR_PASS(lite_fc_fuse_pass);
USE_MIR_PASS(lite_matmul_element_add_fuse_pass);
USE_MIR_PASS(lite_shuffle_channel_fuse_pass);
USE_MIR_PASS(lite_transpose_softmax_transpose_fuse_pass);
USE_MIR_PASS(lite_interpolate_fuse_pass);
USE_MIR_PASS(lite_sequence_pool_concat_fuse_pass);
USE_MIR_PASS(identity_scale_eliminate_pass);
USE_MIR_PASS(identity_dropout_eliminate_pass);
USE_MIR_PASS(lite_conv_elementwise_fuse_pass);
USE_MIR_PASS(lite_conv_activation_fuse_pass);
USE_MIR_PASS(lite_var_conv_2d_activation_fuse_pass);
USE_MIR_PASS(lite_match_matrix_activation_fuse_pass);
USE_MIR_PASS(lite_scales_fuse_pass);
USE_MIR_PASS(lite_sequence_reverse_embedding_fuse_pass);
USE_MIR_PASS(lite_elementwise_activation_fuse_pass);
USE_MIR_PASS(lite_elementwise_scale_fuse_pass);
USE_MIR_PASS(lite_conv_scale_fuse_pass);

USE_MIR_PASS(lite_quant_dequant_fuse_pass);
USE_MIR_PASS(type_precision_cast_pass);
USE_MIR_PASS(type_layout_cast_pass);
USE_MIR_PASS(type_layout_cast_preprocess_pass);
USE_MIR_PASS(memory_optimize_pass);
USE_MIR_PASS(lite_inplace_fuse_pass);
USE_MIR_PASS(multi_stream_analysis_pass);
USE_MIR_PASS(elementwise_mul_constant_eliminate_pass)
USE_MIR_PASS(npu_subgraph_pass);
USE_MIR_PASS(huawei_ascend_npu_subgraph_pass);
USE_MIR_PASS(imagination_nna_subgraph_pass);
USE_MIR_PASS(xpu_subgraph_pass);
USE_MIR_PASS(mlu_subgraph_pass);
USE_MIR_PASS(mlu_postprocess_pass);
USE_MIR_PASS(weight_quantization_preprocess_pass);
USE_MIR_PASS(post_quant_dynamic_pass);
USE_MIR_PASS(fp16_attribute_pass);
USE_MIR_PASS(apu_subgraph_pass);
USE_MIR_PASS(quantized_op_attributes_inference_pass);
USE_MIR_PASS(restrict_quantized_op_with_same_input_output_scale_pass);
USE_MIR_PASS(control_flow_op_unused_inputs_and_outputs_eliminate_pass);
USE_MIR_PASS(control_flow_op_shared_inputs_and_outputs_place_sync_pass);
USE_MIR_PASS(lite_scale_activation_fuse_pass);
USE_MIR_PASS(lite_instance_norm_activation_fuse_pass);
USE_MIR_PASS(ssd_boxes_calc_offline_pass);
USE_MIR_PASS(lite_fc_prelu_fuse_pass);
USE_MIR_PASS(__xpu__graph_dedup_pass);
USE_MIR_PASS(__xpu__resnet_fuse_pass);
USE_MIR_PASS(__xpu__resnet_cbam_fuse_pass);
USE_MIR_PASS(__xpu__multi_encoder_fuse_pass);
USE_MIR_PASS(__xpu__embedding_with_eltwise_add_fuse_pass);
USE_MIR_PASS(__xpu__fc_fuse_pass);
USE_MIR_PASS(__xpu__mmdnn_fuse_pass);
USE_MIR_PASS(__xpu__conv2d_fuse_pass);
USE_MIR_PASS(__xpu__resblock_reduction_fuse_pass);
USE_MIR_PASS(__xpu__resblock_normal_fuse_pass);
USE_MIR_PASS(__xpu__sfa_head_meanstd_fuse_pass);
USE_MIR_PASS(__xpu__sfa_head_moment_fuse_pass);
USE_MIR_PASS(__xpu__softmax_topk_fuse_pass);
USE_MIR_PASS(__xpu__multi_encoder_adaptive_seqlen_fuse_pass);
USE_MIR_PASS(__xpu__multi_encoder_slice_link_fuse_pass);
USE_MIR_PASS(__xpu__generate_sequence_fuse_pass);
USE_MIR_PASS(__xpu__conv2d_concat_pool2d_fuse_pass);
USE_MIR_PASS(__xpu__conv2d_pool2d_fuse_pass);
USE_MIR_PASS(__xpu__consecutive_block_fuse_pass);
USE_MIR_PASS(__xpu__consecutive_conv2d_fuse_pass);
USE_MIR_PASS(__xpu__logit_fuse_pass);
USE_MIR_PASS(__xpu__link_previous_out_max_pass);
USE_MIR_PASS(__xpu__squeeze_excitation_fuse_pass);
USE_MIR_PASS(__xpu__concat_conv2d_fuse_pass);
