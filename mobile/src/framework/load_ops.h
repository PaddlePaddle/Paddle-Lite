/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_MOBILE_CPU
#define LOAD_CPU_OP(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_##cpu();                     \
  static int use_op_itself_##op_type##_##cpu __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_##cpu()
#else
#define LOAD_CPU_OP(op_type)
#endif

#ifdef PADDLE_MOBILE_CL
#define LOAD_GPU_CL_OP(op_type)                                       \
  extern int TouchOpRegistrar_##op_type##_##cl();                     \
  static int use_op_itself_##op_type##_##cl __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_##cl()
#else
#define LOAD_GPU_CL_OP(op_type)
#endif

#ifdef PADDLE_MOBILE_FPGA
#define LOAD_FPGA_OP(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_##fpga();                     \
  static int use_op_itself_##op_type##_##fpga __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_##fpga()
#else
#define LOAD_FPGA_OP(op_type)
#endif

#define LOAD_FUSION_MATCHER(op_type)                                       \
  extern int TouchFusionMatcherRegistrar_##op_type();                      \
  static int use_fusion_matcher_itself_##op_type __attribute__((unused)) = \
      TouchFusionMatcherRegistrar_##op_type();

#define LOAD_OP(op_type)   \
  LOAD_CPU_OP(op_type);    \
  LOAD_GPU_CL_OP(op_type); \
  LOAD_FPGA_OP(op_type);

#define LOAD_OP1(op_type, device_type) LOAD_##device_type##_OP(op_type);

#define LOAD_OP2(op_type, device_type1, device_type2) \
  LOAD_OP1(op_type, device_type1)                     \
  LOAD_OP1(op_type, device_type2)

#define LOAD_OP3(op_type, device_type1, device_type2, device_type3) \
  LOAD_OP2(op_type, device_type1, device_type2)                     \
  LOAD_OP1(op_type, device_type3)

// load requared ops
LOAD_OP(feed)
LOAD_OP(fetch)
#ifdef FILL_CONSTANT_OP
LOAD_OP2(fill_constant, CPU, FPGA)
#endif
#ifdef BATCHNORM_OP
LOAD_OP2(batch_norm, CPU, GPU_CL);
#endif
#ifdef INSTANCENORM_OP
LOAD_OP1(instance_norm, GPU_CL);
#endif
#ifdef BILINEAR_INTERP_OP
LOAD_OP1(bilinear_interp, CPU);
#endif
#ifdef NEAREST_INTERP_OP
LOAD_OP1(nearest_interp, CPU);
#endif
#ifdef LEAKY_RELU_OP
LOAD_OP1(leaky_relu, CPU);
#endif
#ifdef BOXCODER_OP
LOAD_OP2(box_coder, CPU, GPU_CL);
#endif
#ifdef CONCAT_OP
LOAD_OP3(concat, CPU, GPU_CL, FPGA);
#endif
#ifdef CONV_OP
LOAD_OP3(conv2d, CPU, GPU_CL, FPGA);
#endif
#ifdef LRN_OP
LOAD_OP2(lrn, CPU, GPU_CL);
#endif
#ifdef SIGMOID_OP
LOAD_OP1(sigmoid, CPU);
#endif
#ifdef FUSION_FC_RELU_OP
LOAD_OP2(fusion_fc_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_fc_relu);
#endif
#ifdef FUSION_ELEMENTWISEADDRELU_OP
LOAD_OP2(fusion_elementwise_add_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_elementwise_add_relu);
#endif
#ifdef SPLIT_OP
LOAD_OP2(split, CPU, GPU_CL);
#endif
#ifdef RESIZE_OP
LOAD_OP1(resize, CPU);
#endif
#ifdef FUSION_CONVADDBNRELU_OP
LOAD_OP3(fusion_conv_add_bn_relu, CPU, GPU_CL, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_bn_relu);
#endif
#ifdef RESHAPE_OP
LOAD_OP2(reshape, CPU, GPU_CL);
#endif
#ifdef RESHAPE2_OP
LOAD_OP2(reshape2, CPU, GPU_CL);
#endif
#ifdef TRANSPOSE_OP
LOAD_OP2(transpose, CPU, GPU_CL);
#endif
#ifdef TRANSPOSE2_OP
LOAD_OP2(transpose2, CPU, GPU_CL);
#endif
#ifdef PRIORBOX_OP
LOAD_OP2(prior_box, CPU, GPU_CL);
#endif
#ifdef DENSITY_PRIORBOX_OP
LOAD_OP2(density_prior_box, CPU, GPU_CL);
#endif
#ifdef FUSION_CONVADDRELU_OP
LOAD_OP3(fusion_conv_add_relu, CPU, GPU_CL, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_relu);
#endif
#ifdef FUSION_CONVADD_OP
LOAD_OP2(fusion_conv_add, CPU, GPU_CL);
LOAD_FUSION_MATCHER(fusion_conv_add);
#endif
#ifdef SOFTMAX_OP
LOAD_OP2(softmax, CPU, GPU_CL);
#endif
#ifdef SHAPE_OP
LOAD_OP1(shape, CPU);
#endif
#ifdef DEPTHWISECONV_OP
LOAD_OP2(depthwise_conv2d, CPU, GPU_CL);
#endif
#ifdef CONV_TRANSPOSE_OP
LOAD_OP2(conv2d_transpose, CPU, GPU_CL);
#endif
#ifdef SCALE_OP
LOAD_OP2(scale, CPU, GPU_CL);
#endif
#ifdef ELEMENTWISEADD_OP
LOAD_OP2(elementwise_add, CPU, GPU_CL);
#endif
#ifdef PRELU_OP
LOAD_OP1(prelu, CPU);
#endif
#ifdef TANH_OP
LOAD_OP2(tanh, CPU, GPU_CL);
#endif
#ifdef FLATTEN_OP
LOAD_OP1(flatten, CPU);
#endif
#ifdef FLATTEN2_OP
LOAD_OP2(flatten2, CPU, GPU_CL);
#endif
#ifdef FUSION_CONVBNADDRELU_OP
LOAD_OP3(fusion_conv_bn_add_relu, CPU, GPU_CL, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn_add_relu);
#endif
#ifdef FUSION_CONVBNRELU_OP
LOAD_OP3(fusion_conv_bn_relu, CPU, GPU_CL, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn_relu);
#endif
#ifdef FUSION_CONVRELU_OP
LOAD_OP2(fusion_conv_relu, CPU, GPU_CL);
LOAD_FUSION_MATCHER(fusion_conv_relu);
#endif
#ifdef GRU_OP
LOAD_OP1(gru, CPU);
#endif
#ifdef GRU_UNIT_OP
LOAD_OP1(gru_unit, CPU);
#endif
#ifdef FUSION_CONVADDBN_OP
LOAD_OP2(fusion_conv_add_bn, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_bn);
#endif
#ifdef DROPOUT_OP
LOAD_OP3(dropout, CPU, GPU_CL, FPGA);
#endif
#ifdef FUSION_DWCONVBNRELU_OP
LOAD_OP2(fusion_dwconv_bn_relu, CPU, GPU_CL);
LOAD_FUSION_MATCHER(fusion_dwconv_bn_relu);
#endif
#ifdef CRF_OP
LOAD_OP1(crf_decoding, CPU);
#endif
#ifdef MUL_OP
LOAD_OP2(mul, CPU, GPU_CL);
#endif
#ifdef NORM_OP
LOAD_OP1(norm, CPU);
#endif
#ifdef RELU_OP
LOAD_OP2(relu, CPU, GPU_CL);
LOAD_OP2(relu6, CPU, GPU_CL);
#endif
#ifdef IM2SEQUENCE_OP
LOAD_OP1(im2sequence, CPU);
#endif
#ifdef LOOKUP_OP
LOAD_OP1(lookup_table, CPU);
#endif
#ifdef FUSION_FC_OP
LOAD_OP3(fusion_fc, CPU, GPU_CL, FPGA);
LOAD_FUSION_MATCHER(fusion_fc);
#endif
#ifdef POOL_OP
LOAD_OP3(pool2d, CPU, GPU_CL, FPGA);
#endif
#ifdef MULTICLASSNMS_OP
LOAD_OP2(multiclass_nms, CPU, GPU_CL);
#endif
#ifdef POLYGONBOXTRANSFORM_OP
LOAD_OP1(polygon_box_transform, CPU);
#endif
#ifdef SUM_OP
LOAD_OP1(sum, CPU);
#endif
#ifdef ELEMENTWISEMUL_OP
LOAD_OP1(elementwise_mul, CPU);
#endif
#ifdef SLICE_OP
LOAD_OP1(slice, CPU);
#endif
#ifdef FUSION_CONVBN_OP
LOAD_OP2(fusion_conv_bn, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn);
#endif
#ifdef ELEMENTWISESUB_OP
LOAD_OP1(elementwise_sub, CPU)
#endif
#ifdef TOP_K_OP
LOAD_OP1(top_k, CPU)
#endif
#ifdef CAST_OP
LOAD_OP1(cast, CPU)
#endif
#ifdef QUANT_OP
LOAD_OP1(quantize, CPU);
#endif
#ifdef DEQUANT_OP
LOAD_OP1(dequantize, CPU);
#endif
#ifdef FUSION_DEQUANT_BN_OP
LOAD_OP1(fusion_dequant_bn, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_bn);
#endif
#ifdef FUSION_DEQUANT_ADD_BN_OP
LOAD_OP1(fusion_dequant_add_bn, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_add_bn);
#endif
#ifdef FUSION_DEQUANT_BN_RELU_OP
LOAD_OP1(fusion_dequant_bn_relu, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_bn_relu);
#endif
#ifdef FUSION_DEQUANT_ADD_BN_RELU_OP
LOAD_OP1(fusion_dequant_add_bn_relu, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_add_bn_relu);
#endif
#ifdef FUSION_DEQUANT_ADD_BN_QUANT_OP
LOAD_OP1(fusion_dequant_add_bn_quant, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_add_bn_quant);
#endif
#ifdef FUSION_DEQUANT_ADD_BN_RELU_QUANT_OP
LOAD_OP1(fusion_dequant_add_bn_relu_quant, CPU);
LOAD_FUSION_MATCHER(fusion_dequant_add_bn_relu_quant);
#endif
#ifdef SEQUENCE_EXPAND_OP
LOAD_OP1(sequence_expand, CPU);
#endif
#ifdef SEQUENCE_POOL_OP
LOAD_OP1(sequence_pool, CPU);
#endif
#ifdef SEQUENCE_SOFTMAX_OP
LOAD_OP1(sequence_softmax, CPU);
#endif
#ifdef LOG_OP
LOAD_OP1(log, CPU);
#endif
#ifdef LOD_RESET_OP
LOAD_OP1(lod_reset, CPU);
#endif
#ifdef LESS_THAN_OP
LOAD_OP1(less_than, CPU);
#endif
#ifdef LOGICAL_AND_OP
LOAD_OP1(logical_and, CPU);
#endif
#ifdef LOGICAL_OR_OP
LOAD_OP1(logical_or, CPU);
#endif
#ifdef LOGICAL_NOT_OP
LOAD_OP1(logical_not, CPU);
#endif
#ifdef LOGICAL_XOR_OP
LOAD_OP1(logical_xor, CPU);
#endif
#ifdef WHILE_OP
LOAD_OP1(while, CPU);
#endif
#ifdef WRITE_TO_ARRAY_OP
LOAD_OP1(write_to_array, CPU);
#endif
#ifdef READ_FROM_ARRAY_OP
LOAD_OP1(read_from_array, CPU);
#endif
#ifdef IS_EMPTY_OP
LOAD_OP1(is_empty, CPU);
#endif
#ifdef INCREMENT_OP
LOAD_OP1(increment, CPU);
#endif
#ifdef ANCHOR_GENERATOR_OP
LOAD_OP1(anchor_generator, CPU);
#endif
#ifdef PROPOSAL_OP
LOAD_OP1(generate_proposals, CPU);
#endif
#ifdef PSROI_POOL_OP
LOAD_OP1(psroi_pool, CPU);
#endif
#ifdef ROI_PERSPECTIVE_OP
LOAD_OP1(roi_perspective_transform, CPU);
#endif
#ifdef BEAM_SEARCH_OP
LOAD_OP1(beam_search, CPU);
#endif
#ifdef BEAM_SEARCH_DECODE_OP
LOAD_OP1(beam_search_decode, CPU);
#endif
#ifdef PAD2D_OP
LOAD_OP1(pad2d, CPU);
#endif
#ifdef ONE_HOT_OP
LOAD_OP1(one_hot, CPU);
#endif
#ifdef ASSIGN_VALUE_OP
LOAD_OP1(assign_value, CPU);
#endif
#ifdef EXP_OP
LOAD_OP1(exp, CPU);
#endif
#ifdef ASSIGN_OP
LOAD_OP1(assign, CPU);
#endif
#ifdef CONDITIONAL_BLOCK_OP
LOAD_OP1(conditional_block, CPU);
#endif
#ifdef EQUAL_OP
LOAD_OP1(equal, CPU);
#endif
#ifdef FILL_CONSTANT_BATCH_SIZE_LIKE_OP
LOAD_OP1(fill_constant_batch_size_like, CPU);
#endif
#ifdef RANGE_OP
LOAD_OP1(range, CPU);
#endif
#ifdef REDUCE_PROD_OP
LOAD_OP1(reduce_prod, CPU);
#endif
