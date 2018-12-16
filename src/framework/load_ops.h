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

#ifdef PADDLE_MOBILE_MALI_GPU
#define LOAD_MALI_GPU_OP(op_type)                                           \
  extern int TouchOpRegistrar_##op_type##_##mali_gpu();                     \
  static int use_op_itself_##op_type##_##mali_gpu __attribute__((unused)) = \
      TouchOpRegistrar_##op_type##_##mali_gpu()
#else
#define LOAD_MALI_GPU_OP(op_type)
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

#define LOAD_OP(op_type)     \
  LOAD_CPU_OP(op_type);      \
  LOAD_MALI_GPU_OP(op_type); \
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
LOAD_OP(fill_constant)
#endif
#ifdef BATCHNORM_OP
LOAD_OP2(batch_norm, CPU, MALI_GPU);
#endif
#ifdef BILINEAR_INTERP_OP
LOAD_OP1(bilinear_interp, CPU);
#endif
#ifdef BOXCODER_OP
LOAD_OP1(box_coder, CPU);
#endif
#ifdef CONCAT_OP
LOAD_OP3(concat, CPU, MALI_GPU, FPGA);
#endif
#ifdef CONV_OP
LOAD_OP3(conv2d, CPU, MALI_GPU, FPGA);
#endif
#ifdef LRN_OP
LOAD_OP2(lrn, CPU, MALI_GPU);
#endif
#ifdef SIGMOID_OP
LOAD_OP1(sigmoid, CPU);
#endif
#ifdef FUSION_FC_RELU_OP
LOAD_OP3(fusion_fc_relu, CPU, MALI_GPU, FPGA);
LOAD_FUSION_MATCHER(fusion_fc_relu);
#endif
#ifdef FUSION_ELEMENTWISEADDRELU_OP
LOAD_OP3(fusion_elementwise_add_relu, CPU, MALI_GPU, FPGA);
LOAD_FUSION_MATCHER(fusion_elementwise_add_relu);
#endif
#ifdef SPLIT_OP
LOAD_OP1(split, CPU);
#endif
#ifdef RESIZE_OP
LOAD_OP2(resize, CPU, MALI_GPU);
#endif
#ifdef FUSION_CONVADDBNRELU_OP
LOAD_OP2(fusion_conv_add_bn_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_bn_relu);
#endif
#ifdef RESHAPE_OP
LOAD_OP2(reshape, CPU, MALI_GPU);
#endif
#ifdef RESHAPE2_OP
LOAD_OP2(reshape2, CPU, MALI_GPU);
#endif
#ifdef TRANSPOSE_OP
LOAD_OP1(transpose, CPU);
#endif
#ifdef TRANSPOSE2_OP
LOAD_OP1(transpose2, CPU);
#endif
#ifdef PRIORBOX_OP
LOAD_OP1(prior_box, CPU);
#endif
#ifdef FUSION_CONVADDRELU_OP
LOAD_OP2(fusion_conv_add_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_relu);
#endif
#ifdef FUSION_CONVADDADDPRELU_OP
LOAD_OP2(fusion_conv_add_add_prelu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_add_prelu);
#endif
#ifdef FUSION_CONVADD_OP
LOAD_OP2(fusion_conv_add, CPU, MALI_GPU);
LOAD_FUSION_MATCHER(fusion_conv_add);
#endif
#ifdef SOFTMAX_OP
LOAD_OP2(softmax, CPU, MALI_GPU);
#endif
#ifdef SHAPE_OP
LOAD_OP1(shape, CPU);
#endif
#ifdef DEPTHWISECONV_OP
LOAD_OP1(depthwise_conv2d, CPU);
#endif
#ifdef CONV_TRANSPOSE_OP
LOAD_OP1(conv2d_transpose, CPU);
#endif
#ifdef SCALE_OP
LOAD_OP2(scale, CPU, MALI_GPU);
#endif
#ifdef ELEMENTWISEADD_OP
LOAD_OP2(elementwise_add, CPU, MALI_GPU);
#endif
#ifdef PRELU_OP
LOAD_OP2(prelu, CPU, MALI_GPU);
#endif
#ifdef FLATTEN_OP
LOAD_OP1(flatten, CPU);
#endif
#ifdef FUSION_CONVBNADDRELU_OP
LOAD_OP2(fusion_conv_bn_add_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn_add_relu);
#endif
#ifdef FUSION_CONVBNRELU_OP
LOAD_OP2(fusion_conv_bn_relu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn_relu);
#endif
#ifdef GRU_OP
LOAD_OP1(gru, CPU);
#endif
#ifdef FUSION_CONVADDBN_OP
LOAD_OP2(fusion_conv_add_bn, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_bn);
#endif
#ifdef DROPOUT_OP
LOAD_OP2(dropout, CPU, FPGA);
#endif
#ifdef FUSION_CONVADDPRELU_OP
LOAD_OP2(fusion_conv_add_prelu, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_add_prelu);
#endif
#ifdef FUSION_DWCONVBNRELU_OP
LOAD_OP1(fusion_dwconv_bn_relu, CPU);
LOAD_FUSION_MATCHER(fusion_dwconv_bn_relu);
#endif
#ifdef CRF_OP
LOAD_OP1(crf_decoding, CPU);
#endif
#ifdef MUL_OP
LOAD_OP2(mul, CPU, MALI_GPU);
#endif
#ifdef RELU_OP
LOAD_OP2(relu, CPU, MALI_GPU);
LOAD_OP1(relu6, CPU);
#endif
#ifdef IM2SEQUENCE_OP
LOAD_OP1(im2sequence, CPU);
#endif
#ifdef LOOKUP_OP
LOAD_OP1(lookup_table, CPU);
#endif
#ifdef FUSION_FC_OP
LOAD_OP3(fusion_fc, CPU, MALI_GPU, FPGA);
LOAD_FUSION_MATCHER(fusion_fc);
#endif
#ifdef POOL_OP
LOAD_OP3(pool2d, CPU, MALI_GPU, FPGA);
#endif
#ifdef MULTICLASSNMS_OP
LOAD_OP1(multiclass_nms, CPU);
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
LOAD_OP2(slice, CPU, MALI_GPU);
#endif
#ifdef FUSION_CONVBN_OP
LOAD_OP2(fusion_conv_bn, CPU, FPGA);
LOAD_FUSION_MATCHER(fusion_conv_bn);
#endif
#ifdef ELEMENTWISESUB_OP
LOAD_OP1(elementwise_sub, CPU)
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
