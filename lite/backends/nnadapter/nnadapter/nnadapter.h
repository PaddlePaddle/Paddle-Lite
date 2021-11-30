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
//
// The naming style of NNAdapter types and API refers to Google's NNAPI.
// The description of some NNAdapter operations refers to ONNX, PaddlePaddle,
// PyTorch and Tensorflow.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

enum { NNADAPTER_VERSION = 1 };
enum { NNADAPTER_UNKNOWN = -1 };

/**
 * Result codes.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_NO_ERROR = 0,
  NNADAPTER_OUT_OF_MEMORY = 1,
  NNADAPTER_INVALID_PARAMETER = 2,
  NNADAPTER_DEVICE_NOT_FOUND = 3,
  NNADAPTER_DEVICE_INTERNAL_ERROR = 4,
} NNAdapterResultCode;

enum { NNADAPTER_MAX_SIZE_OF_DIMENSIONS = 8 };
enum { NNADAPTER_MAX_SIZE_OF_DYNAMIC_DIMENSIONS = 8 };

/**
 * Operand precision codes.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_BOOL8 = 0,
  NNADAPTER_INT8 = 1,
  NNADAPTER_UINT8 = 2,
  NNADAPTER_INT16 = 3,
  NNADAPTER_UINT16 = 4,
  NNADAPTER_INT32 = 5,
  NNADAPTER_UINT32 = 6,
  NNADAPTER_INT64 = 7,
  NNADAPTER_UINT64 = 8,
  NNADAPTER_FLOAT16 = 9,
  NNADAPTER_FLOAT32 = 10,
  NNADAPTER_FLOAT64 = 11,
  /**
   * A tensor of 8/16/32 bit signed/unsigned integers that represent real
   * numbers.
   * - scale: a 32 bit floating point value greater than zero.
   *
   * The formula is:
   * real_value = integer_value * scale.
   */
  NNADAPTER_QUANT_INT8_SYMM_PER_LAYER = 24,
  NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL = 25,
  NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER = 26,
  NNADAPTER_QUANT_INT16_SYMM_PER_LAYER = 27,
  NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL = 28,
  NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER = 29,
  NNADAPTER_QUANT_INT32_SYMM_PER_LAYER = 30,
  NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL = 31,
  NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER = 32,
} NNAdapterOperandPrecisionCode;

/**
 * Operand layout codes.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_NCHW = 0,
  NNADAPTER_NHWC = 1,
} NNAdapterOperandLayoutCode;

/**
 * Operand life time codes(internal type), where to find the data for a operand.
 * * NNADAPTER_TEMPORARY_VARIABLE: A temporary operand in a newtork, There is no
 need to set its data during building a network.
 * * NNADAPTER_CONSTANT_COPY: A constant operand, copy to an new space allocated
 internally during building a network.
 * * NNADAPTER_CONSTANT_REFERENCE: A constant operand, reference to the external
 buffer, the caller or driver must not change the content of the buffer.
 * * NNADAPTER_MODEL_INPUT: indicate the operand is the input of model.
 * * NNADAPTER_MODEL_OUTPUT: indicate the operand is the output of model.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_TEMPORARY_VARIABLE = 0,
  NNADAPTER_TEMPORARY_SHAPE = 1,
  NNADAPTER_CONSTANT_COPY = 2,
  NNADAPTER_CONSTANT_REFERENCE = 3,
  NNADAPTER_MODEL_INPUT = 4,
  NNADAPTER_MODEL_OUTPUT = 5,
} NNAdapterOperandLifetimeCode;

/**
 * Operation codes.
 *
 * Available since version 1.
 */
typedef enum {
  /**
   * Applies the abs activation to the input tensor element-wise.
   * The output is calculated using this formula:
   * output = abs(input)
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_ABS,

  /**
   * Applies adaptive 2-D average pooling across the input according to input
   * and
   * output size.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: output_shape, a NNADAPTER_INT32 or
   * NNADAPTER_INT64 tensor, with shape [2], with value [H_out, H_out].
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D,

  /**
   * Applies adaptive 2-D max pooling across the input according to input and
   * output size.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: output_shape, a NNADAPTER_INT32 or
   * NNADAPTER_INT64 tensor, with shape [2], with value [H_out, H_out].
   * * 2: return_indices, a NNADAPTER_BOOL8 scalar, whether to return index of
   * output. Defaults to false
   * * 3: return_indices_dtype, a NNADAPTER_INT32 scalar, must be one of
   * NNADAPTER_INT32 or NNADAPTER_INT64, specifies the dtype of
   * the indices.
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   * * 1: indices, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor,
   * with the same shape as output, indicates the indices of the current feature
   * map.
   *
   * Available since version 1.
   */
  NNADAPTER_ADAPTIVE_MAX_POOL_2D,

  /**
   * Performs element-wise binary addition(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_ADD,

  /**
   * Performs element-wise binary and logical operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula: output = input0 && input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_BOOL8 tensor.
   * * 1: input1, a NNADAPTER_BOOL8 tensor.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_AND,

  /**
   * Computes the indices of the max elements of the input tensor’s element
   * along the provided axis.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar, the axis in which to compute
   * the arg indices, it should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as axis+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 scalar, keep the reduced dimension or not,
   * If TRUE, keep the reduced dimension.
   * * 3: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_INT32,
   * NNADAPTER_INT64, specifies the dtype of the result. Default
   * NNADAPTER_INT64.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_ARG_MAX,

  /**
   * Computes the indices of the min elements of the input tensor’s element
   * along the provided axis.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. the axis in which to compute
   * the arg indices, it should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as axis+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 scalar, keep the reduced dimension or not,
   * If TRUE, keep the reduced dimension.
   * * 3: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_INT32,
   * NNADAPTER_INT64, specifies the dtype of the result. Default
   * NNADAPTER_INT64.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_ARG_MIN,

  /**
   * Copy the input to the output.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_ASSIGN,

  /**
   * Performs element-wise binary equal relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     output = input0 == input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_EQUAL,

  /**
   * Applies a 2-D average pooling across the input according to kernel sizes,
   * stride sizes, and pad lengths.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that
   * paddings is used. 1 means "SAME". 2 means "VALID". It must be one of
   * NNAdapterAutoPadCode values.
   * * 2: pads, a NNADAPTER_INT32 tensor, with shape [4] and data
   * {height_top,
   * height_bottom, width_left, width_right}, or with shape[0] and no data.
   * * 3: kernel_shape, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {kernel_height, kernel_width}.
   * * 4: strides, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {height_stride, width_stride}.
   * * 5: ceil_mode, a NNADAPTER_BOOL8 scalar, whether to use ceil or floor
   * (default) to compute the output shape. Defaults to false
   * * 6: count_include_pad, a NNADAPTER_BOOL8 scalar, whether include pad
   * pixels when calculating values for the edges. Defaults to false
   * * 7: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as input.
   *      1) If ceil_mode=false,
   *         H_out = floor((H_in + padding_height_top + padding_height_bottom -
   * filter_height) / stride_height + 1)
   *         W_out = floor((W_in + padding_width_left + padding_width_right -
   * filter_width) / stride_width + 1)
   *      2) If ceil_mode=true,
   *         H_out = ceil((H_in + padding_height_top + padding_height_bottom -
   * filter_height) / stride_height + 1)
   *         W_out = ceil((W_in + padding_width_left + padding_width_right -
   * filter_width) / stride_width + 1)
   *
   * Available since version 1.
   */
  NNADAPTER_AVERAGE_POOL_2D,

  /**
   * Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with
   * additional channel dimension) as described in the paper Batch
   * Normalization: Accelerating Deep Network Training by Reducing Internal
   * Covariate Shift .
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...]
   * * 1: scale, a 1-D tensor with shape [C]. 1) If input's type is
   * NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 2: bias, a 1-D tensor with shape [C]. 1) If input's type is
   * NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 3: mean, a 1-D tensor with shape [C]. 1) If input's type is
   * NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 4: var, a 1-D tensor with shape [C]. 1) If input's type is
   * NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 5: epsilon, a NNADAPTER_FLOAT32 scalar. Defaults to 1e-5. The small value
   * added to the variance to prevent division by zero.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_BATCH_NORMALIZATION,

  /**
   * The operator casts the elements of `input` to a data type specified
   * by the `dtype` argument.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_BOOL8, NNADAPTER_INT8,
   * NNADAPTER_UINT8, NNADAPTER_INT16, NNADAPTER_INT32,
   * NNADAPTER_INT64, NNADAPTER_FLOAT16, NNADAPTER_FLOAT32,
   * NNADAPTER_FLOAT64 tensor.
   * * 1: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_INT32,
   * NNADAPTER_INT64, NNADAPTER_FLOAT32, NNADAPTER_FLOAT64 etc.
   * Specifies the dtype of the result.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape as input.
   *
   * Available since version 1.
   */
  NNADAPTER_CAST,

  /**
   * Clip all elements in input into the range [ min, max ].
   * The output is calculated using this formula:
   *     output = MIN(MAX(input, min), max)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: min, a 1-D tensor with the same type as input with shape[1].
   * * 2: max, a 1-D tensor with the same type as input with shape[1].
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_CLIP,

  /**
   * Concatenates a list of tensors into a single tensor along the given
   * dimension. All input tensors must have the same shape, except for the
   * dimension size of the axis to concatenate on.
   *
   * Inputs:
   * * 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. It represents the dimension along
   * which axis to concat on. It should be in range [-R, R), where R is the rank
   * of input, negative value works the same way as axis+R.
   *
   * Outputs:
   * * 0: output, the result with the same type as the inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_CONCAT,

  /**
   * Performs a normal or depthwise 2-D convolution operation.
   * The CONV_2D op computes a 2-D convolution based on the input, filter,
   * strides, paddings, dilations, groups and etc.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: filter, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor.
   *      1) For a normal convolution, the filter's shape is [C_out, C_in,
   * filter_height, filter_width], where C_out and C_in is the number of the
   * channels of output and input, filter_height and filter_width is the
   * filter's kernel size in the 'H' and 'W' dimension.
   *      2) For a depthwise convolution, the filter's shape is [C_out, 1,
   * filter_height, filter_width], where C_out is the number of the channels of
   * output, filter_height and filter_width is the filter's kernel size in the
   * 'H' and 'W' dimension.
   * * 2: bias, a 1-D tensor with shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 3: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that
   * paddings is used. 1 means "SAME". 2 means "VALID". It must be one of
   * NNAdapterAutoPadCode.
   * * 4: pads, a NNADAPTER_INT32 tensor, with shape [4] and data
   * {height_top,
   * height_bottom, width_left, width_right}, or with shape[0] and no data.
   * * 5: strides, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {height_stride, width_stride}.
   * * 6: group, a NNADAPTER_INT32 scalar.
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * group=C_out=C_in.
   * * 7: dilations, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {dilations_height, dilations_width}.
   * * 8: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   *
   * Outputs:
   * * 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as input.
   *      H_out = (H_in + padding_height_top + padding_height_bottom -
   * (dilation_height * (filter_height
   *              - 1) + 1)) / stride_height + 1
   *      W_out = (W_in + padding_width_left + padding_width_right -
   * (dilation_width * (filter_width - 1)
   *              + 1)) / stride_width + 1
   *
   * Available since version 1.
   */
  NNADAPTER_CONV_2D,

  /**
   * Performs the transpose of 2-D convolution operation(also called
   * deconvolution) based on the input, filter, strides, paddings, dilations,
   * groups and etc.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: filter, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor. The filter's shape
   * is [C_out, C_in, filter_height, filter_width], where C_out and C_in is the
   * number of the channels of output and input, filter_height and filter_width
   * is the filter's kernel size in the 'H' and 'W' dimension.
   * * 2: bias, a 1-D tensor with shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 3: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that
   * paddings is used. 1 means "SAME". 2 means "VALID". It must be one of
   * NNAdapterAutoPadCode.
   * * 4: pads, a NNADAPTER_INT32 tensor, with shape [4] and data
   * {height_top,
   * height_bottom, width_left, width_right}, or shape[0] and no data.
   * * 5: strides, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {height_stride, width_stride}.
   * * 6: group, a NNADAPTER_INT32 scalar.
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * group=C_out=C_in.
   * * 7: dilations, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {dilations_height, dilations_width}.
   * * 8: output_padding, a NNADAPTER_INT32 tensor, with shape [2] and
   * data
   * {output_pad_height, output_pad_width}, or shape[0] and no data.
   * * 9: output_shape, a NNADAPTER_INT32 or NNADAPTER_INT64
   * tensor, with shape [2] and data {output_height, output_width}, or shape[0]
   * and no data.
   * * 10: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as input.
   *      H_out = (H_in - 1) * stride_height - padding_height_top -
   * padding_height_bottom + (dilation_height * (filter_height - 1)) + 1 +
   * output_padding_height
   *      W_out = (W_in - 1) * stride_width - padding_width_left -
   * padding_width_right + (dilation_width * (filter_width - 1) + 1)) + 1 +
   * output_padding_width
   *
   * Available since version 1.
   */
  NNADAPTER_CONV_2D_TRANSPOSE,

  /**
   * Performs cumulative sum of the input elements along the given axis.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. Defaults to -1. It represents the
   * dimension along which softmax will be performed. It should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * axis+R.
   * * 2: exclusive, a NNADAPTER_BOOL8 scalar. If set to true, the top element
   * will not be include. Default false.
   * * 3: reverse, a NNADAPTER_BOOL8 scalar, whether to perform the cumsum in
   * the reversed direction. Default false.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_CUM_SUM,

  /**
   * Compute 2-D deformable convolution on 4-D input.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: offset, a tensor with the same type as input.
   * It's shape is [N, 2 * deformable_groups * H_f * W_f, H_in, W_in]
   * * 2: mask, a tensor with the same type as input.
   * It's shape is [N, deformable_groups * H_f * W_f, H_in, W_in]
   * * 3: filter, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor.
   *      1) For a normal convolution, the filter's shape is [C_out, C_in,
   * filter_height, filter_width], where C_out and C_in is the number of the
   * channels of output and input, filter_height and filter_width is the
   * filter's kernel size in the 'H' and 'W' dimension.
   *      2) For a depthwise convolution, the filter's shape is [C_out, 1,
   * filter_height, filter_width], where C_out is the number of the channels of
   * output, filter_height and filter_width is the filter's kernel size in the
   * 'H' and 'W' dimension.
   * * 4: bias, a 1-D tensor with shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 5: pads, a NNADAPTER_INT32 tensor, with shape [4] and data
   * {height_top, height_bottom, width_left, width_right}, or with shape[0] and
   * no data.
   * * 6: strides, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {height_stride, width_stride}.
   * * 7: group, a NNADAPTER_INT32 scalar.
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * group=C_out=C_in.
   * * 8: deformable_group, a NNADAPTER_INT32 scalar. Specify the c-axis
   * grouping number of input x.
   * * 9: dilations, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {dilations_height, dilations_width}.
   * * 10: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as input.
   *      H_out = (H_in + padding_height_top + padding_height_bottom -
   * (dilation_height * (filter_height
   *              - 1) + 1)) / stride_height + 1
   *      W_out = (W_in + padding_width_left + padding_width_right -
   * (dilation_width * (filter_width - 1)
   *              + 1)) / stride_width + 1
   *
   * Available since version 1.
   */
  NNADAPTER_DEFORMABLE_CONV_2D,

  /**
   * Applies the quantization to the input tensor. The output is calculated
   * using this formula:
   * output = (input - zero_point) * scale,
   * `zero_point` and `scale` is obtained from `input` .
   *
   * Inputs:
   * * 0: input, a NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER and
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_CHANNEL tensor.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_FLOAT32 tensor with the same shape as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_DEQUANTIZE,

  /**
   * Performs element-wise binary division(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_DIV,

  /**
   * Applies the exp activation to the input tensor element-wise.
   * The output is calculated using this formula:
   * output = e^input
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_EXP,

  /**
   * Broadcast the input tensor following the given shape(by Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor. It
   * indicates the shape you want to expand to, following the broadcast rule.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_EXPAND,

  /**
   * Return a Tensor with the 'shape' and 'value'.
   *
   * Inputs:
   * * 0: shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor.
   * * 1: value, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64 or
   * NNADAPTER_BOOL scalar.
   *
   * Outputs:
   * * 0: output, a tensor with the 'shape' and 'value'.
   *
   * Available since version 1.
   */
  NNADAPTER_FILL,

  /*
   * According to the given start_axis and end_axis flattens successive
   * dimensions.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: start_axis, a NNADAPTER_INT32 scalar, first dim to flatten.
   * * 2: end_axis, a NNADAPTER_INT32 scalar, last dim to flatten.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_FLATTEN,

  /*
   * Applies floor to the input tensor element-wise. The output is calculated
   * using this formula: output = floor(input)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_FLOOR,

  /**
   * Add a fully connected layer.
   * The output is calculated using this formula:
   *     output = activation(input * weight' + bias)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor of at least rank 2, If
   * its rank is greater than 2, it will be flattened to a 2-D Tensor with the
   * shape [batch_size, input_size], where input_size represents the number of
   * inputs, matching the second dimension of weight, and batch_size is
   * calculated by dividing the number of elements by input_size
   * * 1: weight, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL 2-D tensor with shape
   * [num_units, input_size], where the num_units represents the number of
   * output units, which also means the feature size of output.
   * * 2: bias, a 1-D tensor with shape [num_units].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If weight's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * weight_scale.
   *      3) If weight's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * weight_scale[i] for each output channel.
   * * 3: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, a 2-D tensor with shape [batch_size, num_units], and its type
   * is the same as input.
   *
   * Available since version 1.
   */
  NNADAPTER_FULLY_CONNECTED,

  /**
   * Output is obtained by gathering entries of axis of x indexed by index and
   * concatenate them together.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32,
   * NNADAPTER_INT64, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: indices, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor,
   * with rank R1, with values between [-k, k-1] along axis of size k.
   * * 2: axis, A NNADAPTER_INT32 scalar. It represents the dimension along
   * which gather will be performed. It should be in range [-R, R), where R is
   * the rank of input, negative value works the same way as axis+R.
   *
   * Outputs
   * * 0: output, a tensor with the same type as input, of rank with rank "R1 +
   * (R - 1)".
   *
   * Available since version 1.
   */
  NNADAPTER_GATHER,

  /**
   * Applies the Gaussian Error Linear Units activation to the input tensor
   * element-wise. Refer to https://arxiv.org/abs/1606.08415 for more details.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: approximate, a NNADAPTER_BOOL8 scalar, whether to enable
   * approximation.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_GELU,

  /**
   * Performs element-wise binary greater relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * output = input0 > input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_GREATER,

  /**
   * Performs element-wise binary greater_equal relational operation(with
   * Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * output = input0 >= input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_GREATER_EQUAL,

  /**
   * Samples input X by using bilinear or nearest interpolation based on flow
   * field grid, which is usually generated by affine_grid. The grid of shape
   * [N, H, W, 2] is the concatenation of (grid_x, grid_y) coordinates with
   * shape [N, H, W] each, where grid_x is indexing the 4th dimension (in width
   * dimension) of input data x and grid_y is indexing the 3rd dimension (in
   * height dimension), finally results is the bilinear interpolation value or
   * nearest value of 4 nearest corner points.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * with shape [N, C, H, W].
   * * 1: grid, a tensor with the same type as input, with shape [N, H, W, 2].
   * * 2: align_corners, a NNADAPTER_BOOL8 tensor with shape [1]. If
   * align_corners is true, it will project -1 and 1 to the centers of the
   * corner pixels. Otherwise, it will project -1 and 1 to the image edges.
   * * 3: mode, a NNADAPTER_INT32 tensor with shape [1]. It must be one of
   * NNAdapterInterpolateMode.
   * * 4: pad_mode, a NNADAPTER_INT32 tensor with shape [1]. Supported modes:
   * `constant`(default), `reflect`, `edge`. It should be a value of
   * NNAdapterPadMode.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_GRID_SAMPLE,

  /*
   * Applies Group Normalization over a ND input
   * (a mini-batch of 2D inputs with additional channel dimension)
   * as described in the paper Group Normalization.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...]
   * * 1: scale, a 1-D tensor, with shape [C].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 2: bias, a tensor with the same shape as scale.
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 3: epsilon, a NNADAPTER_FLOAT32 scalar. Defaults to 1e-5.
   * The small value added to the variance to prevent division by zero.
   * * 4: groups, a NNADAPTER_FLOAT32 tensor with shape [1],
   * that divided from channels.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_GROUP_NORMALIZATION,

  /**
   * Applies the hard-sigmoid activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = max(0, min(1, alpha * input + beta))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 scalar.
   * * 2: beta, a NNADAPTER_FLOAT32 scalar.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_HARD_SIGMOID,

  /**
   * Applies the hard-swish activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = input * max(0, min(1, alpha * input + beta))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 scalar.
   * * 2: beta, a NNADAPTER_FLOAT32 scalar.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_HARD_SWISH,

  /**
   * Applies Instance Normalization over a N-D input (N>2) as described
   * in the paper https://arxiv.org/abs/1607.08022.
   * y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
   * where mean and variance are computed per instance per channel.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...]
   * * 1: scale, a tensor, with shape [C].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 2: bias, a tensor with the same shape as scale.
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 3: epsilon, a NNADAPTER_FLOAT32 scalar. Defaults to 1e-5.
   * The small value added to the variance to prevent division by zero.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_INSTANCE_NORMALIZATION,

  /**
   * Applies Layer Normalization over a N-D input described
   * in the paper Layer Normalization: <https://arxiv.org/pdf/1607.06450v1.pdf>.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N,C,...]
   * * 1: scale, a tensor, shape is performed along the input dimension
   * from begin_norm_axis to input rank.
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 2: bias, a tensor with the same shape as scale.
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   * * 3: begin_norm_axis, a NNADAPTER_INT32 scalar.
   * Indicates that the normalization will be performed along the dimension
   * from begin_norm_axis to rank (input). Default value: 1.
   * * 4: epsilon, a NNADAPTER_FLOAT32 scalar. Defaults to 1e-5.
   * The small value added to the variance to prevent division by zero.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_LAYER_NORMALIZATION,

  /**
   * Applies the Leaky ReLU activation to the input tensor element-wise. The
   * output is calculated using this formula: output = input, if input >=0
   * output = alpha * input, if input < 0
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 scalar.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_LEAKY_RELU,

  /**
   * Performs element-wise binary less relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * output = input0 < input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_LESS,

  /**
   * Performs element-wise binary less_equal relational operation(with
   * Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * output = input0 <= input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_LESS_EQUAL,

  /**
   * Applies the log activation to the input tensor element-wise. The output is
   * calculated using this formula: output = log(input)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_LOG,

  /**
   * Applies the Lp Normalization to the input tensor element-wise.
   * The output is calculated using this formula:
   * output = input / (sum(abs(input)) + epsilon), if p = 1
   * output = input / (sqrt(sum(input^2)) + epsilon), if p = 2
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, an 1-D NNADAPTER_INT32. Defaults to [1].
   * It represents the dimension along which norm will be performed.
   * It should be in range [-R, R), where R is the rank of input,
   * negative value works the same way as axis+R.
   * * 2: p, a NNADAPTER_INT32 scalar. The exponent value in the norm
   * formulation,
   * only 1 or 2 are supported. Defaults to 2.
   * * 3: epsilon, a NNADAPTER_FLOAT32 scalar,
   * specifying the lower limit of normalization
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_LP_NORMALIZATION,

  /**
   * Matrix product that behaves like numpy.matmul.
   *
   * Inputs:
   * * 0: x, A NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: y, a tensor with the same type as input.
   * * 2: transpose_x, a NNADAPTER_BOOL8 scalar, whether to transpose the last
   * two dimensions of x before multiplication.
   * * 3: transpose_y, a NNADAPTER_BOOL8 scalar, whether to transpose the last
   * two dimensions of y before multiplication.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as x.
   *
   * Available since version 1.
   */
  NNADAPTER_MAT_MUL,

  /**
   * Performs element-wise binary maximum(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_MAX,

  /**
   * Applies a 2-D max pooling across the input according to kernel sizes,
   * stride sizes, and pad lengths.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: auto_pad, a NNADAPTER_INT32 scalar. 0 means "EXPLICIT" so that
   * paddings is used. 1 means "SAME". 2 means "VALID". It must be one of
   * NNAdapterAutoPadCode values.
   * * 2: pads, a NNADAPTER_INT32 tensor, with shape [4] and data
   * {height_top,
   * height_bottom, width_left, width_right}, or with shape[0] and no data.
   * * 3: kernel_shape, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {kernel_height, kernel_width}.
   * * 4: strides, a NNADAPTER_INT32 tensor, with shape [2] and data
   * {height_stride, width_stride}.
   * * 5: ceil_mode, a NNADAPTER_BOOL8 scalar, whether to use ceil or floor
   * (default) to compute the output shape. Defaults to false.
   * * 6: return_indices, A NNADAPTER_BOOL8 scalar, whether to return index of
   * output. Defaults to false.
   * * 7: return_indices_dtype, a NNADAPTER_INT32 scalar, must be one of
   * NNADAPTER_INT32 or NNADAPTER_INT64, specifies the dtype of
   * the indices.
   * * 8: fuse_code, a NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, the output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as input.
   *      1) If ceil_mode=false,
   *         H_out = floor((H_in + padding_height_top + padding_height_bottom -
   * filter_height) / stride_height + 1)
   *         W_out = floor((W_in + padding_width_left + padding_width_right -
   * filter_width) / stride_width + 1)
   *      2) If ceil_mode=true,
   *         H_out = ceil((H_in + padding_height_top + padding_height_bottom -
   * filter_height) / stride_height + 1)
   *         W_out = ceil((W_in + padding_width_left + padding_width_right -
   * filter_width) / stride_width + 1)
   * * 1: indices, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor,
   * with the same shape as output, indicates the indices of the current feature
   * map.
   *
   * Available since version 1.
   */
  NNADAPTER_MAX_POOL_2D,

  /**
   * Takes a list of N tensors as inputs, each of which is 1-dimensional vector,
   * and creates N-dimensional grids.
   *
   * Inputs:
   * * input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER  tensor with shape [d0], [d1], ...
   * [dn-1].
   *
   * Outputs:
   * * output0 ~ outputn-1, a  tensor with the same type as input, with shape
   * [d0, d1, ... dn-1].
   *
   * Available since version 1.
   */
  NNADAPTER_MESHGRID,

  /**
   * Performs element-wise binary minimum(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_MIN,

  /**
   * Performs element-wise binary multiplication(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_MUL,

  /**
   * Applies logical not to the input tensor element-wise. The output is
   * calculated using this formula: output = !input
   *
   * Inputs:
   * * 0: input, a NNADAPTER_BOOL8 tensor.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor with the same shape as input.
   *
   * Available since version 1.
   */
  NNADAPTER_NOT,

  /**
   * Performs element-wise binary not_equal relational operation(with
   * Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     output = input0 != input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_NOT_EQUAL,

  /**
   * Pad input by "pads", "mode", "constant_value"
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32,
   * NNADAPTER_INT64, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: pads, a NNADAPTER_INT32 1-D tensor,
   * with shape [2 * input_rank],
   * with value [x0_begin, x0_end, x1_begin, x1_end,...].
   * * 2: mode, a NNADAPTER_INT32 scalar.
   * Supported modes: `constant`(default), `reflect`, `edge`.
   * It should be a value of NNAdapterPadModeCode.
   * * 3: value, a scalar with the same type as input,
   * only be used if the mode is "constant".
   *
   * Outputs:
   * * 0: output, the result with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_PAD,

  /**
   * Performs element-wise binary pow(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html). The output is
   * calculated using this formula: output = input0^input1
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_POW,

  /**
   * Applies the prelu activation to the input tensor. The output is calculated
   * using this formula:
   * output = input, if input >=0;
   * output = slope * input, if input < 0;
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32 or
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
   * * 1: slope, a tensor, with shape [1] or [C].
   * 1) If input's type is NNADAPTER_FLOAT32, its type must be the same
   * type.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_PRELU,

  /**
   * Applies the quantization to the input tensor. The output is calculated
   * using this formula:
   * output = input / scale + zero_point
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32 or NNADAPTER_INT32 tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar, the axis of the quantization dimension
   * of the input tensor. Ignored for per-tensor quantization. It should be in
   * range [-R, R), where R is the rank of input, negative value works the same
   * way as axis+R, default to 1.
   * * 2: scale, a NNADAPTER_FLOAT32 tensor, Scale for input. It can be a
   * scalar, which means a per-tensor/layer dequantization, or a 1-D tensor for
   * per-axis dequantization.
   * * 3: zero_point, a NNADAPTER_INT32  tensor, Zero point for `input`. Shape
   * must match `scale`, default to 0.
   *
   * Outputs:
   * * 0: output, a quantized tensor with the same shape as `input` , its type
   * can be NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER and
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_CHANNEL according to `axis` and
   * `zero_point`.
   *
   * Available since version 1.
   */
  NNADAPTER_QUANTIZE,

  /**
   * Outputs a 1-D Tensor with spaced values within a given interval.
   *
   * Inputs:
   * * 0: start, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape[1].
   * * 1: end, a tensor with the same shape and type as start.
   * * 2: step, a tensor with the same shape and type as start.
   *
   * Outputs:
   * * 0: output, a 1-D tensor with the same type as start.
   *
   * Available since version 1.
   */
  NNADAPTER_RANGE,

  /**
   * Computes the mean of the input’s elements along axis. If axis has no
   * data, mean is calculated over all elements of input.
   * If keepdims equal 0, then the resulted tensor have the reduced dimension
   * pruned.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axes, a NNADAPTER_INT32 tensor. It indicating the dimensions to
   * perform mean calculations. It should be in range [-R, R), where R is the
   * rank of input, negative value works the same way as axis+ndim(input). If
   * axis has no data, mean is calculated over all elements of input.
   * * 2: keepdim, a NNADAPTER_BOOL8 scalar. Keep the reduced dimension or not,
   * default 1 mean keep reduced dimension.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_REDUCE_MEAN,

  /**
   * Computes the sum of the input’s elements along axis. If axis has no
   * data, sum is calculated over all elements of input.
   * If keepdims equal 0, then the resulted tensor have the reduced dimension
   * pruned.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axes, a NNADAPTER_INT32 tensor. It indicating the dimensions to
   * perform mean calculations. It should be in range [-R, R), where R is the
   * rank of input, negative value works the same way as axis+ndim(input). If
   * axis has no data, mean is calculated over all elements of input.
   * * 2: keepdim, a NNADAPTER_BOOL8 scalar. Keep the reduced dimension or not,
   * default 1 mean keep reduced dimension.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_REDUCE_SUM,

  /**
   * Applies rectified linear activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = max(0, input)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_RELU,

  /**
   * Applies rectified linear 6 activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = min(6, max(0, input))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_RELU6,

  /**
   * Reshapes a tensor similar to numpy.reshape.
   * The output tensor has the same data as the input tensor but with a new
   * shape.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: shape, an 1-D NNADAPTER_INT32 or NNADAPTER_INT64 shape
   * tensor which specifies the new shape, At most one dimension of the new
   * shape can be -1. In this case, the value is inferred from the size of the
   * tensor and the remaining dimensions. a dimension could also be 0, in which
   * case the actual dimension value is unchanged.
   *
   * Outputs:
   * * 0: output, a tensor with a new shape, and its type and data is same as
   * input.
   *
   * Available since version 1.
   */
  NNADAPTER_RESHAPE,

  /**
   * Resizes the input tensor using the nearest interpolation.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
   * * 1: shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor. It
   * indicates the target shape of output exclude dim_N and dim_C.
   * * 2: scales, a NNADAPTER_FLOAT32 tensor. It indicates the scale of
   * the output's shape exclude dim_N and dim_C.
   * * 3: align_corners. a NNADAPTER_BOOL scalar.  If True, the centers of the 4
   * corner pixels of the input and output tensors are aligned, preserving the
   * values at the corner pixels.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   */
  NNADAPTER_RESIZE_NEAREST,

  /**
   * Resizes the input tensor using the linear interpolation.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor with shape [N, C, ...].
   * * 1: shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor. It
   * indicates the target shape of output exclude dim_N and dim_C.
   * * 2: scales, a NNADAPTER_FLOAT32 tensor. It indicates the scale of
   * the output's shape exclude dim_N and dim_C.
   * * 3: align_corners, NNADAPTER_BOOL scalar. If True, the centers of the 4
   * corner pixels of the input and output tensors are aligned, preserving the
   * values at the corner pixels.
   * * 4: align_mode, a NNADAPTER_INT32 scalar, optional for linear
   * interpolation. It can be ‘0’ for src_idx = scale_factor*(dst_indx+0.5)-0.5
   * , can be ‘1’ for src_idx = scale_factor*dst_index.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   */
  NNADAPTER_RESIZE_LINEAR,

  /**
   * Perform bilinear interpolation on inputs of nonuniform sizes to obtain
   * fixed-size feature maps (e.g. 7*7), as described in Mask R-CNN.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * with shape [N, C, H, W].
   * * 1: rois, a tensor with the same type as input, with shape [rois_num, 4]
   * given as [[x1, y1, x2, y2], ...].
   * * 2: batch_indices, a tensor with shape [rois_num], denoting the index of
   * the corresponding image in the batch.
   * * 3: output_height, a NNADAPTER_INT32 tensor with shape [1], the pooled
   * output height.
   * * 4: output_width, a NNADAPTER_INT32 tensor with shape [1], the pooled
   * output width.
   * * 5: sampling_ratio, , a NNADAPTER_INT32 tensor with shape [1], number of
   * sampling points in the interpolation grid. If sampling_ratio <= 0, then
   * grid points are adaptive to roi_width and output_width, likewise for
   * height.
   * * 6: spatial_scale, a NNADAPTER_FLOAT32 tensor with shape [1],
   * multiplicative spatial scale factor to translate ROI coords from their
   * input scale to the scale used when pooling.
   * * 7: aligned, a NNADAPTER_BOOL8 tensor. If true, pixel shift it by -0.5 for
   * align more perfectly.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   */
  NNADAPTER_ROI_ALIGN,

  /**
   * Outputs an 1D tensor containing the shape of the input tensor.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_INT32 tensor.
   * * 1: dtype, a NNADAPTER_INT32 scalar, the value of NNADAPTER_INT32
   * or NNADAPTER_INT64. Specifies the dtype of the result.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_INT32 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_SHAPE,

  /**
   * Applies sigmoid activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = 1 / (1 + exp(-input))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SIGMOID,

  /**
   * This operator produces a slice of input along multiple axes. Slice uses
   * axes, starts and ends attributes to specify the start and end dimension for
   * each axis in the list of axes and Slice uses this information to slice the
   * input data tensor. If a negative value is passed to starts or ends such as
   * −i, it represents the reverse position of the axis i−1 (here 0 is the
   * initial position). If the value passed to starts or ends is greater than n
   * (the number of elements in this dimension), it represents n. For slicing to
   * the end of a dimension with unknown size, it is recommended to pass in
   * INT_MAX. The size of axes must be equal to starts and ends.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor that `starts` and `ends` apply
   * to. It's optional. If not present, will be treated as [0, 1, ...,
   * len(`starts`) - 1].
   * * 2: starts, starts indices of corresponding axis in `axes`, a
   * NNADAPTER_INT32 tensor.
   * * 3: ends, ends indices of corresponding axis in `axes`, a
   * NNADAPTER_INT32 tensor.
   * * 4: steps, a NNADAPTER_INT32  1-D tensor, 1-D tensor of slice step
   * of corresponding axis in `axes`. Negative value means slicing backward.
   * 'steps' cannot be 0. Defaults to 1.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SLICE,

  /**
   * Computes the normalized exponential values for the input tensor
   * element-wise.
   * The output is calculated using this formula:
   *     output = exp(input) / reduce_sum(exp(input), axis=axis, keepdims=true)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. Defaults to 1. It represents the
   * dimension along which softmax will be performed. It should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * axis+R.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SOFTMAX,

  /**
   * Split a tensor into a list of tensors along the given dimension.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. It represents the dimension along
   * which axis to split. It should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as axis+R.
   * * 2: split, An 1-D NNADAPTER_INT32, each of values indicates the
   * length of each output. Sum of the values must be equal to the dimension at
   * 'axis' specified.
   *
   * Outputs:
   * * 0 ~ n-1: output0 ~ outputn-1, the results with the same type as the
   * input.
   *
   * Available since version 1.
   */
  NNADAPTER_SPLIT,

  /**
   * Applies square to the input tensor element-wise.
   * The output is calculated using this formula:
   * output = input^2
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SQUARE,

  /**
   * Squeeze the dimension(s) of size 1 of input's shape.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axes, a NNADAPTER_INT32 tensor. It indicating the dimensions to
   * be squeezed. Default is None. The range of axis is [−ndim(x),ndim(x)). It
   * should be in range [-R, R), where R is the rank of input, negative value
   * works the same way as axis+ndim(input).
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SQUEEZE,

  /**
   * Join a sequence of tensors along a new axis.
   * All input tensors must have the same shape.
   *
   * Inputs:
   * * 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 scalar. It represents the dimension along
   * which axis to stack. It should be in range [-R-1, R+1), where R is the rank
   * of input, negative value works the same way as axis+R+1.
   *
   * Outputs:
   * * 0: output, the result with the same type as the inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_STACK,

  /**
   * Performs element-wise binary subtraction(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor with the same type as input0.
   * * 2: fuse_code, a NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_SUB,

  /**
   * Performs element-wise binary addition(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, the result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_SUM,

  /**
   * Applies the Swish activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = input / (1 + e ^ (-input))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SWISH,

  /**
   * Applies the hyperbolic tangent activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = tanh(input)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_TANH,

  /**
   * Repeats the input by given times number
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: repeats, a NNADAPTER_INT32 tensor with shape [rank(input)].
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_TILE,

  /**
   * Retrieve the top-K largest elements along a specified axis.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32,
   * NNADAPTER_INT64, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: k, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor, the number of top
   * elements to look for along the axis.
   * * 2: axis, a NNADAPTER_INT32 scalar, represents the dimension along which
   * top_k will be performed. It should be in range [-R, R), where R is the rank
   * of input, negative value works the same way as axis+R.
   * * 3: largest, a NNADAPTER_BOOL8 scalar, whether to return the top-K largest
   * or smallest elements.
   * * 4: sorted, a NNADAPTER_BOOL8 scalar, whether to return the elements in
   * sorted order.
   * * 5: return_indices_dtype, a NNADAPTER_INT32 scalar, the value of
   * NNADAPTER_INT32 or NNADAPTER_INT64, specifies the dtype of
   * the indices.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input, top K values
   * from the input tensor.
   * * 1: indices, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor,
   * the corresponding input tensor indices for the top K values.
   *
   * Available since version 1.
   */
  NNADAPTER_TOP_K,

  /**
   * Transposes the input according to the perm, similar to numpy.transpose
   * https://numpy.org/doc/stable/reference/generated/numpy.transpose.html.
   * For example, the input with shape (1, 2, 3) and perm=(1, 0, 2), the shape
   * of output will be (2, 1, 3).
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: perm, An optional 1-D NNADAPTER_INT32 tensor, reverse the
   * dimensions of input if perm is not given, otherwise permute the axes
   * according to the values given.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_TRANSPOSE,

  /**
   * Remove dimensions of input which size is 1
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT16, NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axes, A NNADAPTER_INT32 tensor. It indicating the dimensions
   * to be inserted. It should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as axis+R+1.
   *
   * Outputs:
   * * 0: output, a tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_UNSQUEEZE,
} NNAdapterOperationCode;

/**
 * Fused activation function types.
 *
 * Available since version 1.
 */
typedef enum {
  /** No fused activation function. */
  NNADAPTER_FUSED_NONE = 0,
  /** Fused ReLU activation function. */
  NNADAPTER_FUSED_RELU = 1,
  /** Fused ReLU1 activation function. */
  NNADAPTER_FUSED_RELU1 = 2,
  /** Fused ReLU6 activation function. */
  NNADAPTER_FUSED_RELU6 = 3,
} NNAdapterFuseCode;

/**
 * Pad types.
 *
 * Available since version 1.
 */
typedef enum {
  /** Use explicit pads. */
  NNADAPTER_AUTO_PAD_NONE = 0,
  /** Results in padding evenly to the left/right or up/down of the input such
     that output has the same height/width dimension as the input.*/
  NNADAPTER_AUTO_PAD_SAME = 1,
  /** No padding. */
  NNADAPTER_AUTO_PAD_VALID = 2,
} NNAdapterAutoPadCode;

/**
 * Device codes.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_CPU = 0,
  NNADAPTER_GPU = 1,
  NNADAPTER_ACCELERATOR = 2,
} NNAdapterDeviceCode;

/**
 * Pad modes.
 *
 * Available since version 1.
 */
typedef enum {
  /** No pad mode. */
  NNADAPTER_PAD_MODE_NONE = 0,
  NNADAPTER_PAD_MODE_CONSTANT = 1,
  NNADAPTER_PAD_MODE_REFLECT = 2,
  NNADAPTER_PAD_MODE_REPLICATE = 3,
  NNADAPTER_PAD_MODE_EDGE = 4,
} NNAdapterPadModeCode;

/**
 * Interpolate Mode
 *
 * Available since version 1.
 */
typedef enum {
  /** No pad mode. */
  NNADAPTER_INTERPOLATE_MODE_NONE = 0,
  NNADAPTER_INTERPOLATE_MODE_BILINEAR = 1,
  NNADAPTER_INTERPOLATE_MODE_NEAREST = 2,
} NNAdapterInterpolateModeCode;

typedef int32_t NNAdapterDeviceType;

/**
 * The quantization parameters for NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
 * operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterSymmPerLayerQuantParams {
  float scale;
} NNAdapterSymmPerLayerQuantParams;

/**
 * The quantization parameters for NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL
 * operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterSymmPerChannelQuantParams {
  uint32_t channel_dim;
  uint32_t scale_count;
  float* scales;
} NNAdapterSymmPerChannelQuantParams;

/**
 * The quantization parameters for NNADAPTER_QUANT_INT8_ASYMM_PER_LAYER
 * operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterAsymmPerLayerQuantParams {
  float scale;
  int32_t zero_point;
} NNAdapterAsymmPerLayerQuantParams;

/**
 * The type of operand's dimensions, include shape and dynamic shape.
 *
 * Available since version 1.
 */
typedef struct NNAdapterOperandDimensionType {
  uint32_t count;
  int32_t data[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
  uint32_t dynamic_count;
  int32_t dynamic_data[NNADAPTER_MAX_SIZE_OF_DYNAMIC_DIMENSIONS]
                      [NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
} NNAdapterOperandDimensionType;

/**
 * The type of an operand, include both scalars and tensors.
 *
 * Available since version 1.
 */
typedef struct NNAdapterOperandType {
  /**
   * The data precision, e.g NNADAPTER_FLOAT32.
   */
  NNAdapterOperandPrecisionCode precision;

  /**
   * The data layout, e.g NNADAPTER_NCHW.
   */

  NNAdapterOperandLayoutCode layout;

  /**
   * The buffer lifetime, e.g read only, don't set it manually.
   */
  NNAdapterOperandLifetimeCode lifetime;

  /**
   * The data dimensions
   *
   */
  NNAdapterOperandDimensionType dimensions;

  /**
   * The quantization parameters.
   */
  union {
    NNAdapterSymmPerLayerQuantParams symm_per_layer_params;
    NNAdapterSymmPerChannelQuantParams symm_per_channel_params;
    NNAdapterAsymmPerLayerQuantParams asymm_per_layer_params;
  };
} NNAdapterOperandType;

typedef int32_t NNAdapterOperationType;

/**
 * An opaque type for Device.
 *
 * Available since version 1.
 */
typedef struct NNAdapterDevice NNAdapterDevice;
/**
 * An opaque type for Context(multiple-devices).
 *
 * Available since version 1.
 */
typedef struct NNAdapterContext NNAdapterContext;
/**
 * An opaque type for Operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterOperand NNAdapterOperand;
/**
 * An opaque type for Operation.
 *
 * Available since version 1.
 */
typedef struct NNAdapterOperation NNAdapterOperation;
/**
 * An opaque type for Model.
 *
 * Available since version 1.
 */
typedef struct NNAdapterModel NNAdapterModel;
/**
 * An opaque type for Compilation.
 *
 * Available since version 1.
 */
typedef struct NNAdapterCompilation NNAdapterCompilation;
/**
 * An opaque type for Execution.
 *
 * Available since version 1.
 */
typedef struct NNAdapterExecution NNAdapterExecution;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the version of NNAdapter.
 *
 * Available since version 1.
 */
int NNAdapter_getVersion(uint32_t* version);
/**
 * Get the count of the devices which provide the driver libraries.
 *
 * Available since version 1.
 */
int NNAdapter_getDeviceCount(uint32_t* numDevices);
/**
 * Acquire the specified device with the device name.
 *
 * Available since version 1.
 */
int NNAdapterDevice_Acquire(const char* name, NNAdapterDevice** device);
/**
 * Release the target device.
 *
 * Available since version 1.
 */
void NNAdapterDevice_release(NNAdapterDevice* device);
/**
 * Get the name of the specified device.
 *
 * Available since version 1.
 */
int NNAdapterDevice_getName(const NNAdapterDevice* device, const char** name);
/**
 * Get the vendor of the specified device.
 *
 * Available since version 1.
 */
int NNAdapterDevice_getVendor(const NNAdapterDevice* device,
                              const char** vendor);
/**
 * Get the type of the specified device.
 * The supported types are listed in NNAdapterDeviceCode.
 *
 * Available since version 1.
 */
int NNAdapterDevice_getType(const NNAdapterDevice* device,
                            NNAdapterDeviceType* type);
/**
 * Get the driver version of the specified device.
 *
 * Available since version 1.
 */
int NNAdapterDevice_getVersion(const NNAdapterDevice* device, int32_t* version);
/**
 * Create an context with multiple devices.
 *
 * Available since version 1.
 */
int NNAdapterContext_create(NNAdapterDevice** devices,
                            uint32_t num_devices,
                            const char* properties,
                            NNAdapterContext** context);
/**
 * Release the context.
 *
 * Available since version 1.
 */
void NNAdapterContext_destroy(NNAdapterContext* context);
/**
 * Create a hardware-independent neural networks model.
 *
 * Available since version 1.
 */
int NNAdapterModel_create(NNAdapterModel** model);
/**
 * Destroy the model that free all of resources of the model includes the memory
 * of constant operands, quantization parameters, etc.
 *
 * Available since version 1.
 */
void NNAdapterModel_destroy(NNAdapterModel* model);
/**
 * Indicate that we have finished building a model, it must only called once.
 *
 * Available since version 1.
 */
int NNAdapterModel_finish(NNAdapterModel* model);
/**
 * Add an operand to a model.
 *
 * Available since version 1.
 */
int NNAdapterModel_addOperand(NNAdapterModel* model,
                              const NNAdapterOperandType* type,
                              NNAdapterOperand** operand);
/**
 * Set the memory for an constant operand.
 * * When 'copy' is true, the content of the buffer will copied into the model.
 * * When 'copy' is false, the pointer of the buffer will stored in the model,
 * so the caller or driver must not change the content of the buffer.
 *
 * Available since version 1.
 */
int NNAdapterModel_setOperandValue(NNAdapterOperand* operand,
                                   void* buffer,
                                   uint32_t length,
                                   bool copy);
/**
 * Get the type of an operand.
 *
 * Available since version 1.
 */
int NNAdapterModel_getOperandType(NNAdapterOperand* operand,
                                  NNAdapterOperandType** type);
/**
 * Add an operation to a model, and set the input and output operands of the
 * specified operation.
 *
 * Available since version 1.
 */
int NNAdapterModel_addOperation(NNAdapterModel* model,
                                NNAdapterOperationType type,
                                uint32_t input_count,
                                NNAdapterOperand** input_operands,
                                uint32_t output_count,
                                NNAdapterOperand** output_operands,
                                NNAdapterOperation** operation);
/**
 * Indentify the input and output operands of the specified model.
 *
 * Available since version 1.
 */
int NNAdapterModel_identifyInputsAndOutputs(NNAdapterModel* model,
                                            uint32_t input_count,
                                            NNAdapterOperand** input_operands,
                                            uint32_t output_count,
                                            NNAdapterOperand** output_operands);

/**
 * Compile the model to the device-specific binary program or load the cached
 * binary program from memory or file system.
 * If cache_token, cache_buffer and cache_length is specified, load the binary
 * program from memory directly.
 * If cache_token and cache_dir is specified, find and load the cached binary
 * program from the cache files directly.
 * If no cache parameter is specified or the cache files are not found, then
 * compile the given model to the binary program of target devices.
 *
 * Available since version 1.
 */
int NNAdapterCompilation_create(NNAdapterModel* model,
                                const char* cache_token,
                                void* cache_buffer,
                                uint32_t cache_length,
                                const char* cache_dir,
                                NNAdapterContext* context,
                                NNAdapterCompilation** compilation);
/**
 * Destroy the hardware-related binary program.
 *
 * Available since version 1.
 */
void NNAdapterCompilation_destroy(NNAdapterCompilation* compilation);
/**
 * Indicate that we have finished building a compilation and start to compile
 * the model to the hardware-releate binary program, it must only called once.
 *
 * Available since version 1.
 */
int NNAdapterCompilation_finish(NNAdapterCompilation* compilation);
/**
 * Query the information of input and output operands of the hardware-releate
 * binary program, it must called after `NNAdapterCompilation_finish`. The
 * function should be called twice: firstly, only set input_count and
 * output_count to obtain the count of inputs and outputs, secondly, allocate
 * memory for the pointers of input and output types, then call it again.
 *
 * Available since version 1.
 */
int NNAdapterCompilation_queryInputsAndOutputs(
    NNAdapterCompilation* compilation,
    uint32_t* input_count,
    NNAdapterOperandType** input_types,
    uint32_t* output_count,
    NNAdapterOperandType** output_types);

/**
 * Create an execution plan to execute the hardware-related binary program.
 *
 * Available since version 1.
 */
int NNAdapterExecution_create(NNAdapterCompilation* compilation,
                              NNAdapterExecution** execution);
/**
 * Destroy an execution plan.
 *
 * Available since version 1.
 */
void NNAdapterExecution_destroy(NNAdapterExecution* execution);
/**
 * Set the input memory and the function used to access it.
 *
 * typedef struct {
 *   NNAdapterOperandPrecisionCode precision;
 *   uint32_t dimensions_count;
 *   int32_t dimensions_data[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
 *   void* buffer;
 *   size_t length;
 * } Memory;
 *
 * void* access_input_memory(void* memory, NNAdapterOperandType* type) {
 *   Memory* handle = static_cast<Memory*>(memory);
 *   // Return the dimensions and the host buffer to driver HAL
 *   memcpy(type->dimensions.data, handle->dimensions_data,
 * handle->dimensions_count);
 *   return handle->buffer;
 * }
 *
 * Available since version 1.
 */
int NNAdapterExecution_setInput(NNAdapterExecution* execution,
                                int32_t index,
                                void* memory,
                                void* (*access)(void* memory,
                                                NNAdapterOperandType* type));
/**
 * Set the output memory and the function used to access it.
 *
 * void* access_output_memory(void* memory, NNAdapterOperandType* type) {
 *   Memory* handle = static_cast<Memory*>(memory);
 *   // Get the buffer length according to the type->precision and
 * type->dimensions
 *   size_t request_length = GetBufferLength(type);
 *   if (request_length > handle->length) {
 *     free(handle->buffer);
 *     handle->buffer = malloc(request_length);
 *     assert(handle->buffer);
 *     handle->length = request_length;
 *   }
 *   // Tell the output dimensions to user and return the host buffer to driver
 * HAL
 *   memcpy(handle->dimensions_data, type->dimensions.data,
 * type->dimensions.count);
 *   handle->dimensions_count = type->dimensions.count;
 *   return handle->buffer;
 * }
 *
 * Available since version 1.
 */
int NNAdapterExecution_setOutput(NNAdapterExecution* execution,
                                 int32_t index,
                                 void* memory,
                                 void* (*access)(void* memory,
                                                 NNAdapterOperandType* type));
/**
 * Start to run the execution synchronously.
 *
 * Available since version 1.
 */
int NNAdapterExecution_compute(NNAdapterExecution* execution);

#ifdef __cplusplus
}
#endif
