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
enum { NNADAPTER_UNKNOWN = -65535 };

/**
 * Result codes.
 *
 * Available since version 1.
 */
typedef enum {
  NNADAPTER_NO_ERROR = 0,
  NNADAPTER_OUT_OF_MEMORY = 1,
  NNADAPTER_INVALID_DIMENSIONS = 2,
  NNADAPTER_INVALID_PARAMETER = 3,
  NNADAPTER_DEVICE_NOT_FOUND = 4,
  NNADAPTER_DEVICE_INTERNAL_ERROR = 5,
  NNADAPTER_FEATURE_NOT_SUPPORTED = 6
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
  NNADAPTER_HWCN = 2,
  NNADAPTER_HWNC = 3,
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
   * Performs element-wise abs activation.
   * The output is calculated using this formula:
   *     `output` = abs(`input`)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER operand.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_ABS = 0,

  /**
   * Performs adaptive 2-D average pooling.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D
   * tensor of shape [N, C_in, H_in, W_in].
   * * 1: output_shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor of shape
   * [2], its value should be [H_out, W_out].
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_in, H_out, W_out] and has same type as
   * `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D,

  /**
   * Performs adaptive 2-D max pooling.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER 4-D
   * tensor of shape [N, C_in, H_in, W_in].
   * * 1: output_shape, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor of shape
   * [2], its value should be [H_out, W_out].
   * * 2: return_indices, a NNADAPTER_BOOL8 tensor of shape [1], whether to
   * return `indices` along with the outputs, defaults to false.
   * * 3: return_indices_dtype, a NNADAPTER_INT32 tensor of shape [1], specifies
   * the data type of `indices`, its value must be one of NNADAPTER_INT32,
   * NNADAPTER_INT64.
   * Outputs:
   * * 0: output, a tensor of shape [N, C_in, H_out, W_out] and has same type as
   * `input`.
   * * 1: indices, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor and has the same
   * shape as `output`.
   *
   * Available since version 1.
   */
  NNADAPTER_ADAPTIVE_MAX_POOL_2D,

  /**
   * Performs element-wise binary addition(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` + `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_ADD,

  /**
   * Performs element-wise binary logical AND operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` && `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_BOOL8 tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_AND,

  /**
   * Computes the indices of the max elements of the input tensor's element
   * along the provided `axis`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], the axis in which to
   * compute
   * the arg indices, it should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as `axis`+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 tensor of shape [1], whether to keep the
   * reduced dimension.
   * * 3: dtype, a NNADAPTER_INT32 tensor of shape [1], specifies the dtype of
   * the `output`, its value should be NNADAPTER_INT32, NNADAPTER_INT64,
   * defaults to NNADAPTER_INT64.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_INT32 or NNADAPTER_INT64 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_ARG_MAX,

  /**
   * Computes the indices of the min elements of the input tensor's element
   * along the provided `axis`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], the axis in which to
   * compute
   * the arg indices, it should be in range [-R, R), where R is the rank of
   * input, negative value works the same way as `axis` +R.
   * * 2: keepdim, a NNADAPTER_BOOL8 tensor of shape [1], whether to keep the
   * reduced dimension.
   * * 3: dtype, a NNADAPTER_INT32 tensor of shape [1], specifies the dtype of
   * the `output`, its value should be NNADAPTER_INT32, NNADAPTER_INT64,
   * defaults to NNADAPTER_INT64.
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
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_ASSIGN,

  /**
   * Performs 2-D average pooling.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
   * * 1: auto_pad, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterAutoPadCode values, NNADAPTER_AUTO_PAD_NONE means specifying the
   * explicit padding by `pads`, otherwise specifying the implicit padding
   * algorithm, including NNADAPTER_AUTO_PAD_SAME and NNADAPTER_AUTO_PAD_VALID.
   * * 2: pads, an optional NNADAPTER_INT32 tensor of shape [4], specifying
   * height_top, height_bottom, width_left and width_right.
   * * 3: kernel_shape, a NNADAPTER_INT32 tensor of shape [2], specifying
   * kernel_height and kernel_width.
   * * 4: strides, a NNADAPTER_INT32 tensor of shape [2], specifying
   * stride_height and stride_width.
   * * 5: ceil_mode, a NNADAPTER_BOOL8 tensor of shape [1], whether to use ceil
   * or floor to compute the output shape, defaults to false to use floor.
   * * 6: count_include_pad, a NNADAPTER_BOOL8 tensor of shape [1], whether
   * include pad pixels when calculating values for the edges, defaults to
   * false.
   * * 7: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_out, H_out, W_out], has the same type
   * as `input`.
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
   * Covariate Shift https://arxiv.org/pdf/1502.03167.pdf .
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, ...].
   * * 1: scale, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 2: bias, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 3: mean, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 4: variance, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 5: epsilon, a NNADAPTER_FLOAT32 tensor of shape [1], a small value added
   * to the variance to prevent division by zero, defaults to 1e-5.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_BATCH_NORMALIZATION,

  /**
   * The operator casts the elements of `input` to a data type specified
   * by the `dtype` argument.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_BOOL8, NNADAPTER_INT8, NNADAPTER_UINT8,
   * NNADAPTER_INT16, NNADAPTER_INT32, NNADAPTER_INT64, NNADAPTER_FLOAT16,
   * NNADAPTER_FLOAT32, NNADAPTER_FLOAT64 tensor.
   * * 1: dtype, a NNADAPTER_INT32 of shape [1], specifies the dtype of the
   * 'output', must be one of NNAdapterOperandPrecisionCode values, should be
   * NNADAPTER_BOOL8, NNADAPTER_INT8, NNADAPTER_UINT8, NNADAPTER_INT16,
   * NNADAPTER_INT32, NNADAPTER_INT64, NNADAPTER_FLOAT16, NNADAPTER_FLOAT32,
   * NNADAPTER_FLOAT64 .
   *
   * Outputs:
   * * 0: output, a `dtype` tensor of the same shape as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_CAST,

  /**
   * It divides the input channels in each group into several subgroups, and
   * obtain a new order by selecting element from every subgroup one by one as
   * described in the paper https://arxiv.org/pdf/1707.01083.pdf .
   * The output is calculated using this formula:
   *     C_out[k * group + g] = C_in[g * size + k], where size = C_in / group.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
   * tensor.
   * * 1: group, a NNADAPTER_INT32 tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_CHANNEL_SHUFFLE,

  /**
   * Clip all elements in input into the range [`min`, `max`].
   * The output is calculated using this formula:
   *     `output` = min(max(`input`, `min`), `max`)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: min, a tensor of shape [1] and has the same type as `input`.
   * * 2: max, a tensor of shape [1] and has the same type as `input`.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
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
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of `input`, negative value works the same way as
   * `axis`+R, defaults to -1.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as the `input0` ~ `inputn-1`.
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
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
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
   * * 2: bias, a tensor of shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the same
   * type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 3: auto_pad, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterAutoPadCode values, NNADAPTER_AUTO_PAD_NONE means specifying the
   * explicit padding by `pads`, otherwise specifying the implicit padding
   * algorithm, including NNADAPTER_AUTO_PAD_SAME and NNADAPTER_AUTO_PAD_VALID.
   * * 4: pads, an optional NNADAPTER_INT32 tensor of shape [4], specifying
   * height_top, height_bottom, width_left and width_right.
   * * 5: strides, a NNADAPTER_INT32 tensor of shape [2], specifying
   * stride_height and stride_width.
   * * 6: group, a NNADAPTER_INT32 tensor of shape [1].
   *      1) For a normal convolution, `group` must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * `group` = C_out = C_in.
   * * 7: dilations, a NNADAPTER_INT32 tensor of shape [2], specifying
   * dilations_height and dilations_width.
   * * 8: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_out, H_out, W_out], has the same type
   * as `input`.
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
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
   * * 1: filter, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL 4-D tensor. The filter's shape
   * is [C_in, C_out, filter_height, filter_width], where C_out and C_in is the
   * number of the channels of output and input, filter_height and filter_width
   * is the filter's kernel size in the 'H' and 'W' dimension.
   * * 2: bias, a tensor of shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 3: auto_pad, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterAutoPadCode values, NNADAPTER_AUTO_PAD_NONE means specifying the
   * explicit padding by `pads`, otherwise specifying the implicit padding
   * algorithm, including NNADAPTER_AUTO_PAD_SAME and NNADAPTER_AUTO_PAD_VALID.
   * * 4: pads, an optional NNADAPTER_INT32 tensor of shape [4], specifying
   * height_top, height_bottom, width_left and width_right.
   * * 5: strides, a NNADAPTER_INT32 tensor of shape [2], specifying
   * stride_height and stride_width.
   * * 6: group, a NNADAPTER_INT32 tensor of shape [1].
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * `group` = C_out = C_in.
   * * 7: dilations, a NNADAPTER_INT32 tensor of shape [2], specifying
   * dilations_height and dilations_width.
   * * 8: output_padding, an optional NNADAPTER_INT32 tensor of shape [2],
   * specifying output_pad_height and output_pad_width.
   * * 9: output_shape, an optional NNADAPTER_INT32 or NNADAPTER_INT64 tensor of
   * shape [2], specifying output_height and output_width.
   * * 10: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_out, H_out, W_out], has the same type
   * as `input`.
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
   * Performs cumulative sum of the input elements along the given `axis`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * `axis`+R, defaults to -1.
   * * 2: exclusive, a NNADAPTER_BOOL8 tensor of shape [1], whether to exclude
   * the top element, defaults to false.
   * * 3: reverse, a NNADAPTER_BOOL8 tensor of shape [1], whether to perform the
   * cumsum in the reversed direction, defaults to false.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_CUM_SUM,

  /**
   * Compute 2-D deformable convolution on 4-D input.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
   * * 1: offset, a tensor of shape [N, 2 * deformable_groups * H_f * W_f, H_in,
   * W_in] and has the same type as `input`.
   * * 2: mask, a tensor of shape [N, deformable_groups * H_f * W_f, H_in, W_in]
   * and has the same type as `input`.
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
   * * 4: bias, a tensor of shape [C_out].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 5: pads, an optional NNADAPTER_INT32 tensor of shape [4], specifying
   * height_top, height_bottom, width_left, width_right.
   * * 6: strides, a NNADAPTER_INT32 tensor of shape [2], specifying
   * stride_height, stride_width.
   * * 7: group, a NNADAPTER_INT32 tensor of shape [1].
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * `group` = C_out = C_in.
   * * 8: deformable_group, a NNADAPTER_INT32 tensor of shape [1], specifying
   * the c-axis grouping number of `input`.
   * * 9: dilations, a NNADAPTER_INT32 tensor of shape [2], specifying
   * dilations_height, dilations_width.
   * * 10: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_out, H_out, W_out], has the same type
   * as `input`.
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
   * Dequantizes a quantized tensor to a full precision one.
   * The output is calculated using this formula:
   *     `output` = (`input` - zero_point) * scale, where zero_point and scale
   * is obtained from `input`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER and
   * NNADAPTER_QUANT_UINT8_ASYMM_PER_CHANNEL tensor.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_FLOAT32 tensor of the same shape as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_DEQUANTIZE,

  /**
   * Performs element-wise binary division(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` / `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_DIV,

  /**
   * Performs element-wise binary equal relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` == `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
   *
   * Available since version 1.
   */
  NNADAPTER_EQUAL,

  /**
   * Performs element-wise exp activation.
   * The output is calculated using this formula:
   * `output` = e^`input`
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
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
   * * 1: shape, a 1-D NNADAPTER_INT32 or NNADAPTER_INT64 tensor indicates the
   * shape you want to expand to, following the broadcasting rule.
   *
   * Outputs:
   * * 0: output, a tensor of shape `shape` and has the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_EXPAND,

  /**
   * Create a tensor of the 'shape' and filled with 'value'.
   *
   * Inputs:
   * * 0: shape, a NNADAPTER_INT32, NNADAPTER_INT64 tensor.
   * * 1: value, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64 or
   * NNADAPTER_BOOL tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of shape 'shape' and filled with 'value'.
   *
   * Available since version 1.
   */
  NNADAPTER_FILL,

  /**
   * Create a tensor of the same shape as `input` and filled with 'value'.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: value, a NNADAPTER_FLOAT32,  NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_BOOL tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape as 'input' and filled with
   * 'value'.
   *
   * Available since version 1.
   */
  NNADAPTER_FILL_LIKE,

  /*
   * According to the given start_axis and end_axis flattens successive
   * dimensions.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: start_axis, a NNADAPTER_INT32 tensor of shape [1], specifying the
   * start axis to flatten.
   * * 2: end_axis, a NNADAPTER_INT32 tensor of shape [1], specifying the end
   * axis to flatten.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_FLATTEN,

  /*
   * Performs element-wise floor activation.
   * The output is calculated using this formula:
   *     `output` = floor(`input`)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_FLOOR,

  /**
   * Add a fully connected layer.
   * The output is calculated using this formula:
   *     `output` = activation(`input` * `weight`' + `bias`)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor of at least rank 2, if
   * its rank is greater than 2, it will be flattened to a 2-D Tensor with the
   * shape [batch_size, input_size], where input_size represents the number of
   * inputs, matching the second dimension of weight, and batch_size is
   * calculated by dividing the number of elements by input_size.
   * * 1: weight, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL tensor of shape [num_units,
   * input_size], where the num_units represents the number of
   * output units, which also means the feature size of output.
   * * 2: bias, a tensor of shape [num_units].
   *      1) If input's type is NNADAPTER_FLOAT32, its type must be the
   * same type.
   *      2) If weight's type is NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * weight_scale.
   *      3) If weight's type is NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * weight_scale[i] for each output channel.
   * * 3: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [batch_size, num_units], and has the same
   * type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_FULLY_CONNECTED,

  /**
   * Output is obtained by gathering entries of axis of x indexed by index and
   * concatenate them together.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: indices, a NNADAPTER_INT32, NNADAPTER_INT64 tensor, with the rank Q,
   * the values must be in the bounds of the corresponding dimensions of
   * `input`.
   * * 2: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * `axis`+R, defaults to -1.
   *
   * Outputs
   * * 0: output, a tensor of the same type as `input`, with the rank Q + (R -
   * 1).
   *
   * Available since version 1.
   */
  NNADAPTER_GATHER,

  /**
   * Performs element-wise Gaussian Error Linear Units activation, refer to
   * https://arxiv.org/abs/1606.08415 for more details.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: approximate, a NNADAPTER_BOOL8 tensor of shape [1], whether to enable
   * pproximation.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_GELU,

  /**
   * Performs element-wise binary greater relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` == `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
   *
   * Available since version 1.
   */
  NNADAPTER_GREATER,

  /**
   * Performs element-wise binary greater_equal relational operation(with
   * Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` == `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
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
   * of shape [N, C, H, W].
   * * 1: grid, a NNADAPTER_FLOAT32 tensor of shape [N, H, W, 2].
   * * 2: align_corners, a NNADAPTER_BOOL8 tensor of shape [1]. If
   * `align_corners` = true, it will project -1 and 1 to the centers of the
   * corner pixels, otherwise, it will project -1 and 1 to the image edges.
   * * 3: mode, a NNADAPTER_INT32 tensor of shape [1], supported interpolation
   * modes: NNADAPTER_INTERPOLATE_MODE_NONE,
   * NNADAPTER_INTERPOLATE_MODE_BILINEAR, NNADAPTER_INTERPOLATE_MODE_NEAREST,
   * must be one of NNAdapterInterpolateMode.
   * * 4: pad_mode, a NNADAPTER_INT32 tensor of shape [1], supported padding
   * modes: NNADAPTER_PAD_MODE_NONE, NNADAPTER_PAD_MODE_CONSTANT,
   * NNADAPTER_PAD_MODE_REFLECT, NNADAPTER_PAD_MODE_REPLICATE,
   * NNADAPTER_PAD_MODE_EDGE, must be one of NNAdapterPadMode.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
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
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, ...].
   * * 1: scale, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 2: bias, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 3: epsilon, a NNADAPTER_FLOAT32 tensor of shape [1], a small value added
   * to the variance to prevent division by zero, defaults to 1e-5.
   * * 4: groups, a NNADAPTER_INT32 tensor of shape [1], the number of groups
  * that divided from channels.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_GROUP_NORMALIZATION,

  /**
   * Performs element-wise hard-sigmoid activation.
   * The output is calculated using this formula:
   *     `output` = max(0, min(1, `alpha` * `input` + `beta`))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 tensor of shape [1].
   * * 2: beta, a NNADAPTER_FLOAT32 tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_HARD_SIGMOID,

  /**
   * Performs element-wise hard-swish activation.
   * The output is calculated using this formula:
   *     `output` = `input` * max(0, min(1, `alpha` * `input` + `beta`))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 tensor of shape [1].
   * * 2: beta, a NNADAPTER_FLOAT32 tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
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
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: scale, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 2: bias, a NNADAPTER_FLOAT32 tensor of shape [C].
   * * 3: epsilon, a NNADAPTER_FLOAT32 tensor of shape [1], a small value added
   * to the variance to prevent division by zero, defaults to 1e-5.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_INSTANCE_NORMALIZATION,

  /**
   * Applies Layer Normalization over a N-D input described
   * in the paper Layer Normalization: <https://arxiv.org/pdf/1607.06450v1.pdf>.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: scale, a NNADAPTER_FLOAT32 tensor, shape is performed along the input
   * dimension from `begin_norm_axis` to rank(`input`).
   * * 2: bias, a NNADAPTER_FLOAT32 tensor, shape is performed along the input
   * dimension from `begin_norm_axis` to rank(`input`).
   * * 3: begin_norm_axis, a NNADAPTER_INT32 tensor of shape [1], indicates that
   * the normalization will be performed along the dimension from
   * `begin_norm_axis` to rank(`input`), defaults to 1.
   * * 4: epsilon, a NNADAPTER_FLOAT32 tensor of shape [1], a small value added
   * to the variance to prevent division by zero, defaults to 1e-5.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_LAYER_NORMALIZATION,

  /**
   * Performs element-wise Leaky ReLU activation.
   * The output is calculated using this formula:
   *     `output` = `input`, if `input` >= 0
   *     `output` = `alpha` * `input`, if `input` < 0
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: alpha, a NNADAPTER_FLOAT32 tensor of shape [1], slope of the formula
   * at `input` < 0.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_LEAKY_RELU,

  /**
   * Performs element-wise binary less relational operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` < `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
   *
   * Available since version 1.
   */
  NNADAPTER_LESS,

  /**
   * Performs element-wise binary less_equal relational operation(with
   * Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` <= `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
   *
   * Available since version 1.
   */
  NNADAPTER_LESS_EQUAL,

  /**
   * Performs element-wise natural log activation.
   * The output is calculated using this formula:
   *     `output` = ln(`input`)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_LOG,

  /**
   * Performs element-wise log of softmax activation.
   * The output is calculated using this formula:
   *     `output` = log(exp(`input`) / reduce_sum(exp(`input`), axis=`axis`,
   * keepdims=true))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of `input`, negative value works the same way as
   * `axis`+R.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_LOG_SOFTMAX,

  /**
   * Applies Lp Normalization along the provided `axis`.
   * The output is calculated using this formula:
   *     `output` = `input` / (sum(abs(`input`)) + `epsilon`), if `p` = 1
   *     `output` = `input` / (sqrt(sum(`input`^2)) + `epsilon`), if `p` = 2
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * `axis`+R, defaults to 1.
   * * 2: p, a NNADAPTER_INT32 tensor of shape [1], represents the exponent
   * value in the formula, only 1 or 2 is supported, defaults to 2.
   * * 3: epsilon, a NNADAPTER_FLOAT32 tensor of shape [1], a small value added
   * to the variance to prevent division by zero, defaults to 1e-5.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_LP_NORMALIZATION,

  /**
   * Matrix product that behaves like numpy.matmul:
   * https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
   *
   * Inputs:
   * * 0: x, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: y, a tensor of the compatible shape and the same type as `x`.
   * * 2: transpose_x, a NNADAPTER_BOOL8 tensor of shape [1], whether to
   * transpose the last
   * two dimensions of x before multiplication.
   * * 3: transpose_y, a NNADAPTER_BOOL8 tensor of shape [1], whether to
   * transpose the last
   * two dimensions of y before multiplication.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `x`.
   *
   * Available since version 1.
   */
  NNADAPTER_MAT_MUL,

  /**
   * Performs element-wise binary maximum(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = max(`input0`, `input1`)
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_MAX,

  /**
   * Applies a 2-D max pooling across the input according to kernel sizes,
   * stride sizes, and pad lengths.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C_in, H_in, W_in].
   * * 1: auto_pad, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterAutoPadCode values, NNADAPTER_AUTO_PAD_NONE means specifying the
   * explicit padding by `pads`, otherwise specifying the implicit padding
   * algorithm, including NNADAPTER_AUTO_PAD_SAME and NNADAPTER_AUTO_PAD_VALID.
   * * 2: pads, an optional NNADAPTER_INT32 tensor of shape [4], specifying
   * height_top, height_bottom, width_left and width_right.
   * * 3: kernel_shape, a NNADAPTER_INT32 tensor of shape [2], specifying
   * kernel_height and kernel_width.
   * * 4: strides, a NNADAPTER_INT32 tensor of shape [2], specifying
   * stride_height and stride_width.
   * * 5: ceil_mode, a NNADAPTER_BOOL8 tensor of shape [1], whether to use ceil
   * or floor to compute the output shape, defaults to false to use floor.
   * * 6: return_indices, a NNADAPTER_BOOL8 tensor of shape [1], whether to
   * return `indices` along with the outputs, defaults to false.
   * * 7: return_indices_dtype, a NNADAPTER_INT32 tensor of shape [1], specifies
   * the data type of `indices`, its value must be one of NNADAPTER_INT32,
   * NNADAPTER_INT64.
   * * 8: fuse_code, a NNADAPTER_INT32 tensor of shape [1], must be one of
   * NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C_out, H_out, W_out], has the same type
   * as `input`.
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
   * * 1: indices, a NNADAPTER_INT32, NNADAPTER_INT64 tensor and has the same
   * shape as `output`.
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
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor of shape [d0], [d1], ... [dn-1].
   *
   * Outputs:
   * * output0 ~ outputn-1, a tensor of shape [d0, d1, ... dn-1] and has the
   * same type as `input0` ~ `inputn-1`.
   *
   * Available since version 1.
   */
  NNADAPTER_MESHGRID,

  /**
   * Performs element-wise binary minimum(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = min(`input0`, `input1`)
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_MIN,

  /**
   * Performs element-wise binary multiplication(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` * `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_MUL,

  /**
   * Performs element-wise logical NOT operation.
   * The output is calculated using this formula:
   *     `output` = !`input`
   *
   * Inputs:
   * * 0: input, a NNADAPTER_BOOL8 tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_NOT,

  /**
   * Performs element-wise binary not_equal relational operation(with
   * Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` != `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_BOOL8,
   * NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a NNADAPTER_BOOL8 tensor, has the compatible shape as
   * 'input0'.
   *
   * Available since version 1.
   */
  NNADAPTER_NOT_EQUAL,

  /**
   * Performs element-wise binary logical OR operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` || `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_BOOL8 tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_OR,

  /**
   * Pads `input` according to the specified `pads`, `mode` and `value`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: pads, a NNADAPTER_INT32 tensor of shape [2 * rank(`input`)] and its
   * value should be [x0_begin, x0_end, x1_begin, x1_end,...].
   * * 2: mode, a NNADAPTER_INT32 tensor of shape [1], supported modes:
   * NNADAPTER_PAD_MODE_NONE, NNADAPTER_PAD_MODE_CONSTANT,
   * NNADAPTER_PAD_MODE_REFLECT, NNADAPTER_PAD_MODE_REPLICATE,
   * NNADAPTER_PAD_MODE_EDGE, must be one of NNAdapterPadModeCode values.
   * * 3: value, a tensor of shape [1] and has the same type as 'input', value
   * to fill the padded areas only when mode = NNADAPTER_PAD_MODE_CONSTANT.
   *
   * Outputs:
   * * 0: output, a tensor of padded shape and has the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_PAD,

  /**
   * Performs element-wise binary pow(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` ^ `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_POW,

  /**
   * Prior box operator Generate prior boxes for SSD(Single Shot MultiBox
   * Detector) algorithm.
   * https://arxiv.org/abs/1512.02325.
   * Each position of the input produce N prior boxes, N is determined by the
   * count of min_sizes, max_sizes and aspect_ratios,
   * The size of the box is in range(min_size, max_size) interval, which is
   * generated in sequence according to the aspect_ratios.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], feature.
   * * 1: image, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], image.
   * * 2: min_sizes, a NNADAPTER_FLOAT32 1-D tensor, min sizes of generated
   * prior boxes.
   * * 3: max_sizes, a NNADAPTER_FLOAT32 1-D tensor, max sizes of generated
   * prior boxes.
   * * 4: aspect_ratios, a NNADAPTER_FLOAT32 1-D tensor, aspect ratios of
   * generated prior boxes.
   * * 5: variances, a NNADAPTER_FLOAT32 1-D tensor, variances to be encoded in
   * prior boxes.
   * * 6: flip, a NNADAPTER_BOOL tensor of shape [1], whether to flip aspect
   * ratios, defaults to false.
   * * 7: clip, a NNADAPTER_BOOL tensor of shape [1], whether to clip
   * out-of-boundary boxes, defaults to false.
   * * 8: step_w, a NNADAPTER_FLOAT32 tensor of shape [1], prior boxes step
   * across width, 0.0 for auto calculation, defaults to 0.0.
   * * 9: step_h, a NNADAPTER_FLOAT32 tensor of shape [1], prior boxes step
   * across height, 0.0 for auto calculation, defaults to 0.0.
   * * 10: offset, a NNADAPTER_FLOAT32 tensor of shape [1], prior boxes center
   * offset, defaults to 0.5.
   * * 11: min_max_aspect_ratios_order, a NNADAPTER_BOOL tensor of shape [1], if
   * set to true, the output prior box is in order of [min, max, aspect_ratios],
   * which is consistent with Caffe. Please note,
   * this order affects the weights order of convolution layer followed by and
   * does not affect the final detection results, defaults to false.
   *
   * Outputs:
   * * 0: boxes, a NNADAPTER_FLOAT32 tensor of shape [H, W, num_priors, 4],
   * prior boxes, where num_priors is the box count of each position.
   * * 1: variances, a NNADAPTER_FLOAT32 tensor of shape [H, W, num_priors, 4],
   * expanded variances, where num_priors is the box count of each position.
   *
   * Available since version 1.
   */
  NNADAPTER_PRIOR_BOX,

  /**
   * Performs element-wise PReLU activation.
   * The output is calculated using this formula:
   *     `output` = `input`, if `input` >= 0
   *     `output` = `slope` * `input`, if `input` < 0
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, ...].
   * * 1: slope, a NNADAPTER_FLOAT32 tensor of shape [1] or [C].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_PRELU,

  /**
   * Quantizes a full precision tensor to a quantized one.
   * The output is calculated using this formula:
   *     `output` = `input` / `scale` + `zero_point`
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32 tensor of shape [N, C,
   * ...] .
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the axis of
   * the quantization dimension of the input tensor, which is only for
   * per-channel/per-axis quantization, should be in range [-R, R), where R is
   * the rank of input, negative value works the same way as axis+R, defaults to
   * 1.
   * * 2: scale, a NNADAPTER_FLOAT32 tensor of shape [1] or [C], scale for
   * quantization, can be a scalar, which means a per-tensor/per-layer
   * quantization, or a 1-D tensor for per-channel/per-axis quantization.
   * * 3: zero_point, a NNADAPTER_INT32 tensor of shape [1] or [C], zero point
   * for quantization, shape must match `scale`, default to 0.
   *
   * Outputs:
   * * 0: output, a quantized tensor of the same shape as `input` , its type
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
   * Generate a tensor containing a sequence of numbers that begin at `start`
   * and extends by increments of `step` up to `end` (exclusive).
   *
   * Inputs:
   * * 0: start, a NNADAPTER_FLOAT32, NNADAPTER_INT32 tensor of shape [1], first
   * entry.
   * * 1: end, a tensor of the same shape and type as `start`, exclusive upper
   * limmit.
   * * 2: step, a tensor of the same shape and type as `start`, value to step
   * by.
   *
   * Outputs:
   * * 0: output, a 1-D tensor of the same type as `start`.
   *
   * Available since version 1.
   */
  NNADAPTER_RANGE,

  /**
   * Computes the mean of the input tensor’s element along the provided `axes`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axes, a 1-D NNADAPTER_INT32 tensor, represents the dimension
   * along which reduce operation will be performed, if `axes` is empty,
   * `output` is calculated over all elements of `input`, should be in range
   * [-R, R), where R is the rank of input, negative value works the same way as
   * axis+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 tensor of shape [1], whether to keep the
   * reduced dimension, defaults to true.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_REDUCE_MEAN,

  /**
   * Computes the max of the input tensor’s element along the provided `axes`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axes, a 1-D NNADAPTER_INT32 tensor, represents the dimension
   * along which reduce operation will be performed, if `axes` is empty,
   * `output` is calculated over all elements of `input`, should be in range
   * [-R, R), where R is the rank of input, negative value works the same way as
   * axis+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 tensor of shape [1], whether to keep the
   * reduced dimension, defaults to true.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_REDUCE_MAX,

  /**
   * Computes the sum of the input tensor’s element along the provided `axes`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axes, a 1-D NNADAPTER_INT32 tensor, represents the dimension
   * along which reduce operation will be performed, if `axes` is empty,
   * `output` is calculated over all elements of `input`, should be in range
   * [-R, R), where R is the rank of input, negative value works the same way as
   * axis+R.
   * * 2: keepdim, a NNADAPTER_BOOL8 tensor of shape [1], whether to keep the
   * reduced dimension, defaults to true.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_REDUCE_SUM,

  /**
  * Performs element-wise rectified linear activation.
  * The output is calculated using this formula:
  *     `output` = max(0, `input`)
  *
  * Inputs:
  * * 0: input, a NNADAPTER_FLOAT32,
  * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
  *
  * Outputs:
  * * 0: output, a tensor of the same shape and type as `input`.
  *
  * Available since version 1.
  */
  NNADAPTER_RELU,

  /**
   * Performs element-wise rectified linear 6 activation.
   * The output is calculated using this formula:
   *     `output` = min(6, max(0, `input`))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_RELU6,

  /**
   * Returns a tensor with the same data and number of elements as `input`, but
   * with a newly specified shape, similar to numpy.reshape.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: shape, a 1-D NNADAPTER_INT32 or NNADAPTER_INT64 tensor, specifies the
   * new shape. At most one dimension of the new shape can be -1. In this case,
   * the value is inferred from the size of the tensor and the remaining
   * dimensions. A dimension could also be 0, in which case the actual dimension
   * value is unchanged.
   *
   * Outputs:
   * * 0: output, a tensor of shape specified by the `shape`, and has the same
   * type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_RESHAPE,

  /**
   * Resizes a tensor to given size using the nearest interpretation, output
   * height and width is determined by `shape` , `scales` in priority order.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, H, W].
   * * 1: shape, a NNADAPTER_INT32, NNADAPTER_INT64 tensor of shape [2],
   * indicates the output height and width.
   * * 2: scales, a NNADAPTER_FLOAT32 tensor of shape [2], indicates the scale
   * factor of the input height and width to calculate the output height and
   * width.
   * * 3: align_corners, a NNADAPTER_BOOL tensor of shape [1], if set to true,
   * the centers of the 4 corner pixels of the input and output tensors are
   * aligned, and preserving the values at the corner pixels.
   *
   * Outputs:
   * * 0: output, a tensor of shape specified by the `shape` or `scales`, and
   * has the same type as `input`.
   */
  NNADAPTER_RESIZE_NEAREST,

  /**
   * Resizes a tensor using the linear interpolation, output height and width is
   * determined by `shape` , `scales` in priority order.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, H, W].
   * * 1: shape, a NNADAPTER_INT32, NNADAPTER_INT64 tensor of shape [2],
   * indicates the output height and width.
   * * 2: scales, a NNADAPTER_FLOAT32 tensor of shape [2], indicates the scale
   * factor of the input height and width to calculate the output height and
   * width.
   * * 3: align_corners, a NNADAPTER_BOOL tensor of shape [1], if set to true,
   * the centers of the 4 corner pixels of the input and output tensors are
   * aligned, and preserving the values at the corner pixels.
   * * 4: align_mode, an optional NNADAPTER_INT32 tensor of shape [1], can be
   * ‘0’ for src_idx = `scale` * (dst_indx + 0.5) - 0.5 , can be ‘1’ for src_idx
   * = `scale` * dst_index.
   *
   * Outputs:
   * * 0: output, a tensor of shape specified by the `shape` or `scales`, and
   * has the same type as `input`.
   */
  NNADAPTER_RESIZE_LINEAR,

  /**
   * Perform bilinear interpolation on inputs of nonuniform sizes to obtain
   * fixed-size feature maps (e.g. 7*7) described in the paper Mask R-CNN
   * https://arxiv.org/abs/1703.06870.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor
   * of shape [N, C, H, W].
   * * 1: rois, a NNADAPTER_FLOAT32 tensor of shape [rois_num, 4] and has the
   * same type as `input`, where rois_num is the number of the ROI boxes, its
   * value is [[x1, y1, x2, y2], ...].
   * * 2: batch_indices, a NNADAPTER_INT32 tensor of shape [rois_num], denoting
   * the index of the corresponding image in the batch.
   * * 3: output_height, a NNADAPTER_INT32 tensor of shape [1], pooled output
   * height.
   * * 4: output_width, a NNADAPTER_INT32 tensor of shape [1], pooled output
   * width.
   * * 5: sampling_ratio, a NNADAPTER_INT32 tensor of shape [1], number of
   * sampling points in the interpolation grid used to compute the output value
   * of each pooled output bin. If > 0, then exactly sampling_ratio x
   * sampling_ratio sampling points per bin are used. If <= 0, then an adaptive
   * number of grid points are used (computed as ceil(roi_width / output_width),
   * and likewise for height).
   * * 6: spatial_scale, a NNADAPTER_FLOAT32 tensor of shape [1], multiplicative
   * spatial scale factor to translate ROI coords from their input scale to the
   * scale used when pooling.
   * * 7: aligned, a NNADAPTER_BOOL8 tensor of shape [1], if set to true, pixel
   * shift it by -0.5 for align more perfectly.
   *
   * Outputs:
   * * 0: output, a tensor of shape [N, C, output_height, output_width] and has
   * the same type as `input`.
   */
  NNADAPTER_ROI_ALIGN,

  /**
   * Outputs an 1-D tensor containing the shape of the input tensor.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: dtype, a NNADAPTER_INT32 tensor of shape [1], specifies the data type
   * of `output`, its value must be one of NNADAPTER_INT32, NNADAPTER_INT64.
   *
   * Outputs:
   * * 0: output, a 1-D NNADAPTER_INT32, NNADAPTER_INT64 tensor.
   *
   * Available since version 1.
   */
  NNADAPTER_SHAPE,

  /**
   * Performs element-wise sigmoid activation.
   * The output is calculated using this formula:
   *     `output` = 1 / (1 + exp(-`input`))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SIGMOID,

  /**
   * Produces a slice of `input` along multiple axes. Similar to numpy:
   * https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html.
   * Slice uses the `axes`, `starts`, `ends` and `steps` inputs to select a
   * sub-tensor from `input` tensor. All negative values in `starts[i]` and
   * `ends[i]` have `dims[axes[i]]` added to them, where `dims` are the
   * dimensions of `input`. Then `start[axes[i]]` is the adjusted `starts[i]` is
   * clamped into the range `[0, dims[axes[i]]]` for positive stepping and `[0,
   * dims[axes[i]]-1]` for negative stepping. For slicing to the end of a
   * dimension with unknown size, it is recommended to pass in INT_MAX when
   * slicing forward and 'INT_MIN' when slicing backward.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axes, a 1-D optional NNADAPTER_INT32 tensor that `starts` and `ends`
   * apply to, will be treated as [0, 1, ..., len(`starts`) - 1] if not
   * provided.
   * * 2: starts, a 1-D NNADAPTER_INT32 tensor of the same shape as `axes`,
   * starting indices of corresponding axis in `axes`.
   * * 3: ends, a 1-D NNADAPTER_INT32 tensor of the same shape as `axes`, ending
   * indices of corresponding axis in `axes`.
   * * 4: steps, a 1-D NNADAPTER_INT32 tensor of the same shape as `axes`, slice
   * step of corresponding axis in `axes`. Negative value means slicing
   * backward. 'steps' cannot be 0. Defaults to 1.
   *
   * Outputs:
   * * 0: output, a tensor of shape specified by the `axes`, `starts` , `ends`,
   * `steps` and the shape of `input`, and has the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SLICE,

  /**
   * Performs element-wise softmax activation.
   * The output is calculated using this formula:
   *     `output` = exp(`input`) / reduce_sum(exp(`input`), axis=`axis`,
   * keepdims=true)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the
   * dimension along which softmax will be performed, should be in range [-R,
   * R), where R is the rank of `input`, negative value works the same way as
   * `axis`+R.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SOFTMAX,

  /**
   * Performs element-wise Softplus activation.
   * The output is calculated using this formula:
   *     `output` = log(1 + exp^(`beta` * `input`)) / `beta`
   * Fornumerical stability, the implementation reverts to the linear function
   * when: `beta` * `input` > threshold:
   *     `output` = `input`
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: beta, a NNADAPTER_FLOAT32 tensor of shape [1].
   * * 2: threshold, a NNADAPTER_FLOAT32 tensor of shape [1].
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SOFTPLUS,

  /**
   * Split a tensor into a list of tensors along the specified `axis`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents which axis to
   * split on, should be in range [-R, R), where R is the rank of `input`,
   * negative value works the same way as `axis`+R.
   * * 2: split, a 1-D NNADAPTER_INT32 tensor, each of values indicates the
   * length of each output. Sum of the values must be equal to the dimension at
   * `axis` specified.
   *
   * Outputs:
   * * 0 ~ n-1: output0 ~ outputn-1, one or more outputs forming list of tensors
   * after splitting, has the same type as the `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SPLIT,

  /**
  * Performs element-wise square operation.
  * The output is calculated using this formula:
  *     `output` = `input`^2
  *
  * Inputs:
  * * 0: input, a NNADAPTER_FLOAT32,
  * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
  *
  * Outputs:
  * * 0: output, a tensor of the same shape and type as `input`.
  *
  * Available since version 1.
  */
  NNADAPTER_SQUARE,

  /**
   * Remove single-dimensional entries from the shape of tensor along the
   * specified `axes`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axes, a 1-D NNADAPTER_INT32 tensor, indicates the dimensions to
   * squeeze, all the single dimensions will be removed if `axes` is not
   * provided, should be in range [-R, R), where R is the rank of `input`,
   * negative value works the same way as `axis`+R.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SQUEEZE,

  /**
   * Concatenates a sequence of tensors along a new `axis`, all tensors need to
   * be the same shape.
   *
   * Inputs:
   * * 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * n: axis, a NNADAPTER_INT32 tensor of shape [1], represents the dimension
   * along which axis to concatenate, should be in range [-R, R), where R is the
   * rank of `input`, negative value works the same way as `axis`+R.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as the `input0` ~ `inputn-1`.
   *
   * Available since version 1.
   */
  NNADAPTER_STACK,

  /**
   * Performs element-wise binary subtraction(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` - `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   * * 2: fuse_code, a NNADAPTER_INT32 tensor of shape [1], specifies the
   * activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_SUB,

  /**
   * Performs element-wise sum of each of the input tensors (with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *      `output` = `input0` + `input1` + ... + `inputn-1`
   *
   * Inputs:
   * * 0 ~ n-1: input0 ~ inputn-1, a NNADAPTER_FLOAT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_SUM,

  /**
   * Performs element-wise swish activation.
   * The output is calculated using this formula:
   *     `output` = `input` / (1 + e ^ (-`input`))
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_SWISH,

  /**
   * Performs element-wise hyperbolic tangent activation.
   * The output is calculated using this formula:
   *     `output` = tanh(`input`)
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   *
   * Outputs:
   * * 0: output, a tensor of the same shape and type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_TANH,

  /**
   * Constructs a tensor by tiling a given tensor.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: repeats, a NNADAPTER_INT32 tensor of shape [rank(`input`)].
   *
   * Outputs:
   * * 0: output, a tensor of the same dimensions and type as `input`, and
   * output_dims[i] = input_dims[i] * repeats[i], and has the same type as
   * `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_TILE,

  /**
   * Retrieve the top-K largest or smallest elements along a specified axis.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_INT32, NNADAPTER_INT64,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: k, a 1-D NNADAPTER_INT32, NNADAPTER_INT64 tensor, containing a single
   * positive value corresponding to the number of top elements to retrieve.
   * * 2: axis, a NNADAPTER_INT32 tensor of shape [1], represents the dimension
   * on which to do the sort, should be in range [-R, R), where R is the rank of
   * `input`, negative value works the same way as `axis`+R.
   * * 3: largest, a NNADAPTER_BOOL8 tensor of shape [1], whether to return the
   * top-K largest or smallest elements.
   * * 4: sorted, a NNADAPTER_BOOL8 tensor of shape [1], whether to return the
   * elements in sorted order.
   * * 5: return_indices_dtype, a NNADAPTER_INT32 tensor of shape [1], its value
   * shoud be NNADAPTER_INT32 or NNADAPTER_INT64, specifies the data type of the
   * `indices`.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as input, containing top K values
   * from the input tensor.
   * * 1: indices, a NNADAPTER_INT32, NNADAPTER_INT64 tensor, containing the
   * corresponding input tensor indices for the top K values.
   *
   * Available since version 1.
   */
  NNADAPTER_TOP_K,

  /**
   * Transposes the input tensor, permuting the dimensions according to the perm
   * tensor, similar to numpy.transpose
   * https://numpy.org/doc/stable/reference/generated/numpy.transpose.html.
   * For example, the input with shape (1, 2, 3) and perm=(1, 0, 2), the shape
   * of output will be (2, 1, 3).
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: perm, an optional 1-D NNADAPTER_INT32 tensor, reverse the dimensions
   * if it is empty, otherwise permute the axes according to the values given.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_TRANSPOSE,

  /**
   * Insert single-dimensional entries to the shape of tensor along the
   * specified `axes`.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER
   * tensor.
   * * 1: axes, a 1-D NNADAPTER_INT32 tensor, indicates the dimensions to be
   * inserted, should be in range [-R, R), where R is the rank of `input`,
   * negative value works the same way as `axis`+R.
   *
   * Outputs:
   * * 0: output, a tensor of the same type as `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_UNSQUEEZE,

  /**
   * Unpacks the given axis of a rank-R tensor into rank-(R-1) tensors.
   *
   * Inputs:
   * * 0, a NNADAPTER_FLOAT32, NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, a NNADAPTER_INT32 tensor of shape [1], represents the dimension
   * along which axis to unpack, should be in range [-R, R), where R is the
   * rank of `input`, negative value works the same way as `axis`+R.
   * * 2: num, a NNADAPTER_INT32 tensor of shape [1], represents the length of
   * axis.
   *
   * Outputs:
   * * 0 ~ n-1: output0 ~ outputn-1, one or more outputs forming list of tensors
   * after unpack, has the same type as the `input`.
   *
   * Available since version 1.
   */
  NNADAPTER_UNSTACK,

  /**
   * Return elements, either from `input0` or `input1`, depending on `condition`
   * (with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html), similar to
   * numpy.where
   * https://numpy.org/doc/stable/reference/generated/numpy.where.html.
   *
   * Inputs:
   * * 0: condition, a NNADAPTER_BOOL8 tensor, when true, yield `input0`,
   * otherwise yield `input1`.
   * * 1: input0, a NNADAPTER_FLOAT32, NNADAPTER_INT32,
   * NNADAPTER_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 2: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0` and
   * `input1`.
   *
   * Available since version 1.
   */
  NNADAPTER_WHERE,

  /**
   * Performs element-wise binary logical XOR operation(with Numpy-style
   * broadcasting https://numpy.org/doc/stable/user/basics.broadcasting.html).
   * The output is calculated using this formula:
   *     `output` = `input0` ^ `input1`
   *
   * Inputs:
   * * 0: input0, a NNADAPTER_BOOL8 tensor.
   * * 1: input1, a tensor of the compatible shape and the same type as
   * `input0`.
   *
   * Outputs:
   * * 0: output, a tensor of the compatible shape and type as `input0`.
   *
   * Available since version 1.
   */
  NNADAPTER_XOR,

  /**
   * Multi-class non maximum suppression (NMS) on a batched of boxes and scores.
   * In the NMS step, this operator greedily selects a subset of detection
   * bounding boxes that have high scores larger than score_threshold,
   * if providing this threshold, then selects the largest nms_top_k confidences
   * scores if nms_top_k is larger than -1.
   * Then this operator pruns away boxes that have high IOU (intersection over
   * union) overlap with already selected boxes by adaptive threshold NMS based
   * on parameters of nms_threshold and nms_eta.
   * Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
   * per image if keep_top_k is larger than -1.
   * https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/layers/multiclass_nms_cn.html
   *
   * Inputs:
   * * 0: bboxes, a NNADAPTER_FLOAT32 tensor.
   * * Two types of bboxes are supported:
   *     1. A 3-D Tensor with shape [N, M, 4/8/16/24/32] represents
   * the predicted locations of M bounding bboxes, N is the batch size. 
   *        Each bounding box has four coordinate values and the layout
   * is [xmin, ymin, xmax, ymax], when box size equals to 4.
   *     2. A 3-D Tensor with shape [M, C, 4]. M is the number of bounding
   * boxes, C is the class number.
   * * 1: scores,scores, a NNADAPTER_FLOAT32 tensor.
   * * Two types of scores are supported:
   *     1. A 3-D Tensor with shape [N, C, M] represents the predicted
   * confidence predictions.
   *        N is the batch size, C is the class number, M is number of bounding
   * boxes.
   *        For each category there are total M scores which corresponding M
   * bounding boxes.
   *        In this case, input bboxes should be the first case with shape [N,
   * M, 4/8/16/24/32].
   *     2. A 2-D LoDTensor with shape [M, C]. M is the number of bbox, C is the
   * class number.
   *        In this case, input bboxes should be the second case with shape [M,
   * C, 4].
   * * 2: rois_num(optional), a NNADAPTER_INT32 tensor with shape [B], B is the
   * number of images.
   * rois_nums exist only if bboxes and scores is in the second case.
   * * 3: background_label, a NNADAPTER_INT32 tensor with shape [1], the index
   * of background label.
   * If set to 0, the background label will be ignored.
   * If set to -1, then all categories will be considered.
   * * 4: score_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], threshold
   * to filter out bounding boxes with low confidence score.
   * * 5: nms_top_k, a NNADAPTER_INT32 tensor with shape [1], maximum number of
   * detections to be kept according to the confidences after the filtering
   * detections based on score_threshold.
   * * 6: nms_threshold, a NNADAPTER_FLOAT32 tensor with shape [1], the
   * parameter for NMS.
   * * 7: nms_eta, a NNADAPTER_FLOAT32 tensor with shape [1], the parameter for
   * adaptive NMS.
   * * 8: keep_top_k, a NNADAPTER_INT32 tensor with shape [1], number of total
   * bboxes to be kept per image after NMS step.
   * "-1" means keeping all bboxes after NMS step.
   * * 9: normalized, a NNADAPTER_BOOL8 tensor with shape [1], whether
   * detections are normalized.
   * * 10: return_index,  a NNADAPTER_BOOL8 tensor with shape [1], whether to
   * return index of RoIs.
   *
   * Outputs:
   * * 0: output, a tensor with the same type as bboxes, with shape [No, 6].
   * "No" is the number of all RoIs. Each row has 6 values: [label, confidence,
   * xmin, ymin, xmax, ymax]
   * * 1: out_rois_num, a NNADAPTER_INT32 tensor of shape [B], B is the number
   * of images.
   * The number of NMS RoIs in each image.
   * * 2: index, a NNADAPTER_INT32 tensor with shape [No] represents the index
   * of selected bbox.
   * The index is the absolute index cross batches.
   * It is valid only if "return_index" is true.
   */
  NNADAPTER_NON_MAX_SUPPRESSION,

  /**
   * Generate YOLO detection boxes from output of YOLOv3 network, refer to
   * https://www.paddlepaddle.org.cn/documentation/docs/zh/2.1/api/paddle/vision/ops/yolo_box_cn.html#yolo-box
   * for more details.
   *
   * Inputs:
   * * 0: input, a NNADAPTER_FLOAT32 tensor of shape [N, C, H, W], its
   * dimension(C) stores box locations, confidence score and classification
   * one-hot keys of each anchor box.
   * * 1: imgsize, a NNADAPTER_INT32 tensor of shape [N, 2], holds height and
   * width of each input image used for resizing output box in input image
   * scale.
   * * 2: anchors, a NNADAPTER_INT32 tensor of shape [2], represents the anchor
   * width and height, it will be parsed pair by pair.
   * * 3: class_num, a NNADAPTER_INT32 tensor of shape [1], represents number of
   * classes.
   * * 4: conf_thresh, a NNADAPTER_FLOAT32 tensor of shape [1], the confidence
   * scores threshold of detection boxes, boxes with confidence scores under
   * threshold should be ignored.
   * * 5: downsample_ratio, a NNADAPTER_INT32 tensor of shape [1], down-sampling
   * rate from network input to this operation input.
   * * 6: clip_bbox, a NNADAPTER_BOOL8 tensor of shape [1], whether clip output
   * bonding box in `imgsize` boundary, defaults to true.
   * * 7: scale_x_y, a NNADAPTER_FLOAT32 tensor of shape [1], scale the center
   * point of decoded bounding box, defaults to 1.0.
   * * 8: iou_aware, a NNADAPTER_BOOL8 tensor of shape [1], whether to use iou
   * aware, defaults to false.
   * * 9: iou_aware_factor, a NNADAPTER_FLOAT32 tensor of shape [1], iou aware
   * factor, defaults to 0.5.
   *
   * Outputs:
   * * 0: boxes, a 3-D NNADAPTER_FLOAT32 tensor of shape [N, M, 4], N is the
   * batch size, M is the number of output boxes, and the coordinates of boxes
   * [xmin, ymin, xmax, ymax].
   * * 1: scores, a 3-D NNADAPTER_FLOAT32 tensor of shape [N, M, `class_num`].
   *
   * Available since version 1.
   */
  NNADAPTER_YOLO_BOX,
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
                            int (*callback)(int event_id, void* user_data),
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
 * Check whether the operations of the target model are supported by the target
 * devices.
 *
 * Available since version 1.
 */
int NNAdapterModel_getSupportedOperations(const NNAdapterModel* model,
                                          NNAdapterContext* context,
                                          bool* supported_operations);

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
                                                NNAdapterOperandType* type,
                                                void* device_buffer));
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
                                                 NNAdapterOperandType* type,
                                                 void* device_buffer));
/**
 * Start to run the execution synchronously.
 *
 * Available since version 1.
 */
int NNAdapterExecution_compute(NNAdapterExecution* execution);

#ifdef __cplusplus
}
#endif
