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

#include <stddef.h>
#include <stdint.h>
#include <sys/cdefs.h>

enum { NNADAPTER_VERSION = 1 };

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

/**
 * Operand precision codes.
 *
 * Available since version 1.
 */
typedef enum {
  /**
   * An 8 bit boolean scalar value.
   */
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
   * A tensor of 8 bit boolean values.
   */
  NNADAPTER_TENSOR_BOOL8 = 12,
  NNADAPTER_TENSOR_INT8 = 13,
  NNADAPTER_TENSOR_UINT8 = 14,
  NNADAPTER_TENSOR_INT16 = 15,
  NNADAPTER_TENSOR_UINT16 = 16,
  NNADAPTER_TENSOR_INT32 = 17,
  NNADAPTER_TENSOR_UINT32 = 18,
  NNADAPTER_TENSOR_INT64 = 19,
  NNADAPTER_TENSOR_UINT64 = 20,
  NNADAPTER_TENSOR_FLOAT16 = 21,
  NNADAPTER_TENSOR_FLOAT32 = 22,
  NNADAPTER_TENSOR_FLOAT64 = 23,
  /**
   * A tensor of 8 bit signed integers that represent real numbers.
   * - scale: a 32 bit floating point value greater than zero.
   *
   * The formula is:
   * real_value = integer_value * scale.
   */
  NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER = 24,
  NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL = 25,
  NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER = 26,
  NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER = 27,
  NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL = 28,
  NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER = 29,
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
  NNADAPTER_CONSTANT_COPY = 1,
  NNADAPTER_CONSTANT_REFERENCE = 2,
  NNADAPTER_MODEL_INPUT = 3,
  NNADAPTER_MODEL_OUTPUT = 4,
} NNAdapterOperandLifetimeCode;

/**
 * Operation codes.
 *
 * Available since version 1.
 */
typedef enum {
  /**
   * Performs element-wise binary addition(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, A tensor with the same type as input0.
   * * 2: fuse_code, A NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, The result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_ADD = 0,

  /**
   * Applies a 2-D average pooling across the input according to kernel sizes,
   * stride sizes, and pad lengths.
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: padding_width_left, A NNADAPTER_INT32 scalar.
   * * 2: padding_width_right, A NNADAPTER_INT32 scalar.
   * * 3: padding_height_top, A NNADAPTER_INT32 scalar.
   * * 4: padding_height_bottom, A NNADAPTER_INT32 scalar.
   * * 5: stride_width, A NNADAPTER_INT32 scalar.
   * * 6: stride_height, A NNADAPTER_INT32 scalar.
   * * 7: filter_width, A NNADAPTER_INT32 scalar, filter_width=W_in and
   * filter_height=H_in represents a global 2-D average pooling.
   * * 8: filter_height, A NNADAPTER_INT32 scalar, filter_width=W_in and
   * filter_height=H_in represents a global 2-D average pooling.
   * * 9: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   * * 10: ceil_mode, A NNADAPTER_BOOL8 scalar, whether to use ceil or floor
   * (default) to compute the output shape. Defaults to false.
   * * 11: count_include_pad, A NNADAPTER_BOOL8 scalar, whether include pad
   * pixels when calculating values for the edges. Defaults to false.
   *
   * Outputs:
   * * 0: output, The output 4-D tensor with shape [N, C_out, H_out, W_out], its
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
  NNADAPTER_AVERAGE_POOL_2D = 1,

  /**
   * Performs a normal or depthwise 2-D convolution operation.
   * The CONV_2D op computes a 2-D convolution based on the input, filter,
   * strides, paddings, dilations, groups and etc.
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: filter, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor.
   *      1) For a normal convolution, the filter's shape is [C_out, C_in,
   * filter_height, filter_width], where C_out and C_in is the number of the
   * channels of output and input, filter_height and filter_width is the
   * filter's kernel size in the 'H' and 'W' dimension.
   *      2) For a depthwise convolution, the filter's shape is [C_out, 1,
   * filter_height, filter_width], where C_out is the number of the channels of
   * output, filter_height and filter_width is the filter's kernel size in the
   * 'H' and 'W' dimension.
   * * 2: bias, A 1-D tensor with shape [C_out].
   *      1) If input's type is NNADAPTER_TENSOR_FLOAT16 or
   * NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
   *      2) If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * filter_scale.
   *      3) If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * filter_scale[i] for each output channel.
   * * 3: padding_width_left, A NNADAPTER_INT32 scalar.
   * * 4: padding_width_right, A NNADAPTER_INT32 scalar.
   * * 5: padding_height_top, A NNADAPTER_INT32 scalar.
   * * 6: padding_height_bottom, A NNADAPTER_INT32 scalar.
   * * 7: stride_width, A NNADAPTER_INT32 scalar.
   * * 8: stride_height, A NNADAPTER_INT32 scalar.
   * * 9: group, A NNADAPTER_INT32 scalar.
   *      1) For a normal convolution, group must be 1.
   *      2) For a depthwise convolution, the formula should be satisfied:
   * group=C_out=C_in.
   * * 10: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   * * 11: dilation_width, A NNADAPTER_INT32 scalar. Defaults to 1.
   * * 12: dilation_height, A NNADAPTER_INT32 scalar. Defaults to 1.
   *
   * Outputs:
   * * 0: output, The output 4-D tensor with shape [N, C_out, H_out, W_out], its
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
  NNADAPTER_CONV_2D = 2,

  /**
   * Performs element-wise binary division(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, A tensor with the same type as input0.
   * * 2: fuse_code, A NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, The result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_DIV = 3,

  /**
   * Add a fully connected layer.
   * The output is calculated using this formula:
   *     output = activation(input * weight' + bias)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor of at least rank 2, If
   * its rank is greater than 2, it will be flattened to a 2-D Tensor with the
   * shape [batch_size, input_size], where input_size represents the number of
   * inputs, matching the second dimension of weight, and batch_size is
   * calculated by dividing the number of elements by input_size
   * * 1: weight, A 2-D tensor with shape [num_units, input_size], where the
   * num_units represents the number of output units, which also means the
   * feature size of output.
   * * 2: bias, A 1-D tensor with shape [num_units].
   *      1) If input's type is NNADAPTER_TENSOR_FLOAT16 or
   * NNADAPTER_TENSOR_FLOAT32, its type must be the same type.
   *      2) If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER, its
   * type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER, and bias_scale
   * == input_scale * weight_scale.
   *      3) If filter's type is NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL,
   * its type should be NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL, and
   * bias_scale[i] = input_scale * weight_scale[i] for each output channel.
   * * 3: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   *
   * Outputs:
   * * 0: output, A 2-D tensor with shape [batch_size, num_units], and its type
   * is the same as input.
   *
   * Available since version 1.
   */
  NNADAPTER_FULLY_CONNECTED = 4,

  /**
   * Applies the hard-sigmoid activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = max(0, min(1, alpha * input + beta))
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_HARD_SIGMOID = 5,

  /**
   * Applies the hard-swish activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = input * max(0, min(1, alpha * input + beta))
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_HARD_SWISH = 6,

  /**
   * Applies a 2-D max pooling across the input according to kernel sizes,
   * stride sizes, and pad lengths.
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor with shape [N, C_in,
   * H_in, W_in].
   * * 1: padding_width_left, A NNADAPTER_INT32 scalar.
   * * 2: padding_width_right, A NNADAPTER_INT32 scalar.
   * * 3: padding_height_top, A NNADAPTER_INT32 scalar.
   * * 4: padding_height_bottom, A NNADAPTER_INT32 scalar.
   * * 5: stride_width, A NNADAPTER_INT32 scalar.
   * * 6: stride_height, A NNADAPTER_INT32 scalar.
   * * 7: filter_width, A NNADAPTER_INT32 scalar, filter_width=W_in and
   * filter_height=H_in represents a global 2-D max pooling.
   * * 8: filter_height, A NNADAPTER_INT32 scalar, filter_width=W_in and
   * filter_height=H_in represents a global 2-D max pooling.
   * * 9: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   * * 10: ceil_mode, A NNADAPTER_BOOL8 scalar, whether to use ceil or floor
   * (default) to compute the output shape. Defaults to false.
   * * 11: count_include_pad, A NNADAPTER_BOOL8 scalar, whether include pad
   * pixels when calculating values for the edges. Defaults to false.
   *
   * Outputs:
   * * 0: output, The output 4-D tensor with shape [N, C_out, H_out, W_out], its
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
  NNADAPTER_MAX_POOL_2D = 7,

  /**
   * Performs element-wise binary multiplication(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, A tensor with the same type as input0.
   * * 2: fuse_code, A NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, The result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_MUL = 8,

  /**
   * Applies rectified linear activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = max(0, input)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_RELU = 9,

  /**
   * Applies rectified linear 6 activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = min(6, max(0, input))
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_RELU6 = 10,

  /**
   * Applies sigmoid activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = 1 / (1 + exp(-input))
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SIGMOID = 11,

  /**
   * Computes the normalized exponential values for the input tensor
   * element-wise.
   * The output is calculated using this formula:
   *     output = exp(input) / reduce_sum(exp(input), axis=axis, keepdims=true)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: axis, A NNADAPTER_INT32 scalar. Defaults to 1. It represents the
   * dimension along which softmax will be performed. It should be in range [-R,
   * R), where R is the rank of input, negative value works the same way as
   * axis+R.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_SOFTMAX = 12,

  /**
   * Performs element-wise binary subtraction(with Numpy-style broadcasting
   * https://numpy.org/doc/stable/user/basics.broadcasting.html).
   *
   * Inputs:
   * * 0: input0, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: input1, A tensor with the same type as input0.
   * * 2: fuse_code, A NNADAPTER_INT32 scalar, Specifies the activation to the
   * result, must be one of NNAdapterFuseCode values.
   *
   * Outputs:
   * * 0: output, The result with the same type as two inputs.
   *
   * Available since version 1.
   */
  NNADAPTER_SUB = 13,

  /**
   * Applies the hyperbolic tangent activation to the input tensor element-wise.
   * The output is calculated using this formula:
   *     output = tanh(input)
   *
   * Inputs:
   * * 0: input, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   *
   * Outputs:
   * * 0: output, A tensor with the same shape and type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_TANH = 14,

  /**
   * Transposes the input according to the perm, similar to numpy.transpose
   * https://numpy.org/doc/stable/reference/generated/numpy.transpose.html.
   * For example, The input with shape (1, 2, 3) and perm=(1, 0, 2), the shape
   * of output will be (2, 1, 3).
   *
   * Inputs:
   * * 0: input0, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER tensor.
   * * 1: perm, An optional 1-D NNADAPTER_TENSOR_INT32 Tensor, reverse the
   * dimensions of input if perm is not given, otherwise permute the axes
   * according to the values given.
   *
   * Outputs:
   * * 0: output, A tensor with the same type as input.
   *
   * Available since version 1.
   */
  NNADAPTER_TRANSPOSE = 15,
} NNAdapterOperationCode;

/**
 * Fused activation function types.
 *
 * Available since version 1.
 */
typedef enum {
  /** NO fused activation function. */
  NNADAPTER_FUSED_NONE = 0,
  /** Fused ReLU activation function. */
  NNADAPTER_FUSED_RELU = 1,
  /** Fused ReLU1 activation function. */
  NNADAPTER_FUSED_RELU1 = 2,
  /** Fused ReLU6 activation function. */
  NNADAPTER_FUSED_RELU6 = 3,
} NNAdapterFuseCode;

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

typedef int32_t NNAdapterDeviceType;

/**
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER
 * operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterSymmPerLayerQuantParams {
  float scale;
} NNAdapterSymmPerLayerQuantParams;

/**
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL
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
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_ASYMM_PER_LAYER
 * operand.
 *
 * Available since version 1.
 */
typedef struct NNAdapterAsymmPerLayerQuantParams {
  float scale;
  int32_t zero_point;
} NNAdapterAsymmPerLayerQuantParams;

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
   * The number of dimensions.
   *
   * Must be 0 for scalars.
   */
  uint32_t dimension_count;

  /**
   * The dimensions of the tensor.
   * -1 means Any for supporting dynamic shape.
   */
  int32_t dimensions[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];

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
