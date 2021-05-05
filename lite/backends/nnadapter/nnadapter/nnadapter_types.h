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

/**
 * Result codes.
 */
typedef enum {
  NNADAPTER_NO_ERROR = 0,
  NNADAPTER_OUT_OF_MEMORY = 1,
  NNADAPTER_INVALID_PARAMETER = 2,
  NNADAPTER_DEVICE_NOT_FOUND = 3,
} NNAdapterResultCode;

enum { NNADAPTER_MAX_SIZE_OF_DIMENSIONS = 8 };

/**
 * Operand precision codes.
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
  NNADAPTER_NETWORK = 27,
} NNAdapterOperandPrecisionCode;

/**
 * Operand layout codes.
 */
typedef enum {
  NNADAPTER_NCHW = 0,
  NNADAPTER_NHWC = 1,
} NNAdapterOperandLayoutCode;

/**
 * Operand life time codes, where to find the data for a operand.
 * * NNADAPTER_TEMPORARY_VARIABLE: A temporary operand in a newtork, There is no
 need to set its data during building a network.
 * * NNADAPTER_CONSTANT: A constant operand, copy to an new space allocated
 internally during building a network.
 */
typedef enum {
  NNADAPTER_TEMPORARY_VARIABLE = 0,
  NNADAPTER_CONSTANT = 1,
  NNADAPTER_INPUT = 2,
  NNADAPTER_OUTPUT = 3,
} NNAdapterOperandLifetimeCode;

/**
 * Operation codes.
 */
typedef enum {
  /**
   * 2-D convolution operation.
   * The CONV_2D op computes a 2-D convolution based on the input, filter,
   * strides, paddings, dilations, groups and etc.
   *
   * Inputs:
   * * 0: input, A 4-D tensor with shape [N, C_in, H_in, W_in].
   * * 1: filter, A NNADAPTER_TENSOR_FLOAT16, NNADAPTER_TENSOR_FLOAT32,
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER or
   * NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER 4-D tensor, The convolution
   * kernel with shape [C_out, C_in, filter_height, filter_width], where C_out
   * and C_in is the number of the channels of output and input, filter_height
   * and filter_width is the filter's kernel size in the 'H' and 'W' dimension.
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
   * * 9: fuse_code, A NNADAPTER_INT32 scalar, must be one of NNAdapterFuseCode
   * values.
   * * 10: dilation_width, optional, A NNADAPTER_INT32 scalar. Defaults to 1. If
   * this input is set, input 11 (dilation_height) must be specified as well.
   * * 11: dilation_height, optional, A NNADAPTER_INT32 scalar. Defaults to 1.
   * If this input is set, input 10 (dilation_width) must be specified as well.
   *
   * Outputs:
   * * 0: output, The output 4-D tensor with shape [N, C_out, H_out, W_out], its
   * type is the same as x.
   *      H_out = (H_in + padding_height_top + padding_height_bottom -
   * (dilation_height * (filter_height
   *              - 1) + 1)) / stride_height + 1
   *      W_out = (W_in + padding_width_left + padding_width_right -
   * (dilation_width * (filter_width - 1)
   *              + 1)) / stride_width + 1
   */
  NNADAPTER_CONV_2D = 3,
} NNAdapterOperationCode;

/**
 * Fused activation function types.
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
 */
typedef struct NNAdapterSymmPerLayerQuantParams {
  float scale;
} NNAdapterSymmPerLayerQuantParams;

/**
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL
 * operand.
 */
typedef struct NNAdapterSymmPerChannelQuantParams {
  uint32_t channel_dim;
  uint32_t scale_count;
  float* scales;
} NNAdapterSymmPerChannelQuantParams;

/**
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_ASYMM_PER_LAYER
 * operand.
 */
typedef struct NNAdapterAsymmPerLayerQuantParams {
  float scale;
  int32_t zero_point;
} NNAdapterAsymmPerLayerQuantParams;

/**
 * The type of an operand, include both scalars and tensors.
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

typedef struct NNAdapterDevice NNAdapterDevice;
typedef struct NNAdapterOperand NNAdapterOperand;
typedef struct NNAdapterOperation NNAdapterOperation;
typedef struct NNAdapterModel NNAdapterModel;
typedef struct NNAdapterCompilation NNAdapterCompilation;
typedef struct NNAdapterExecution NNAdapterExecution;
