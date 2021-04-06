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
  NNADAPTER_INVALID_OBJECT = 2,
  NNADAPTER_RESOURCE_NOT_FOUND = 3,
  NNADAPTER_SYMBOL_NOT_FOUND = 4,
  NNADAPTER_OUT_OF_BOUNDS = 5,
  NNADAPTER_NOT_INITIALIZED = 6,
} NNAdapterResultCode;

enum { NNADAPTER_MAX_SIZE_OF_DIMENSIONS = 8 };

/**
 * Operand precision codes.
 */
typedef enum {
  /**
   * An 8 bit boolean scalar value.
   */
  NNADAPTER_BOOL = 0,
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
  NNADAPTER_TENSOR_BOOL = 12,
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
  NNADAPTER_TENSOR_QUANT_INT8_ASYMM_PER_LAYER = 26,
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
 * Operation codes.
 */
typedef enum {
  /**
   * 2-D convolution operation.
   */
  NNADAPTER_CONV_2D = 0
} NNAdapterOperationCode;

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
  uint32_t channelDim;
  uint32_t scaleCount;
  const float* scales;
} NNAdapterSymmPerChannelQuantParams;

/**
 * The quantization parameters for NNADAPTER_TENSOR_QUANT_INT8_ASYMM_PER_LAYER
 * operand.
 */
typedef struct NNAdapterAsymmPerLayerQuantParams {
  float scale;
  int32_t zeroPoint;
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
   * The number of dimensions.
   *
   * Must be 0 for scalars.
   */
  uint32_t dimensionCount;

  /**
   * The dimensions of the tensor.
   */
  const uint32_t dimensions[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];

  /**
   * The quantization parameters.
   */
  union {
    NNAdapterSymmPerLayerQuantParams symmPerLayerParams;
    NNAdapterSymmPerChannelQuantParams symmPerChannelParams;
    NNAdapterAsymmPerLayerQuantParams asymmPerLayerParams;
  };
} NNAdapterOperandType;

typedef int32_t NNAdapterOperationType;

typedef struct NNAdapterDevice NNAdapterDevice;
typedef struct NNAdapterNetwork NNAdapterNetwork;
typedef struct NNAdapterOperand NNAdapterOperand;
typedef struct NNAdapterOperation NNAdapterOperation;
typedef struct NNAdapterModel NNAdapterModel;
typedef struct NNAdapterExecution NNAdapterExecution;
