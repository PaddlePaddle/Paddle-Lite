// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <vector>

namespace fake_ddk {

/* The type of return value of the function call */
typedef enum {
  SUCCESS = 0,
  FAILURE = -1,
  INVALID_INPUTS = -2,
  INVALID_OUTPUTS = -3,
  INVALID_MODEL = -4,
  INVALID_PARAM = -5,
  NO_MEMORY = -6,
  DEVICE_UNAVAILABLE = -7,
  INVALID_TENSOR = -8,
  INVALID_OP = -9,
} StatusType;

/* The precision of Tensor */
typedef enum {
  BOOL8 = 0,
  INT8,
  INT16,
  INT32,
  INT64,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  QUANT_INT8_SYMM_PER_LAYER,
  QUANT_INT8_SYMM_PER_CHANNEL,
  QUANT_INT32_SYMM_PER_LAYER,
  QUANT_INT32_SYMM_PER_CHANNEL,
  QUANT_UINT8_ASYMM_PER_LAYER,
} PrecisionType;

/* The data layout of Tensor */
typedef enum {
  NCHW = 0,
  NHWC,
  ANY,
} DataLayoutType;

/* The life time of Tensor */
typedef enum {
  TEMPORARY_VARIABLE = 0,
  CONSTANT,
  INPUT,
  OUTPUT,
} LifeTimeType;

/* Fused activation function types */
typedef enum {
  FUSE_NONE = 0,
  FUSE_RELU = 1,
  FUSE_RELU1 = 2,
  FUSE_RELU6 = 3,
} FuseType;

/* Pad type enum */
typedef enum {
  /*
   * Decided by driver.
  */
  AUTO = 0,
  /*
   * VALID Padding: it means no padding and it assumes that all the dimensions
   * are valid
   * So that the input image gets fully covered by a filter and the stride
   * specified by you.
  */
  VALID,
  /*
   * SAME Padding: it applies padding to the input image so that the input image
   * gets fully covered by the filter and specified stride
   * It is called SAME because,
   * for stride 1, the output will be the same as the input.
  */
  SAME
} PadType;

}  // namespace fake_ddk
