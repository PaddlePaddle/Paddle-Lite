// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

namespace nnadapter {
namespace operation {
namespace math {

typedef enum {
  DATA_TYPE_BOOL8 = 0,
  DATA_TYPE_INT8 = 1,
  DATA_TYPE_UINT8 = 2,
  DATA_TYPE_INT16 = 3,
  DATA_TYPE_UINT16 = 4,
  DATA_TYPE_INT32 = 5,
  DATA_TYPE_UINT32 = 6,
  DATA_TYPE_INT64 = 7,
  DATA_TYPE_UINT64 = 8,
  DATA_TYPE_FLOAT16 = 9,
  DATA_TYPE_FLOAT32 = 10,
  DATA_TYPE_FLOAT64 = 11,
} DataTypeCode;

typedef enum { RELU = 1, RELU1 = 2, RELU6 = 3, SIGMOID = 4 } ActivationTypeCode;

typedef enum {
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
  MAX = 5,
  MIN = 6,
  POW = 7
} ElementwiseTypeCode;

// Fused activation function types
typedef enum {
  FUSE_NONE = 0,
  FUSE_RELU = 1,
  FUSE_RELU1 = 2,
  FUSE_RELU6 = 3,
} FuseCode;

// Get the slice of the shape
std::vector<int32_t> shape_slice(const std::vector<int32_t>& input_shape,
                                 int start,
                                 int end);
// Get the production of the shape
int64_t shape_production(const std::vector<int32_t>& input_shape);

// Get the broadcasted shape
std::vector<int32_t> shape_broadcast(const std::vector<int32_t>& input0_shape,
                                     const std::vector<int32_t>& input1_shape);

// Get the strides of the shape
std::vector<int64_t> shape_strides(const std::vector<int32_t>& input_shape);

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
