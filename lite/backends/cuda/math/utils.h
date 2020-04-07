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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <limits>
#include <string>

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

enum class BinaryOperation {
  kADD = 0,
  kMUL = 1,
  kDIV = 2,
  kSUB = 3,
};

template <typename T>
__device__ T binary_calc(T x, T y, BinaryOperation type);

template <>
__device__ __forceinline__ float binary_calc(float x,
                                             float y,
                                             BinaryOperation type) {
  if (type == BinaryOperation::kADD) return x + y;
  if (type == BinaryOperation::kMUL) return x * y;
  if (type == BinaryOperation::kDIV) return x / y;
  if (type == BinaryOperation::kSUB) return x - y;
}

template <typename T>
__device__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ int8_t from_float<int8_t>(float x) {
  x = fmaxf(x, std::numeric_limits<char>::min());
  x = fminf(x, std::numeric_limits<char>::max());
  return __float2int_rn(x);
}

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
