// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename T>
__device__ inline T math_log(T a);

template <>
__device__ inline float math_log<float>(float a) {
  return logf(a);
}

template <typename T>
__device__ inline T math_exp(T a);

template <>
__device__ inline float math_exp<float>(float a) {
  return expf(a);
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
