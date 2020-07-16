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
#include <cudnn.h>

#include <string>
#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/activation.h"
#include "lite/core/context.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename Dtype>
inline __device__ Dtype Sigmoid(const Dtype a) {
  return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-a));
}

template <typename Dtype>
inline __device__ Dtype ReLU(const Dtype a) {
  return a > static_cast<Dtype>(0.f) ? a : static_cast<Dtype>(0.f);
}

template <typename Dtype>
inline __device__ Dtype Tanh(const Dtype a) {
  Dtype tmp = static_cast<Dtype>(-2.0) * a;
  return (static_cast<Dtype>(2.0) / (static_cast<Dtype>(1.0) + expf(tmp))) -
         static_cast<Dtype>(1.0);
}

template <typename T>
__global__ void GruForwardResetOutput(
    T* gate_value,
    T* reset_output_value,
    T* prev_output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_gate,
    bool is_batch);

template <typename T>
__global__ void GruForwardFinalOutput(
    T* gate_value,
    T* prev_output_value,
    T* output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_node,
    bool origin_mode,
    bool is_batch);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
