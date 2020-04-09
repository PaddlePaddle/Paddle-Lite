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
#include <cuda_runtime.h>
#include <string>
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename Dtype>
void elementwise(const Dtype* x_data,
                 const Dtype* y_data,
                 Dtype* out_data,
                 int pre,
                 int n,
                 int post,
                 BinaryOperation type,
                 cudaStream_t stream);

template <typename Dtype>
void elementwise_act(const Dtype* x_data,
                     const Dtype* y_data,
                     Dtype* out_data,
                     int pre,
                     int n,
                     int post,
                     std::string act,
                     BinaryOperation type,
                     cudaStream_t stream);

template <typename Dtype>
void elementwise_add(int num,
                     const Dtype* x_data,
                     const Dtype* y_data,
                     Dtype* out_data,
                     cudaStream_t stream);

void elementwise_add_int8(int num,
                          const float* x_data,
                          const float* y_data,
                          const float alpha,
                          int8_t* out_data,
                          cudaStream_t stream);
// input type is float32
// output type is int8
void elementwise_add_nhwc4_int8(int num,
                                const void* x_data,
                                const void* y_data,
                                const float alpha,
                                void* out_data,
                                cudaStream_t stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
