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

#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

enum ActivationType {
  kSigmoid,
  kReLU,
  kTanh,
  kIdentity,
};

ActivationType GetActiveType(const std::string& act);

// fp32 and half
template <typename T>
void relu(int num, const T* din, T* dout, float alpha, cudaStream_t stream);

template <typename out_type>
void relu_int8_nhwc4(int num,
                     const void* in,
                     void* out,
                     int N,
                     int K,
                     int H,
                     int W,
                     const void* scale,
                     float alpha,
                     cudaStream_t stream);

template <typename T>
void bias_relu(int num,
               const T* din,
               const float* bias,
               T* dout,
               float alpha,
               cudaStream_t stream);

// For int8
template <typename out_type>
void bias_relu_int8_nhwc(int num,
                         const void* in,
                         const void* bias,
                         void* out,
                         int N,
                         int C,
                         int H,
                         int W,
                         const void* scale,
                         float alpha,
                         cudaStream_t stream);

template <typename out_type>
void bias_int8_nhwc(int num,
                    const void* in,
                    const void* bias,
                    void* out,
                    int N,
                    int C,
                    int H,
                    int W,
                    const void* scale,
                    cudaStream_t stream);

template <typename T>
void sigmoid(const int num, const T* din, T* dout, cudaStream_t stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
