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
#include "lite/backends/cuda/cuda_utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

void fp32_to_int8_nhwc(int num,
                       const void* din,
                       void* dout,
                       const void* scale,
                       int N,
                       int C,
                       int H,
                       int W,
                       cudaStream_t stream);

void fp32_to_fp16(int num, const float* din, half* dout, cudaStream_t stream);
void fp32_to_fp16(int num, const float* din, half* dout);

void fp16_to_fp32(int num, const half* din, float* dout, cudaStream_t stream);
void fp16_to_fp32(int num, const half* din, float* dout);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
