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
#include "lite/backends/cuda/math/bias.h"

#include <iostream>

#include "lite/backends/cuda/cuda_utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
__global__ void RowwiseAddKernel(
    const T* a, const T* b, T* c, int width, int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i / width;
    int w = i - h * width;
    c[i] = a[i] + b[w];
  }
}

template <>
__global__ void RowwiseAddKernel(
    const half* a, const half* b, half* c, int width, int num) {
  CUDA_KERNEL_LOOP(i, num) {
    int h = i / width;
    int w = i - h * width;
    c[i] = __hadd(a[i], b[w]);
  }
}

template <typename T>
void RowwiseAdd<T>::operator()(const T* input,
                               const T* bias,
                               T* output,
                               const int width,
                               const int count,
                               const cudaStream_t& stream) {
  RowwiseAddKernel<T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
      input, bias, output, width, count);
  CUDA_POST_KERNEL_CHECK;
}

template struct RowwiseAdd<float>;
template struct RowwiseAdd<half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
