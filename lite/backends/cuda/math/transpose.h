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
#include <vector>
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
class Transpose {
 public:
  void NCHW2NHWC(int N, int C, int HxW, const T* X, T* Y, cudaStream_t* stream);

  void NHWC2NCHW(int N, int C, int HxW, const T* X, T* Y, cudaStream_t* stream);

  void transpose(T* dst,
                 const T* src,
                 const std::vector<int64_t>& src_dims,
                 const std::vector<int>& axes,
                 cudaStream_t* stream);

  // void transpose(T* dst,
  //               const T* src,
  //               const std::vector<int>& src_dims,
  //               const std::vector<int>& axes,
  //               cudaStream_t* stream);

 private:
  lite::Tensor Y_dims_, strides_;  // for transpose.
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
