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

template <typename T>
struct RowwiseAdd {
  void operator()(const T* input,
                  const T* bias,
                  T* output,
                  const int width,
                  const int count,
                  const cudaStream_t& stream);
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
