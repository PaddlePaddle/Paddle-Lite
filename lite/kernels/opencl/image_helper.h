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

#include <map>
#include <string>
#include <vector>
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

static std::map<std::string, size_t> InitImageDimInfoWith(
    const DDim& tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;
  return std::map<std::string, size_t>({{"width", width}, {"height", height}});
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
