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

#include <sys/time.h>
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

inline static int maptofactor(int i, int factor) {
  return (i + factor - 1) / factor;
}

static std::vector<size_t> DefaultGlobalWorkSize(const DDim& image_dim,
                                                 const DDim& image_shape) {
  // n c h w
  //  auto image_dim = image.dims();
  if (image_dim.size() == 4) {
    auto n = image_dim[0];
    auto h = image_dim[2];
    auto w = image_dim[3];
    auto image_width = image_shape[0];
    size_t work_size_0 = image_width / w;
    size_t work_size_1 = w;
    size_t work_size_2 = n * h;
    return {work_size_0, work_size_1, work_size_2};
  } else if (image_dim.size() == 2) {
    auto h = image_dim[0];
    auto w = image_dim[1];
    return {1,
            static_cast<unsigned int>(image_shape[0]),
            static_cast<unsigned int>(image_shape[1])};
  } else if (image_dim.size() == 1) {
    return {1, static_cast<unsigned int>(image_shape[0]), 1};
  } else if (image_dim.size() == 3) {
    size_t c = image_dim[0];
    size_t h = image_dim[1];
    size_t w = image_dim[2];
    return {(c + 3) / 4, w, h};
  }
  LOG(FATAL) << " not support this dim, need imp ";
}

static const std::string GetTimeStamp() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return std::to_string(time.tv_usec);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
