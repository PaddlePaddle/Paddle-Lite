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
#include "lite/utils/log/cp_logging.h"
#include "lite/utils/timer.h"

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

static std::vector<size_t> DefaultGlobalWorkSize(const DDim& tensor_dim,
                                                 const DDim& image_shape) {
  if (tensor_dim.size() == 4) {
    auto n = tensor_dim[0];
    auto h = tensor_dim[2];
    auto w = tensor_dim[3];
    auto image_width = image_shape[0];
    size_t ws0 = image_width / w;
    size_t ws1 = w;
    size_t ws2 = n * h;
    return {ws0, ws1, ws2};
  } else if (tensor_dim.size() == 2) {
    return {1,
            static_cast<unsigned int>(image_shape[0]),
            static_cast<unsigned int>(image_shape[1])};
  } else if (tensor_dim.size() == 1) {
    return {1, static_cast<unsigned int>(image_shape[0]), 1};
  } else if (tensor_dim.size() == 3) {
    size_t c = tensor_dim[0];
    size_t h = tensor_dim[1];
    size_t w = tensor_dim[2];
    return {(c + 3) / 4, w, h};
  }
  LOG(FATAL) << "Unsupport DefaultGlobalWorkSize with tensor_dim.size():"
             << tensor_dim.size()
             << ", image_shape.size():" << image_shape.size();
  return {};
}

static DDim Broadcast2GpuShape(const DDim& src_dim) {
  CHECK_LE(src_dim.size(), 4);
  DDim dst_dim({1, 1, 1, 1});
  size_t offset = 4 - src_dim.size();
  for (auto i = 0; i < src_dim.size(); ++i) {
    dst_dim[offset + i] = src_dim[i];
  }
  return dst_dim;
}

static const std::string GetTimeStamp() {
  uint64_t usec = lite::Timer::GetCurrentUS();
  return std::to_string(usec);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
