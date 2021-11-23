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
#include <stdint.h>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

DDimLite stride_flip(const DDimLite& ddim) {
  std::vector<int64_t> tmp(ddim.size(), 0);
  DDimLite strides(tmp);
  strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return strides;
}

template <typename T>
class FlipCompute : public KernelLite<TARGET(kHost), PRECISION(kAny)> {
 public:
  using param_t = operators::FcParam;

  void Run() {
    auto& param = this->Param<operators::FlipParam>();
    auto x = param.X;
    auto out = param.Out;
    auto flip_dims = param.axis;

    auto x_dims = x->dims();
    const int total_dims = x_dims.size();
    std::vector<bool> dim_bitset(64);
    for (size_t i = 0; i < flip_dims.size(); ++i) {
      int dim = flip_dims[i];
      if (flip_dims[i] < 0) {
        dim += total_dims;
      }
      dim_bitset[dim] = true;
    }
    auto x_strides = stride_flip(x_dims);
    auto numel = x->numel();
    const T* x_data = x->template data<T>();
    T* out_data = out->template mutable_data<T>();
    LITE_PARALLEL_BEGIN(i, tid, numel) {
      int64_t cur_indices = i;
      int64_t rem = 0;
      int64_t dst_offset = 0;

      for (int d = 0; d < total_dims; ++d) {
        int64_t temp = cur_indices;
        cur_indices = cur_indices / x_strides[d];
        rem = temp - cur_indices * x_strides[d];
        dst_offset += dim_bitset[d]
                          ? (x_dims[d] - 1 - cur_indices) * x_strides[d]
                          : cur_indices * x_strides[d];
        cur_indices = rem;
      }
      out_data[i] = x_data[dst_offset];
    }
    LITE_PARALLEL_END()
  }

  ~FlipCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
