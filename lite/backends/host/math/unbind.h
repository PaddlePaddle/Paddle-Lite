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

#include <vector>
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
void unbind(const lite::Tensor* in,
            const std::vector<lite::Tensor*>& outs,
            const int axis) {
  auto in_dim = in->dims();
  const T* din = in->template data<T>();
  int out_stride = 1;
  for (int i = in_dim.size() - 1; i > axis; i--) {
    out_stride *= in_dim[i];
  }
  int in_stride = axis == 0 ? out_stride : out_stride * in_dim[axis];
  int loop_cnt = in_dim[0];
  for (int i = 1; i < axis; i++) {
    loop_cnt *= in_dim[i];
  }
  for (auto out : outs) {
    auto temp_din = din;
    T* out_data = out->template mutable_data<T>();
    for (int loop = 0; loop < loop_cnt; loop++) {
      std::memcpy(out_data, temp_din, sizeof(T) * out_stride);
      out_data += out_stride;
      temp_din += in_stride;
    }
    din += out_stride;
  }
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
