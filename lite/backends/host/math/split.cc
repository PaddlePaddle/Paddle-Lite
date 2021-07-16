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

#include "lite/backends/host/math/split.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
void split(const T* din,
           const std::vector<lite::Tensor*>& dout,
           const int axis,
           const std::vector<int>& in_strides) {
  int input_offset = 0;
  for (auto out : dout) {
    auto out_dim = out->dims();
    std::vector<int> out_strides(out_dim.size());
    out_strides[out_dim.size() - 1] = out_dim[out_dim.size() - 1];
    for (int i = static_cast<int>(out_dim.size()) - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

    T* out_data = out->mutable_data<T>();
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    const T* din_ptr = din + input_offset;

    for (int i = 0; i < before; ++i) {
      std::memcpy(out_data, din_ptr, sizeof(T) * out_after);
      din_ptr += in_after;
      out_data += out_after;
    }
    input_offset += out_strides[axis];
  }
}

template void split(const float* din,
                    const std::vector<lite::Tensor*>& dout,
                    const int axis,
                    const std::vector<int>& in_strides);
template void split(const int* din,
                    const std::vector<lite::Tensor*>& dout,
                    const int axis,
                    const std::vector<int>& in_strides);
template void split(const int64_t* din,
                    const std::vector<lite::Tensor*>& dout,
                    const int axis,
                    const std::vector<int>& in_strides);

#ifdef ENABLE_ARM_FP16
template void split(const lite_api::float16_t* din,
                    const std::vector<lite::Tensor*>& dout,
                    const int axis,
                    const std::vector<int>& in_strides);
#endif

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
