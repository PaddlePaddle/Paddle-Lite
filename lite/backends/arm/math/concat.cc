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

#include "lite/backends/arm/math/concat.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void concat_func(const std::vector<lite::Tensor *> &input,
                 const int axis,
                 lite::Tensor *output) {
  int64_t concat_input_size = 1;
  int64_t num_cancats = 1;
  auto dim_0 = input[0]->dims();
  size_t num = input.size();
  for (int i = axis + 1; i < dim_0.size(); i++) {
    concat_input_size *= dim_0[i];
  }
  for (int i = 0; i < axis; i++) {
    num_cancats *= dim_0[i];
  }
  float *dst_ptr = output->mutable_data<float>();
  const int out_concat_axis = output->dims()[axis];
  int64_t offset_concat_axis = 0;
  int64_t out_sum = out_concat_axis * concat_input_size;
  for (int n = 0; n < num; n++) {  // input.size()
    auto dims = input[n]->dims();
    const float *src_ptr = input[n]->data<float>();
    int64_t in_concat_axis = dims[axis];
    float *dout_ptr = dst_ptr + offset_concat_axis * concat_input_size;
    int64_t in_sum = in_concat_axis * concat_input_size;
    for (int i = 0; i < num_cancats; i++) {
      std::memcpy(dout_ptr, src_ptr, sizeof(float) * in_sum);
      dout_ptr += out_sum;
      src_ptr += in_sum;
    }
    offset_concat_axis += in_concat_axis;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
