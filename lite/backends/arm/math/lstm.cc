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

#include "lite/backends/arm/math/lstm.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void add_bias_rowwise(Tensor* input,
                      const Tensor* bias,
                      int start_w,
                      int end_w) {
  auto in_dim = input->dims();
  int width = input->numel() / in_dim[0];
  int w_adds = width < end_w ? width : end_w;
  float* i_data = input->mutable_data<float>();
  const float* b_data = bias->data<float>();
  for (int i = 0; i < in_dim[0]; ++i) {
    for (int w = start_w; w < w_adds; ++w) {
      i_data[w] += b_data[w];
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
