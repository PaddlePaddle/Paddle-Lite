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

#include "lite/backends/arm/math/sin.h"
#include <algorithm>
#include <cmath>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sin_func(const lite::Tensor* input, lite::Tensor* output) {
  auto input_ddim = input->dims();
  int num = 1;
  for (int i = 0; i < input_ddim.size(); i++) {
    num *= input_ddim[i];
  }

  const float* inp_ptr = input->data<float>();
  float* out_ptr = output->mutable_data<float>();
  for (int n = 0; n < num; n++) {
    out_ptr[n] = sin(inp_ptr[n]);
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
