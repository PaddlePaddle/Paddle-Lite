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

#include <cmath>
#include "lite/core/context.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
void norm(const float* input,
          const int pre_n,
          const int n,
          const int post_n,
          const float epsilon,
          float* out,
          Context<TARGET(kARM)>* ctx);

void matrix_norm_row(const float* x_data,
                     const float* scale_data,
                     const float* bias_data,
                     float* out_data,
                     float* mean_out,
                     float* var_out,
                     float epsilon,
                     int batch_size,
                     int feature_size);
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
