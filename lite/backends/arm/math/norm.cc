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

#include "lite/backends/arm/math/norm.h"
#include <arm_neon.h>
#include <cmath>
#include "lite/utils/cp_logging.h"

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
          Context<TARGET(kARM)>* ctx) {
  for (int i = 0; i < pre_n; i++) {
    for (int k = 0; k < post_n; k++) {
      float sum = epsilon;
      const float* in_tmp = input + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        sum += in_tmp[j * post_n] * in_tmp[j * post_n];
      }
      sum = std::sqrt(sum);
      float* out_tmp = out + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        out_tmp[j * post_n] = in_tmp[j * post_n] / sum;
      }
    }
  }
  LOG(INFO) << "norm math finished";
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
