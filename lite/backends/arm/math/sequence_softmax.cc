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

#include "lite/backends/arm/math/sequence_softmax.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

bool sequence_softmax(const float* input,
                      const std::vector<uint64_t>& seq_offset,
                      float* out,
                      Context<TARGET(kARM)>* ctx) {
  int seq_num = seq_offset.size() - 1;
  for (int i = 0; i < seq_num; i++) {
    float seq_max = input[seq_offset[i]];
    float exp_sum = 0.f;
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      seq_max = std::max(seq_max, input[j]);
    }
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      exp_sum += expf(input[j] - seq_max);
    }
    for (int j = seq_offset[i]; j < seq_offset[i + 1]; j++) {
      out[j] = expf(input[j] - seq_max) / exp_sum;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
