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

#include "lite/backends/arm/math/clip.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void clip_kernel_fp32(
    const float* input, int64_t num, float min, float max, float* output) {
  float tmp;
  for (int64_t i = 0; i < num; i++) {
    tmp = *input;
    tmp = tmp > min ? tmp : min;
    *output = tmp < max ? tmp : max;
    input++;
    output++;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
