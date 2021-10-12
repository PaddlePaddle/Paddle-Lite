// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {
void fp32_to_int8(const float* din,
                  int8_t* dout,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size);

void int8_to_fp32(const int8_t* in,
                  float* out,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size);

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
