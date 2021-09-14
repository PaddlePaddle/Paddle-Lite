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

#include <algorithm>
#include <string>
#include <vector>
#include "lite/operators/op_params.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void axpy_kernel_fp32(const float* scale,
                      const float* din,
                      const float* bias,
                      float* dout,
                      int num,
                      int channel,
                      int size,
                      int in_channel);

void axpy_kernel_int8(const int8_t* scale,
                      const int8_t* din,
                      const int8_t* bias,
                      int8_t* dout,
                      int num,
                      int channel,
                      int size,
                      int in_channel);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
