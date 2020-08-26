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

#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void pixel_shuffle_scale2_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout);
void pixel_shuffle_scale3_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout);
void pixel_shuffle_scale4_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout);
void pixel_shuffle_native_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout,
                               const int upscale_factor);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
