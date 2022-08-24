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
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
typedef __fp16 float16_t;

void pooling_global_avg_sve(const float* din,
                            float* dout,
                            int num,
                            int chout,
                            int hout,
                            int wout,
                            int chin,
                            int hin,
                            int win);

void pooling_global_avg_fp16_sve(const float16_t* din,
                                 float16_t* dout,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win);

#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
