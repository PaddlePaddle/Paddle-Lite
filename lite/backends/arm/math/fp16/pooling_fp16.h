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
namespace fp16 {
typedef __fp16 float16_t;
#define POOLING_PARAM                                                  \
  const float16_t *din, float16_t *dout, int num, int chout, int hout, \
      int wout, int chin, int hin, int win

void pooling_basic_fp16(POOLING_PARAM,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        bool global_pooling,
                        bool exclusive,
                        bool adaptive,
                        bool ceil_mode,
                        bool use_quantizer,
                        const std::string& pooling_type);

void pooling_global_max_fp16(POOLING_PARAM);

void pooling_global_avg_fp16(POOLING_PARAM);

void pooling3x3s2p0_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right);

void pooling3x3s2p0_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right);

void pooling3x3s2p1_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right);

void pooling3x3s2p1_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right);

void pooling3x3s1p0_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right);

void pooling3x3s1p1_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right);

void pooling3x3s1p0_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right);

void pooling3x3s1p1_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
