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
namespace fp16 {

typedef __fp16 float16_t;
#define PAD2D_PARAM                                                  \
  const float16_t *din, float16_t *dout, int n, int c, int h, int w, \
      const int pad_top, const int pad_bottom, const int pad_left,   \
      const int pad_right, const float16_t pad_value

void pad_constant_fp16(PAD2D_PARAM);
void pad_edge_fp16(PAD2D_PARAM);
void pad_reflect_fp16(PAD2D_PARAM);
void pad2d_func_fp16(const lite::Tensor* input,
                     lite::Tensor* output,
                     int _mode,
                     std::vector<int> _pad_h,
                     std::vector<int> _pad_w,
                     float16_t _pad_value);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
