// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <smmintrin.h>
#include <xmmintrin.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

inline void transpose4_ps(__m128& row0,
                          __m128& row1,
                          __m128& row2,
                          __m128& row3) {
  __m128 tmp3, tmp2, tmp1, tmp0;
  tmp0 = _mm_unpacklo_ps((row0), (row1));
  tmp2 = _mm_unpacklo_ps((row2), (row3));
  tmp1 = _mm_unpackhi_ps((row0), (row1));
  tmp3 = _mm_unpackhi_ps((row2), (row3));
  row0 = _mm_movelh_ps(tmp0, tmp2);
  row1 = _mm_movehl_ps(tmp2, tmp0);
  row2 = _mm_movelh_ps(tmp1, tmp3);
  row3 = _mm_movehl_ps(tmp3, tmp1);
}

void packC4_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel);

void unpackC4_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel);
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
