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

#include "lite/backends/host/math/temporal_shift.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <>
void temporalshiftNCHW_func<float>(const float* input,
                                   float* output,
                                   const int ntchw,
                                   const int tchw,
                                   const int chw,
                                   const int hw,
                                   const int t,
                                   const int c1,
                                   const int c2) {
  int src_it = 0;
  for (int i = 0; i < ntchw; i++) {
    int it = (i % tchw) / chw;
    int ic = (i % chw) / hw;
    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }
    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * chw];
    }
  }
}

template <>
void temporalshiftNHWC_func<float>(const float* input,
                                   float* output,
                                   const int nthwc,
                                   const int thwc,
                                   const int hwc,
                                   const int t,
                                   const int c,
                                   const int c1,
                                   const int c2) {
  int src_it = 0;
  for (int i = 0; i < nthwc; i++) {
    int it = (i % thwc) / hwc;
    int ic = i % c;

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[i] = 0;
    } else {
      output[i] = input[i + (src_it - it) * hwc];
    }
  }
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
