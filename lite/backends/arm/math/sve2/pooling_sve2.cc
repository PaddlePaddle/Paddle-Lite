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

#include "lite/backends/arm/math/sve2/pooling_sve2.h"
#include <algorithm>
#include <limits>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
#include <arm_sve.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
void pooling_global_avg_sve2(const float* din,
                             float* dout,
                             int num,
                             int chout,
                             int hout,
                             int wout,
                             int chin,
                             int hin,
                             int win) {
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);
  std::vector<float> vec_tmp;
  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
    LITE_PARALLEL_BEGIN(c, tid, chout) {
      const float* data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      float* data_out_channel = data_out_batch + c;
      vec_tmp.clear();
      for (int i = 0; i < size_channel_in; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, size_channel_in);
        svfloat32_t vec_x = svld1(pg, &data_in_channel[i]);
        float psum = svaddv(pg, vec_x);
        vec_tmp.emplace_back(psum);
      }
      float sum = 0.f;
      int size = vec_tmp.size();
      float* tmp_data = vec_tmp.data();
      for (int i = 0; i < size; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t vec_x = svld1(pg, &tmp_data[i]);
        float psum = svaddv(pg, vec_x);
        sum += psum;
      }
      data_out_channel[0] = sum / size_channel_in;
    }
    LITE_PARALLEL_END();
  }
}
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
