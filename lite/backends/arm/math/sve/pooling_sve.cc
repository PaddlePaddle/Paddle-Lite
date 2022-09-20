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

#include "lite/backends/arm/math/sve/pooling_sve.h"
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
void pooling_global_avg_sve(const float* din,
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
  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
    LITE_PARALLEL_BEGIN(c, tid, chout) {
      const float* data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      float* data_out_channel = data_out_batch + c;
      auto cntw = svcntw();
      auto cntb = svcntb();
      asm volatile(
          "mov x0, xzr\n"
          "eor z1.s, z1.s, z1.s\n"
          "whilelt p0.s, x0, %x[size_channel_in]\n"
          "1:\n"
          "add x0, x0, %x[cntw]\n"
          "ld1w {z0.s}, p0/Z, [%x[data_in_channel]]\n"
          "add %x[data_in_channel], %x[data_in_channel], %x[cntb]\n"
          "fadd z1.s, p0/M, z1.s, z0.s\n"
          "whilelt p0.s, x0, %x[size_channel_in]\n"
          "b.any 1b\n"
          "ptrue p0.s\n"
          "faddv s0, p0, z1.s\n"
          "str s0, [%x[data_out_channel]]\n"
          : [data_in_channel] "+r"(data_in_channel)
          : [size_channel_in] "r"(size_channel_in),
            [cntw] "r"(cntw),
            [cntb] "r"(cntb),
            [data_out_channel] "r"(data_out_channel)
          : "cc", "memory", "z0", "z1", "p0", "x0", "s0");
      data_out_channel[0] = data_out_channel[0] / size_channel_in;
    }
    LITE_PARALLEL_END();
  }
}

void pooling_global_avg_fp16_sve(const float16_t* din,
                                 float16_t* dout,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win) {
  int size_channel_in = win * hin;
  auto data_out = static_cast<float16_t*>(dout);
  auto data_in = static_cast<const float16_t*>(din);
  for (int n = 0; n < num; ++n) {
    float16_t* data_out_batch = data_out + n * chout;
    const float16_t* data_in_batch = data_in + n * chin * size_channel_in;
    LITE_PARALLEL_BEGIN(c, tid, chout) {
      const float16_t* data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      float16_t* data_out_channel = data_out_batch + c;
      auto cnth = svcnth();
      auto cntb = svcntb();
      asm volatile(
          "mov x0, xzr\n"
          "eor z1.h, z1.h, z1.h\n"
          "whilelt p0.h, x0, %x[size_channel_in]\n"
          "1:\n"
          "add x0, x0, %x[cnth]\n"
          "ld1h {z0.h}, p0/Z, [%x[data_in_channel]]\n"
          "add %x[data_in_channel], %x[data_in_channel], %x[cntb]\n"
          "fadd z1.h, p0/M, z1.h, z0.h\n"
          "whilelt p0.h, x0, %x[size_channel_in]\n"
          "b.any 1b\n"
          "ptrue p0.h\n"
          "faddv h0, p0, z1.h\n"
          "str h0, [%[data_out_channel]]\n"
          : [data_in_channel] "+r"(data_in_channel)
          : [size_channel_in] "r"(size_channel_in),
            [cnth] "r"(cnth),
            [cntb] "r"(cntb),
            [data_out_channel] "r"(data_out_channel)
          : "cc", "memory", "z0", "z1", "p0", "x0");
      data_out_channel[0] = data_out_channel[0] / size_channel_in;
    }
    LITE_PARALLEL_END();
  }
}

#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
