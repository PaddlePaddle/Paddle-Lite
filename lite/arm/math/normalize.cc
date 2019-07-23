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

#include "lite/arm/math/normalize.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
void compute_mean(const float* input,
                  lite::Tensor* mean,
                  int num,
                  int channel,
                  int height,
                  int width) {
  int spatial_size = height * width;
  float* out = mean->mutable_data<float>();
  int cnt = spatial_size / 8;
  for (int n = 0; n < num; ++n) {
    const float* in_batch = input + n * channel * spatial_size;
    float* out_batch = out + n * channel;
#pragma omp parallel for
    for (int c = 0; c < channel; ++c) {
      const float* in_channel = in_batch + c * spatial_size;
      int i = 0;
      float32x4_t vsum = vdupq_n_f32(0.0f);
      float32x4_t vc = vdupq_n_f32(0.f);

#ifdef __aarch64__
      int loop = cnt;
      if (loop > 0) {
        asm volatile(
            "1:                                         \n"
            "ld1   {v0.4s}, [%[in_channel]], #16        \n"
            "ld1   {v1.4s}, [%[in_channel]], #16        \n"
            "fsub   v6.4s, v0.4s, %[c].4s               \n"  // y
            "fadd   v7.4s, %[vsum].4s, v6.4s            \n"  // t
            "fsub   %[c].4s, v7.4s, %[vsum].4s          \n"
            "fsub   %[c].4s, %[c].4s, v6.4s             \n"
            "mov    %[vsum].16b, v7.16b                 \n"

            "fsub   v4.4s, v1.4s, %[c].4s               \n"  // y
            "fadd   v5.4s, %[vsum].4s, v4.4s            \n"  // t
            "fsub   %[c].4s, v5.4s, %[vsum].4s          \n"
            "fsub   %[c].4s, %[c].4s, v4.4s             \n"
            "mov    %[vsum].16b, v5.16b                 \n"
            "subs       %w[loop], %w[loop], #1          \n"
            "bne        1b                              \n"
            : [in_channel] "+r"(in_channel),
              [loop] "+r"(loop),
              [vsum] "+w"(vsum),
              [c] "+w"(vc)
            : "r"(in_channel), "r"(num), "w"(vsum)
            : "cc", "memory", "v0", "v1", "v4", "v5", "v6", "v7");
      }
#else
      int loop = cnt;
      if (loop > 0) {
        asm volatile(
            "1:                                     \n"
            "vld1.f32   {d0-d1}, [%[in_channel]]!   \n"
            "vld1.f32   {d2,d3}, [%[in_channel]]!   \n"
            "vsub.f32   q6, q0, %q[c]               \n"  // y
            "vadd.f32   q7, %q[vsum], q6            \n"  // t
            "vsub.f32   %q[c], q7, %q[vsum]         \n"
            "vsub.f32   %q[c], %q[c], q6            \n"
            "vmov.32    %q[vsum], q7                \n"
            "vsub.f32   q4, q1, %q[c]               \n"
            "vadd.f32   q5, %q[vsum], q4            \n"
            "vsub.f32   %q[c], q5, %q[vsum]         \n"
            "vsub.f32   %q[c], %q[c], q4            \n"
            "vmov.32    %q[vsum], q5                \n"
            "subs       %[loop], #1                 \n"
            "bne        1b                          \n"
            : [in_channel] "+r"(in_channel),
              [loop] "+r"(loop),
              [vsum] "+w"(vsum),
              [c] "+w"(vc)
            : "r"(in_channel), "r"(num), "w"(vsum)
            : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q7");
      }
#endif  // __aarch64__
      float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
      float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
      for (i = cnt * 8; i < spatial_size; i++) {
        sum += in_channel[0];
        in_channel++;
      }
      out_batch[c] = sum / (spatial_size * num);
    }
  }

  // add mean in num
  for (int c = 0; c < channel; ++c) {
    for (int n = 1; n < num; ++n) {
      out[c] += out[n * channel + c];
    }
  }
}

void compute_variance(const float* input,
                      lite::Tensor* mean,
                      lite::Tensor* variance,
                      int num,
                      int channel,
                      int height,
                      int width) {
  int spatial_size = height * width;
  float* out = variance->mutable_data<float>();
  const float* mean_data = mean->data<float>();

  int cnt = spatial_size / 8;
  for (int n = 0; n < num; ++n) {
    const float* in_batch = input + n * channel * spatial_size;
    float* out_batch = out + n * channel;

#pragma omp parallel for
    for (int c = 0; c < channel; ++c) {
      const float* in_channel = in_batch + c * spatial_size;
      int i = 0;
      float mean_val = mean_data[c];
      float32x4_t vsum = vdupq_n_f32(0.0f);
      float32x4_t vc = vdupq_n_f32(0.f);
#ifdef __aarch64__
      int loop = cnt;
      if (loop > 0) {
        asm volatile(
            "1:                                        \n"
            "ld1   {v0.4s}, [%[in_channel]], #16       \n"
            "ld1   {v3.4s}, [%[in_channel]], #16       \n"
            "dup   v10.4s, %w[mean]                    \n"
            "fsub   v1.4s, v0.4s, v10.4s               \n"
            "fmul   v2.4s, v1.4s, v1.4s                \n"
            "fsub   v4.4s, v3.4s, v10.4s               \n"
            "fmul   v5.4s, v4.4s, v4.4s                \n"

            "faddp  v6.4s, v2.4s, v5.4s                \n"
            "fsub   v7.4s, v6.4s, %[c].4s              \n"  // y
            "fadd   v8.4s, %[vsum].4s, v7.4s           \n"  // t
            "fsub   %[c].4s, v8.4s, %[vsum].4s         \n"
            "fsub   %[c].4s, %[c].4s, v7.4s            \n"
            "mov    %[vsum].16b, v8.16b                \n"
            "subs       %w[loop], %w[loop], #1         \n"
            "bne        1b                             \n"
            : [in_channel] "+r"(in_channel),
              [loop] "+r"(loop),
              [vsum] "+w"(vsum),
              [mean] "+r"(mean_val),
              [c] "+w"(vc)
            : "r"(in_channel), "r"(loop), "w"(vsum)
            : "cc",
              "memory",
              "v0",
              "v1",
              "v2",
              "v3",
              "v4",
              "v5",
              "v6",
              "v7",
              "v8",
              "v10");
      }
#else
      int loop = cnt;
      if (loop > 0) {
        asm volatile(
            "1:                                     \n"
            "vld1.f32   {d0-d1}, [%[in_channel]]!   \n"
            "vdup.f32   q10, %[mean]                \n"
            "vsub.f32   q1, q0, q10                 \n"
            "vmul.f32   q2, q1, q1                  \n"

            "vld1.f32   {d6-d7}, [%[in_channel]]!   \n"
            "vsub.f32   q4, q3, q10                 \n"
            "vmul.f32   q5, q4, q4                  \n"

            "vpadd.f32  d12, d4, d5                 \n"
            "vpadd.f32  d13, d10, d11               \n"
            "vsub.f32   q7, q6, %q[c]               \n"  // y
            "vadd.f32   q8, %q[vsum], q7            \n"  // t
            "vsub.f32   %q[c], q8, %q[vsum]         \n"
            "vsub.f32   %q[c], %q[c], q7            \n"
            "vmov.32    %q[vsum], q8                \n"
            "subs       %[loop], #1                 \n"
            "bne        1b                          \n"
            : [in_channel] "+r"(in_channel),
              [loop] "+r"(loop),
              [vsum] "+w"(vsum),
              [mean] "+r"(mean_val),
              [c] "+w"(vc)
            : "r"(in_channel), "r"(loop), "w"(vsum)
            : "cc",
              "memory",
              "q0",
              "q1",
              "q2",
              "q3",
              "q4",
              "q5",
              "q6",
              "q7",
              "q8",
              "q10");
      }
#endif  // __aarch64__
      float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
      float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
      float sum_tmp = 0.f;
      for (i = cnt * 8; i < spatial_size; i++) {
        float in_data = in_channel[0];
        in_data = powf(in_data - mean_val, 2);
        sum += in_data;
        in_channel++;
      }
      sum += sum_tmp;
      out_batch[c] = sum / (spatial_size * num);
    }
  }
  // add variance in num
  for (int c = 0; c < channel; ++c) {
    for (int n = 1; n < num; ++n) {
      out[c] += out[n * channel + c];
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
