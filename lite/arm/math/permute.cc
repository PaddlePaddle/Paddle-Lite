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

#include "lite/arm/math/permute.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void permute_basic<float>(const int count,
                          const float* din,
                          const int* permute_order,
                          const int* old_steps,
                          const int* new_steps,
                          const int num_axes,
                          float* dout) {
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    dout[i] = din[old_idx];
  }
}

template <>
void transpose_mat<float>(const float* din,
                          float* dout,
                          const int num,
                          const int width,
                          const int height) {
  int nw = width >> 2;
  int nh = height >> 2;
  int size_in = width * height;
  for (int i = 0; i < num; ++i) {
    float* ptr_out = dout + i * size_in;
    const float* ptr_in = din + i * size_in;
#pragma omp parallel for
    for (int h = 0; h < nh; h++) {
      const float* ptr_din_row = ptr_in + h * 4 * width;
      for (int w = 0; w < nw; w++) {
        float* data_out_ptr = ptr_out + w * 4 * height + h * 4;
        const float* din0 = ptr_din_row;
        const float* din1 = din0 + width;
        const float* din2 = din1 + width;
        const float* din3 = din2 + width;

        float* dout0 = data_out_ptr;
        float* dout1 = dout0 + height;
        float* dout2 = dout1 + height;
        float* dout3 = dout2 + height;
#ifdef __aarch64__
        float32x4_t vr0 = vld1q_f32(din0);
        float32x4_t vr1 = vld1q_f32(din1);
        float32x4_t vr2 = vld1q_f32(din2);
        float32x4_t vr3 = vld1q_f32(din3);
        float32x4_t re0 = vtrn1q_f32(vr0, vr1);
        float32x4_t re1 = vtrn2q_f32(vr0, vr1);
        float32x4_t re2 = vtrn1q_f32(vr2, vr3);
        float32x4_t re3 = vtrn2q_f32(vr2, vr3);
        vst1_f32(dout0, vget_low_f32(re0));
        dout0 += 2;
        vst1_f32(dout0, vget_low_f32(re2));
        vst1_f32(dout1, vget_low_f32(re1));
        dout1 += 2;
        vst1_f32(dout1, vget_low_f32(re3));
        vst1_f32(dout2, vget_high_f32(re0));
        dout2 += 2;
        vst1_f32(dout2, vget_high_f32(re2));
        vst1_f32(dout3, vget_high_f32(re1));
        dout3 += 2;
        vst1_f32(dout3, vget_high_f32(re3));
#else
        asm("vld1.32 {d0, d1}, [%[in0]]    \n"
            "vld1.32 {d2, d3}, [%[in1]]    \n"
            "vld1.32 {d4, d5}, [%[in2]]    \n"
            "vld1.32 {d6, d7}, [%[in3]]    \n"
            "vtrn.32 q0, q1                \n"
            "vtrn.32 q2, q3                \n"
            "vswp d1, d4                   \n"
            "vswp d3, d6                   \n"
            "vst1.32 {d0, d1}, [%[out0]]   \n"
            "vst1.32 {d2, d3}, [%[out1]]   \n"
            "vst1.32 {d4, d5}, [%[out2]]   \n"
            "vst1.32 {d6, d7}, [%[out3]]   \n"
            :
            : [out0] "r"(dout0),
              [out1] "r"(dout1),
              [out2] "r"(dout2),
              [out3] "r"(dout3),
              [in0] "r"(din0),
              [in1] "r"(din1),
              [in2] "r"(din2),
              [in3] "r"(din3)
            : "q0", "q1", "q2", "q3");
#endif
        ptr_din_row += 4;
      }
    }
    // remian
    for (int h = 0; h < height; h++) {
      for (int w = nw * 4; w < width; w++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
    for (int w = 0; w < width; w++) {
      for (int h = nh * 4; h < height; h++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
