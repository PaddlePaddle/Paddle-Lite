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

#include "lite/backends/arm/math/pixel_shuffle.h"
#include <arm_neon.h>

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
                               const int wout) {
  const int upscale_factor = 2;
  const int feat_size_in = win * hin;
  const int feat_size_out = wout * hout;

  const int cnt = win >> 2;
  const int remain = win - (cnt << 2);

#pragma omp parallel for
  // batch * out_channel loop
  for (int nc = 0; nc < num * chout; nc++) {
    const float* inptr = input + nc * feat_size_out;
    float* outptr = output + nc * feat_size_out;

    // out_height loop
    for (int h = 0; h < hin; h++) {
      for (int sh = 0; sh < upscale_factor; sh++) {
        const float* inptr_loc0 =
            inptr + h * win + sh * feat_size_in * upscale_factor;
        const float* inptr_loc1 = inptr_loc0 + feat_size_in;

        // out_width loop
        for (int i = 0; i < cnt; i++) {
          float32x4_t vin0 = vld1q_f32(inptr_loc0);
          float32x4_t vin1 = vld1q_f32(inptr_loc1);

          float32x4x2_t vin = {vin0, vin1};

          vst2q_f32(outptr, vin);
          outptr += 8;

          inptr_loc0 += 4;
          inptr_loc1 += 4;
        }

        for (int j = 0; j < remain; j++) {
          outptr[0] = inptr_loc0[0];
          outptr[1] = inptr_loc1[0];
          inptr_loc0++;
          inptr_loc1++;
          outptr += upscale_factor;
        }
      }
    }
  }
}

void pixel_shuffle_scale3_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout) {
  const int upscale_factor = 3;
  const int feat_size_in = win * hin;
  const int feat_size_out = wout * hout;

  const int cnt = win >> 2;
  const int remain = win - (cnt << 2);

#pragma omp parallel for
  // batch * out_channel loop
  for (int nc = 0; nc < num * chout; nc++) {
    const float* inptr = input + nc * feat_size_out;
    float* outptr = output + nc * feat_size_out;

    // out_height loop
    for (int h = 0; h < hin; h++) {
      for (int sh = 0; sh < upscale_factor; sh++) {
        const float* inptr_loc0 =
            inptr + h * win + sh * feat_size_in * upscale_factor;
        const float* inptr_loc1 = inptr_loc0 + feat_size_in;
        const float* inptr_loc2 = inptr_loc1 + feat_size_in;

        // out_width loop
        for (int i = 0; i < cnt; i++) {
          float32x4_t vin0 = vld1q_f32(inptr_loc0);
          float32x4_t vin1 = vld1q_f32(inptr_loc1);
          float32x4_t vin2 = vld1q_f32(inptr_loc2);

          float32x4x3_t vin = {vin0, vin1, vin2};

          vst3q_f32(outptr, vin);
          outptr += 12;

          inptr_loc0 += 4;
          inptr_loc1 += 4;
          inptr_loc2 += 4;
        }

        for (int j = 0; j < remain; j++) {
          outptr[0] = inptr_loc0[0];
          outptr[1] = inptr_loc1[0];
          outptr[2] = inptr_loc2[0];
          inptr_loc0++;
          inptr_loc1++;
          inptr_loc2++;
          outptr += upscale_factor;
        }
      }
    }
  }
}

void pixel_shuffle_scale4_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout) {
  const int upscale_factor = 4;
  const int feat_size_in = win * hin;
  const int feat_size_out = wout * hout;

  const int cnt = win >> 2;
  const int remain = win - (cnt << 2);

#pragma omp parallel for
  // batch * out_channel loop
  for (int nc = 0; nc < num * chout; nc++) {
    const float* inptr = input + nc * feat_size_out;
    float* outptr = output + nc * feat_size_out;

    // out_height loop
    for (int h = 0; h < hin; h++) {
      for (int sh = 0; sh < upscale_factor; sh++) {
        const float* inptr_loc0 =
            inptr + h * win + sh * feat_size_in * upscale_factor;
        const float* inptr_loc1 = inptr_loc0 + feat_size_in;
        const float* inptr_loc2 = inptr_loc1 + feat_size_in;
        const float* inptr_loc3 = inptr_loc2 + feat_size_in;

        // out_width loop
        for (int i = 0; i < cnt; i++) {
          float32x4_t vin0 = vld1q_f32(inptr_loc0);
          float32x4_t vin1 = vld1q_f32(inptr_loc1);
          float32x4_t vin2 = vld1q_f32(inptr_loc2);
          float32x4_t vin3 = vld1q_f32(inptr_loc3);

          float32x4x4_t vin = {vin0, vin1, vin2, vin3};

          vst4q_f32(outptr, vin);
          outptr += 16;

          inptr_loc0 += 4;
          inptr_loc1 += 4;
          inptr_loc2 += 4;
          inptr_loc3 += 4;
        }

        for (int j = 0; j < remain; j++) {
          outptr[0] = inptr_loc0[0];
          outptr[1] = inptr_loc1[0];
          outptr[2] = inptr_loc2[0];
          outptr[3] = inptr_loc3[0];
          inptr_loc0++;
          inptr_loc1++;
          inptr_loc2++;
          inptr_loc3++;
          outptr += upscale_factor;
        }
      }
    }
  }
}

void pixel_shuffle_native_fp32(const float* input,
                               float* output,
                               const int num,
                               const int hin,
                               const int win,
                               const int chout,
                               const int hout,
                               const int wout,
                               const int upscale_factor) {
#pragma omp parallel for
  for (int nc = 0; nc < num * chout; nc++) {
    const float* inptr = input + nc * hout * wout;
    float* outptr_nc = output + nc * hout * wout;

    for (int sh = 0; sh < upscale_factor; sh++) {
      for (int sw = 0; sw < upscale_factor; sw++) {
        float* outptr = outptr_nc + sh * wout + sw;
        for (int h = 0; h < hin; h++) {
          for (int w = 0; w < win; w++) {
            outptr[0] = inptr[0];
            inptr++;
            outptr += upscale_factor;
          }
          outptr += (upscale_factor - 1) * wout;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
