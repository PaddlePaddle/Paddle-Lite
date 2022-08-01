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

#include "lite/backends/arm/math/norm.h"
#include <arm_neon.h>
#include <cmath>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void matrix_norm_row(const float* x_data,
                     const float* scale_data,
                     const float* bias_data,
                     float* out_data,
                     float* mean_out,
                     float* var_out,
                     float epsilon,
                     int batch_size,
                     int feature_size) {
  int cnt = feature_size >> 4;
  int remain = feature_size & 0xf;

  LITE_PARALLEL_BEGIN(bi, tid, batch_size) {
    int offset = bi * feature_size;
    const float* x_ptr = x_data + offset;
    float mean = 0.f;
    float variance = 0.f;

    // get mean and variance
    float32x4_t mean_v = vdupq_n_f32(0);
    float32x4_t var_v = vdupq_n_f32(0);
    for (int oi = 0; oi < cnt; ++oi) {
      float32x4_t odim1 = vld1q_f32(x_ptr);
      float32x4_t odim2 = vld1q_f32(x_ptr + 4);
      float32x4_t odim3 = vld1q_f32(x_ptr + 8);
      float32x4_t odim4 = vld1q_f32(x_ptr + 12);

      mean_v = vaddq_f32(mean_v, odim1);
      mean_v = vaddq_f32(mean_v, odim2);
      mean_v = vaddq_f32(mean_v, odim3);
      mean_v = vaddq_f32(mean_v, odim4);

      var_v = vmlaq_f32(var_v, odim1, odim1);
      var_v = vmlaq_f32(var_v, odim2, odim2);
      var_v = vmlaq_f32(var_v, odim3, odim3);
      var_v = vmlaq_f32(var_v, odim4, odim4);

      x_ptr += 16;
    }
    mean = vgetq_lane_f32(mean_v, 0) + vgetq_lane_f32(mean_v, 1) +
           vgetq_lane_f32(mean_v, 2) + vgetq_lane_f32(mean_v, 3);
    variance = vgetq_lane_f32(var_v, 0) + vgetq_lane_f32(var_v, 1) +
               vgetq_lane_f32(var_v, 2) + vgetq_lane_f32(var_v, 3);
    for (int i = 0; i < remain; ++i) {
      mean += *x_ptr;
      variance += (*x_ptr) * (*x_ptr);
      ++x_ptr;
    }
    mean /= feature_size;
    variance = variance / feature_size - mean * mean;
    mean_out[bi] = mean;
    var_out[bi] = variance;

    variance = sqrtf(variance + epsilon);
    float rvar = 1 / variance;
    // compute norm_out
    float* out_ptr = out_data + offset;
    x_ptr = x_data + offset;

    auto* scale_ptr = scale_data;
    auto* bias_ptr = bias_data;

    float32x4_t vneg = vdupq_n_f32(-1);

    float32x4_t scale1 = vdupq_n_f32(1);
    float32x4_t scale2 = vdupq_n_f32(1);
    float32x4_t scale3 = vdupq_n_f32(1);
    float32x4_t scale4 = vdupq_n_f32(1);

    float32x4_t bias1 = vdupq_n_f32(0);
    float32x4_t bias2 = vdupq_n_f32(0);
    float32x4_t bias3 = vdupq_n_f32(0);
    float32x4_t bias4 = vdupq_n_f32(0);

    for (int oi = 0; oi < cnt; ++oi) {
      float32x4_t odim1 = vld1q_f32(x_ptr);
      float32x4_t odim2 = vld1q_f32(x_ptr + 4);
      float32x4_t odim3 = vld1q_f32(x_ptr + 8);
      float32x4_t odim4 = vld1q_f32(x_ptr + 12);

      odim1 = vmlaq_n_f32(odim1, vneg, mean);
      odim2 = vmlaq_n_f32(odim2, vneg, mean);
      odim3 = vmlaq_n_f32(odim3, vneg, mean);
      odim4 = vmlaq_n_f32(odim4, vneg, mean);

      if (scale_data) {
        scale1 = vld1q_f32(scale_ptr);
        scale2 = vld1q_f32(scale_ptr + 4);
        scale3 = vld1q_f32(scale_ptr + 8);
        scale4 = vld1q_f32(scale_ptr + 12);
        scale_ptr += 16;
      }
      if (bias_data) {
        bias1 = vld1q_f32(bias_ptr);
        bias2 = vld1q_f32(bias_ptr + 4);
        bias3 = vld1q_f32(bias_ptr + 8);
        bias4 = vld1q_f32(bias_ptr + 12);
        bias_ptr += 16;
      }

      float32x4_t os1 = vmulq_n_f32(scale1, rvar);
      float32x4_t os2 = vmulq_n_f32(scale2, rvar);
      float32x4_t os3 = vmulq_n_f32(scale3, rvar);
      float32x4_t os4 = vmulq_n_f32(scale4, rvar);

      odim1 = vmlaq_f32(bias1, odim1, os1);
      odim2 = vmlaq_f32(bias2, odim2, os2);
      odim3 = vmlaq_f32(bias3, odim3, os3);
      odim4 = vmlaq_f32(bias4, odim4, os4);

      vst1q_f32(out_ptr, odim1);
      vst1q_f32(out_ptr + 4, odim2);
      vst1q_f32(out_ptr + 8, odim3);
      vst1q_f32(out_ptr + 12, odim4);

      x_ptr += 16;
      out_ptr += 16;
    }
    for (int i = 0; i < remain; ++i) {
      auto out_value = (*x_ptr - mean) / variance;
      if (scale_data) {
        out_value = out_value * (*scale_ptr);
        ++scale_ptr;
      }
      if (bias_data) {
        out_value = out_value + *bias_ptr;
        ++bias_ptr;
      }
      *out_ptr = out_value;

      ++out_ptr;
      ++x_ptr;
    }
  }  // for bi
  LITE_PARALLEL_END()
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
