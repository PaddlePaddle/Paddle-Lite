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

#include "lite/backends/arm/math/interpolate.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

// The following function bilinear_interp is partially base on
// https://github.com/Tencent/ncnn/blob/master/src/layer/arm/interp_arm.cpp
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
void bilinear_interp(const float* src,
                     int w_in,
                     int h_in,
                     float* dst,
                     int w_out,
                     int h_out,
                     float scale_x,
                     float scale_y,
                     bool align_corners,
                     bool align_mode) {
  int* buf = new int[w_out + h_out + w_out * 2 + h_out * 2];

  int* xofs = buf;
  int* yofs = buf + w_out;

  float* alpha = reinterpret_cast<float*>(buf + w_out + h_out);
  float* beta = reinterpret_cast<float*>(buf + w_out + h_out + w_out * 2);
  bool with_align = (align_mode == 0 && !align_corners);

  float fx = 0.0f;
  float fy = 0.0f;
  int sx = 0;
  int sy = 0;
  if (!with_align) {
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = dx * scale_x;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = dy * scale_y;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  } else {
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = scale_x * (dx + 0.5f) - 0.5f;
      fx = fx < 0 ? 0.f : fx;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = scale_y * (dy + 0.5f) - 0.5f;
      fy = fy < 0 ? 0.f : fy;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  }
  float* rowsbuf0 = new float[w_out];
  float* rowsbuf1 = new float[w_out];
  float* rows0 = rowsbuf0;
  float* rows1 = rowsbuf1;
  // output w , h boundary
  int w_bound = w_out;
  int h_bound = h_out;
  if (with_align) {
    w_bound = ceil((w_in - 1) / scale_x);
    h_bound = ceil((h_in - 1) / scale_y);
  } else {
    w_bound = ceil((w_in - 0.5f) / scale_x - 0.5f);
    h_bound = ceil((h_in - 0.5f) / scale_y - 0.5f);
  }
  // h_bound loop
  for (int dy = 0; dy < h_bound; dy++) {
    int sy = yofs[dy];

    const float* s0 = src + sy * w_in;
    const float* s1 = src + (sy + 1) * w_in;

    const float* alphap = alpha;
    float* rows0p = rows0;
    float* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;
      const float* s0np = s0 + sxn;
      const float* s1np = s1 + sxn;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }
    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;

      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    const float buffer1[2] = {*(src + sy * w_in + w_in - 1),
                              *(src + sy * w_in + w_in - 1)};
    const float buffer2[2] = {*(src + (sy + 1) * w_in + w_in - 1),
                              *(src + (sy + 1) * w_in + w_in - 1)};
    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float* s0p = buffer1;
      const float* s1p = buffer2;
      const float* s0np = buffer1;
      const float* s1np = buffer2;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }
    // w_bound - w_out remain loop
    for (; dx < w_out; dx++) {
      const float* s0p = buffer1;
      const float* s1p = buffer2;

      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    float b0 = beta[0];
    float b1 = beta[1];

    float* dp = dst + dy * w_out;

    int nn = w_out >> 3;
    int remain = w_out - (nn << 3);

#ifdef __aarch64__
    float32x4_t _b0 = vdupq_n_f32(b0);
    float32x4_t _b1 = vdupq_n_f32(b1);
    // calculate and store results
    for (; nn > 0; nn--) {
      float32x4_t _rows0 = vld1q_f32(rows0p);
      float32x4_t _d = vmulq_f32(_rows0, _b0);
      float32x4_t _rows1 = vld1q_f32(rows1p);
      _d = vmlaq_f32(_d, _rows1, _b1);

      float32x4_t _rows0n = vld1q_f32(rows0p + 4);
      float32x4_t _rows1n = vld1q_f32(rows1p + 4);

      float32x4_t _dn = vmulq_f32(_rows0n, _b0);
      vst1q_f32(dp, _d);
      _dn = vmlaq_f32(_dn, _rows1n, _b1);
      vst1q_f32(dp + 4, _dn);

      dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

#else
    if (nn > 0) {
      asm volatile(
          "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
          "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
          "1:                                                      \n"
          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows0p]]                     @preload rows0p\n"

          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows1p]]                     @preload rows1p\n"
          "subs %[loopc], #1                   @loop count minus #1\n"
          "bne 1b                              @jump to 1\n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [out] "+r"(dp),
            [loopc] "+r"(nn)
          : [b0] "r"(b0), [b1] "r"(b1)
          : "cc", "memory", "q0", "q1", "q2", "q3");
    }
#endif
    // calculate and store remain resluts
    for (; remain; --remain) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }
    beta += 2;
  }

  // h_bound - h_out loop
  for (int dy = h_bound; dy < h_out; dy++) {
    int sy = h_in - 1;
    const float* s0 = src + sy * w_in;
    const float* s1 = s0;
    const float* alphap = alpha;
    float* rows0p = rows0;
    float* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;
      const float* s0np = s0 + sxn;
      const float* s1np = s1 + sxn;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }
    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float* s0p = s0 + sx;
      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];

      alphap += 2;
    }

    const float buffer1[2] = {*(src + sy * w_in + w_in - 1),
                              *(src + sy * w_in + w_in - 1)};
    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float* s0p = buffer1;
      const float* s1p = buffer1;
      const float* s0np = buffer1;
      const float* s1np = buffer1;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }
    // w_bound - wout remain loop
    for (; dx < w_out; dx++) {
      const float* s0p = buffer1;
      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];
      alphap += 2;
    }

    float b0 = beta[0];
    float b1 = beta[1];

    float* dp = dst + dy * w_out;

    int nn = w_out >> 3;
    int remain = w_out - (nn << 3);

#ifdef __aarch64__
    float32x4_t _b0 = vdupq_n_f32(b0);
    float32x4_t _b1 = vdupq_n_f32(b1);
    // calculate and store results
    for (; nn > 0; nn--) {
      float32x4_t _rows0 = vld1q_f32(rows0p);
      float32x4_t _d = vmulq_f32(_rows0, _b0);
      float32x4_t _rows1 = vld1q_f32(rows1p);
      _d = vmlaq_f32(_d, _rows1, _b1);

      float32x4_t _rows0n = vld1q_f32(rows0p + 4);
      float32x4_t _rows1n = vld1q_f32(rows1p + 4);

      float32x4_t _dn = vmulq_f32(_rows0n, _b0);
      vst1q_f32(dp, _d);
      _dn = vmlaq_f32(_dn, _rows1n, _b1);
      vst1q_f32(dp + 4, _dn);

      dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

#else
    if (nn > 0) {
      asm volatile(
          "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
          "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
          "1:                                                      \n"
          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows0p]]                     @preload rows0p\n"

          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows1p]]                     @preload rows1p\n"
          "subs %[loopc], #1                   @loop count minus #1\n"
          "bne 1b                              @jump to 1\n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [out] "+r"(dp),
            [loopc] "+r"(nn)
          : [b0] "r"(b0), [b1] "r"(b1)
          : "cc", "memory", "q0", "q1", "q2", "q3");
    }
#endif
    // calculate and store remain results
    for (; remain; --remain) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }

    beta += 2;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

void nearest_interp(const float* src,
                    int w_in,
                    int h_in,
                    float* dst,
                    int w_out,
                    int h_out,
                    float scale_w_new,
                    float scale_h_new,
                    bool with_align) {
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      float* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w + 0.5);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      float* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  }
}

void interpolate(lite::Tensor* X,
                 lite::Tensor* OutSize,
                 std::vector<const lite::Tensor*> SizeTensor,
                 lite::Tensor* Scale,
                 lite::Tensor* Out,
                 int out_height,
                 int out_width,
                 float scale,
                 bool align_corners,
                 bool align_mode,
                 std::string interpolate_type) {
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];
  if (SizeTensor.size() > 0) {
    auto new_size = get_new_shape(SizeTensor);
    out_height = new_size[0];
    out_width = new_size[1];
  } else {
    auto scale_tensor = Scale;
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    }
    if (scale > 0) {
      out_height = static_cast<int>(in_h * scale);
      out_width = static_cast<int>(in_w * scale);
    }
    auto out_size = OutSize;
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_height = out_size_data[0];
      out_width = out_size_data[1];
    }
  }
  // float height_scale = scale;
  // float width_scale = scale;
  // if (out_width > 0 && out_height > 0) {
  //   height_scale = static_cast<float>(out_height / X->dims()[2]);
  //   width_scale = static_cast<float>(out_width / X->dims()[3]);
  // }
  int num_cout = X->dims()[0];
  int c_cout = X->dims()[1];
  Out->Resize({num_cout, c_cout, out_height, out_width});

  float* dout = Out->mutable_data<float>();
  const float* din = X->data<float>();
  int out_num = Out->dims()[0];
  int out_c = Out->dims()[1];
  int count = out_num * out_c;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int spatial_in = in_h * in_w;
  int spatial_out = out_h * out_w;

  float scale_x = (align_corners) ? (static_cast<float>(in_w - 1) / (out_w - 1))
                                  : (static_cast<float>(in_w) / (out_w));
  float scale_y = (align_corners) ? (static_cast<float>(in_h - 1) / (out_h - 1))
                                  : (static_cast<float>(in_h) / (out_h));
  if ("Bilinear" == interpolate_type) {
#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      bilinear_interp(din + spatial_in * i,
                      in_w,
                      in_h,
                      dout + spatial_out * i,
                      out_w,
                      out_h,
                      scale_x,
                      scale_y,
                      align_corners,
                      align_mode);
    }
  } else if ("Nearest" == interpolate_type) {
#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      nearest_interp(din + spatial_in * i,
                     in_w,
                     in_h,
                     dout + spatial_out * i,
                     out_w,
                     out_h,
                     scale_x,
                     scale_y,
                     align_corners);
    }
  }
}

} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
