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

#include "lite/backends/arm/math/fp16/interpolate_fp16.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

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

#define ALPHA_COMPUTE                     \
  float16x4_t va = vld1_f16(alphap);      \
  float16x4_t vs0 = vld1_f16(s0p);        \
  float16x4_t vs1 = vld1_f16(s1p);        \
  float16x4_t vs0n = vld1_f16(s0np);      \
  float16x4_t vs1n = vld1_f16(s1np);      \
  vs0[2] = vs0n[0];                       \
  vs0[3] = vs0n[1];                       \
  vs1[2] = vs1n[0];                       \
  vs1[3] = vs1n[1];                       \
  float16x4_t vms0 = vmul_f16(vs0, va);   \
  float16x4_t vms1 = vmul_f16(vs1, va);   \
  float16_t vmres0_0 = vms0[0] + vms0[1]; \
  float16_t vmres0_1 = vms0[2] + vms0[3]; \
  float16_t vmres1_0 = vms1[0] + vms1[1]; \
  float16_t vmres1_1 = vms1[2] + vms1[3]; \
  rows0p[dx] = vmres0_0;                  \
  rows0p[dx + 1] = vmres0_1;              \
  rows1p[dx] = vmres1_0;                  \
  rows1p[dx + 1] = vmres1_1;

#define BRTA_COMPUTE                           \
  float16x8_t vrows0 = vld1q_f16(rows0p);      \
  float16x8_t vrows2 = vld1q_f16(rows0p + 8);  \
  float16x8_t vrows4 = vld1q_f16(rows0p + 16); \
  float16x8_t vrows6 = vld1q_f16(rows0p + 24); \
  float16x8_t vrows1 = vld1q_f16(rows1p);      \
  float16x8_t vsum0 = vmulq_f16(vrows0, vb0);  \
  float16x8_t vrows3 = vld1q_f16(rows1p + 8);  \
  float16x8_t vsum1 = vmulq_f16(vrows2, vb0);  \
  float16x8_t vrows5 = vld1q_f16(rows1p + 16); \
  float16x8_t vsum2 = vmulq_f16(vrows4, vb0);  \
  float16x8_t vrows7 = vld1q_f16(rows1p + 24); \
  float16x8_t vsum3 = vmulq_f16(vrows6, vb0);  \
  vsum0 = vfmaq_f16(vsum0, vrows1, vb1);       \
  vsum1 = vfmaq_f16(vsum1, vrows3, vb1);       \
  vsum2 = vfmaq_f16(vsum2, vrows5, vb1);       \
  vsum3 = vfmaq_f16(vsum3, vrows7, vb1);       \
  rows0p += 32;                                \
  vst1q_f16(dp, vsum0);                        \
  vst1q_f16(dp + 8, vsum1);                    \
  rows1p += 32;                                \
  vst1q_f16(dp + 16, vsum2);                   \
  vst1q_f16(dp + 24, vsum3);                   \
  dp += 32;

void bilinear_interp(const float16_t* src,
                     int w_in,
                     int h_in,
                     float16_t* dst,
                     int w_out,
                     int h_out,
                     float scale_x,
                     float scale_y,
                     bool with_align,
                     int align_mode) {
  int* buf = new int[w_out + h_out + w_out * 2 + h_out * 2];

  int* xofs = buf;
  int* yofs = buf + w_out;

  float16_t* alpha = reinterpret_cast<float16_t*>(buf + w_out + h_out);
  float16_t* beta =
      reinterpret_cast<float16_t*>(buf + w_out + h_out + w_out * 2);

  float fx = 0.0f;
  float fy = 0.0f;
  int sx = 0;
  int sy = 0;
  if (with_align) {
    scale_x = static_cast<float>(w_in - 1) / (w_out - 1);
    scale_y = static_cast<float>(h_in - 1) / (h_out - 1);
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = dx * scale_x;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - static_cast<float16_t>(fx);
      alpha[dx * 2 + 1] = static_cast<float16_t>(fx);
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = dy * scale_y;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - static_cast<float16_t>(fy);
      beta[dy * 2 + 1] = static_cast<float16_t>(fy);
    }
  } else {
    scale_x = static_cast<float>(w_in) / w_out;
    scale_y = static_cast<float>(h_in) / h_out;
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = align_mode ? scale_x * dx : scale_x * (dx + 0.5f) - 0.5f;
      fx = fx < 0 ? 0.f : fx;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - static_cast<float16_t>(fx);
      alpha[dx * 2 + 1] = static_cast<float16_t>(fx);
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = align_mode ? scale_y * dy : scale_y * (dy + 0.5f) - 0.5f;
      fy = fy < 0 ? 0.f : fy;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - static_cast<float16_t>(fy);
      beta[dy * 2 + 1] = static_cast<float16_t>(fy);
    }
  }
  float16_t* rowsbuf0 = new float16_t[w_out];
  float16_t* rowsbuf1 = new float16_t[w_out];
  float16_t* rows0 = rowsbuf0;
  float16_t* rows1 = rowsbuf1;
  // output w , h boundary
  int w_bound = w_out;
  int h_bound = h_out;

  int cnt = w_out >> 5;
  int remain = (w_out & 31);
  int remain_cnt = remain >> 3;
  int remain_rem = remain & 7;
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

    const float16_t* s0 = src + sy * w_in;
    const float16_t* s1 = src + (sy + 1) * w_in;

    const float16_t* alphap = alpha;
    float16_t* rows0p = rows0;
    float16_t* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float16_t* s0p = s0 + sx;
      const float16_t* s1p = s1 + sx;
      const float16_t* s0np = s0 + sxn;
      const float16_t* s1np = s1 + sxn;

      ALPHA_COMPUTE

      alphap += 4;
    }
    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float16_t* s0p = s0 + sx;
      const float16_t* s1p = s1 + sx;

      float16_t a0 = alphap[0];
      float16_t a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    const float16_t buffer1[2] = {*(src + sy * w_in + w_in - 1),
                                  *(src + sy * w_in + w_in - 1)};
    const float16_t buffer2[2] = {*(src + (sy + 1) * w_in + w_in - 1),
                                  *(src + (sy + 1) * w_in + w_in - 1)};
    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float16_t* s0p = buffer1;
      const float16_t* s1p = buffer2;
      const float16_t* s0np = buffer1;
      const float16_t* s1np = buffer2;
      ALPHA_COMPUTE

      alphap += 4;
    }
    // w_bound - w_out remain loop
    for (; dx < w_out; dx++) {
      const float16_t* s0p = buffer1;
      const float16_t* s1p = buffer2;

      float16_t a0 = alphap[0];
      float16_t a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    float16_t b0 = beta[0];
    float16_t b1 = beta[1];

    float16_t* dp = dst + dy * w_out;

    float16x8_t vb0 = vdupq_n_f16(b0);
    float16x8_t vb1 = vdupq_n_f16(b1);
    // calculate and store results
    for (int i = 0; i < cnt; i++) {
      BRTA_COMPUTE
    }
    for (int i = 0; i < remain_cnt; i++) {
      float16x8_t vrows0 = vld1q_f16(rows0p);
      float16x8_t vrows1 = vld1q_f16(rows1p);
      float16x8_t vsum0 = vmulq_f16(vrows0, vb0);
      vsum0 = vfmaq_f16(vsum0, vrows1, vb1);

      rows0p += 8;
      vst1q_f16(dp, vsum0);
      rows1p += 8;
      dp += 8;
    }
    // calculate and store remain resluts
    for (int i = 0; i < remain_rem; i++) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }
    beta += 2;
  }

  // h_bound - h_out loop
  for (int dy = h_bound; dy < h_out; dy++) {
    int sy = h_in - 1;
    const float16_t* s0 = src + sy * w_in;
    const float16_t* s1 = s0;
    const float16_t* alphap = alpha;
    float16_t* rows0p = rows0;
    float16_t* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float16_t* s0p = s0 + sx;
      const float16_t* s1p = s1 + sx;
      const float16_t* s0np = s0 + sxn;
      const float16_t* s1np = s1 + sxn;

      ALPHA_COMPUTE

      alphap += 4;
    }
    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float16_t* s0p = s0 + sx;
      float16_t a0 = alphap[0];
      float16_t a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];

      alphap += 2;
    }

    const float16_t buffer1[2] = {*(src + sy * w_in + w_in - 1),
                                  *(src + sy * w_in + w_in - 1)};
    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float16_t* s0p = buffer1;
      const float16_t* s1p = buffer1;
      const float16_t* s0np = buffer1;
      const float16_t* s1np = buffer1;
      ALPHA_COMPUTE

      alphap += 4;
    }
    // w_bound - wout remain loop
    for (; dx < w_out; dx++) {
      const float16_t* s0p = buffer1;
      float16_t a0 = alphap[0];
      float16_t a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];
      alphap += 2;
    }

    float16_t b0 = beta[0];
    float16_t b1 = beta[1];

    float16_t* dp = dst + dy * w_out;

    float16x8_t vb0 = vdupq_n_f16(b0);
    float16x8_t vb1 = vdupq_n_f16(b1);
    // calculate and store results
    for (int i = 0; i < cnt; i++) {
      BRTA_COMPUTE
    }
    for (int i = 0; i < remain_cnt; i++) {
      float16x8_t vrows0 = vld1q_f16(rows0p);
      float16x8_t vrows1 = vld1q_f16(rows1p);
      float16x8_t vsum0 = vmulq_f16(vrows0, vb0);
      vsum0 = vfmaq_f16(vsum0, vrows1, vb1);

      rows0p += 8;
      vst1q_f16(dp, vsum0);
      rows1p += 8;
      dp += 8;
    }

    // calculate and store remain results
    for (int i = 0; i < remain_rem; i++) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }

    beta += 2;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

void nearest_interp(const float16_t* src,
                    int w_in,
                    int h_in,
                    float16_t* dst,
                    int w_out,
                    int h_out,
                    float scale_x,
                    float scale_y,
                    bool with_align) {
  float scale_w_new = (with_align)
                          ? (static_cast<float>(w_in - 1) / (w_out - 1))
                          : (static_cast<float>(w_in) / (w_out));
  float scale_h_new = (with_align)
                          ? (static_cast<float>(h_in - 1) / (h_out - 1))
                          : (static_cast<float>(h_in) / (h_out));
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      float16_t* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w + 0.5);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      float16_t* dst_p = dst + h * w_out;
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
                 bool with_align,
                 int align_mode,
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
  float height_scale = scale;
  float width_scale = scale;
  if (out_width > 0 && out_height > 0) {
    height_scale = static_cast<float>(out_height / X->dims()[2]);
    width_scale = static_cast<float>(out_width / X->dims()[3]);
  }
  int num_cout = X->dims()[0];
  int c_cout = X->dims()[1];
  Out->Resize({num_cout, c_cout, out_height, out_width});

  float16_t* dout = Out->mutable_data<float16_t>();
  const float16_t* din = X->data<float16_t>();
  int out_num = Out->dims()[0];
  int out_c = Out->dims()[1];
  int count = out_num * out_c;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int spatial_in = in_h * in_w;
  int spatial_out = out_h * out_w;

  if (interpolate_type == "Bilinear") {
#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      bilinear_interp(din + spatial_in * i,
                      in_w,
                      in_h,
                      dout + spatial_out * i,
                      out_w,
                      out_h,
                      1.f / width_scale,
                      1.f / height_scale,
                      with_align,
                      align_mode);
    }
  } else if (interpolate_type == "Nearest") {
#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      nearest_interp(din + spatial_in * i,
                     in_w,
                     in_h,
                     dout + spatial_out * i,
                     out_w,
                     out_h,
                     1.f / width_scale,
                     1.f / height_scale,
                     with_align);
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
