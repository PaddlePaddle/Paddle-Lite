/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/interpolate.h"
#include <string>
#include <vector>
#include "lite/backends/x86/math/math_function.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void bilinear_interp(const float* input_data,
                     float* output_data,
                     const float ratio_h,
                     const float ratio_w,
                     const int h_in,
                     const int w_in,
                     const int n,
                     const int c,
                     const int h_out,
                     const int w_out,
                     const bool align_corners,
                     const bool align_mode) {
  int* buf = static_cast<int*>(
      lite::host::malloc(sizeof(int) * (w_out * 4 + h_out * 4)));
  int* xofs = buf;
  int* yofs = buf + w_out * 2;

  float* alpha = reinterpret_cast<float*>(buf + w_out * 2 + h_out * 2);
  float* beta = reinterpret_cast<float*>(buf + h_out * 2 + w_out * 4);

  float fx = 0.0f;
  float fy = 0.0f;
  int sx = 0;
  int sy = 0;
  if (align_corners) {
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = dx * ratio_w;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx * 2] = sx;
      xofs[dx * 2 + 1] = (sx + 1) < w_in - 1 ? (sx + 1) : (w_in - 1);
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = dy * ratio_h;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy * 2] = sy;
      yofs[dy * 2 + 1] = (sy + 1) < h_in - 1 ? (sy + 1) : (h_in - 1);
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  } else {
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = align_mode ? ratio_w * dx : ratio_w * (dx + 0.5f) - 0.5f;
      fx = fx < 0 ? 0.f : fx;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx * 2] = sx;
      xofs[dx * 2 + 1] = (sx + 1) < w_in - 1 ? (sx + 1) : (w_in - 1);
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = align_mode ? ratio_h * dy : ratio_h * (dy + 0.5f) - 0.5f;
      fy = fy < 0 ? 0.f : fy;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy * 2] = sy;
      yofs[dy * 2 + 1] = (sy + 1) < h_in - 1 ? (sy + 1) : (h_in - 1);
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  }
  // output w , h boundary
  int w_bound = w_out;
  int h_bound = h_out;
  if (ratio_w > 0 && ratio_h > 0) {
    if (align_corners) {
      w_bound = ceil((w_in - 1) / ratio_w);
      h_bound = ceil((h_in - 1) / ratio_h);
    } else {
      w_bound = ceil((w_in - 0.5f) / ratio_w - 0.5f);
      h_bound = ceil((h_in - 0.5f) / ratio_h - 0.5f);
    }
  }
  int in_stride = h_in * w_in;
  int out_stride = h_out * w_out;
  int total = n * c;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int nc = 0; nc < total; ++nc) {
    const float* src = input_data + nc * in_stride;
    float* dst = output_data + nc * out_stride;
    const float* betap = beta;

    float* rowsbuf0 =
        static_cast<float*>(lite::host::malloc(sizeof(int) * w_out));
    float* rowsbuf1 =
        static_cast<float*>(lite::host::malloc(sizeof(int) * w_out));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    // h_bound loop
    for (int dy = 0; dy < h_bound; dy++) {
      int sy0 = yofs[dy * 2];
      int sy1 = yofs[dy * 2 + 1];

      const float* s0 = src + sy0 * w_in;
      const float* s1 = src + sy1 * w_in;

      const float* alphap = alpha;
      float* rows0p = rows0;
      float* rows1p = rows1;

      int dx = 0;
// w_bound loop
#ifdef __AVX__
      for (; dx + 3 < w_bound; dx += 4) {
        int x0 = xofs[dx * 2];
        int x1 = xofs[(dx + 1) * 2];
        int x2 = xofs[(dx + 2) * 2];
        int x3 = xofs[(dx + 3) * 2];
        int x01 = xofs[dx * 2 + 1];
        int x11 = xofs[(dx + 1) * 2 + 1];
        int x21 = xofs[(dx + 2) * 2 + 1];
        int x31 = xofs[(dx + 3) * 2 + 1];

        const float* s0p0 = s0 + x0;
        const float* s0p1 = s0 + x1;
        const float* s0p2 = s0 + x2;
        const float* s0p3 = s0 + x3;

        const float* s0p0_1 = s0 + x01;
        const float* s0p1_1 = s0 + x11;
        const float* s0p2_1 = s0 + x21;
        const float* s0p3_1 = s0 + x31;

        const float* s1p0 = s1 + x0;
        const float* s1p1 = s1 + x1;
        const float* s1p2 = s1 + x2;
        const float* s1p3 = s1 + x3;

        const float* s1p0_1 = s1 + x01;
        const float* s1p1_1 = s1 + x11;
        const float* s1p2_1 = s1 + x21;
        const float* s1p3_1 = s1 + x31;

        __m256 _a = _mm256_loadu_ps(alphap);

        __m256 _s0p0p3 = _mm256_set_ps(
            *s0p3_1, *s0p3, *s0p2_1, *s0p2, *s0p1_1, *s0p1, *s0p0_1, *s0p0);

        __m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
        __m256 _s1p0p3 = _mm256_set_ps(
            *s1p3_1, *s1p3, *s1p2_1, *s1p2, *s1p1_1, *s1p1, *s1p0_1, *s1p0);
        __m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

        __m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
        __m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

        __m256 _rs0 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
        __m256 _rs1 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
        _mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
        _mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

        alphap += 8;
      }
#endif

      // w_bound remain loop
      for (; dx < w_bound; ++dx) {
        int sx = xofs[dx * 2];
        int sx1 = xofs[dx * 2 + 1];
        const float* s0p = s0 + sx;
        const float* s1p = s1 + sx;
        const float* s0p1 = s0 + sx1;
        const float* s1p1 = s1 + sx1;

        float a0 = alphap[0];
        float a1 = alphap[1];
        rows0p[dx] = s0p[0] * a0 + s0p1[0] * a1;
        rows1p[dx] = s1p[0] * a0 + s1p1[0] * a1;
        alphap += 2;
      }

      float param0 = *(src + sy0 * w_in + w_in - 1);
      float param1 = *(src + sy1 * w_in + w_in - 1);
      const float buffer0[2] = {param0, param0};
      const float buffer1[2] = {param1, param1};
#ifdef __AVX__
      __m256 _s0p0p3 = _mm256_set1_ps(param0);
      __m256 _s1p0p3 = _mm256_set1_ps(param1);
      for (; dx + 3 < w_out; dx += 4) {
        __m256 _a = _mm256_loadu_ps(alphap);

        __m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
        __m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

        __m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
        __m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

        __m256 _rs0 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
        __m256 _rs1 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
        _mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
        _mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

        alphap += 8;
      }
#endif

      // w_bound - w_out remain loop
      for (; dx < w_out; dx++) {
        const float* s0p = buffer0;
        const float* s1p = buffer1;

        float a0 = alphap[0];
        float a1 = alphap[1];
        rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
        rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

        alphap += 2;
      }

      float b0 = betap[0];
      float b1 = betap[1];

      // output pos
      float* dp = dst + dy * w_out;

      int nn = 0;

#ifdef __AVX__
      // 8 float
      __m256 _b0 = _mm256_set1_ps(b0);
      __m256 _b1 = _mm256_set1_ps(b1);
      // calculate and store results
      for (; nn + 7 < w_out; nn += 8) {
        __m256 _rows0 = _mm256_loadu_ps(rows0p);
        __m256 _rows1 = _mm256_loadu_ps(rows1p);

        __m256 _d = _mm256_add_ps(_mm256_mul_ps(_rows0, _b0),
                                  _mm256_mul_ps(_rows1, _b1));
        _mm256_storeu_ps(dp, _d);

        dp += 8;
        rows0p += 8;
        rows1p += 8;
      }

      // 4 float
      __m128 _c0 = _mm_set1_ps(b0);
      __m128 _c1 = _mm_set1_ps(b1);
      for (; nn + 3 < w_out; nn += 4) {
        __m128 _rows0 = _mm_loadu_ps(rows0p);
        __m128 _rows1 = _mm_loadu_ps(rows1p);

        __m128 _d =
            _mm_add_ps(_mm_mul_ps(_rows0, _c0), _mm_mul_ps(_rows1, _c1));
        _mm_storeu_ps(dp, _d);

        dp += 4;
        rows0p += 4;
        rows1p += 4;
      }
#endif

      // calculate and store remain resluts
      for (; nn < w_out; ++nn) {
        *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
      }
      betap += 2;
    }  // end h_bound loop

    // h_bound - h_out loop
    for (int dy = h_bound; dy < h_out; dy++) {
      int sy = h_in - 1;
      const float* s0 = src + sy * w_in;
      const float* alphap = alpha;
      float* rows0p = rows0;
      float* rows1p = rows1;

      int dx = 0;
#ifdef __AVX__
      const float* s1 = s0;

      // w_bound loop
      for (; dx + 3 < w_bound; dx += 4) {
        int x0 = xofs[dx * 2];
        int x1 = xofs[(dx + 1) * 2];
        int x2 = xofs[(dx + 2) * 2];
        int x3 = xofs[(dx + 3) * 2];
        int x01 = xofs[dx * 2 + 1];
        int x11 = xofs[(dx + 1) * 2 + 1];
        int x21 = xofs[(dx + 2) * 2 + 1];
        int x31 = xofs[(dx + 3) * 2 + 1];

        const float* s0p0 = s0 + x0;
        const float* s0p1 = s0 + x1;
        const float* s0p2 = s0 + x2;
        const float* s0p3 = s0 + x3;

        const float* s0p0_1 = s0 + x01;
        const float* s0p1_1 = s0 + x11;
        const float* s0p2_1 = s0 + x21;
        const float* s0p3_1 = s0 + x31;

        const float* s1p0 = s1 + x0;
        const float* s1p1 = s1 + x1;
        const float* s1p2 = s1 + x2;
        const float* s1p3 = s1 + x3;

        const float* s1p0_1 = s1 + x01;
        const float* s1p1_1 = s1 + x11;
        const float* s1p2_1 = s1 + x21;
        const float* s1p3_1 = s1 + x31;

        __m256 _a = _mm256_loadu_ps(alphap);

        __m256 _s0p0p3 = _mm256_set_ps(
            *s0p3_1, *s0p3, *s0p2_1, *s0p2, *s0p1_1, *s0p1, *s0p0_1, *s0p0);
        __m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
        __m256 _s1p0p3 = _mm256_set_ps(
            *s1p3_1, *s1p3, *s1p2_1, *s1p2, *s1p1_1, *s1p1, *s1p0_1, *s1p0);
        __m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

        __m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
        __m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

        __m256 _rs0 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
        __m256 _rs1 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
        _mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
        _mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

        alphap += 8;
      }
#endif

      // w_bound remain loop
      for (; dx < w_bound; ++dx) {
        int sx = xofs[dx * 2];
        int sx1 = xofs[dx * 2 + 1];
        const float* s0p = s0 + sx;
        const float* s0p1 = s0 + sx1;
        float a0 = alphap[0];
        float a1 = alphap[1];
        rows0p[dx] = s0p[0] * a0 + s0p1[0] * a1;
        rows1p[dx] = rows0p[dx];

        alphap += 2;
      }

      float param = *(src + sy * w_in + w_in - 1);
      const float buffer1[2] = {param, param};

#ifdef __AVX__
      __m256 _s0p0p3 = _mm256_set1_ps(param);
      __m256 _s1p0p3 = _mm256_set1_ps(param);

      // w_bound - w_out loop
      for (; dx + 3 < w_out; dx += 4) {
        __m256 _a = _mm256_loadu_ps(alphap);

        __m256 _ms0 = _mm256_mul_ps(_s0p0p3, _a);
        __m256 _ms1 = _mm256_mul_ps(_s1p0p3, _a);

        __m256 _rows0 = _mm256_hadd_ps(_ms0, _ms0);
        __m256 _rows1 = _mm256_hadd_ps(_ms1, _ms1);

        __m256 _rs0 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows0), 0b11011000));
        __m256 _rs1 = _mm256_castpd_ps(
            _mm256_permute4x64_pd(_mm256_castps_pd(_rows1), 0b11011000));
        _mm_storeu_ps(rows0p + dx, _mm256_castps256_ps128(_rs0));
        _mm_storeu_ps(rows1p + dx, _mm256_castps256_ps128(_rs1));

        alphap += 8;
      }
#endif

      // w_bound - wout remain loop
      for (; dx < w_out; dx++) {
        const float* s0p = buffer1;
        float a0 = alphap[0];
        float a1 = alphap[1];
        rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
        rows1p[dx] = rows0p[dx];
        alphap += 2;
      }

      float b0 = betap[0];
      float b1 = betap[1];

      float* dp = dst + dy * w_out;

      int nn = 0;

#ifdef __AVX__
      // 8 float
      __m256 _b0 = _mm256_set1_ps(b0);
      __m256 _b1 = _mm256_set1_ps(b1);
      // calculate and store results
      for (; nn + 7 < w_out; nn += 8) {
        __m256 _rows0 = _mm256_loadu_ps(rows0p);
        __m256 _rows1 = _mm256_loadu_ps(rows1p);

        __m256 _d = _mm256_add_ps(_mm256_mul_ps(_rows0, _b0),
                                  _mm256_mul_ps(_rows1, _b1));
        _mm256_storeu_ps(dp, _d);

        dp += 8;
        rows0p += 8;
        rows1p += 8;
      }

      // 4 float
      __m128 _c0 = _mm_set1_ps(b0);
      __m128 _c1 = _mm_set1_ps(b1);
      for (; nn + 3 < w_out; nn += 4) {
        __m128 _rows0 = _mm_loadu_ps(rows0p);
        __m128 _rows1 = _mm_loadu_ps(rows1p);

        __m128 _d =
            _mm_add_ps(_mm_mul_ps(_rows0, _c0), _mm_mul_ps(_rows1, _c1));
        _mm_storeu_ps(dp, _d);

        dp += 4;
        rows0p += 4;
        rows1p += 4;
      }
#endif
      // calculate and store remain results
      for (; nn < w_out; ++nn) {
        *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
      }

      betap += 2;
    }  // end h_bound - h_out loop
    lite::host::free(rowsbuf0);
    lite::host::free(rowsbuf1);
  }
  lite::host::free(buf);
}

void nearest_interp(const float* input_data,
                    float* output_data,
                    const float ratio_h,
                    const float ratio_w,
                    const int n,
                    const int c,
                    const int in_h,
                    const int in_w,
                    const int out_h,
                    const int out_w,
                    const bool align_corners) {
  int total_count = n * c;
  if (align_corners) {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
    for (int i = 0; i < total_count; ++i) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          const float* input_data_ptr = input_data + i * in_h * in_w;
          float* output_data_ptr =
              output_data + i * out_h * out_w + h * out_w + w;
          int near_y = static_cast<int>(ratio_h * h + 0.5);
          int near_x = static_cast<int>(ratio_w * w + 0.5);
          *output_data_ptr = input_data_ptr[near_y * in_w + near_x];
        }
      }
    }
  } else {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
    for (int i = 0; i < total_count; ++i) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          const float* input_data_ptr = input_data + i * in_h * in_w;
          float* output_data_ptr =
              output_data + i * out_h * out_w + h * out_w + w;
          int near_y = static_cast<int>(ratio_h * h);
          int near_x = static_cast<int>(ratio_w * w);
          *output_data_ptr = input_data_ptr[near_y * in_w + near_x];
        }
      }
    }
  }
}

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

void interpolate(lite::Tensor* input,
                 lite::Tensor* out_size,
                 std::vector<const lite::Tensor*> list_new_size_tensor,
                 lite::Tensor* scale_tensor,
                 lite::Tensor* output,
                 float scale,
                 std::vector<float> scale_v,
                 int out_h,
                 int out_w,
                 const int align_mode,
                 const bool align_corners,
                 const std::string interpolate_type) {
  // format NCHW
  int n = input->dims()[0];
  int c = input->dims()[1];
  int in_h = input->dims()[2];
  int in_w = input->dims()[3];
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  } else if (scale_v.size() == 2) {
    if (scale_v[0] > 0 && scale_v[1] > 0) {
      out_h = static_cast<int>(in_h * scale_v[0]);
      out_w = static_cast<int>(in_w * scale_v[1]);
    }
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  } else {
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    }
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
    }
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  output->Resize({n, c, out_h, out_w});

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(in_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(in_w) / out_w;
  }

  const float* input_data = input->data<float>();
  float* output_data = output->mutable_data<float>();
  if ("Bilinear" == interpolate_type) {
    bilinear_interp(input_data,
                    output_data,
                    ratio_h,
                    ratio_w,
                    in_h,
                    in_w,
                    n,
                    c,
                    out_h,
                    out_w,
                    align_corners,
                    align_mode);
  } else if ("Nearest" == interpolate_type) {
    nearest_interp(input_data,
                   output_data,
                   ratio_h,
                   ratio_w,
                   n,
                   c,
                   in_h,
                   in_w,
                   out_h,
                   out_w,
                   align_corners);
  } else {
    LOG(FATAL) << "Not supported interpolate_type: " << interpolate_type;
  }
}

void interpolate_v2(lite::Tensor* input,
                    lite::Tensor* out_size,
                    std::vector<const lite::Tensor*> list_new_size_tensor,
                    lite::Tensor* scale_tensor,
                    lite::Tensor* output,
                    float scale,
                    std::vector<float> scale_v,
                    int out_h,
                    int out_w,
                    const int align_mode,
                    const bool align_corners,
                    const std::string interpolate_type) {
  // format NCHW
  int n = input->dims()[0];
  int c = input->dims()[1];
  int in_h = input->dims()[2];
  int in_w = input->dims()[3];
  float scale_h = -1;
  float scale_w = -1;
  if (list_new_size_tensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(list_new_size_tensor);
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }
    } else {
      if (scale_v.size() > 1 && scale_v[0] > 0 && scale_v[1] > 0) {
        scale_h = scale_v[0];
        scale_w = scale_v[1];
      }
    }

    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }

    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  output->Resize({n, c, out_h, out_w});

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    float new_scale_h = 0.f;
    new_scale_h = (scale_h > 0) ? static_cast<float>(1. / scale_h)
                                : static_cast<float>(in_h) / out_h;
    ratio_h = (align_corners) ? static_cast<float>(in_h - 1) / (out_h - 1)
                              : static_cast<float>(new_scale_h);
  }
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }

  const float* input_data = input->data<float>();
  float* output_data = output->mutable_data<float>();
  if ("Bilinear" == interpolate_type) {
    bilinear_interp(input_data,
                    output_data,
                    ratio_h,
                    ratio_w,
                    in_h,
                    in_w,
                    n,
                    c,
                    out_h,
                    out_w,
                    align_corners,
                    align_mode);
  } else if ("Nearest" == interpolate_type) {
    nearest_interp(input_data,
                   output_data,
                   ratio_h,
                   ratio_w,
                   n,
                   c,
                   in_h,
                   in_w,
                   out_h,
                   out_w,
                   align_corners);
  } else {
    LOG(FATAL) << "Not supported interpolate_type: " << interpolate_type;
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
