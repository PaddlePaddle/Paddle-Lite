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
                     const int in_h,
                     const int in_w,
                     const int n,
                     const int c,
                     const int out_h,
                     const int out_w,
                     const bool align_corners,
                     const bool align_mode) {
  bool align_flag = (align_mode == 0 && !align_corners);

  std::vector<int> vy_n, vy_s;
  std::vector<float> vd_n, vd_s;
  vy_n.reserve(out_h);
  vy_s.reserve(out_h);
  vd_n.reserve(out_h);
  vd_s.reserve(out_h);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int k = 0; k < out_h; k++) {
    float k_r = ratio_h * k;
    float z_r = ratio_h * 0.5;
    int y_n = align_flag ? static_cast<int>(k_r + z_r - 0.5)
                         : static_cast<int>(k_r);
    y_n = (y_n > 0) ? y_n : 0;
    int y_s = (y_n + 1) < (in_h - 1) ? (y_n + 1) : (in_h - 1);
    float idx_src_y = k_r + z_r - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float d_n = align_flag ? idx_src_y - y_n : k_r - y_n;
    float d_s = 1.f - d_n;
    {
      vy_n[k] = y_n;
      vy_s[k] = y_s;
      vd_n[k] = d_n;
      vd_s[k] = d_s;
    }
  }

  std::vector<int> vx_w, vx_e;
  std::vector<float> vd_w, vd_e;
  vx_w.reserve(out_w);
  vx_e.reserve(out_w);
  vd_w.reserve(out_w);
  vd_e.reserve(out_w);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int l = 0; l < out_w; l++) {
    float w_r = ratio_h * l;
    float z_r = ratio_h * 0.5;
    int x_w = (align_mode == 0 && !align_corners)
                  ? static_cast<int>(w_r + z_r - 0.5)
                  : static_cast<int>(w_r);
    x_w = (x_w > 0) ? x_w : 0;
    int x_e = (x_w + 1) < (in_w - 1) ? (x_w + 1) : (in_w - 1);
    float idx_src_x = w_r + z_r - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float d_w = align_flag ? idx_src_x - x_w : w_r - x_w;
    float d_e = 1.f - d_w;
    {
      vx_w[l] = x_w;
      vx_e[l] = x_e;
      vd_w[l] = d_w;
      vd_e[l] = d_e;
    }
  }

  int total_count = n * c;
  float* buf = (float*)malloc(out_w * 2 * sizeof(float));
  int in_stride = in_h * in_w, out_stride = out_h * out_w;
  for (int i = 0; i < total_count; i++) {
    const float* input_data_ptr = input_data + i * in_stride;
    for (int h = 0; h < out_h; h++) {
      float* output_ptr = output_data + i * out_stride + h * out_w;
      // load input 
      const float* in_row0 = input_data + i * in_stride + vy_n[h] * in_w;
      const float* in_row1 = input_data + i * in_stride + vy_s[h] * in_w;
      for(int idx = 0; idx < out_w; ++ idx) {
        buf[idx] = in_row0[vx_w[idx]] * vd_e[idx] + in_row0[vx_e[idx]] * vd_w[idx];
        buf[idx + out_w] = in_row1[vx_w[idx]] * vd_e[idx] + in_row1[vx_e[idx]] * vd_w[idx];        
      }
      
      __m256 yt0 = _mm256_set1_ps(vd_s[h]);
      __m256 yt1 = _mm256_set1_ps(vd_n[h]);
      __m128 ys0 = _mm_set1_ps(vd_s[h]);
      __m128 ys1 = _mm_set1_ps(vd_n[h]);
      int w = 0;
      for(; w + 8 < out_w; w += 8) {
        __m256 xr0 = _mm256_loadu_ps(reinterpret_cast<float*>(buf + w));
        __m256 xr1 = _mm256_loadu_ps(reinterpret_cast<float*>(buf + w + out_w));

        __m256 r0 = _mm256_mul_ps(yt0, xr0);
        __m256 r1 = _mm256_mul_ps(yt1, xr1);
        __m256 r  = _mm256_add_ps(r0, r1);
        
        _mm256_storeu_ps(output_ptr + w, r);
      }
      for(; w + 4 < out_w; w += 4) {
        __m128 xr0 = _mm_loadu_ps(buf + w);
        __m128 xr1 = _mm_loadu_ps(buf + w + out_w);

        __m128 r0 = _mm_mul_ps(ys0, xr0);
        __m128 r1 = _mm_mul_ps(ys1, xr1);
        __m128 r  = _mm_add_ps(r0, r1);

        _mm_storeu_ps(output_ptr + w, r);
      }

      for (;w < out_w; ++ w) {
        output_ptr[w] = vd_s[h] * buf[w] + vd_n[h] * buf[w + out_w];
      }
    }
  }
  free(buf);
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
#pragma omp parallel for collapse(3)
#endif
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
#pragma omp parallel for collapse(3)
#endif
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

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
