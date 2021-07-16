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

#include "lite/backends/x86/math/box_coder.h"
#include <string>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void encode_center_size(const int64_t row,  // N
                        const int64_t col,  // M
                        const int64_t len,  // 4
                        const float* target_box_data,
                        const float* prior_box_data,
                        const float* prior_box_var_data,
                        const bool normalized,
                        const std::vector<float> variance,
                        float* output) {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(2)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      size_t offset = i * col * len + j * len;
      float prior_box_width = prior_box_data[j * len + 2] -
                              prior_box_data[j * len] + (normalized == false);
      float prior_box_height = prior_box_data[j * len + 3] -
                               prior_box_data[j * len + 1] +
                               (normalized == false);
      float prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[j * len + 1] + prior_box_height / 2;

      float target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      float target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      float target_box_width = target_box_data[i * len + 2] -
                               target_box_data[i * len] + (normalized == false);
      float target_box_height = target_box_data[i * len + 3] -
                                target_box_data[i * len + 1] +
                                (normalized == false);

      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width));
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height));
    }
  }

  if (prior_box_var_data) {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int64_t k = 0; k < len; ++k) {
          size_t offset = i * col * len + j * len;
          int prior_var_offset = j * len;
          output[offset + k] /= prior_box_var_data[prior_var_offset + k];
        }
      }
    }
  } else if (!(variance.empty())) {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
    for (int64_t i = 0; i < row; ++i) {
      for (int64_t j = 0; j < col; ++j) {
        for (int64_t k = 0; k < len; ++k) {
          size_t offset = i * col * len + j * len;
          output[offset + k] /= variance[k];
        }
      }
    }
  }
}

void decode_center_size(const int axis,
                        const int var_size,
                        const int64_t row,
                        const int64_t col,
                        const int64_t len,
                        const float* target_box_data,
                        const float* prior_box_data,
                        const float* prior_box_var_data,
                        const bool normalized,
                        const std::vector<float> variance,
                        float* output) {
#ifdef PADDLE_WITH_MKLML
#if !defined(WIN32)
#pragma omp parallel for collapse(2)
#else
#pragma omp parallel for
#endif  // WIN32
#endif  // PADDLE_WITH_MKLML
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      float var_data[4] = {1., 1., 1., 1.};
      float* var_ptr = var_data;
      size_t offset = i * col * len + j * len;
      int prior_box_offset = axis == 0 ? j * len : i * len;

      float prior_box_width = prior_box_data[prior_box_offset + 2] -
                              prior_box_data[prior_box_offset] +
                              (normalized == false);
      float prior_box_height = prior_box_data[prior_box_offset + 3] -
                               prior_box_data[prior_box_offset + 1] +
                               (normalized == false);
      float prior_box_center_x =
          prior_box_data[prior_box_offset] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[prior_box_offset + 1] + prior_box_height / 2;

      float target_box_center_x = 0, target_box_center_y = 0;
      float target_box_width = 0, target_box_height = 0;
      int prior_var_offset = axis == 0 ? j * len : i * len;
      if (var_size == 2) {
        std::memcpy(
            var_ptr, prior_box_var_data + prior_var_offset, 4 * sizeof(float));
      } else if (var_size == 1) {
        var_ptr = const_cast<float*>(variance.data());
      }
      float box_var_x = *var_ptr;
      float box_var_y = *(var_ptr + 1);
      float box_var_w = *(var_ptr + 2);
      float box_var_h = *(var_ptr + 3);

      target_box_center_x =
          box_var_x * target_box_data[offset] * prior_box_width +
          prior_box_center_x;
      target_box_center_y =
          box_var_y * target_box_data[offset + 1] * prior_box_height +
          prior_box_center_y;
      target_box_width =
          std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width;
      target_box_height =
          std::exp(box_var_h * target_box_data[offset + 3]) * prior_box_height;

      output[offset] = target_box_center_x - target_box_width / 2;
      output[offset + 1] = target_box_center_y - target_box_height / 2;
      output[offset + 2] =
          target_box_center_x + target_box_width / 2 - (normalized == false);
      output[offset + 3] =
          target_box_center_y + target_box_height / 2 - (normalized == false);
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
