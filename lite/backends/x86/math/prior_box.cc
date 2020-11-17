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

#include "lite/backends/x86/math/prior_box.h"
#include <algorithm>
#include <string>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void density_prior_box(const int64_t img_width,
                       const int64_t img_height,
                       const int64_t feature_width,
                       const int64_t feature_height,
                       const float* input_data,
                       const float* image_data,
                       const bool clip,
                       const std::vector<float> variances,
                       const std::vector<float> fixed_sizes,
                       const std::vector<float> fixed_ratios,
                       const std::vector<int> densities,
                       const float step_width,
                       const float step_height,
                       const float offset,
                       const int num_priors,
                       float* boxes_data,
                       float* vars_data) {
  int step_average = static_cast<int>((step_width + step_height) * 0.5);

  std::vector<float> sqrt_fixed_ratios;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (size_t i = 0; i < fixed_ratios.size(); i++) {
    sqrt_fixed_ratios.push_back(sqrt(fixed_ratios[i]));
  }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
  for (int64_t h = 0; h < feature_height; ++h) {
    for (int64_t w = 0; w < feature_width; ++w) {
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      int64_t offset = (h * feature_width + w) * num_priors * 4;
      // Generate density prior boxes with fixed sizes.
      for (size_t s = 0; s < fixed_sizes.size(); ++s) {
        auto fixed_size = fixed_sizes[s];
        int density = densities[s];
        int shift = step_average / density;
        // Generate density prior boxes with fixed ratios.
        for (size_t r = 0; r < fixed_ratios.size(); ++r) {
          float box_width_ratio = fixed_size * sqrt_fixed_ratios[r];
          float box_height_ratio = fixed_size / sqrt_fixed_ratios[r];
          float density_center_x = center_x - step_average / 2. + shift / 2.;
          float density_center_y = center_y - step_average / 2. + shift / 2.;
          for (int di = 0; di < density; ++di) {
            for (int dj = 0; dj < density; ++dj) {
              float center_x_temp = density_center_x + dj * shift;
              float center_y_temp = density_center_y + di * shift;
              boxes_data[offset++] = (std::max)(
                  (center_x_temp - box_width_ratio / 2.) / img_width, 0.);
              boxes_data[offset++] = (std::max)(
                  (center_y_temp - box_height_ratio / 2.) / img_height, 0.);
              boxes_data[offset++] = (std::min)(
                  (center_x_temp + box_width_ratio / 2.) / img_width, 1.);
              boxes_data[offset++] = (std::min)(
                  (center_y_temp + box_height_ratio / 2.) / img_height, 1.);
            }
          }
        }
      }
    }
  }
  //! clip the prior's coordinate such that it is within [0, 1]
  if (clip) {
    int channel_size = feature_height * feature_width * num_priors * 4;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int d = 0; d < channel_size; ++d) {
      boxes_data[d] = (std::min)((std::max)(boxes_data[d], 0.f), 1.f);
    }
  }
//! set the variance.
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(3)
#endif
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      for (int i = 0; i < num_priors; ++i) {
        int idx = ((h * feature_width + w) * num_priors + i) * 4;
        vars_data[idx++] = variances[0];
        vars_data[idx++] = variances[1];
        vars_data[idx++] = variances[2];
        vars_data[idx++] = variances[3];
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
