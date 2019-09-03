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

#include "lite/backends/arm/math/anchor_generator.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void anchor_generator_func(int feature_height,
                           int feature_width,
                           std::vector<float> anchor_sizes,
                           std::vector<float> aspect_ratios,
                           std::vector<float> stride,
                           std::vector<float> variances,
                           float offset,
                           float* anchors_ptr,
                           float* vars_ptr) {
  float stride_width = stride[0];
  float stride_height = stride[1];
  int num_anchors = aspect_ratios.size() * anchor_sizes.size();
  for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
    float* anchors_ptr_h =
        anchors_ptr + h_idx * feature_width * num_anchors * 4;
    for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
      float* anchors_ptr_w = anchors_ptr_h + w_idx * num_anchors * 4;
      float x_ctr = (w_idx * stride_width) + offset * (stride_width - 1);
      float y_ctr = (h_idx * stride_height) + offset * (stride_height - 1);
      float area, area_ratios;
      float base_w, base_h;
      float scale_w, scale_h;
      float anchor_width, anchor_height;
      int idx = 0;
      for (size_t r = 0; r < aspect_ratios.size(); ++r) {
        auto ar = aspect_ratios[r];
        for (size_t s = 0; s < anchor_sizes.size(); ++s) {
          auto anchor_size = anchor_sizes[s];
          area = stride_width * stride_height;
          area_ratios = area / ar;
          base_w = round(sqrt(area_ratios));
          base_h = round(base_w * ar);
          scale_w = anchor_size / stride_width;
          scale_h = anchor_size / stride_height;
          anchor_width = scale_w * base_w;
          anchor_height = scale_h * base_h;
          anchors_ptr_w[idx++] = x_ctr - 0.5 * (anchor_width - 1);
          anchors_ptr_w[idx++] = y_ctr - 0.5 * (anchor_height - 1);
          anchors_ptr_w[idx++] = x_ctr + 0.5 * (anchor_width - 1);
          anchors_ptr_w[idx++] = y_ctr + 0.5 * (anchor_height - 1);
        }
      }
    }
  }

  int64_t hwn = feature_height * feature_width * num_anchors * 4;
  for (int64_t i = 0; i < hwn; i++) {
    *vars_ptr = variances[i % 4];
    vars_ptr++;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
