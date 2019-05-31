/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef DENSITY_PRIORBOX_OP
#pragma once

#include <operators/kernel/prior_box_kernel.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace paddle_mobile {
namespace operators {

template <typename T>
struct ClipFunctor {
  inline T operator()(T in) const {
    return std::min<T>(std::max<T>(in, 0.), 1.);
  }
};

template <typename P>
void DensityPriorBoxCompute(const DensityPriorBoxParam<CPU> &param) {
  const auto *input_ = param.Input();
  const auto &input_dims = input_->dims();

  const auto *input_image = param.InputImage();
  const auto &input_image_dims = input_image->dims();

  auto densities = param.Densities();
  auto fixed_ratios = param.FixedRatios();

  auto fixed_sizes = param.FixedSizes();

  const auto &variances = param.Variances();
  const bool &clip = param.Clip();

  const float &step_w = param.StepW();
  const float &step_h = param.StepH();
  const float &offset = param.Offset();

  Tensor *output_boxes = param.OutputBoxes();
  auto output_boxes_dataptr = output_boxes->mutable_data<float>();
  Tensor *output_variances = param.OutputVariances();
  auto output_variances_dataptr = output_variances->mutable_data<float>();

  auto img_width = input_image_dims[3];
  auto img_height = input_image_dims[2];

  auto feature_width = input_dims[3];
  auto feature_height = input_dims[2];

  auto stride0 = output_boxes->dims()[1] * output_boxes->dims()[2] *
                 output_boxes->dims()[3];
  auto stride1 = output_boxes->dims()[2] * output_boxes->dims()[3];
  auto stride2 = output_boxes->dims()[3];

  float step_width, step_height;
  /// 300 / 19
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = step_w;
    step_height = step_h;
  }

  int num_priors = 0;
  for (size_t i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }

  auto box_dim = output_variances->dims();

  output_boxes->Resize({feature_height, feature_width, num_priors, 4});
  int step_average = static_cast<int>((step_width + step_height) * 0.5);

  std::vector<float> sqrt_fixed_ratios;
  for (size_t i = 0; i < fixed_ratios.size(); i++) {
    sqrt_fixed_ratios.push_back(sqrt(fixed_ratios[i]));
  }

  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      /// map origin image
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      int idx = 0;
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
              output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                   0] =
                  std::max((center_x_temp - box_width_ratio / 2.) / img_width,
                           0.);
              output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                   1] =
                  std::max((center_y_temp - box_height_ratio / 2.) / img_height,
                           0.);
              output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                   2] =
                  std::min((center_x_temp + box_width_ratio / 2.) / img_width,
                           1.);
              output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                   3] =
                  std::min((center_y_temp + box_height_ratio / 2.) / img_height,
                           1.);
              idx++;
            }
          }
        }
      }
    }
  }
  if (clip) {
    math::Transform trans;
    ClipFunctor<float> clip_func;
    trans(output_boxes_dataptr, output_boxes_dataptr + output_boxes->numel(),
          output_boxes_dataptr, clip_func);
  }

  if ((variances.size() != 4)) {
    LOG(kLOG_ERROR) << " variances.size() must be 4.";
  }

  int64_t box_num = feature_height * feature_width * num_priors;

  for (int i = 0; i < box_num; i++) {
    output_variances_dataptr[4 * i] = variances[0];
    output_variances_dataptr[4 * i + 1] = variances[1];
    output_variances_dataptr[4 * i + 2] = variances[2];
    output_variances_dataptr[4 * i + 3] = variances[3];
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
