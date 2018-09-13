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

#ifdef PRIORBOX_OP
#pragma once

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
void PriorBoxCompute(const PriorBoxParam<CPU> &param) {
  const auto *input_ = param.Input();
  const auto &input_dims = input_->dims();

  const auto *input_image = param.InputImage();
  const auto &input_image_dims = input_image->dims();

  const auto &min_sizes = param.MinSizes();
  const auto &max_sizes = param.MaxSizes();
  const auto &variances = param.Variances();
  const auto &input_aspect_ratio = param.AspectRatios();
  const bool &flip = param.Flip();
  const bool &clip = param.Clip();
  const float &step_w = param.StepW();
  const float &step_h = param.StepH();
  const float &offset = param.Offset();

  Tensor *output_boxes = param.OutputBoxes();
  auto output_boxes_dataptr = output_boxes->mutable_data<float>();
  Tensor *output_variances = param.OutputVariances();
  auto output_variances_dataptr = output_variances->mutable_data<float>();

  std::vector<float> aspect_ratios;
  ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

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

  int num_priors = aspect_ratios.size() * min_sizes.size();
  if (!max_sizes.empty()) {
    num_priors += max_sizes.size();
  }

  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      /// map origin image
      float center_x = (w + offset) * step_width;
      float center_y = (h + offset) * step_height;
      float box_width, box_height;
      int idx = 0;
      for (size_t s = 0; s < min_sizes.size(); ++s) {
        auto min_size = min_sizes[s];
        if (param.MinMaxAspectRatiosOrder()) {
          box_width = box_height = min_size / 2.;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 0] =
              (center_x - box_width) / img_width;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 1] =
              (center_y - box_height) / img_height;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 2] =
              (center_x + box_width) / img_width;
          output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 + 3] =
              (center_y + box_height) / img_height;
          idx++;

          if (max_sizes.size() > 0) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
          }

          // priors with different aspect ratios
          for (float ar : aspect_ratios) {
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            /// box_width/2 , / img_width 为了得到feature map 相对于
            /// 原图的归一化位置的比例。
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
          }

        } else {
          // priors with different aspect ratios
          for (float ar : aspect_ratios) {
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            /// box_width/2 , / img_width 为了得到feature map 相对于
            /// 原图的归一化位置的比例。
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
          }
          if (!max_sizes.empty()) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 0] = (center_x - box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 1] = (center_y - box_height) / img_height;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 2] = (center_x + box_width) / img_width;
            output_boxes_dataptr[h * stride0 + w * stride1 + idx * stride2 +
                                 3] = (center_y + box_height) / img_height;
            idx++;
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
