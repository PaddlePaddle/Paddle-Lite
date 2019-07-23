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

#include "lite/arm/math/sroi_align.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void sroi_align_func(const std::vector<lite::Tensor*>& inputs,
                     int _channels,
                     int _height,
                     int _width,
                     int _pooled_height,
                     int _pooled_width,
                     float _spatial_scale,
                     lite::Tensor* output) {
  // Number of ROIs
  std::vector<lite::DDim> input_dims;
  for (auto p : inputs) {
    input_dims.push_back(p->dims());
  }

  lite::DDim output_dims = output->dims();
  int num_rois = input_dims[1][0];
  int batch_size = input_dims[0][0];
  int top_count = output->dims().count(0, output_dims.size());

  int in_0_c = input_dims[0][1];
  int in_0_h = input_dims[0][2];
  int in_0_w = input_dims[0][3];
  int in_1_c = input_dims[1][1];
  int in_1_h = input_dims[1][2];
  int in_1_w = input_dims[1][3];
  int out_0_h = output_dims[2];
  int out_0_w = output_dims[3];

  const float* bottom_data = inputs[0]->data<float>();
  const float* bottom_rois = inputs[1]->data<float>();
  float* top_data = output->mutable_data<float>();

// For each ROI R = [batch_index x1 y1 x2 y2]: roi align over R
#pragma omp parallel for
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = static_cast<int>(bottom_rois[0]);
    float roi_start_w = bottom_rois[1] * _spatial_scale;
    float roi_start_h = bottom_rois[2] * _spatial_scale;
    float roi_end_w = bottom_rois[3] * _spatial_scale;
    float roi_end_h = bottom_rois[4] * _spatial_scale;
    CHECK_GE(roi_batch_ind, 0) << "roi_batch_ind must be >= 0 \n";
    CHECK_LT(roi_batch_ind, batch_size)
        << "roi_batch_ind must be < batch_size \n";
    float roi_height =
        std::max(roi_end_h - roi_start_h + 1, static_cast<float>(0.));
    float roi_width =
        std::max(roi_end_w - roi_start_w + 1, static_cast<float>(0.));
    const float bin_size_h = static_cast<float>(roi_height) /
                             static_cast<float>(_pooled_height - 1.);
    const float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(_pooled_width - 1.);

    int offset_roi_batch_ind = roi_batch_ind * in_0_c * in_0_h * in_0_w;
    const float* batch_data = bottom_data + offset_roi_batch_ind;

    for (int c = 0; c < _channels; ++c) {
      for (int ph = 0; ph < _pooled_height; ++ph) {
        for (int pw = 0; pw < _pooled_width; ++pw) {
          float h = static_cast<float>(ph) * bin_size_h + roi_start_h;
          float w = static_cast<float>(pw) * bin_size_w + roi_start_w;

          int hstart = std::min(static_cast<int>(floor(h)), _height - 2);
          int wstart = std::min(static_cast<int>(floor(w)), _width - 2);

          bool is_empty(h < 0 || h >= _height || w < 0 || w >= _width);
          const int pool_index = ph * _pooled_width + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
          } else {
            float h_ratio = h - static_cast<float>(hstart);
            float w_ratio = w - static_cast<float>(wstart);
            int upleft = hstart * _width + wstart;
            int upright = upleft + 1;
            int downleft = upleft + _width;
            int downright = downleft + 1;
            top_data[pool_index] =
                batch_data[upleft] * (1. - h_ratio) * (1. - w_ratio) +
                batch_data[upright] * (1. - h_ratio) * w_ratio +
                batch_data[downleft] * h_ratio * (1. - w_ratio) +
                batch_data[downright] * h_ratio * w_ratio;
          }
        }
      }
      batch_data += in_0_h * in_0_w;
      top_data += out_0_h * out_0_w;
    }
    bottom_rois += in_1_c * in_1_h * in_1_w;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
