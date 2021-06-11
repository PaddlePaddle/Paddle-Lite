// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

const double kBBoxClipDefault = std::log(1000.0 / 16.0);

template <typename T>
void BoxCoder(Tensor *all_anchors,
              Tensor *bbox_deltas,
              Tensor *variances,
              Tensor *proposals,
              const bool pixel_offset = true) {
  T *proposals_data = proposals->mutable_data<T>();

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<T>();
  auto *anchor_data = all_anchors->data<T>();
  const T *variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + offset;
    T anchor_height =
        anchor_data[i * len + 3] - anchor_data[i * len + 1] + offset;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(variances_data[i * len + 2] *
                                            bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(variances_data[i * len + 3] *
                                             bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - offset;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - offset;
  }
}

template <typename T>
void AppendTensor(Tensor *dst, int64_t offset, const Tensor &src) {
  auto *out_data = dst->mutable_data<T>();
  auto *to_add_data = src.data<T>();
  std::memcpy(out_data + offset, to_add_data, src.numel() * sizeof(T));
}

template <typename T>
void ClipTiledBoxes(const Tensor &im_info,
                    const Tensor &input_boxes,
                    Tensor *out,
                    const bool is_scale = true,
                    const bool pixel_offset = true) {
  T *out_data = out->mutable_data<T>();
  const T *im_info_data = im_info.data<T>();
  const T *input_boxes_data = input_boxes.data<T>();
  T offset = pixel_offset ? static_cast<T>(1) : 0;
  T zero(0);
  T im_w =
      is_scale ? round(im_info_data[1] / im_info_data[2]) : im_info_data[1];
  T im_h =
      is_scale ? round(im_info_data[0] / im_info_data[2]) : im_info_data[0];
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if (i % 4 == 0) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_w - offset), zero);
    } else if (i % 4 == 1) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_h - offset), zero);
    } else if (i % 4 == 2) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_w - offset), zero);
    } else {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_h - offset), zero);
    }
  }
}

template <typename T>
void FilterBoxes(Tensor *boxes,
                 float min_size,
                 const Tensor &im_info,
                 bool is_scale,
                 Tensor *keep,
                 bool pixel_offset = true) {
  const T *im_info_data = im_info.data<T>();
  const T *boxes_data = boxes->data<T>();
  keep->Resize({boxes->dims()[0]});
  min_size = std::max(min_size, 1.0f);
  int *keep_data = keep->mutable_data<int>();
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + offset;
    T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + offset;
    if (pixel_offset) {
      T x_ctr = boxes_data[4 * i] + ws / 2;
      T y_ctr = boxes_data[4 * i + 1] + hs / 2;

      if (is_scale) {
        ws = (boxes_data[4 * i + 2] - boxes_data[4 * i]) / im_info_data[2] + 1;
        hs = (boxes_data[4 * i + 3] - boxes_data[4 * i + 1]) / im_info_data[2] +
             1;
      }
      if (ws >= min_size && hs >= min_size && x_ctr <= im_info_data[1] &&
          y_ctr <= im_info_data[0]) {
        keep_data[keep_len++] = i;
      }
    } else {
      if (ws >= min_size && hs >= min_size) {
        keep_data[keep_len++] = i;
      }
    }
  }
  keep->Resize({static_cast<int64_t>(keep_len)});
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
