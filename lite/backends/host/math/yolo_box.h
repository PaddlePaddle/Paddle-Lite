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

#pragma once
#include <cmath>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
inline T Sigmoid(T x) {
  return (T)1. / ((T)1. + std::exp(-x));
}

template <>
inline float Sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

template <typename T>
inline void GetYoloBox(T* box,
                       const T* x,
                       const int* anchors,
                       int i,
                       int j,
                       int an_idx,
                       int grid_size,
                       int input_size,
                       int index,
                       int stride,
                       int img_height,
                       int img_width,
                       T scale,
                       T bias) {
  box[0] = (i + Sigmoid(x[index]) * scale + bias) * img_width / grid_size;
  box[1] =
      (j + Sigmoid(x[index + stride]) * scale + bias) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

inline int GetEntryIndex(int batch,
                         int an_idx,
                         int hw_idx,
                         int an_num,
                         int an_stride,
                         int stride,
                         int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
inline void CalcDetectionBox(T* boxes,
                             T* box,
                             const int box_idx,
                             const int img_height,
                             const int img_width,
                             bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  if (clip_bbox) {
    boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<T>(0);
    boxes[box_idx + 1] =
        boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<T>(0);
    boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                             ? boxes[box_idx + 2]
                             : static_cast<T>(img_width - 1);
    boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                             ? boxes[box_idx + 3]
                             : static_cast<T>(img_height - 1);
  }
}

template <typename T>
inline void CalcLabelScore(T* scores,
                           const T* input,
                           const int label_idx,
                           const int score_idx,
                           const int class_num,
                           const T conf,
                           const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * Sigmoid(input[label_idx + i * stride]);
  }
}

template <typename T>
void YoloBox(lite::Tensor* X,
             lite::Tensor* ImgSize,
             lite::Tensor* Boxes,
             lite::Tensor* Scores,
             std::vector<int> anchors,
             int class_num,
             T conf_thresh,
             int downsample_ratio,
             bool clip_bbox,
             T scale,
             T bias) {
  const int n = X->dims()[0];
  const int h = X->dims()[2];
  const int w = X->dims()[3];
  const int b_num = Boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int X_size = downsample_ratio * h;

  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;

  auto anchors_data = anchors.data();

  const T* X_data = X->data<T>();
  int* ImgSize_data = ImgSize->mutable_data<int>();

  T* Boxes_data = Boxes->mutable_data<T>();
  memset(Boxes_data, 0, Boxes->numel() * sizeof(T));

  T* Scores_data = Scores->mutable_data<T>();
  memset(Scores_data, 0, Scores->numel() * sizeof(T));

  T box[4];
  for (int i = 0; i < n; i++) {
    int img_height = ImgSize_data[2 * i];
    int img_width = ImgSize_data[2 * i + 1];

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx =
              GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 4);
          T conf = Sigmoid(X_data[obj_idx]);
          if (conf < conf_thresh) {
            continue;
          }

          int box_idx =
              GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 0);
          GetYoloBox(box,
                     X_data,
                     anchors_data,
                     l,
                     k,
                     j,
                     h,
                     X_size,
                     box_idx,
                     stride,
                     img_height,
                     img_width,
                     scale,
                     bias);
          box_idx = (i * b_num + j * stride + k * w + l) * 4;
          CalcDetectionBox(
              Boxes_data, box, box_idx, img_height, img_width, clip_bbox);

          int label_idx =
              GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 5);
          int score_idx = (i * b_num + j * stride + k * w + l) * class_num;
          CalcLabelScore(Scores_data,
                         X_data,
                         label_idx,
                         score_idx,
                         class_num,
                         conf,
                         stride);
        }
      }
    }
  }
}
}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
