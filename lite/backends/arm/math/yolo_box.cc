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

#include "lite/backends/arm/math/yolo_box.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

namespace {
inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

inline void get_yolo_box(float* box,
                         const float* x,
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
                         float scale,
                         float bias) {
  box[0] = (i + sigmoid(x[index]) * scale + bias) * img_width / grid_size;
  box[1] =
      (j + sigmoid(x[index + stride]) * scale + bias) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

inline int get_entry_index(int batch,
                           int an_idx,
                           int hw_idx,
                           int an_num,
                           int an_stride,
                           int stride,
                           int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

inline void calc_detection_box(float* boxes,
                               float* box,
                               const int box_idx,
                               const int img_height,
                               const int img_width,
                               bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  if (clip_bbox) {
    boxes[box_idx] =
        boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<float>(0);
    boxes[box_idx + 1] =
        boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<float>(0);
    boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                             ? boxes[box_idx + 2]
                             : static_cast<float>(img_width - 1);
    boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                             ? boxes[box_idx + 3]
                             : static_cast<float>(img_height - 1);
  }
}

inline void calc_label_score(float* scores,
                             const float* input,
                             const int label_idx,
                             const int score_idx,
                             const int class_num,
                             const float conf,
                             const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid(input[label_idx + i * stride]);
  }
}
}  // namespace

void yolobox(lite::Tensor* X,
             lite::Tensor* ImgSize,
             lite::Tensor* Boxes,
             lite::Tensor* Scores,
             std::vector<int> anchors,
             int class_num,
             float conf_thresh,
             int downsample_ratio,
             bool clip_bbox,
             float scale,
             float bias) {
  const int n = X->dims()[0];
  const int h = X->dims()[2];
  const int w = X->dims()[3];
  const int b_num = Boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int X_size = downsample_ratio * h;

  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;

  auto anchors_data = anchors.data();

  const float* X_data = X->data<float>();
  int* ImgSize_data = ImgSize->mutable_data<int>();

  float* Boxes_data = Boxes->mutable_data<float>();
  memset(Boxes_data, 0, Boxes->numel() * sizeof(float));

  float* Scores_data = Scores->mutable_data<float>();
  memset(Scores_data, 0, Scores->numel() * sizeof(float));

  float box[4];
  for (int i = 0; i < n; i++) {
    int img_height = ImgSize_data[2 * i];
    int img_width = ImgSize_data[2 * i + 1];

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 4);
          float conf = sigmoid(X_data[obj_idx]);
          if (conf < conf_thresh) {
            continue;
          }

          int box_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 0);
          get_yolo_box(box,
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
          calc_detection_box(
              Boxes_data, box, box_idx, img_height, img_width, clip_bbox);

          int label_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 5);
          int score_idx = (i * b_num + j * stride + k * w + l) * class_num;
          calc_label_score(Scores_data,
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
}  // namespace arm
}  // namespace lite
}  // namespace paddle
