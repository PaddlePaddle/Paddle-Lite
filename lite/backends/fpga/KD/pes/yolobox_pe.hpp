/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

inline void GetYoloBox(float* box,
                       const float* x,
                       const int* anchors,
                       int w,
                       int h,
                       int an_idx,
                       int grid_size,
                       int input_size,
                       int index,
                       int img_height,
                       int img_width) {
  box[0] = (w + sigmoid(x[index])) * img_width * 1.0f / grid_size;
  box[1] = (h + sigmoid(x[index + 1])) * img_height * 1.0f / grid_size;
  box[2] = std::exp(x[index + 2]) * anchors[2 * an_idx] * img_width * 1.0f /
           input_size;
  box[3] = std::exp(x[index + 3]) * anchors[2 * an_idx + 1] * img_height *
           1.0f / input_size;
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

inline void CalcDetectionBox(float* boxes,
                             float* box,
                             const int box_idx,
                             const int img_height,
                             const int img_width) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : 0;
  boxes[box_idx + 1] = boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : 0;
  boxes[box_idx + 2] =
      boxes[box_idx + 2] < img_width - 1 ? boxes[box_idx + 2] : (img_width - 1);
  boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1 ? boxes[box_idx + 3]
                                                           : (img_height - 1);
}

inline void CalcLabelScore(float* scores,
                           const float* input,
                           const int label_idx,
                           const int score_idx,
                           const int class_num,
                           const float conf) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid(input[label_idx + i]);
  }
}

class YoloBoxPE : public PE {
 public:
  bool init() {
    param_.outputBoxes->setAligned(false);
    param_.outputScores->setAligned(false);
    param_.outputBoxes->setDataLocation(CPU);
    param_.outputScores->setDataLocation(CPU);
    return true;
  }

  bool dispatch() {
    auto* input = param_.input;
    auto* imgsize = param_.imgSize;
    auto* boxes = param_.outputBoxes;
    auto* scores = param_.outputScores;
    auto anchors = param_.anchors;
    int class_num = param_.classNum;
    float conf_thresh = param_.confThresh;
    int downsample_ratio = param_.downsampleRatio;

    const int num = input->shape().num();
    const int height = input->shape().height();
    const int width = input->shape().width();
    const int box_num = boxes->shape().channel();
    const int an_num = anchors.size() / 2;
    int input_size = downsample_ratio * height;

    const int stride = height * width;
    const int an_stride = (class_num + 5) * stride;

    Tensor anchors_;
    Shape anchors_shape(N, {an_num * 2});
    auto anchors_data = anchors_.mutableData<int32_t>(INT32, anchors_shape);
    std::copy(anchors.begin(), anchors.end(), anchors_data);

    input->syncToCPU();
    Tensor input_float;
    input_float.setDataLocation(CPU);
    float* input_data = input_float.mutableData<float>(FP32, input->shape());
    input_float.copyFrom(input);
    input_float.setAligned(input->aligned());
    input_float.unalignImage();
    input_float.setAligned(false);

    Tensor boxes_float;
    Tensor scores_float;

    boxes_float.setDataLocation(CPU);
    float* boxes_float_data =
        boxes_float.mutableData<float>(FP32, boxes->shape());
    memset(boxes_float_data, 0, boxes->shape().numel() * sizeof(float));

    scores_float.setDataLocation(CPU);
    float* scores_float_data =
        scores_float.mutableData<float>(FP32, scores->shape());
    memset(scores_float_data, 0, scores->shape().numel() * sizeof(float));

    float box[4];
    int img_height = 0;
    int img_width = 0;
    if (imgsize->dataType() == FP32) {
      float* imgsize_data = imgsize->mutableData<float>();
      img_height = imgsize_data[0];
      img_width = imgsize_data[1];
    } else {
      int32_t* imgsize_data = imgsize->mutableData<int32_t>();
      img_height = imgsize_data[0];
      img_width = imgsize_data[1];
    }

    int channel = input_float.shape().channel();
    int count = 0;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int n = 0; n < an_num; n++) {
          int obj_idx =
              channel * width * h + channel * w + n * (5 + class_num) + 4;
          float conf = sigmoid(input_data[obj_idx]);
          if (conf < conf_thresh) {
            count++;
            continue;
          }

          int box_idx =
              channel * width * h + channel * w + n * (5 + class_num) + 0;
          GetYoloBox(box,
                     input_data,
                     anchors_data,
                     w,
                     h,
                     n,
                     height,
                     input_size,
                     box_idx,
                     img_height,
                     img_width);

          box_idx = h * an_num * 4 * width + an_num * 4 * w + n * 4;
          CalcDetectionBox(
              boxes_float_data, box, box_idx, img_height, img_width);

          int label_idx =
              channel * width * h + channel * w + n * (5 + class_num) + 5;
          int score_idx = h * an_num * class_num * width +
                          an_num * class_num * w + n * class_num;
          CalcLabelScore(scores_float_data,
                         input_data,
                         label_idx,
                         score_idx,
                         class_num,
                         conf);
        }
      }
    }

    boxes->copyFrom(&boxes_float);
    scores->copyFrom(&scores_float);
  }

  void apply() {}

  YoloBoxParam& param() { return param_; }

 private:
  YoloBoxParam param_;
};
}  // namespace zynqmp
}  // namespace paddle
