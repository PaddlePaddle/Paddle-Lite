// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/__custom__yolo_det_compute.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
inline T Sigmoid(T x) {
  return (T)1. / ((T)1. + std::exp(-x));
}

float CalConfScore(const float* data, int cls_idx) {
  const float objectness = Sigmoid(data[4]);
  const float confidence = Sigmoid(data[cls_idx + 5]);

  return objectness * confidence;
}

void NMS(const std::vector<std::pair<float, int>>& score_index,
         const std::vector<float>& bboxes,
         std::vector<int>& indices,  // NOLINT
         float nms_threshold) {
  auto overlap1D = [](
      float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };
  auto computeIoU = [&overlap1D](
      const std::vector<float>& bbox, int idx1, int idx2) -> float {
    float overlapX = overlap1D(
        bbox[idx1 * 5], bbox[idx1 * 5 + 2], bbox[idx2 * 5], bbox[idx2 * 5 + 2]);
    float overlapY = overlap1D(bbox[idx1 * 5 + 1],
                               bbox[idx1 * 5 + 3],
                               bbox[idx2 * 5 + 1],
                               bbox[idx2 * 5 + 3]);
    float area1 = (bbox[idx1 * 5 + 2] - bbox[idx1 * 5]) *
                  (bbox[idx1 * 5 + 3] - bbox[idx1 * 5 + 1]);
    float area2 = (bbox[idx2 * 5 + 2] - bbox[idx2 * 5]) *
                  (bbox[idx2 * 5 + 3] - bbox[idx2 * 5 + 1]);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };
  for (auto i : score_index) {
    int idx = i.second;
    bool keep = true;
    for (unsigned int k = 0; k < indices.size(); k++) {
      if (keep) {
        int keptIdx = indices[k];
        float overlap = computeIoU(bboxes, idx, keptIdx);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices.emplace_back(idx);
    }
  }
}

bool ForwardInClassNMS(const std::vector<lite::Tensor*>& inputs,
                       const int* img_size,
                       std::vector<float>& outputs,  // NOLINT
                       std::vector<int> anchors,
                       std::vector<int> downsample_ratios,
                       int class_num,
                       float conf_thresh,
                       int keep_top_k,
                       float nms_threshold) {
  CHECK_EQ(inputs[0]->dims()[0], 1) << "Only support batch_size is 1";
  bool ret = true;
  int input_num = inputs.size();
  int anchor_stride = anchors.size() / 2 / input_num;
  int data_offset = class_num + 5;
  int input_h = img_size[0];
  int input_w = img_size[1];

  for (int c = 0; c < class_num; c++) {
    std::vector<float> bboxes;
    std::vector<std::pair<float, int>> score_idx;
    int bbox_idx = 0;
    for (int i = 0; i < input_num; i++) {
      const int grid_h = inputs[i]->dims()[2];
      const int grid_w = inputs[i]->dims()[3];
      const float* input_data = inputs[i]->data<float>();
      for (int yid = 0; yid < grid_h; yid++) {
        for (int xid = 0; xid < grid_w; xid++) {
          for (int b = 0; b < anchor_stride; b++) {
            float score = CalConfScore(input_data, c);
            if (score >= conf_thresh) {
              // decode bbox
              const float pred_x = input_data[0];
              const float pred_y = input_data[1];
              const float pred_w = input_data[2];
              const float pred_h = input_data[3];
              float dec_x = (xid + Sigmoid(pred_x)) / grid_w;
              float dec_y = (yid + Sigmoid(pred_y)) / grid_h;
              float dec_w = exp(pred_w) *
                            anchors[i * anchor_stride * 2 + b * 2] / input_w;
              float dec_h = exp(pred_h) *
                            anchors[i * anchor_stride * 2 + b * 2 + 1] /
                            input_h;
              bboxes.emplace_back(dec_x - dec_w * 0.5);
              bboxes.emplace_back(dec_y - dec_h * 0.5);
              bboxes.emplace_back(dec_x + dec_w * 0.5);
              bboxes.emplace_back(dec_y + dec_h * 0.5);
              bboxes.emplace_back(score);
              score_idx.emplace_back(std::make_pair(score, bbox_idx));
              bbox_idx++;
            }
            input_data += data_offset;
          }
        }
      }
    }
    /* nms */
    std::sort(score_idx.begin(),
              score_idx.end(),
              [](std::pair<float, int>& pair1, std::pair<float, int>& pair2) {
                return pair1.first > pair2.first;
              });
    std::vector<int> indices;
    NMS(score_idx, bboxes, indices, nms_threshold);
    for (int k = 0; k < indices.size(); k++) {
      float xmin = std::min(std::max(bboxes[indices[k] * 5], 0.f), 1.f);
      float ymin = std::min(std::max(bboxes[indices[k] * 5 + 1], 0.f), 1.f);
      float xmax = std::min(std::max(bboxes[indices[k] * 5 + 2], 0.f), 1.f);
      float ymax = std::min(std::max(bboxes[indices[k] * 5 + 3], 0.f), 1.f);
      if (xmax <= xmin || ymax <= ymin) {
        continue;
      }
      outputs.emplace_back(c);
      outputs.emplace_back(bboxes[indices[k] * 5 + 4]);
      outputs.emplace_back(xmin * input_w);
      outputs.emplace_back(ymin * input_h);
      outputs.emplace_back(xmax * input_w);
      outputs.emplace_back(ymax * input_h);
    }
  }

  return ret;
}

template <typename T, TargetType TType, PrecisionType PType>
void CustomYoloDetCompute<T, TType, PType>::Run() {
  auto& param = this->template Param<operators::CustomYoloDetParam>();
  auto* X0 = param.X0;
  auto* X1 = param.X1;
  auto* X2 = param.X2;
  auto* ImgSize = param.ImgSize;
  auto* Output = param.Output;

  std::vector<int> anchors = param.anchors;
  std::vector<int> downsample_ratios = param.downsample_ratios;
  int class_num = param.class_num;
  float conf_thresh = param.conf_thresh;
  int keep_top_k = param.keep_top_k;
  float nms_threshold = param.nms_threshold;
  int* img_size = ImgSize->template mutable_data<int>();
  Output->clear();

  std::vector<lite::Tensor*> inputs{X0, X1, X2};
  std::vector<float> outputs;
  ForwardInClassNMS(inputs,
                    img_size,
                    outputs,
                    anchors,
                    downsample_ratios,
                    class_num,
                    conf_thresh,
                    keep_top_k,
                    nms_threshold);
  int output_num = outputs.size() / 6;
  Output->Resize({output_num, 6});
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using yolo_det_fp32 =
    paddle::lite::kernels::host::CustomYoloDetCompute<float,
                                                      TARGET(kHost),
                                                      PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    __custom__yolo_det, kHost, kFloat, kNCHW, yolo_det_fp32, def)
    .BindInput("X0", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("X1", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("X2", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
