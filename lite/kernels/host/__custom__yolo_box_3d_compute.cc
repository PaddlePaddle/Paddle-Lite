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

#include "lite/kernels/host/__custom__yolo_box_3d_compute.h"

#include <cmath>
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

template <>
inline float Sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

inline int GetEntryIndex(int batch_idx,
                         int an_idx,
                         int hw_idx,
                         int an_num,
                         int an_stride,
                         int stride,
                         int entry) {
  return (batch_idx * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
inline void GetYoloBox(T* box,
                       const T* x_data,
                       const int* anchors,
                       int w_idx,
                       int h_idx,
                       int an_idx,
                       int h,
                       int w,
                       int downsample_ratio,
                       int index,
                       int stride,
                       int img_height,
                       int img_width,
                       T scale,
                       T bias) {
  box[0] = (w_idx + Sigmoid(x_data[index]) * scale + bias) / w;
  box[1] = (h_idx + Sigmoid(x_data[index + stride]) * scale + bias) / h;
  box[2] = std::exp(x_data[index + 2 * stride]) * anchors[2 * an_idx] /
           (downsample_ratio * w);
  box[3] = std::exp(x_data[index + 3 * stride]) * anchors[2 * an_idx + 1] /
           (downsample_ratio * h);
}

template <typename T>
inline void CalcDetectionBox(T* boxes_data,
                             T* box,
                             const int box_idx,
                             const int img_height,
                             const int img_width) {
  boxes_data[box_idx] = (box[0] - box[2] * 0.5f) * img_width;
  boxes_data[box_idx + 1] = (box[1] - box[3] * 0.5f) * img_height;
  boxes_data[box_idx + 2] = (box[0] + box[2] * 0.5f) * img_width;
  boxes_data[box_idx + 3] = (box[1] + box[3] * 0.5f) * img_height;
}

template <typename T>
std::vector<T> softmax(std::vector<T> input) {
  T total = 0;
  auto max_value = *std::max_element(input.begin(), input.end());
  for (auto x : input) {
    total += std::exp(x - max_value);
  }
  std::vector<T> result;
  for (auto x : input) {
    result.push_back(std::exp(x - max_value) / total);
  }
  return result;
}

template <typename T>
inline void CalcLabelScore(T* scores,
                           const T* input,
                           const int score_input_idx,
                           const int score_output_idx,
                           const int class_num,
                           const T conf,
                           const int stride) {
  std::vector<T> softmax_inputs;
  for (int i = 0; i < class_num; i++) {
    softmax_inputs.push_back(input[score_input_idx + i * stride]);
  }
  auto softmax_result = softmax(softmax_inputs);
  for (int i = 0; i < class_num; i++) {
    scores[score_output_idx + i] = conf * softmax_result[i];
  }
}

template <typename T>
void YoloBox3d(lite::Tensor* X,
               lite::Tensor* ImgSize,
               lite::Tensor* Boxes,
               lite::Tensor* Scores,
               lite::Tensor* Location,
               lite::Tensor* Dim,
               lite::Tensor* Alpha,
               std::vector<int> anchors,
               int class_num,
               T conf_thresh,
               int downsample_ratio,
               T scale,
               T bias) {
  const int n = X->dims()[0];
  const int h = X->dims()[2];
  const int w = X->dims()[3];
  const int b_num = Boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  const int stride = h * w;
  const int an_stride = (5 + 8 + class_num) * stride;

  auto anchors_data = anchors.data();
  // Input
  const T* x_data = X->data<T>();
  int* img_size_data = ImgSize->mutable_data<int>();
  // Output
  T* boxes_data = Boxes->mutable_data<T>();
  memset(boxes_data, 0, Boxes->numel() * sizeof(T));
  T* scores_data = Scores->mutable_data<T>();
  memset(scores_data, 0, Scores->numel() * sizeof(T));
  T* location_data = Location->mutable_data<T>();
  memset(location_data, 0, Location->numel() * sizeof(T));
  T* dim_data = Dim->mutable_data<T>();
  memset(dim_data, 0, Dim->numel() * sizeof(T));
  T* alpha_data = Alpha->mutable_data<T>();
  memset(alpha_data, 0, Alpha->numel() * sizeof(T));

  T box[4];
  for (int batch_idx = 0; batch_idx < n; batch_idx++) {
    int img_height = img_size_data[2 * batch_idx];
    int img_width = img_size_data[2 * batch_idx + 1];

    for (int an_idx = 0; an_idx < an_num; an_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          // Calc boxes output
          int box_input_idx = GetEntryIndex(batch_idx,
                                            an_idx,
                                            h_idx * w + w_idx,
                                            an_num,
                                            an_stride,
                                            stride,
                                            0);
          GetYoloBox(box,
                     x_data,
                     anchors_data,
                     w_idx,
                     h_idx,
                     an_idx,
                     h,
                     w,
                     downsample_ratio,
                     box_input_idx,
                     stride,
                     img_height,
                     img_width,
                     scale,
                     bias);
          int box_output_idx =
              (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 4;
          CalcDetectionBox(
              boxes_data, box, box_output_idx, img_height, img_width);

          // Calc score output
          int obj_idx = GetEntryIndex(batch_idx,
                                      an_idx,
                                      h_idx * w + w_idx,
                                      an_num,
                                      an_stride,
                                      stride,
                                      4);
          T conf = Sigmoid(x_data[obj_idx]);
          int score_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              an_num,
                                              an_stride,
                                              stride,
                                              13);
          int score_output_idx =
              (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) *
              class_num;
          CalcLabelScore(scores_data,
                         x_data,
                         score_input_idx,
                         score_output_idx,
                         class_num,
                         conf,
                         stride);

          // Calc location output
          int location_input_idx = GetEntryIndex(batch_idx,
                                                 an_idx,
                                                 h_idx * w + w_idx,
                                                 an_num,
                                                 an_stride,
                                                 stride,
                                                 5);
          int location_output_idx =
              (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 3;
          location_data[location_output_idx] =
              (w_idx + Sigmoid(x_data[location_input_idx]) * scale + bias) *
              img_width / w;
          location_data[location_output_idx + 1] =
              (h_idx + Sigmoid(x_data[location_input_idx + stride]) * scale +
               bias) *
              img_height / h;
          location_data[location_output_idx + 2] =
              x_data[location_input_idx + 2 * stride];

          // Calc dim output
          int dim_input_idx = GetEntryIndex(batch_idx,
                                            an_idx,
                                            h_idx * w + w_idx,
                                            an_num,
                                            an_stride,
                                            stride,
                                            8);
          int dim_output_idx =
              (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 3;
          dim_data[dim_output_idx] = x_data[dim_input_idx];
          dim_data[dim_output_idx + 1] = x_data[dim_input_idx + stride];
          dim_data[dim_output_idx + 2] = x_data[dim_input_idx + 2 * stride];

          // Calc alpha output
          int alpha_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              an_num,
                                              an_stride,
                                              stride,
                                              11);
          int alpha_output_idx =
              (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 2;
          alpha_data[alpha_output_idx] = x_data[alpha_input_idx];
          alpha_data[alpha_output_idx + 1] = x_data[alpha_input_idx + stride];
        }
      }
    }
  }
}

template <typename T, TargetType TType, PrecisionType PType>
void CustomYoloBox3dCompute<T, TType, PType>::Run() {
  auto& param = this->template Param<operators::CustomYoloBox3dParam>();
  auto* X = param.X;
  auto* ImgSize = param.ImgSize;
  auto* Boxes = param.Boxes;
  auto* Scores = param.Scores;
  auto* Location = param.Location;
  auto* Dim = param.Dim;
  auto* Alpha = param.Alpha;

  std::vector<int> anchors = param.anchors;
  int class_num = param.class_num;
  T conf_thresh = static_cast<T>(param.conf_thresh);
  int downsample_ratio = param.downsample_ratio;
  T scale_x_y = static_cast<T>(param.scale_x_y);
  T bias = static_cast<T>(-0.5 * (scale_x_y - 1.f));
  Boxes->clear();
  Scores->clear();
  Location->clear();
  Dim->clear();
  Alpha->clear();
  YoloBox3d<T>(X,
               ImgSize,
               Boxes,
               Scores,
               Location,
               Dim,
               Alpha,
               anchors,
               class_num,
               conf_thresh,
               downsample_ratio,
               scale_x_y,
               bias);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using yolo_box_3d_fp32 =
    paddle::lite::kernels::host::CustomYoloBox3dCompute<float,
                                                        TARGET(kHost),
                                                        PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(
    __custom__yolo_box_3d, kHost, kFloat, kNCHW, yolo_box_3d_fp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Location", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Dim", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Alpha", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
