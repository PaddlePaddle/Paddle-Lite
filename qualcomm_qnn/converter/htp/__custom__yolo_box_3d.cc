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

#include "driver/qualcomm_qnn/converter/htp/__custom__yolo_box_3d.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

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
std::vector<T> Softmax(std::vector<T> input) {
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
  auto softmax_result = Softmax(softmax_inputs);
  for (int i = 0; i < class_num; i++) {
    scores[score_output_idx + i] = conf * softmax_result[i];
  }
}

template <typename T>
inline void CustomYoloBox3dKernel(T* out_boxes_data,
                                  T* out_scores_data,
                                  T* out_location_data,
                                  T* out_dim_data,
                                  T* out_alpha_data,
                                  const T* input_data,
                                  const int32_t* img_size_data,
                                  const int32_t* anchors_data,
                                  const int32_t class_num_data,
                                  const float conf_thresh_data,
                                  const int32_t downsample_ratio_data,
                                  const float scale_x_y_data,
                                  const int32_t batch_size,
                                  const int32_t h,
                                  const int32_t w,
                                  const int32_t box_num,
                                  const int32_t anchor_num) {
  float bias = -0.5 * (scale_x_y_data - 1.f);
  int stride = h * w;
  int an_stride = (5 + 8 + class_num_data) * stride;
  float box[4];
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int img_height = img_size_data[2 * batch_idx];
    int img_width = img_size_data[2 * batch_idx + 1];

    for (int an_idx = 0; an_idx < anchor_num; an_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          // Calc boxes output
          int box_input_idx = GetEntryIndex(batch_idx,
                                            an_idx,
                                            h_idx * w + w_idx,
                                            anchor_num,
                                            an_stride,
                                            stride,
                                            0);
          GetYoloBox(box,
                     input_data,
                     anchors_data,
                     w_idx,
                     h_idx,
                     an_idx,
                     h,
                     w,
                     downsample_ratio_data,
                     box_input_idx,
                     stride,
                     img_height,
                     img_width,
                     scale_x_y_data,
                     bias);
          int box_output_idx =
              (batch_idx * box_num + an_idx * stride + h_idx * w + w_idx) * 4;
          CalcDetectionBox(
              out_boxes_data, box, box_output_idx, img_height, img_width);

          // Calc score output
          int obj_idx = GetEntryIndex(batch_idx,
                                      an_idx,
                                      h_idx * w + w_idx,
                                      anchor_num,
                                      an_stride,
                                      stride,
                                      4);
          float conf = Sigmoid(input_data[obj_idx]);
          int score_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              anchor_num,
                                              an_stride,
                                              stride,
                                              13);
          int score_output_idx =
              (batch_idx * box_num + an_idx * stride + h_idx * w + w_idx) *
              class_num_data;
          CalcLabelScore(out_scores_data,
                         input_data,
                         score_input_idx,
                         score_output_idx,
                         class_num_data,
                         conf,
                         stride);

          // Calc location output
          int location_input_idx = GetEntryIndex(batch_idx,
                                                 an_idx,
                                                 h_idx * w + w_idx,
                                                 anchor_num,
                                                 an_stride,
                                                 stride,
                                                 5);
          int location_output_idx =
              (batch_idx * box_num + an_idx * stride + h_idx * w + w_idx) * 3;
          out_location_data[location_output_idx] =
              (w_idx +
               Sigmoid(input_data[location_input_idx]) * scale_x_y_data +
               bias) *
              img_width / w;
          out_location_data[location_output_idx + 1] =
              (h_idx +
               Sigmoid(input_data[location_input_idx + stride]) *
                   scale_x_y_data +
               bias) *
              img_height / h;
          out_location_data[location_output_idx + 2] =
              input_data[location_input_idx + 2 * stride];

          // Calc dim output
          int dim_input_idx = GetEntryIndex(batch_idx,
                                            an_idx,
                                            h_idx * w + w_idx,
                                            anchor_num,
                                            an_stride,
                                            stride,
                                            8);
          int dim_output_idx =
              (batch_idx * box_num + an_idx * stride + h_idx * w + w_idx) * 3;
          out_dim_data[dim_output_idx] = input_data[dim_input_idx];
          out_dim_data[dim_output_idx + 1] = input_data[dim_input_idx + stride];
          out_dim_data[dim_output_idx + 2] =
              input_data[dim_input_idx + 2 * stride];

          // Calc alpha output
          int alpha_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              anchor_num,
                                              an_stride,
                                              stride,
                                              11);
          int alpha_output_idx =
              (batch_idx * box_num + an_idx * stride + h_idx * w + w_idx) * 2;
          out_alpha_data[alpha_output_idx] = input_data[alpha_input_idx];
          out_alpha_data[alpha_output_idx + 1] =
              input_data[alpha_input_idx + stride];
        }
      }
    }
  }
}

template <typename TensorType>
int CustomYoloBox3dImpl(TensorType& out_boxes,     // NOLINT
                        TensorType& out_scores,    // NOLINT
                        TensorType& out_location,  // NOLINT
                        TensorType& out_dim,       // NOLINT
                        TensorType& out_alpha,     // NOLINT
                        const TensorType& input,
                        const TensorType& img_size,
                        const Tensor& anchors,
                        const Tensor& class_num,
                        const Tensor& conf_thresh,
                        const Tensor& downsample_ratio,
                        const Tensor& scale_x_y) {
  infolog("CustomYoloBox3dImpl execute... ");
  // Input
  auto* input_data = reinterpret_cast<const float*>(input.raw_data_const());
  auto* img_size_data = reinterpret_cast<const int*>(img_size.raw_data_const());
  // Output
  auto* boxes_data = reinterpret_cast<float*>(out_boxes.raw_data());
  auto* scores_data = reinterpret_cast<float*>(out_scores.raw_data());
  auto* location_data = reinterpret_cast<float*>(out_location.raw_data());
  auto* dim_data = reinterpret_cast<float*>(out_dim.raw_data());
  auto* alpha_data = reinterpret_cast<float*>(out_alpha.raw_data());
  // Attr
  auto* anchors_data = reinterpret_cast<const int*>(anchors.raw_data_const());
  int class_num_data = class_num(0, 0, 0, 0);
  float conf_thresh_data = conf_thresh(0, 0, 0, 0);
  int downsample_ratio_data = downsample_ratio(0, 0, 0, 0);
  float scale_x_y_data = scale_x_y(0, 0, 0, 0);
  const int batch_size = input.dim(0);
  const int h = input.dim(2);
  const int w = input.dim(3);
  const int box_num = out_boxes.dim(1);
  const int anchor_num = anchors.dim(3) / 2;
  CustomYoloBox3dKernel<float>(boxes_data,
                               scores_data,
                               location_data,
                               dim_data,
                               alpha_data,
                               input_data,
                               img_size_data,
                               anchors_data,
                               class_num_data,
                               conf_thresh_data,
                               downsample_ratio_data,
                               scale_x_y_data,
                               batch_size,
                               h,
                               w,
                               box_num,
                               anchor_num);
  return GraphStatus::Success;
}

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter

using namespace nnadapter::qualcomm_qnn::htp;  // NOLINT
using namespace hnnx;                          // NOLINT

const char* op_name_yolobox3d = "CustomYoloBox3d";

DEF_PACKAGE_OP(CustomYoloBox3dImpl<Tensor>, op_name_yolobox3d)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((CustomYoloBox3dImpl<PlainFloatTensor>),
                                  op_name_yolobox3d,
                                  "SNAIL",
                                  Flags::RESOURCE_HVX)
