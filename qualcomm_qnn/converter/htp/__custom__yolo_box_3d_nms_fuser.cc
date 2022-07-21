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

#include "driver/qualcomm_qnn/converter/htp/__custom__yolo_box_3d_nms_fuser.h"

#include "driver/qualcomm_qnn/converter/htp/__custom__yolo_box_3d.h"
#include "driver/qualcomm_qnn/converter/htp/multiclass_nms.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

template <typename T>
int ConcatKernel(const std::vector<T*>& input_datas,
                 const std::vector<std::vector<int32_t>>& input_shapes,
                 const int axis,
                 T* output_data) {
  size_t num = input_datas.size();
  auto input_dim_0 = input_shapes[0];
  int64_t input_size = 1;
  int64_t num_cancats = 1;
  for (int i = axis + 1; i < input_dim_0.size(); i++) {
    input_size *= input_dim_0[i];
  }
  for (int i = 0; i < axis; i++) {
    num_cancats *= input_dim_0[i];
  }

  std::vector<int32_t> output_dims = input_dim_0;
  for (uint32_t i = 1; i < num; i++) {
    for (uint32_t j = 0; j < output_dims.size(); j++) {
      if (j == axis) {
        output_dims[j] += input_shapes[i][j];
      }
    }
  }
  auto* dst_ptr = output_data;
  const int out_axis = output_dims[axis];
  int64_t offset_axis = 0;
  int64_t out_sum = out_axis * input_size;
  for (int n = 0; n < num; n++) {
    auto dims = input_shapes[n];
    auto* src_ptr = input_datas[n];
    int64_t in_axis = dims[axis];
    auto* dout_ptr = dst_ptr + offset_axis * input_size;
    int64_t in_sum = in_axis * input_size;
    for (int i = 0; i < num_cancats; i++) {
      std::memcpy(dout_ptr, src_ptr, sizeof(T) * in_sum);
      dout_ptr += out_sum;
      src_ptr += in_sum;
    }
    offset_axis += in_axis;
  }
  return 0;
}

// template <typename T>
// void TransposeKernel(T* input,
//                     const std::vector<int32_t>& input_shape,
//                     T* output,
//                     const std::vector<int32_t>& output_shape,
//                     const std::vector<int32_t>& orders) {
//   int input_size = input_shape.size();
//   int count = 1;
//   for (int32_t i = 0; i < input_size; i++) {
//     count *= input_shape[i];
//     infolog("input_shape-%d ", input_shape[i]);
//   }

//   std::vector<int> old_steps(
//       {static_cast<int>(input_shape[1] * input_shape[2] * input_shape[3]),
//        static_cast<int>(input_shape[2] * input_shape[3]),
//        static_cast<int>(input_shape[3]),
//        1});
//   std::vector<int> new_steps(
//       {static_cast<int>(output_shape[1] * output_shape[2] * output_shape[3]),
//        static_cast<int>(output_shape[2] * output_shape[3]),
//        static_cast<int>(output_shape[3]),
//        1});
//   infolog("AAAAAAAAAAAAAAAAA... ");
//   for (int i = 0; i < count; ++i) {
//     int old_idx = 0;
//     int idx = i;
//     for (int j = 0; j < input_size; ++j) {
//       int order = orders[j];
//       infolog("new_steps[j]-%d ", new_steps[j]);
//       infolog("old_steps[order]-%d ", old_steps[order]);
//       old_idx += (idx / new_steps[j]) * old_steps[order];
//       idx %= new_steps[j];
//     }
//     output[i] = input[old_idx];
//   }
// }

// A naive implementation of transpose operation
bool IsIdentityPermutation(const std::vector<int32_t>& permutation) {
  auto rank = permutation.size();
  for (size_t i = 0; i < rank; i++) {
    if (permutation[i] != i) return false;
  }
  return true;
}

int64_t ProductionOfDimensions(const int32_t* input_dimensions_data,
                               uint32_t input_dimensions_count) {
  int64_t production = 1;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions_data[i];
    production *= dimension;
  }
  return production;
}

int64_t ProductionOfDimensions(const std::vector<int32_t>& input_dimensions) {
  return !input_dimensions.empty()
             ? ProductionOfDimensions(&input_dimensions[0],
                                      input_dimensions.size())
             : 1;
}

template <typename T>
void TransposeKernel(const T* input,
                     T* output,
                     const std::vector<int32_t>& input_dimensions,
                     const std::vector<int32_t>& permutation) {
  auto permutation_count = permutation.size();
  std::vector<int32_t> output_dimensions(permutation_count);
  for (size_t i = 0; i < permutation_count; i++) {
    output_dimensions[i] = input_dimensions[i];
  }
  for (size_t i = 0; i < permutation_count; i++) {
    output_dimensions[i] = input_dimensions[permutation[i]];
  }
  if (!IsIdentityPermutation(permutation)) {
    std::vector<int64_t> input_strides(permutation_count, 1);
    std::vector<int64_t> output_strides(permutation_count, 1);
    for (int i = permutation_count - 2; i >= 0; i--) {
      input_strides[i] = input_strides[i + 1] * input_dimensions[i + 1];
      output_strides[i] = output_strides[i + 1] * output_dimensions[i + 1];
    }
    auto element_count = input_strides[0] * input_dimensions[0];
    for (int64_t i = 0; i < element_count; i++) {
      // Calculate the indexes for input
      int64_t input_offset = i;
      std::vector<int64_t> input_index(permutation_count, 0);
      for (size_t j = 0; j < permutation_count; j++) {
        input_index[j] = input_offset / input_strides[j];
        input_offset %= input_strides[j];
      }
      // Calculate the transposed indexes for output
      std::vector<int64_t> output_index(permutation_count, 0);
      for (size_t j = 0; j < permutation_count; j++) {
        output_index[j] = input_index[permutation[j]];
      }
      // Calculate the element offset for output
      int64_t output_offset = 0;
      for (size_t j = 0; j < permutation_count; j++) {
        output_offset += output_strides[j] * output_index[j];
      }
      output[output_offset] = input[i];
    }
  } else {
    memcpy(
        output, input, sizeof(T) * ProductionOfDimensions(output_dimensions));
  }
}

template <typename TensorType>
int CustomYoloBox3dNmsFuserImpl(TensorType& out_nms_boxes,     // NOLINT
                                TensorType& out_nms_rois_num,  // NOLINT
                                TensorType& out_nms_index,     // NOLINT
                                TensorType& out_locations,     // NOLINT
                                TensorType& out_dims,          // NOLINT
                                TensorType& out_alphas,        // NOLINT
                                const TensorType& input0,
                                const TensorType& input1,
                                const TensorType& input2,
                                const TensorType& img_size,
                                const Tensor& anchors0,
                                const Tensor& anchors1,
                                const Tensor& anchors2,
                                const Tensor& class_num,
                                const Tensor& conf_thresh,
                                const Tensor& downsample_ratio0,
                                const Tensor& downsample_ratio1,
                                const Tensor& downsample_ratio2,
                                const Tensor& scale_x_y,
                                const Tensor& background_label,
                                const Tensor& score_threshold,
                                const Tensor& nms_top_k,
                                const Tensor& nms_threshold,
                                const Tensor& nms_eta,
                                const Tensor& keep_top_k,
                                const Tensor& normalized) {
  infolog("CustomYoloBox3dNmsFuserImpl execute... ");
  // Input
  auto* input0_data = reinterpret_cast<const float*>(input0.raw_data_const());
  auto* input1_data = reinterpret_cast<const float*>(input1.raw_data_const());
  auto* input2_data = reinterpret_cast<const float*>(input2.raw_data_const());
  auto* img_size_data = reinterpret_cast<const int*>(img_size.raw_data_const());
  // Output
  auto* out_nms_boxes_data = reinterpret_cast<float*>(out_nms_boxes.raw_data());
  auto* out_nms_rois_num_data =
      reinterpret_cast<int32_t*>(out_nms_rois_num.raw_data());
  auto* out_nms_index_data =
      reinterpret_cast<int32_t*>(out_nms_index.raw_data());
  auto* out_locations_data = reinterpret_cast<float*>(out_locations.raw_data());
  auto* out_dims_data = reinterpret_cast<float*>(out_dims.raw_data());
  auto* out_alphas_data = reinterpret_cast<float*>(out_alphas.raw_data());
  // Attr
  auto* anchors0_data = reinterpret_cast<const int*>(anchors0.raw_data_const());
  auto* anchors1_data = reinterpret_cast<const int*>(anchors1.raw_data_const());
  auto* anchors2_data = reinterpret_cast<const int*>(anchors2.raw_data_const());
  int class_num_data = class_num(0, 0, 0, 0);
  float conf_thresh_data = conf_thresh(0, 0, 0, 0);
  int downsample_ratio0_data = downsample_ratio0(0, 0, 0, 0);
  int downsample_ratio1_data = downsample_ratio1(0, 0, 0, 0);
  int downsample_ratio2_data = downsample_ratio2(0, 0, 0, 0);
  float scale_x_y_data = scale_x_y(0, 0, 0, 0);
  const int batch_size = input0.dim(0);
  const int h0 = input0.dim(2);
  const int w0 = input0.dim(3);
  const int h1 = input1.dim(2);
  const int w1 = input1.dim(3);
  const int h2 = input2.dim(2);
  const int w2 = input2.dim(3);
  const int anchor_num0 = anchors0.dim(3) / 2;
  const int anchor_num1 = anchors1.dim(3) / 2;
  const int anchor_num2 = anchors2.dim(3) / 2;
  const int box_num0 = anchor_num0 * h0 * w0;
  const int box_num1 = anchor_num1 * h1 * w1;
  const int box_num2 = anchor_num2 * h2 * w2;
  // YoloBox3d intermediate output
  float* out_box0 =
      static_cast<float*>(malloc(batch_size * box_num0 * 4 * sizeof(float)));
  float* out_box1 =
      static_cast<float*>(malloc(batch_size * box_num1 * 4 * sizeof(float)));
  float* out_box2 =
      static_cast<float*>(malloc(batch_size * box_num2 * 4 * sizeof(float)));
  float* out_score0 = static_cast<float*>(
      malloc(batch_size * box_num0 * class_num_data * sizeof(float)));
  float* out_score1 = static_cast<float*>(
      malloc(batch_size * box_num1 * class_num_data * sizeof(float)));
  float* out_score2 = static_cast<float*>(
      malloc(batch_size * box_num2 * class_num_data * sizeof(float)));
  float* out_location0 =
      static_cast<float*>(malloc(batch_size * box_num0 * 3 * sizeof(float)));
  float* out_location1 =
      static_cast<float*>(malloc(batch_size * box_num1 * 3 * sizeof(float)));
  float* out_location2 =
      static_cast<float*>(malloc(batch_size * box_num2 * 3 * sizeof(float)));
  float* out_dim0 =
      static_cast<float*>(malloc(batch_size * box_num0 * 3 * sizeof(float)));
  float* out_dim1 =
      static_cast<float*>(malloc(batch_size * box_num1 * 3 * sizeof(float)));
  float* out_dim2 =
      static_cast<float*>(malloc(batch_size * box_num2 * 3 * sizeof(float)));
  float* out_alpha0 =
      static_cast<float*>(malloc(batch_size * box_num0 * 2 * sizeof(float)));
  float* out_alpha1 =
      static_cast<float*>(malloc(batch_size * box_num1 * 2 * sizeof(float)));
  float* out_alpha2 =
      static_cast<float*>(malloc(batch_size * box_num2 * 2 * sizeof(float)));
  CustomYoloBox3dKernel(out_box0,
                        out_score0,
                        out_location0,
                        out_dim0,
                        out_alpha0,
                        input0_data,
                        img_size_data,
                        anchors0_data,
                        class_num_data,
                        conf_thresh_data,
                        downsample_ratio0_data,
                        scale_x_y_data,
                        batch_size,
                        h0,
                        w0,
                        box_num0,
                        anchor_num0);
  CustomYoloBox3dKernel(out_box1,
                        out_score1,
                        out_location1,
                        out_dim1,
                        out_alpha1,
                        input1_data,
                        img_size_data,
                        anchors1_data,
                        class_num_data,
                        conf_thresh_data,
                        downsample_ratio1_data,
                        scale_x_y_data,
                        batch_size,
                        h1,
                        w1,
                        box_num1,
                        anchor_num1);
  CustomYoloBox3dKernel(out_box2,
                        out_score2,
                        out_location2,
                        out_dim2,
                        out_alpha2,
                        input2_data,
                        img_size_data,
                        anchors2_data,
                        class_num_data,
                        conf_thresh_data,
                        downsample_ratio2_data,
                        scale_x_y_data,
                        batch_size,
                        h2,
                        w2,
                        box_num2,
                        anchor_num2);

  // Transpose Score
  std::vector<int32_t> score0_input_shape = {
      batch_size, box_num0, class_num_data};
  std::vector<int32_t> score1_input_shape = {
      batch_size, box_num1, class_num_data};
  std::vector<int32_t> score2_input_shape = {
      batch_size, box_num2, class_num_data};
  std::vector<int32_t> score0_output_shape = {
      batch_size, class_num_data, box_num0};
  std::vector<int32_t> score1_output_shape = {
      batch_size, class_num_data, box_num1};
  std::vector<int32_t> score2_output_shape = {
      batch_size, class_num_data, box_num2};
  std::vector<int32_t> permutation = {0, 2, 1};
  float* out_score0_transpose = static_cast<float*>(
      malloc(batch_size * box_num0 * class_num_data * sizeof(float)));
  float* out_score1_transpose = static_cast<float*>(
      malloc(batch_size * box_num1 * class_num_data * sizeof(float)));
  float* out_score2_transpose = static_cast<float*>(
      malloc(batch_size * box_num2 * class_num_data * sizeof(float)));
  TransposeKernel<float>(
      out_score0, out_score0_transpose, score0_input_shape, permutation);
  TransposeKernel<float>(
      out_score1, out_score1_transpose, score1_input_shape, permutation);
  TransposeKernel<float>(
      out_score2, out_score2_transpose, score2_input_shape, permutation);
  // Concat
  float* out_boxes_concat = static_cast<float*>(malloc(
      batch_size * (box_num0 + box_num1 + box_num2) * 4 * sizeof(float)));
  float* out_scores_concat =
      static_cast<float*>(malloc(batch_size * (box_num0 + box_num1 + box_num2) *
                                 class_num_data * sizeof(float)));
  std::vector<float*> boxes_concat_input{out_box0, out_box1, out_box2};
  std::vector<float*> scores_concat_input{
      out_score0_transpose, out_score1_transpose, out_score2_transpose};
  std::vector<float*> location_concat_input{
      out_location0, out_location1, out_location2};
  std::vector<float*> dim_concat_input{out_dim0, out_dim1, out_dim2};
  std::vector<float*> alpha_concat_input{out_alpha0, out_alpha1, out_alpha2};
  std::vector<std::vector<int32_t>> boxes_concat_input_shape = {
      {batch_size, box_num0, 4},
      {batch_size, box_num1, 4},
      {batch_size, box_num2, 4}};
  std::vector<std::vector<int32_t>> scores_concat_input_shape = {
      {batch_size, class_num_data, box_num0},
      {batch_size, class_num_data, box_num1},
      {batch_size, class_num_data, box_num2}};
  std::vector<std::vector<int32_t>> location_concat_input_shape = {
      {batch_size, box_num0, 3},
      {batch_size, box_num1, 3},
      {batch_size, box_num2, 3}};
  std::vector<std::vector<int32_t>> dim_concat_input_shape = {
      {batch_size, box_num0, 3},
      {batch_size, box_num1, 3},
      {batch_size, box_num2, 3}};
  std::vector<std::vector<int32_t>> alpha_concat_input_shape = {
      {batch_size, box_num0, 2},
      {batch_size, box_num1, 2},
      {batch_size, box_num2, 2}};
  ConcatKernel<float>(
      boxes_concat_input, boxes_concat_input_shape, 1, out_boxes_concat);
  ConcatKernel<float>(
      scores_concat_input, scores_concat_input_shape, 2, out_scores_concat);
  ConcatKernel<float>(location_concat_input,
                      location_concat_input_shape,
                      1,
                      out_locations_data);
  ConcatKernel<float>(
      dim_concat_input, dim_concat_input_shape, 1, out_dims_data);
  ConcatKernel<float>(
      alpha_concat_input, alpha_concat_input_shape, 1, out_alphas_data);
  // Nms
  int32_t background_label_data = background_label(0, 0, 0, 0);
  float score_threshold_data = score_threshold(0, 0, 0, 0);
  int32_t nms_top_k_data = nms_top_k(0, 0, 0, 0);
  float nms_threshold_data = nms_threshold(0, 0, 0, 0);
  float nms_eta_data = nms_eta(0, 0, 0, 0);
  int32_t keep_top_k_data = keep_top_k(0, 0, 0, 0);
  bool normalized_data = normalized(0, 0, 0, 0);
  // The total boxes for each instance.
  int32_t box_num = box_num0 + box_num1 + box_num2;
  int32_t box_size = 4;
  int32_t box_out_dim = box_size + 2;
  std::vector<uint32_t> batch_starts;

  MulticlassNmsKernel<float>(out_nms_boxes_data,
                             out_nms_rois_num_data,
                             out_nms_index_data,
                             out_boxes_concat,
                             out_scores_concat,
                             background_label_data,
                             score_threshold_data,
                             nms_top_k_data,
                             nms_threshold_data,
                             nms_eta_data,
                             keep_top_k_data,
                             normalized_data,
                             batch_size,
                             class_num_data,
                             box_num,
                             box_size,
                             batch_starts);

  size_t output_nms_rois_num_dims[] = {static_cast<size_t>(batch_size)};
  out_nms_rois_num.set_dims(output_nms_rois_num_dims);

  uint32_t num_kept = batch_starts.back();
  if (num_kept == 0) {
    size_t output_box_dims[] = {0, static_cast<size_t>(box_out_dim)};
    out_nms_boxes.set_dims(output_box_dims);
    size_t output_index_dims[] = {0, 1};
    out_nms_index.set_dims(output_index_dims);
  } else {
    size_t output_box_dims[] = {
        1, 1, static_cast<size_t>(num_kept), static_cast<size_t>(box_out_dim)};
    out_nms_boxes.set_dims(output_box_dims);
    size_t output_index_dims[] = {static_cast<size_t>(num_kept), 1};
    out_nms_index.set_dims(output_index_dims);
  }

  // clean
  free(out_box0);
  free(out_box1);
  free(out_box2);
  free(out_score0);
  free(out_score1);
  free(out_score2);
  free(out_location0);
  free(out_location1);
  free(out_location2);
  free(out_dim0);
  free(out_dim1);
  free(out_dim2);
  free(out_alpha0);
  free(out_alpha1);
  free(out_alpha2);
  free(out_score0_transpose);
  free(out_score1_transpose);
  free(out_score2_transpose);
  free(out_boxes_concat);
  free(out_scores_concat);
  return GraphStatus::Success;
}

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter

using namespace hnnx;                          // NOLINT
using namespace nnadapter::qualcomm_qnn::htp;  // NOLINT

const char* op_name_yolobox3d_nms_fuser = "CustomYoloBox3dNmsFuser";

DEF_PACKAGE_OP(CustomYoloBox3dNmsFuserImpl<Tensor>, op_name_yolobox3d_nms_fuser)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS(
    (CustomYoloBox3dNmsFuserImpl<PlainFloatTensor>),
    op_name_yolobox3d_nms_fuser,
    "SNAIL",
    Flags::RESOURCE_HVX)
