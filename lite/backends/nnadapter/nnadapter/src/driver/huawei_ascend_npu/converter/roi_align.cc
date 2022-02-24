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

#include "operation/roi_align.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertRoiAlign(Converter* converter, core::Operation* operation) {
  ROI_ALIGN_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(aligned, false)
      << "ROIAlignNPU only support Aligned attribute equaled to False";

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto rois_operator = converter->GetMappedOperator(rois_operand);
  if (!rois_operator) {
    rois_operator = converter->ConvertOperand(rois_operand);
  }
  auto batch_indices_operator =
      converter->GetMappedOperator(batch_indices_operand);
  if (!batch_indices_operator) {
    batch_indices_operator = converter->ConvertOperand(batch_indices_operand);
  }
  // Unsqueeze batch indices shape to [N, 1]
  auto unsqueeze_op =
      converter->AddOperator<ge::op::Unsqueeze>(output_operand, "unsqueeze");
  unsqueeze_op->set_attr_axes(
      ge::Operator::OpListInt(std::vector<int64_t>({1})));
  SET_INPUT(unsqueeze_op, x, batch_indices_operator);
  auto roi_nums_unsqueeze = MAP_OUTPUT(unsqueeze_op, y, output_operand);
  // Change batch indices date type to fp32
  auto cast_op = converter->AddOperator<ge::op::Cast>(output_operand, "cast");
  cast_op->set_attr_dst_type(ge::DT_FLOAT);
  SET_INPUT(cast_op, x, roi_nums_unsqueeze);
  auto roi_nums_fp32 = MAP_OUTPUT(cast_op, y, output_operand);
  // Concat to make (N, 5)
  auto concat_op =
      converter->AddOperator<ge::op::ConcatD>(output_operand, "concat");
  concat_op->set_attr_concat_dim(1);
  concat_op->set_attr_N(2);
  concat_op->create_dynamic_input_x(2);
  SET_DYNAMIC_INPUT(concat_op, x, 0, roi_nums_fp32);
  SET_DYNAMIC_INPUT(concat_op, x, 1, rois_operator);
  auto roisN5_operator = MAP_OUTPUT(concat_op, y, output_operand);
  // Roi align
  auto roi_align_op = converter->AddOperator<ge::op::ROIAlign>(output_operand);
  roi_align_op->set_attr_pooled_height(output_height);
  roi_align_op->set_attr_pooled_width(output_width);
  roi_align_op->set_attr_sample_num(sampling_ratio);
  roi_align_op->set_attr_spatial_scale(spatial_scale);
  roi_align_op->set_attr_roi_end_mode(0);
  SET_INPUT(roi_align_op, features, input_operator);
  SET_INPUT(roi_align_op, rois, roisN5_operator);
  MAP_OUTPUT(roi_align_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
