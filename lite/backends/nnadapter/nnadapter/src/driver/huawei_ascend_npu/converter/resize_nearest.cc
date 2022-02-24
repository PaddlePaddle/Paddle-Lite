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

#include "operation/resize_nearest.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto resize_nearest_op =
      converter->AddOperator<ge::op::ResizeNearestNeighborV2>(output_operand);
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  SET_INPUT(resize_nearest_op, x, input_operator);
  if (shape_operand != nullptr) {
    auto shape_operator = converter->GetMappedOperator(shape_operand);
    if (shape_operator == nullptr) {
      shape_operator = converter->ConvertOperand(shape_operand);
    }
    SET_INPUT(resize_nearest_op, size, shape_operator);
  } else if (scales_operand != nullptr) {
    auto scale_operator = converter->GetMappedOperator(scales_operand);
    if (scale_operator == nullptr) {
      scale_operator = converter->ConvertOperand(scales_operand);
    }
    // Shape op
    auto shape_op =
        converter->AddOperator<ge::op::Shape>(output_operand, "shape");
    SET_INPUT(shape_op, x, input_operator);
    auto shape_tensor_desc = std::make_shared<ge::TensorDesc>();
    shape_op->update_output_desc_y(*shape_tensor_desc);
    auto shape_operator = converter->UpdateOperatorMap(
        shape_operand,
        std::make_shared<Operator>(shape_op, shape_tensor_desc, "y", -1));
    // Slice op
    auto slice_op =
        converter->AddOperator<ge::op::Slice>(output_operand, "slice");
    std::vector<int> offsets{2};
    std::vector<int> size{2};
    auto offsets_operator = converter->AddInt32ConstantOperator(offsets);
    auto size_operator = converter->AddInt32ConstantOperator(size);
    SET_INPUT(slice_op, x, shape_operator);
    SET_INPUT(slice_op, offsets, offsets_operator);
    SET_INPUT(slice_op, size, size_operator);
    auto slice_tensor_desc = std::make_shared<ge::TensorDesc>();
    slice_op->update_output_desc_y(*slice_tensor_desc);
    auto slice_operator = converter->UpdateOperatorMap(
        nullptr,
        std::make_shared<Operator>(slice_op, slice_tensor_desc, "y", -1));
    // Cast op
    auto slice_cast_op =
        converter->AddOperator<ge::op::Cast>(output_operand, "slice_cast");
    slice_cast_op->set_attr_dst_type(ge::DT_FLOAT);
    SET_INPUT(slice_cast_op, x, slice_operator);
    auto slice_cast_tensor_desc = std::make_shared<ge::TensorDesc>();
    slice_cast_op->update_output_desc_y(*slice_cast_tensor_desc);
    auto slice_cast_operator = converter->UpdateOperatorMap(
        nullptr,
        std::make_shared<Operator>(
            slice_cast_op, slice_cast_tensor_desc, "y", -1));
    // Mul op
    auto mul_op = converter->AddOperator<ge::op::Mul>(output_operand, "mul");
    SET_INPUT(mul_op, x1, slice_cast_operator);
    SET_INPUT(mul_op, x2, scale_operator);
    auto mul_tensor_desc = std::make_shared<ge::TensorDesc>();
    mul_op->update_output_desc_y(*mul_tensor_desc);
    auto mul_operator = converter->UpdateOperatorMap(
        nullptr, std::make_shared<Operator>(mul_op, mul_tensor_desc, "y", -1));
    // Cast op
    auto mul_cast_op =
        converter->AddOperator<ge::op::Cast>(output_operand, "mul_cast");
    mul_cast_op->set_attr_dst_type(ge::DT_INT32);
    SET_INPUT(mul_cast_op, x, mul_operator);
    auto mul_cast_tensor_desc = std::make_shared<ge::TensorDesc>();
    mul_cast_op->update_output_desc_y(*mul_cast_tensor_desc);
    auto mul_cast_operator = converter->UpdateOperatorMap(
        nullptr,
        std::make_shared<Operator>(mul_cast_op, mul_cast_tensor_desc, "y", -1));
    SET_INPUT(resize_nearest_op, size, mul_cast_operator);
  } else {
    NNADAPTER_LOG(WARNING) << "Either shape_operand or scales_operand should "
                              "be set.";
    return NNADAPTER_INVALID_PARAMETER;
  }
  resize_nearest_op->set_attr_align_corners(align_corners);
  MAP_OUTPUT(resize_nearest_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
