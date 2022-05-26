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

#include "operation/resize_linear.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!(align_mode == 0 && align_corners))
      << "Unsupported align_mode=0 when align_corners=true.";

  // Convert to GE operators
  auto resize_linear_op =
      converter->AddOperator<ge::op::ResizeBilinearV2>(output_operand);
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  SET_INPUT(resize_linear_op, x, input_operator);

  if (shape_operand != nullptr) {
    auto shape_operator = converter->GetMappedOperator(shape_operand);
    if (shape_operator == nullptr) {
      shape_operator = converter->ConvertOperand(shape_operand);
    }
    SET_INPUT(resize_linear_op, size, shape_operator);
  } else {
    // shape -> cast -> slice -> mul -> cast
    output_operand->type.precision = NNADAPTER_INT32;
    auto shape_op = converter->AddOperator<ge::op::Shape>(output_operand);
    shape_op->set_attr_dtype(ge::DT_INT32);
    SET_INPUT(shape_op, x, input_operator);
    auto shape_operator = MAP_OUTPUT(shape_op, y, output_operand);

    output_operand->type.precision = NNADAPTER_FLOAT32;
    auto cast0_op = converter->AddOperator<ge::op::Cast>(output_operand);
    cast0_op->set_attr_dst_type(ge::DT_FLOAT);
    SET_INPUT(cast0_op, x, shape_operator);
    auto cast0_operator = MAP_OUTPUT(cast0_op, y, output_operand);

    auto starts_operator =
        converter->AddInt32ConstantOperator(std::vector<int>{2});
    auto ends_operator =
        converter->AddInt32ConstantOperator(std::vector<int>{4});
    auto axes_operator =
        converter->AddInt32ConstantOperator(std::vector<int>{0});
    auto steps_operator =
        converter->AddInt32ConstantOperator(std::vector<int>{1});
    auto slice_op =
        converter->AddOperator<ge::op::StridedSliceV2>(output_operand);
    SET_INPUT(slice_op, x, cast0_operator);
    SET_INPUT(slice_op, begin, starts_operator);
    SET_INPUT(slice_op, end, ends_operator);
    SET_INPUT(slice_op, axes, axes_operator);
    SET_INPUT(slice_op, strides, steps_operator);
    auto slice_operator = MAP_OUTPUT(slice_op, y, output_operand);

    auto scales_operator = converter->GetMappedOperator(scales_operand);
    if (scales_operator == nullptr) {
      scales_operator = converter->ConvertOperand(scales_operand);
    }
    auto mul_op = converter->AddOperator<ge::op::Mul>(output_operand);
    SET_INPUT(mul_op, x1, slice_operator);
    SET_INPUT(mul_op, x2, scales_operator);
    auto mul_operator = MAP_OUTPUT(mul_op, y, output_operand);

    output_operand->type.precision = NNADAPTER_INT32;
    auto cast1_op = converter->AddOperator<ge::op::Cast>(output_operand);
    cast1_op->set_attr_dst_type(ge::DT_INT32);
    SET_INPUT(cast1_op, x, mul_operator);
    shape_operator = MAP_OUTPUT(cast1_op, y, output_operand);
    output_operand->type.precision = NNADAPTER_FLOAT32;

    SET_INPUT(resize_linear_op, size, shape_operator);
  }

  resize_linear_op->set_attr_align_corners(align_corners);
  if (align_mode == 0) {
    resize_linear_op->set_attr_half_pixel_centers(true);
  } else {
    resize_linear_op->set_attr_half_pixel_centers(false);
  }
  MAP_OUTPUT(resize_linear_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
