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

#include "core/operation/resize_linear.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertResizeLinear(Converter* converter, hal::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (align_mode == 0 && !align_corners) {
    NNADAPTER_LOG(FATAL) << "align_mode = 0 && align_corners = false isn't "
                            "supported in Huawei Ascend NPU DDK";
    return NNADAPTER_INVALID_PARAMETER;
  }

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
  } else if (scales_operand != nullptr) {
    NNADAPTER_LOG(WARNING) << "Not support scales now.";
    return NNADAPTER_INVALID_PARAMETER;
  } else {
    NNADAPTER_LOG(WARNING) << "Either shape_operand or scales_operand should "
                              "be set.";
    return NNADAPTER_INVALID_PARAMETER;
  }
  resize_linear_op->set_attr_align_corners(align_corners);
  MAP_OUTPUT(resize_linear_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
