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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  std::shared_ptr<Operator> shape_operator = nullptr;
  if (shape_operand != nullptr) {
    shape_operator = converter->GetMappedOperator(shape_operand);
    if (shape_operator == nullptr) {
      shape_operator = converter->ConvertOperand(shape_operand);
    }
  } else {
    auto out_h = output_operand->type.dimensions.data[2];
    auto out_w = output_operand->type.dimensions.data[3];
    shape_operator =
        converter->AddInt32ConstantOperator(std::vector<int32_t>{out_h, out_w});
  }
  auto resize_nearest_op =
      converter->AddOperator<hiai::op::ResizeNearestNeighborV2>(output_operand);
  resize_nearest_op->set_attr_align_corners(align_corners);
  SET_INPUT(resize_nearest_op, x, input_operator);
  SET_INPUT(resize_nearest_op, size, shape_operator);
  MAP_OUTPUT(resize_nearest_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
