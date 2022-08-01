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

#include "operation/meshgrid.h"
#include <vector>
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertMeshgrid(Converter* converter, core::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  for (int i = 0; i < output_count; i++) {
    auto input_operand = input_operands[i];
    auto input_operator = converter->GetMappedOperator(input_operand);
    if (!input_operator) {
      input_operator = converter->ConvertOperand(input_operand);
    }
    std::vector<int32_t> shape(input_operand->type.dimensions.data,
                               input_operand->type.dimensions.data +
                                   input_operand->type.dimensions.count);
    auto shape_operator = converter->AddInt32ConstantOperator(shape);
    auto output_operand = output_operands[i];
    auto broadcastToOp =
        converter->AddOperator<ge::op::BroadcastTo>(output_operands[i]);
    SET_INPUT(broadcastToOp, x, input_operator);
    SET_INPUT(broadcastToOp, shape, shape_operator);
    MAP_OUTPUT(broadcastToOp, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
