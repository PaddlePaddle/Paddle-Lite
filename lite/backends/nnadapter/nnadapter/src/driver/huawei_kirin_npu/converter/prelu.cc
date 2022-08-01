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

#include "operation/prelu.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertPRelu(Converter* converter, core::Operation* operation) {
  PRELU_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(input_operand->type.dimensions.count, 4)
      << "PRelu only supports 4-D input tensor!";
  auto input_channel_size = input_operand->type.dimensions.data[1];
  NNADAPTER_CHECK_NE(input_channel_size, NNADAPTER_UNKNOWN)
      << "The input channel size of PRelu should not be dynamic!";
  NNADAPTER_CHECK(IsConstantOperand(slope_operand))
      << "PRelu only supports constant slope tensor!";
  NNADAPTER_CHECK_EQ(slope_operand->type.dimensions.count, 1)
      << "PRelu only supports 1-D slope tensor!";
  auto slope_element_size =
      ProductionOfDimensions(slope_operand->type.dimensions.data,
                             slope_operand->type.dimensions.count);
  NNADAPTER_CHECK(slope_element_size == 1 ||
                  slope_element_size == input_channel_size)
      << "The element size of slope is not 1 or does not match the input "
         "channel size!";

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  // The slope tensor of PRelu only supports (1,c,1,1) or (c,1,1) in hiai
  std::shared_ptr<Operator> slope_operator = nullptr;
  if (input_channel_size == slope_element_size) {
    slope_operator =
        converter->ConvertOperand(slope_operand, {input_channel_size, 1, 1});
  } else {
    auto slope = *reinterpret_cast<float*>(slope_operand->buffer);
    slope_operator = converter->AddFloat32ConstantOperator(
        std::vector<float>(input_channel_size, slope),
        {input_channel_size, 1, 1});
  }
  auto prelu_op = converter->AddOperator<hiai::op::PRelu>(output_operand);
  SET_INPUT(prelu_op, x, input_operator);
  SET_INPUT(prelu_op, weight, slope_operator);
  MAP_OUTPUT(prelu_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
