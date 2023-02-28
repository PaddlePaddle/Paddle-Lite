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

#include "operation/split.h"
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));
  NNADAPTER_CHECK(IsConstantOperand(split_operand));

  // Convert to eeasynpu operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors;
  for (auto output_operand : output_operands) {
    auto output_tensor = converter->ConvertOperand(output_operand);
    output_tensors.push_back(output_tensor);
  }
  eeasy::nn::SplitAttr attr;
  attr.axis = axis;
  attr.slices = std::vector<uint32_t>(split.begin(), split.end());
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors = {
      input_tensor};
  converter->AddOperator(
      eeasy::nn::OperatorType::SPLIT, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
