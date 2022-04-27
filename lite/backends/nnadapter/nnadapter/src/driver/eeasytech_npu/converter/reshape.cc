// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/reshape.h"
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertReshape(Converter* converter, core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_LE(output_operand->type.dimensions.count, 4);

  // Convert to eeasynpu tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  eeasy::nn::ReshapeAttr attr;
  for (uint32_t i = 0; i < output_operand->type.dimensions.count; i++) {
    attr.shapes.push_back(output_operand->type.dimensions.data[i]);
  }
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors = {
      input_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors = {
      output_tensor};
  converter->AddOperator(
      eeasy::nn::OperatorType::RESHAPE, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
