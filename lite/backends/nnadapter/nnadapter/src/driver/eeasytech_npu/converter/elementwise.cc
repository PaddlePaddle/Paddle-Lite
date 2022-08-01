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

#include "operation/elementwise.h"
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to eeasynpu tensors and operators
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors = {
      input0_tensor, input1_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors = {
      output_tensor};
  eeasy::nn::OperatorType op_type;
  if (operation->type == NNADAPTER_ADD) {
    op_type = eeasy::nn::OperatorType::ADD;
  } else if (operation->type == NNADAPTER_SUB) {
    op_type = eeasy::nn::OperatorType::SUBTRACT;
  } else if (operation->type == NNADAPTER_MUL) {
    op_type = eeasy::nn::OperatorType::MULTIPLY;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  converter->AddOperator(op_type, input_tensors, output_tensors, &fuse_code);
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
