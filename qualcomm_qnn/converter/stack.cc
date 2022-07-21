// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/stack.h"
#include "driver/qualcomm_qnn/converter/converter.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertStack(Converter* converter, core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  std::vector<Qnn_Tensor_t> input_tensors;
  for (int i = 0; i < input_count - 1; i++) {
    auto input_operand = input_operands[i];
    auto input_tensor = converter->GetMappedTensor(input_operand);
    input_tensors.push_back(input_tensor);
  }
  auto output_tensor = converter->GetMappedTensor(output_operand);
  auto axis_param =
      converter->GetParam(QNN_OP_PACK_PARAM_AXIS, static_cast<uint32_t>(axis));
  converter->AddNode(QNN_OP_PACK, input_tensors, {output_tensor}, {axis_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
