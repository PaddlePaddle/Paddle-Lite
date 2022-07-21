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

#include "operation/split.h"
#include "driver/qualcomm_qnn/converter/converter.h"

namespace nnadapter {
namespace qualcomm_qnn {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to qnn tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  std::vector<Qnn_Tensor_t> output_tensors;
  for (int i = 0; i < split.size(); i++) {
    auto output_operand = output_operands[i];
    auto output_tensor = converter->GetMappedTensor(output_operand);
    output_tensors.push_back(output_tensor);
  }
  auto axis_param =
      converter->GetParam(QNN_OP_SPLIT_PARAM_AXIS, static_cast<uint32_t>(axis));
  std::vector<uint32_t> split_index{0};
  uint32_t index = 0;
  for (uint32_t i = 0; i < split.size() - 1; i++) {
    index += split[i];
    split_index.push_back(index);
  }
  auto split_index_param =
      converter->GetParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX, split_index);
  converter->AddNode(QNN_OP_SPLIT,
                     {input_tensor},
                     output_tensors,
                     {axis_param, split_index_param});
  return NNADAPTER_NO_ERROR;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
