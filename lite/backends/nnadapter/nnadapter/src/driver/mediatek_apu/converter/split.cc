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

#include "operation/split.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto axis_index = converter->AddInt32ConstantOperand(axis);
  auto num_index = converter->AddInt32ConstantOperand(output_count);
  std::vector<uint32_t> output_indexes;
  for (int i = 0; i < output_count; i++) {
    auto output_operand = output_operands[i];
    auto output_index = converter->ConvertOperand(output_operand);
    output_indexes.push_back(output_index);
  }
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          NEURON_SPLIT, {input_index, axis_index, num_index}, output_indexes),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
