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

#include "operation/concat.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertConcat(Converter* converter, core::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  std::vector<uint32_t> input_indexes;
  for (int i = 0; i < input_count - 1; i++) {
    auto input_operand = input_operands[i];
    auto input_index = converter->GetMappedIndex(input_operand);
    if (input_index == INVALID_INDEX) {
      input_index = converter->ConvertOperand(input_operand);
    }
    input_indexes.push_back(input_index);
  }
  auto axis_index = converter->AddInt32ConstantOperand(axis);
  input_indexes.push_back(axis_index);
  auto output_index = converter->ConvertOperand(output_operand);
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(converter->AddOperation(
                         NEURON_CONCATENATION, input_indexes, output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
