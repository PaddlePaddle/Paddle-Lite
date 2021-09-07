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

#include "core/operation/softmax.h"
#include "driver/mediatek_apu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int Program::ConvertSoftmax(hal::Operation* operation) {
  SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  auto input_index = GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = ConvertOperand(input_operand);
  }
  auto beta_index = AddFloat32ConstantOperand(1.0f);
  auto axis_index = AddInt32ConstantOperand(axis);
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index, beta_index, axis_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(
      AddOperation(NEURON_SOFTMAX, &input_indexes, &output_indexes),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
