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

#include "driver/mediatek_apu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int Program::ConvertActivation(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 1);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  auto output_index = ConvertOperand(output_operand);
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_SIGMOID) {
    op_type = NEURON_LOGISTIC;
  } else if (operation->type == NNADAPTER_RELU) {
    op_type = NEURON_RELU;
  } else if (operation->type == NNADAPTER_RELU6) {
    op_type = NEURON_RELU6;
  } else if (operation->type == NNADAPTER_TANH) {
    op_type = NEURON_TANH;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  std::vector<uint32_t> input_indexes = {input_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(AddOperation(op_type, &input_indexes, &output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
