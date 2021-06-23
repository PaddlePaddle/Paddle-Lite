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

int Program::ConvertElementwise(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input0
  auto input0_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input0: " << OperandToString(input0_operand);
  // Input1
  auto input1_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "input1: " << OperandToString(input1_operand);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to Neuron operands and operations
  auto input0_index = ConvertOperand(input0_operand);
  auto input1_index = ConvertOperand(input1_operand);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_ADD) {
    op_type = NEURON_ADD;
  } else if (operation->type == NNADAPTER_SUB) {
    op_type = NEURON_SUB;
  } else if (operation->type == NNADAPTER_MUL) {
    op_type = NEURON_MUL;
  } else if (operation->type == NNADAPTER_DIV) {
    op_type = NEURON_DIV;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  std::vector<uint32_t> input_indexes = {
      input0_index, input1_index, fuse_code_index};
  std::vector<uint32_t> output_indexes = {output_index};
  NNADAPTER_CHECK_EQ(AddOperation(op_type, &input_indexes, &output_indexes),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
