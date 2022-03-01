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

#include "operation/pool2d.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto padding_width_left_index =
      converter->AddInt32ConstantOperand(pad_width_left);
  auto padding_width_right_index =
      converter->AddInt32ConstantOperand(pad_width_right);
  auto padding_height_top_index =
      converter->AddInt32ConstantOperand(pad_height_top);
  auto padding_height_bottom_index =
      converter->AddInt32ConstantOperand(pad_height_bottom);
  auto stride_width_index = converter->AddInt32ConstantOperand(stride_width);
  auto stride_height_index = converter->AddInt32ConstantOperand(stride_height);
  auto filter_width_index = converter->AddInt32ConstantOperand(kernel_width);
  auto filter_height_index = converter->AddInt32ConstantOperand(kernel_height);
  auto fuse_code_index = converter->AddInt32ConstantOperand(
      ConvertFuseCodeToNeuronFuseCode(fuse_code));
  auto output_index = converter->ConvertOperand(output_operand);
  NeuronOperationType op_type;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    op_type = NEURON_AVERAGE_POOL_2D;
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    op_type = NEURON_MAX_POOL_2D;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  NNADAPTER_CHECK_EQ(converter->AddOperation(op_type,
                                             {input_index,
                                              padding_width_left_index,
                                              padding_width_right_index,
                                              padding_height_top_index,
                                              padding_height_bottom_index,
                                              stride_width_index,
                                              stride_height_index,
                                              filter_width_index,
                                              filter_height_index,
                                              fuse_code_index},
                                             {output_index}),
                     NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
