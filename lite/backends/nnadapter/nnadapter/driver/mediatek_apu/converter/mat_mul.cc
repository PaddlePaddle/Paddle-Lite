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

#include "core/operation/mat_mul.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertMatMul(Converter* converter, hal::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(y_operand))
      << "Only support constant y now.";
  auto input_size = y_operand->type.dimensions.data[0];
  NNADAPTER_VLOG(5) << "input_size: " << input_size;
  auto num_units = y_operand->type.dimensions.data[1];
  NNADAPTER_VLOG(5) << "num_units: " << num_units;

  // Convert to Neuron operands and operations
  auto input_index = converter->GetMappedIndex(x_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(x_operand);
  }
  auto weight_index = converter->ConvertOperand(y_operand);

  auto bias_dimensions = std::vector<int32_t>({num_units});
  auto bias_dimensions_data = std::vector<int32_t>(num_units, 0);
  auto bias_index = converter->AddQuant32ConstantOperand(
      &bias_dimensions_data[0],
      &bias_dimensions[0],
      1,
      x_operand->type.symm_per_layer_params.scale);
  // auto bias_index = converter->ConvertOperand(bias_operand);
  auto fuse_code_index =
      converter->AddInt32ConstantOperand(NNADAPTER_FUSED_NONE);
  auto output_index = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(
          NEURON_FULLY_CONNECTED,
          {input_index, weight_index, bias_index, fuse_code_index},
          {output_index}),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
