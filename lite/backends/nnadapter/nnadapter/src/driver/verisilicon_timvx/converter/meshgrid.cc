// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/meshgrid.h"
#include <vector>
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertMeshgrid(Converter* converter, core::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  for (int i = 0; i < output_count; i++) {
    auto input_operand = input_operands[i];
    auto input_tensor = converter->GetMappedTensor(input_operand);
    if (!input_tensor) {
      input_tensor = converter->ConvertOperand(input_operand);
    }

    auto output_operand = output_operands[i];
    std::vector<int32_t> output_shape(
        output_operand->type.dimensions.data,
        output_operand->type.dimensions.data +
            output_operand->type.dimensions.count);

    std::vector<int32_t> operand_shape(output_count, 1);
    operand_shape[i] = output_shape[i];
    std::vector<uint32_t> view_shape(output_count, 1);
    view_shape[i] = static_cast<uint32_t>(output_shape[i]);
    auto reshape_tensor =
        converter->ConvertOperand(output_operand, operand_shape);
    auto reshape_op =
        converter->graph()->CreateOperation<tim::vx::ops::Reshape>(view_shape);
    reshape_op->BindInputs({input_tensor});
    reshape_op->BindOutputs({reshape_tensor});

    auto output_tensor = converter->ConvertOperand(output_operand);
    auto meshgrid_op =
        converter->graph()->CreateOperation<tim::vx::ops::Broadcast>(
            output_shape);
    meshgrid_op->BindInputs({reshape_tensor});
    meshgrid_op->BindOutputs({output_tensor});
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
