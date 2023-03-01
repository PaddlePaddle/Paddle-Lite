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

#include "operation/layer_normalization.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto last_dimension = input_operand->type.dimensions.count - 1;
  NNADAPTER_CHECK((begin_norm_axis == last_dimension) ||
                  (begin_norm_axis == -1))
      << "Tim-VX only support the last dimension";

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  auto scale_tensor = converter->ConvertOperand(scale_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto output_tensor = converter->ConvertOperand(output_operand);
  auto layer_norm_op =
      converter->graph()->CreateOperation<tim::vx::ops::LayerNormalization>(
          ConvertToTimVXAxis(begin_norm_axis,
                             output_operand->type.dimensions.count) /* WHCN */,
          epsilon);
  layer_norm_op->BindInputs({input_tensor, bias_tensor, scale_tensor});
  layer_norm_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
