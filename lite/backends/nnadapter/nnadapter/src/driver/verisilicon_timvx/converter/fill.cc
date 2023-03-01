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

#include "operation/fill.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto value_tensor = converter->GetMappedTensor(value_operand);
  if (!value_tensor) {
    value_tensor = converter->ConvertOperand(value_operand);
  }

  auto shape_count = output_operand->type.dimensions.count;
  auto shape_data = output_operand->type.dimensions.data;
  std::vector<int32_t> shape;
  for (int i = shape_count - 1; i >= 0; i--) {
    shape.push_back(shape_data[i]);
  }

  auto output_tensor = converter->ConvertOperand(output_operand);
  auto expand_op =
      converter->graph()->CreateOperation<tim::vx::ops::Broadcast>(shape);
  expand_op->BindInputs({value_tensor});
  expand_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
