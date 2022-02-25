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
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertConcat(Converter* converter, core::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors;
  for (int i = 0; i < input_count - 1; i++) {
    auto input_operand = input_operands[i];
    auto input_tensor = converter->GetMappedTensor(input_operand);
    if (!input_tensor) {
      input_tensor = converter->ConvertOperand(input_operand);
    }
    input_tensors.push_back(input_tensor);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  auto concat_op = converter->graph()->CreateOperation<tim::vx::ops::Concat>(
      ConvertToTimVXAxis(axis,
                         output_operand->type.dimensions.count) /* WHCN */,
      input_tensors.size());
  concat_op->BindInputs(input_tensors);
  concat_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
