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

#include "operation/comparisons.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input0_tensor = converter->GetMappedTensor(input0_operand);
  if (!input0_tensor) {
    input0_tensor = converter->ConvertOperand(input0_operand);
  }
  auto input1_tensor = converter->GetMappedTensor(input1_operand);
  if (!input1_tensor) {
    input1_tensor = converter->ConvertOperand(input1_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  switch (operation->type) {
#define CONVERT_COMPARISON(type, class_name)                             \
  case NNADAPTER_##type: {                                               \
    auto comp_op =                                                       \
        converter->graph()->CreateOperation<tim::vx::ops::class_name>(); \
    comp_op->BindInputs({input0_tensor, input1_tensor});                 \
    comp_op->BindOutputs({output_tensor});                               \
  } break;
    CONVERT_COMPARISON(EQUAL, Equal);
    CONVERT_COMPARISON(NOT_EQUAL, NotEqual);
    CONVERT_COMPARISON(GREATER, Greater);
    CONVERT_COMPARISON(GREATER_EQUAL, GreaterOrEqual);
    CONVERT_COMPARISON(LESS, Less);
    CONVERT_COMPARISON(LESS_EQUAL, LessOrEqual);

#undef CONVERT_COMPARISON
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported comparison operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
