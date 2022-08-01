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

#include "operation/elementwise.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertElementwise(Converter* converter, core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

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
#define CONVERT_ELEMENTWISE(TYPE, CLASS_NAME)                            \
  case NNADAPTER_##TYPE: {                                               \
    auto eltwise_op =                                                    \
        converter->graph()->CreateOperation<tim::vx::ops::CLASS_NAME>(); \
    eltwise_op->BindInputs({input0_tensor, input1_tensor});              \
    eltwise_op->BindOutput({output_tensor});                             \
  } break;
    CONVERT_ELEMENTWISE(ADD, Add);
    CONVERT_ELEMENTWISE(DIV, Div);
    CONVERT_ELEMENTWISE(MAX, Maximum);
    CONVERT_ELEMENTWISE(MIN, Minimum);
    CONVERT_ELEMENTWISE(MUL, Multiply);
    CONVERT_ELEMENTWISE(POW, Pow);
    CONVERT_ELEMENTWISE(SUB, Sub);
#undef CONVERT_ELEMENTWISE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE)
      << "Missing the processing of fuse_code(" << fuse_code
      << ") in unpack_op_fusion.cc";
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
