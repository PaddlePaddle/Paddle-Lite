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

#include "operation/fully_connected.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertFullyConnected(Converter* converter, core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto weight_tensor = converter->ConvertOperand(weight_operand);
  auto bias_tensor = converter->GetMappedTensor(bias_operand);
  if (!bias_tensor) {
    bias_tensor = converter->ConvertOperand(bias_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  // Find the first axis to be flattened into a 2-D matrix [-1, input_size]
  int input_axis = input_operand->type.dimensions.count - 1;
  int64_t input_prod = 1;
  for (; input_axis >= 0; input_axis--) {
    input_prod *= input_operand->type.dimensions.data[input_axis];
    if (input_prod == input_size) break;
  }
  auto fc_op =
      converter->graph()->CreateOperation<tim::vx::ops::FullyConnected>(
          input_operand->type.dimensions.count - 1 - input_axis /* WHCN */,
          num_units);
  fc_op->BindInputs({input_tensor, weight_tensor, bias_tensor});
  fc_op->BindOutputs({output_tensor});
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE)
      << "Missing the processing of fuse_code(" << fuse_code
      << ") in unpack_op_fusion.cc";
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
