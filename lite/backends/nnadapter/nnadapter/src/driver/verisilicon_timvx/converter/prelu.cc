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

#include "operation/prelu.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertPRelu(Converter* converter, core::Operation* operation) {
  PRELU_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(input_operand->type.dimensions.count, 4)
      << "PRelu only supports 4-D input tensor!";

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  auto input_channel_size = input_operand->type.dimensions.data[1];
  auto input_element_size =
      ProductionOfDimensions(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.count);
  auto slope_element_size =
      ProductionOfDimensions(slope_operand->type.dimensions.data,
                             slope_operand->type.dimensions.count);

  std::shared_ptr<tim::vx::Tensor> slope_tensor = nullptr;
  if (input_element_size == slope_element_size || slope_element_size == 1) {
    slope_tensor = converter->ConvertOperand(slope_operand);
  } else if (input_channel_size == slope_element_size) {
    slope_tensor =
        converter->ConvertOperand(slope_operand, {input_channel_size, 1, 1});
  } else {
    NNADAPTER_LOG(FATAL) << "slope size doesn't match with input size";
  }

  auto output_tensor = converter->ConvertOperand(output_operand);
  int axis = 1;
  auto prelu_op = converter->graph()->CreateOperation<tim::vx::ops::Prelu>(
      ConvertToTimVXAxis(axis,
                         output_operand->type.dimensions.count) /* WHCN */);
  prelu_op->BindInputs({input_tensor, slope_tensor});
  prelu_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
