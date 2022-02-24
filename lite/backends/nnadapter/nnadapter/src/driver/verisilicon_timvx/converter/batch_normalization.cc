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

#include "operation/batch_normalization.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to tim-vx tensors and operators
  auto input_operator = converter->GetMappedTensor(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }

  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto offset_operator = converter->ConvertOperand(bias_operand);
  auto mean_operator = converter->ConvertOperand(mean_operand);
  auto variance_operator = converter->ConvertOperand(variance_operand);
  auto output_tensor = converter->ConvertOperand(output_operand);

  auto batch_norm_op =
      converter->graph()->CreateOperation<tim::vx::ops::BatchNorm>(epsilon);
  batch_norm_op->BindInputs({input_operator,
                             mean_operator,
                             variance_operator,
                             scale_operator,
                             offset_operator});
  batch_norm_op->BindOutputs({output_tensor});
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
