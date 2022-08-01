// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include "driver/eeasytech_npu/converter/converter.h"
#include "operation/hard_sigmoid_swish.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertHardSwish(Converter* converter, core::Operation* operation) {
  HARD_SIGMOID_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(fabs(alpha - 1.0f / 6) <= 1e-5f && fabs(beta - 0.5) <= 1e-5f)
      << "Only supports alpha = 0.2f and beta = 0.5f!";

  // Convert to eeasynpu operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> input_tensors = {
      input_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> output_tensors = {
      output_tensor};
  converter->AddOperator(
      eeasy::nn::OperatorType::RELU6, input_tensors, output_tensors, nullptr);

  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
