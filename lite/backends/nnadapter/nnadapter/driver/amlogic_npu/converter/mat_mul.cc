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

#include "core/operation/mat_mul.h"
#include "driver/amlogic_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace amlogic_npu {

int ConvertMatMul(Converter* converter, hal::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  NNADAPTER_CHECK(IsConstantOperand(y_operand))
      << "Only support constant y now.";
  auto num_units = y_operand->type.dimensions.data[0];
  NNADAPTER_VLOG(5) << "num_units: " << num_units;
  auto input_size = y_operand->type.dimensions.data[1];
  NNADAPTER_VLOG(5) << "input_size: " << input_size;

  // Convert to amlnpu tensors and operators
  auto input_tensor = converter->GetMappedTensor(x_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(x_operand);
  }
  auto weight_tensor = converter->ConvertOperand(y_operand);
  auto output_tensor = converter->ConvertOperand(output_operand);
  aml::nn::FCAttr attr;
  attr.weights = num_units;
  attr.has_relu = false;

  auto bias_dimensions_data = std::vector<int32_t>({input_size});
  auto bias_tensor = converter->AddQuant32ConstantTensor(
      {0}, &bias_dimensions_data[0], 1, 0.f);

  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors = {
      input_tensor, weight_tensor, bias_tensor};
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors = {
      output_tensor};
  converter->AddOperator(
      aml::nn::OperatorType::FULLCONNECT, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace amlogic_npu
}  // namespace nnadapter
