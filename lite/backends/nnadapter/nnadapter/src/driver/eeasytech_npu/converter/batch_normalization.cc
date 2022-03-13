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
#include "operation/batch_normalization.h"
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);

  auto new_scale_operand = converter->GetOperand(scale_operand);
  auto mul_output_operand = converter->GetOperand(input_operand, false);
  auto new_bias_operand = converter->GetOperand(bias_operand);

  auto new_scale_tensor = converter->ConvertOperand(new_scale_operand);
  auto mul_output_tensor = converter->ConvertOperand(mul_output_operand);
  auto new_bias_tensor = converter->ConvertOperand(new_bias_operand);

  int size = scale_operand->type.dimensions.data[0];
  NNADAPTER_VLOG(5) << "scale_operand->length: " << scale_operand->length << " data[0] " << size;

  float *scale_data = (float *)scale_operand->buffer;
  float *bias_data = (float *)bias_operand->buffer;
  float *mean_data = (float *)mean_operand->buffer;
  float *var_data = (float *)variance_operand->buffer;

  float *new_scale_data = (float *)new_scale_operand->buffer;
  float *new_bias_data = (float *)new_bias_operand->buffer;

  for (int i = 0; i < size; i++) {
  	float denominator = sqrt(pow(var_data[i], 2) + epsilon);
    new_scale_data[i] = scale_data[i] / denominator;
	new_bias_data[i] = bias_data[i] - mean_data[i] / denominator;
  }

  std::vector<std::shared_ptr<eeasy::nn::Tensor>> mul_input_tensors = {input_tensor, new_scale_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> mul_output_tensors = {mul_output_tensor};

  converter->AddOperator(eeasy::nn::OperatorType::MULTIPLY, mul_input_tensors, mul_output_tensors, nullptr);

  std::vector<std::shared_ptr<eeasy::nn::Tensor>> add_input_tensors = {mul_output_tensor, new_bias_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> add_output_tensors = {output_tensor};

  converter->AddOperator(eeasy::nn::OperatorType::ADD, add_input_tensors, add_output_tensors, nullptr);

  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
