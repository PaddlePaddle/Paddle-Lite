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

#include "operation/batch_normalization.h"
#include <math.h>
#include "driver/eeasytech_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace eeasytech_npu {

int ConvertBatchNormalization(Converter *converter,
                              core::Operation *operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to eeasy operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  int size = scale_operand->type.dimensions.data[0];
  NNADAPTER_VLOG(5) << "scale_operand->length: " << scale_operand->length
                    << " data[0] " << size;
  std::vector<float> new_scale_data;
  std::vector<float> new_bias_data;
  for (int i = 0; i < size; i++) {
    float denominator = sqrt(pow(variance_data[i], 2) + epsilon);
    new_scale_data.push_back(scale_data[i] / denominator);
    new_bias_data.push_back(bias_data[i] - mean_data[i] / denominator);
  }
  auto mul_output_tensor = converter->AddVariableTensor(
      converter->GetTensorName(input_operand) + "_mul_output",
      input_operand->type.dimensions.data,
      input_operand->type.dimensions.count,
      ConvertToEznnPrecisionType(input_operand->type.precision),
      reinterpret_cast<const float *>(
          &(input_operand->type.symm_per_layer_params.scale)));
  auto new_scale_tensor =
      converter->AddConstantTensor(new_scale_data.data(),
                                   scale_operand->type.dimensions.data,
                                   scale_operand->type.dimensions.count,
                                   eeasy::nn::PrecisionType::FLOAT32);
  auto new_bias_tensor =
      converter->AddConstantTensor(new_bias_data.data(),
                                   bias_operand->type.dimensions.data,
                                   bias_operand->type.dimensions.count,
                                   eeasy::nn::PrecisionType::FLOAT32);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> mul_input_tensors = {
      input_tensor, new_scale_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> mul_output_tensors = {
      mul_output_tensor};
  converter->AddOperator(eeasy::nn::OperatorType::MULTIPLY,
                         mul_input_tensors,
                         mul_output_tensors,
                         nullptr);
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> add_input_tensors = {
      mul_output_tensor, new_bias_tensor};
  std::vector<std::shared_ptr<eeasy::nn::Tensor>> add_output_tensors = {
      output_tensor};
  converter->AddOperator(eeasy::nn::OperatorType::ADD,
                         add_input_tensors,
                         add_output_tensors,
                         nullptr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace eeasytech_npu
}  // namespace nnadapter
