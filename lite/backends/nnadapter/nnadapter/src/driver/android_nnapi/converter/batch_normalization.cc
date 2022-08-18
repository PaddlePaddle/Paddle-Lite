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
#include <cmath>
#include <sstream>
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateBatchNormalization(Validator* validator,
                                const core::Operation* operation) {
  return true;
}

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto output_index = converter->ConvertOperand(output_operand);
  // The formula for BATCH_NORMALIZATION: output = scale * (input - mean) /
  // sqrt(variance + epsilon) + bias
  // Equivalent to: output = alpha * input + beta, where alpha = scale /
  // sqrt(variance + epsilon), beta = -scale * mean / sqrt(variance + epsilon) +
  // bias
  std::vector<float> alpha(scale_count), beta(scale_count);
  for (uint32_t i = 0; i < scale_count; i++) {
    double coeff = scale_data[i] /
                   std::sqrt(variance_data[i] + static_cast<double>(epsilon));
    alpha[i] = coeff;
    beta[i] = -mean_data[i] * coeff + bias_data[i];
  }
  for (uint32_t i = 0; i < scale_count && i < 8; i++) {
    NNADAPTER_VLOG(5) << "alpha[" << i << "]=" << alpha[i];
  }
  for (uint32_t i = 0; i < scale_count && i < 8; i++) {
    NNADAPTER_VLOG(5) << "beta[" << i << "]=" << beta[i];
  }
  uint32_t alpha_index = INVALID_INDEX;
  uint32_t beta_index = INVALID_INDEX;
  uint32_t intermediate_index = INVALID_INDEX;
  if (IsUInt8AsymmPerLayerQuantType(input_operand->type.precision) &&
      IsUInt8AsymmPerLayerQuantType(output_operand->type.precision)) {
    std::vector<uint8_t> quantized_alpha(scale_count),
        quantized_beta(scale_count);
    float alpha_max_value = alpha[0];
    float beta_max_value = beta[0];
    float alpha_min_value = alpha_max_value;
    float beta_min_value = beta_max_value;
    for (uint32_t i = 1; i < scale_count; i++) {
      if (alpha[i] < alpha_min_value) {
        alpha_min_value = alpha[i];
      } else if (alpha[i] > alpha_max_value) {
        alpha_max_value = alpha[i];
      }
      if (beta[i] < beta_min_value) {
        beta_min_value = beta[i];
      } else if (beta[i] > beta_max_value) {
        beta_max_value = beta[i];
      }
    }
    float alpha_scale, beta_scale, intermediate_scale;
    int32_t alpha_zero_point, beta_zero_point, intermediate_zero_point;
    NNADAPTER_CHECK(CalcAsymmQuantParams(
        alpha_min_value, alpha_max_value, &alpha_scale, &alpha_zero_point));
    NNADAPTER_CHECK(CalcAsymmQuantParams(
        beta_min_value, beta_max_value, &beta_scale, &beta_zero_point));
    NNADAPTER_VLOG(5) << "alpha=[" << alpha_min_value << "," << alpha_max_value
                      << "] scale=" << alpha_scale
                      << " zero_point=" << alpha_zero_point;
    NNADAPTER_VLOG(5) << "beta=[" << beta_min_value << "," << beta_max_value
                      << "] scale=" << beta_scale
                      << " zero_point=" << beta_zero_point;
    NNADAPTER_CHECK(QuantizeData<uint8_t>(alpha.data(),
                                          &scale_count,
                                          1,
                                          &alpha_scale,
                                          &alpha_zero_point,
                                          -1,
                                          0,
                                          255,
                                          quantized_alpha.data()));
    NNADAPTER_CHECK(QuantizeData<uint8_t>(beta.data(),
                                          &scale_count,
                                          1,
                                          &beta_scale,
                                          &beta_zero_point,
                                          -1,
                                          0,
                                          255,
                                          quantized_beta.data()));
    for (uint32_t i = 0; i < scale_count && i < 8; i++) {
      NNADAPTER_VLOG(5) << "quantized_alpha[" << i
                        << "]=" << static_cast<int32_t>(quantized_alpha[i]);
    }
    for (uint32_t i = 0; i < scale_count && i < 8; i++) {
      NNADAPTER_VLOG(5) << "quantized_beta[" << i
                        << "]=" << static_cast<int32_t>(quantized_beta[i]);
    }
    alpha_index = converter->AddQuant8ConstantOperand(
        quantized_alpha.data(), scale_count, alpha_scale, alpha_zero_point);
    beta_index = converter->AddQuant8ConstantOperand(
        quantized_beta.data(), scale_count, beta_scale, beta_zero_point);
    auto output_min_value =
        (0 - output_operand->type.asymm_per_layer_params.zero_point) *
        output_operand->type.asymm_per_layer_params.scale;
    auto output_max_value =
        (255 - output_operand->type.asymm_per_layer_params.zero_point) *
        output_operand->type.asymm_per_layer_params.scale;
    float intermediate_min_value = output_min_value - beta_max_value;
    float intermediate_max_value = output_max_value - beta_min_value;
    NNADAPTER_CHECK(CalcAsymmQuantParams(intermediate_min_value,
                                         intermediate_max_value,
                                         &intermediate_scale,
                                         &intermediate_zero_point));
    intermediate_index = converter->AddQuant8VariableOperand(
        output_operand->type.dimensions.data,
        output_operand->type.dimensions.count,
        intermediate_scale,
        intermediate_zero_point);
  } else {
    NNADAPTER_CHECK_EQ(input_operand->type.precision, NNADAPTER_FLOAT32);
    NNADAPTER_CHECK_EQ(output_operand->type.precision, NNADAPTER_FLOAT32);
    alpha_index =
        converter->AddFloat32ConstantOperand(alpha.data(), scale_count);
    beta_index = converter->AddFloat32ConstantOperand(beta.data(), scale_count);
    intermediate_index = converter->AddFloat32VariableOperand(
        output_operand->type.dimensions.data,
        output_operand->type.dimensions.count);
  }
  // alpha * input
  auto mul_fuse_code_index =
      converter->AddInt32ConstantOperand(ANEURALNETWORKS_FUSED_NONE);
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(ANEURALNETWORKS_MUL,
                              {input_index, alpha_index, mul_fuse_code_index},
                              {intermediate_index}),
      ANEURALNETWORKS_NO_ERROR);
  // + beta
  auto add_fuse_code_index =
      converter->AddInt32ConstantOperand(ANEURALNETWORKS_FUSED_NONE);
  NNADAPTER_CHECK_EQ(converter->AddOperation(
                         ANEURALNETWORKS_ADD,
                         {intermediate_index, beta_index, add_fuse_code_index},
                         {output_index}),
                     ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
