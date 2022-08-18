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

#include "operation/elementwise.h"
#include <algorithm>
#include <unordered_map>
#include "core/types.h"
#include "operation/math/elementwise.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void CalcEltwiseBinaryOperationsOutputSize(
    const NNAdapterOperandType& input0_type,
    const NNAdapterOperandType& input1_type,
    NNAdapterOperandType* output_type) {
  // Infer the shape and type of output operands
  int32_t input0_size = input0_type.dimensions.count;
  int32_t input1_size = input1_type.dimensions.count;
  int32_t max_size = std::max(input0_size, input1_size);
  auto infer_output_shape = [&](const int32_t* input0_dimensions_data,
                                const int32_t* input1_dimensions_data,
                                int32_t* output_dimensions_data) {
    int32_t input0_i = input0_size - 1;
    int32_t input1_i = input1_size - 1;
    for (int32_t i = max_size - 1; i >= 0; i--) {
      if (input0_i < 0) {
        NNADAPTER_CHECK_GE(input1_i, 0);
        output_dimensions_data[i] = input1_dimensions_data[input1_i];
      } else if (input1_i < 0) {
        NNADAPTER_CHECK_GE(input0_i, 0);
        output_dimensions_data[i] = input0_dimensions_data[input0_i];
      } else {
        int32_t input0_data = input0_dimensions_data[input0_i];
        int32_t input1_data = input1_dimensions_data[input1_i];
        if (input0_data == input1_data) {
          output_dimensions_data[i] = input0_data;
        } else if (input0_data == 1) {
          output_dimensions_data[i] = input1_data;
        } else if (input1_data == 1) {
          output_dimensions_data[i] = input0_data;
        } else {
          NNADAPTER_LOG(ERROR) << "Cannot broadcast input0: " << input0_data
                               << ", input1: " << input1_data;
        }
      }
      input0_i--;
      input1_i--;
    }
  };
  output_type->dimensions.count = max_size;
  output_type->dimensions.dynamic_count = input0_type.dimensions.dynamic_count;
  infer_output_shape(input0_type.dimensions.data,
                     input1_type.dimensions.data,
                     output_type->dimensions.data);
  for (uint32_t i = 0; i < input0_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input0_type.dimensions.dynamic_data[i],
                       input1_type.dimensions.dynamic_data[i],
                       output_type->dimensions.dynamic_data[i]);
  }
}

static std::unordered_map<NNAdapterOperationType, math::ElementwiseTypeCode>
    kSupportedElementwise = {{NNADAPTER_ADD, math::ADD},
                             {NNADAPTER_SUB, math::SUB},
                             {NNADAPTER_MUL, math::MUL}};

NNADAPTER_EXPORT bool ValidateElementwise(const core::Operation* operation) {
  return kSupportedElementwise.count(operation->type) > 0;
}

NNADAPTER_EXPORT int PrepareElementwise(core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  if (IsConstantOperand(input0_operand) && !IsConstantOperand(input1_operand)) {
    input0_operand->type.dimensions.dynamic_count =
        input1_operand->type.dimensions.dynamic_count;
    for (size_t i = 0; i < input0_operand->type.dimensions.dynamic_count; i++) {
      for (size_t j = 0; j < input1_operand->type.dimensions.count; j++) {
        input0_operand->type.dimensions.dynamic_data[i][j] = 1;
      }
    }
  } else if (IsConstantOperand(input1_operand) &&
             !IsConstantOperand(input0_operand)) {
    input1_operand->type.dimensions.dynamic_count =
        input0_operand->type.dimensions.dynamic_count;
    for (size_t i = 0; i < input1_operand->type.dimensions.dynamic_count; i++) {
      for (size_t j = 0; j < input0_operand->type.dimensions.count; j++) {
        input1_operand->type.dimensions.dynamic_data[i][j] = 1;
      }
    }
  }

  CalcEltwiseBinaryOperationsOutputSize(
      input0_operand->type, input1_operand->type, &output_operand->type);
  output_operand->type.precision = input0_operand->type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteElementwise(core::Operation* operation) {
  if (!kSupportedElementwise.count(operation->type))
    return NNADAPTER_FEATURE_NOT_SUPPORTED;
  auto eltwise_type = kSupportedElementwise[operation->type];

  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto& input0_type = input0_operand->type;
  auto input0_shape = std::vector<int32_t>(
      input0_type.dimensions.data,
      input0_type.dimensions.data + input0_type.dimensions.count);
  const auto input0_buffer = input0_operand->buffer;
  NNADAPTER_CHECK(input0_buffer);
  auto& input1_type = input1_operand->type;
  auto input1_shape = std::vector<int32_t>(
      input1_type.dimensions.data,
      input1_type.dimensions.data + input1_type.dimensions.count);
  const auto input1_buffer = input1_operand->buffer;
  NNADAPTER_CHECK(input1_buffer);
  auto& output_type = output_operand->type;
  auto output_shape = std::vector<int32_t>(
      output_type.dimensions.data,
      output_type.dimensions.data + output_type.dimensions.count);
  auto output_buffer = AllocateOperand(output_operand);
  if (input0_type.precision == NNADAPTER_FLOAT32 &&
      input1_type.precision == NNADAPTER_FLOAT32) {
    const auto input0_data = reinterpret_cast<const float*>(input0_buffer);
    const auto input1_data = reinterpret_cast<const float*>(input1_buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    status = math::elementwise<float>(eltwise_type,
                                      input0_data,
                                      input0_shape,
                                      input1_data,
                                      input1_shape,
                                      static_cast<math::FuseCode>(fuse_code),
                                      output_data);
  } else if (input0_type.precision == NNADAPTER_INT32 &&
             input1_type.precision == NNADAPTER_INT32) {
    const auto input0_data = reinterpret_cast<const int32_t*>(input0_buffer);
    const auto input1_data = reinterpret_cast<const int32_t*>(input1_buffer);
    auto output_data = reinterpret_cast<int32_t*>(output_buffer);
    status = math::elementwise<int32_t>(eltwise_type,
                                        input0_data,
                                        input0_shape,
                                        input1_data,
                                        input1_shape,
                                        static_cast<math::FuseCode>(fuse_code),
                                        output_data);
  } else if (input0_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER &&
             input1_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    const auto input0_data = reinterpret_cast<const int8_t*>(input0_buffer);
    const auto input1_data = reinterpret_cast<const int8_t*>(input1_buffer);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    status = math::elementwise(eltwise_type,
                               input0_data,
                               input0_shape,
                               input0_type.symm_per_layer_params.scale,
                               input1_data,
                               input1_shape,
                               input1_type.symm_per_layer_params.scale,
                               static_cast<math::FuseCode>(fuse_code),
                               output_data,
                               output_type.symm_per_layer_params.scale);
  } else if (input0_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER &&
             input1_type.precision == NNADAPTER_FLOAT32) {
    const auto input0_data = reinterpret_cast<const int8_t*>(input0_buffer);
    auto input0_count = math::shape_production(input0_shape);
    std::vector<float> dequantized_input0_data(input0_count);
    status = math::dequantize(input0_data,
                              input0_shape,
                              input0_type.symm_per_layer_params.scale,
                              dequantized_input0_data.data());
    NNADAPTER_CHECK_EQ(status, 0);
    const auto input1_data = reinterpret_cast<const float*>(input1_buffer);
    auto output_count = math::shape_production(output_shape);
    std::vector<float> dequantized_output_data(output_count);
    status = math::elementwise<float>(eltwise_type,
                                      dequantized_input0_data.data(),
                                      input0_shape,
                                      input1_data,
                                      input1_shape,
                                      static_cast<math::FuseCode>(fuse_code),
                                      dequantized_output_data.data());
    NNADAPTER_CHECK_EQ(status, 0);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    status = math::quantize(dequantized_output_data.data(),
                            output_shape,
                            output_type.symm_per_layer_params.scale,
                            output_data);
  } else if (input0_type.precision == NNADAPTER_FLOAT32 &&
             input1_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    const auto input1_data = reinterpret_cast<const int8_t*>(input1_buffer);
    auto input1_count = math::shape_production(input1_shape);
    std::vector<float> dequantized_input1_data(input1_count);
    status = math::dequantize(input1_data,
                              input1_shape,
                              input1_type.symm_per_layer_params.scale,
                              dequantized_input1_data.data());
    NNADAPTER_CHECK_EQ(status, 0);
    const auto input0_data = reinterpret_cast<const float*>(input0_buffer);
    auto output_count = math::shape_production(output_shape);
    std::vector<float> dequantized_output_data(output_count);
    status = math::elementwise<float>(eltwise_type,
                                      input0_data,
                                      input0_shape,
                                      dequantized_input1_data.data(),
                                      input1_shape,
                                      static_cast<math::FuseCode>(fuse_code),
                                      dequantized_output_data.data());
    NNADAPTER_CHECK_EQ(status, 0);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    status = math::quantize(dequantized_output_data.data(),
                            output_shape,
                            output_type.symm_per_layer_params.scale,
                            output_data);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported input0 precision code("
                         << OperandPrecisionCodeToString(input0_type.precision)
                         << ") and input1 precision code("
                         << OperandPrecisionCodeToString(input1_type.precision)
                         << ") for " << OperationTypeToString(operation->type)
                         << " is found!";
  }
  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
