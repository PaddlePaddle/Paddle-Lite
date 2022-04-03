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

#include "operation/fully_connected.h"
#include <vector>
#include "core/types.h"
#include "operation/math/fully_connected.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateFullyConnected(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareFullyConnected(core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  const auto& input_type = input_operand->type;
  const auto& weight_type = weight_operand->type;
  NNADAPTER_CHECK(IsConstantOperandType(weight_type));
  auto infer_output_shape = [&](const int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                const int32_t* weight_dimensions_data,
                                uint32_t weight_dimensions_count,
                                int32_t* output_dimensions_data,
                                uint32_t* output_dimensions_count) {
    NNADAPTER_CHECK_GE(input_dimensions_count, 2U);
    NNADAPTER_CHECK_EQ(weight_dimensions_count, 2U);
    NNADAPTER_CHECK_EQ(input_dimensions_data[input_dimensions_count - 1],
                       weight_dimensions_data[1]);
    int batch_size = 1;
    for (size_t i = 0; i < input_dimensions_count - 1; i++) {
      auto dimension = input_dimensions_data[i];
      if (dimension == NNADAPTER_UNKNOWN) {
        batch_size = NNADAPTER_UNKNOWN;
        break;
      }
      batch_size *= dimension;
    }
    int num_units = weight_dimensions_data[0];
    NNADAPTER_CHECK(num_units != NNADAPTER_UNKNOWN);
    output_dimensions_data[0] = batch_size;
    output_dimensions_data[1] = num_units;
    *output_dimensions_count = 2;
  };
  auto& output_type = output_operand->type;
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     weight_type.dimensions.data,
                     weight_type.dimensions.count,
                     output_type.dimensions.data,
                     &(output_type.dimensions.count));
  output_type.dimensions.dynamic_count = input_type.dimensions.dynamic_count;
  for (uint32_t i = 0; i < output_type.dimensions.dynamic_count; i++) {
    uint32_t output_dimensions_count = 0;
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       weight_type.dimensions.dynamic_data[i],
                       weight_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i],
                       &output_dimensions_count);
    NNADAPTER_CHECK_EQ(output_dimensions_count, output_type.dimensions.count);
  }
  output_type.precision = input_type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteFullyConnected(core::Operation* operation) {
  FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto& input_type = input_operand->type;
  auto input_shape = std::vector<int32_t>(
      input_type.dimensions.data,
      input_type.dimensions.data + input_type.dimensions.count);
  const auto input_buffer = input_operand->buffer;
  NNADAPTER_CHECK(input_buffer);
  auto& weight_type = weight_operand->type;
  auto weight_shape = std::vector<int32_t>(
      weight_type.dimensions.data,
      weight_type.dimensions.data + weight_type.dimensions.count);
  const auto weight_buffer = weight_operand->buffer;
  NNADAPTER_CHECK(weight_buffer);
  const auto bias_buffer = bias_operand->buffer;
  NNADAPTER_CHECK(bias_buffer);
  auto& output_type = output_operand->type;
  auto output_buffer = AllocateOperand(output_operand);
  NNADAPTER_CHECK_EQ(input_type.precision, output_type.precision);
  if (input_type.precision == NNADAPTER_FLOAT32) {
    const auto input_data = reinterpret_cast<const float*>(input_buffer);
    const auto weight_data = reinterpret_cast<const float*>(weight_buffer);
    const auto bias_data = reinterpret_cast<const float*>(bias_buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    status =
        math::fully_connected<float>(input_data,
                                     input_shape,
                                     weight_data,
                                     weight_shape,
                                     bias_data,
                                     static_cast<math::FuseCode>(fuse_code),
                                     output_data);
  } else if (input_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER &&
             (weight_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER ||
              weight_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL)) {
    const auto input_data = reinterpret_cast<const int8_t*>(input_buffer);
    const auto weight_data = reinterpret_cast<const int8_t*>(weight_buffer);
    const auto bias_data = reinterpret_cast<const int32_t*>(bias_buffer);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    auto weight_scales = std::make_pair(
        std::vector<float>({weight_type.symm_per_layer_params.scale}), -1);
    if (weight_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL) {
      weight_scales.first = std::vector<float>(
          weight_type.symm_per_channel_params.scales,
          weight_type.symm_per_channel_params.scales +
              weight_type.symm_per_channel_params.scale_count);
      weight_scales.second = weight_type.symm_per_channel_params.channel_dim;
    }
    status = math::fully_connected(input_data,
                                   input_shape,
                                   input_type.symm_per_layer_params.scale,
                                   weight_data,
                                   weight_shape,
                                   weight_scales,
                                   bias_data,
                                   static_cast<math::FuseCode>(fuse_code),
                                   output_data,
                                   output_type.symm_per_layer_params.scale);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported precision code("
                         << OperandPrecisionCodeToString(input_type.precision)
                         << ") for " << OperationTypeToString(operation->type)
                         << " is found!";
  }
  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
