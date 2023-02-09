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

#include "operation/quantize.h"
#include "core/types.h"
#include "operation/math/quantize.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateQuantize(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareQuantize(core::Operation* operation) {
  QUANTIZE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  if (is_per_layer_quant && is_symm_quant) {
    output_operand->type.precision = NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
  } else if (!is_per_layer_quant && is_symm_quant) {
    output_operand->type.precision = NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL;
  } else if (is_per_layer_quant && !is_symm_quant) {
    output_operand->type.precision = NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported quant mode.";
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteQuantize(core::Operation* operation) {
  QUANTIZE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto& input_type = input_operand->type;
  auto input_shape = std::vector<int32_t>(
      input_type.dimensions.data,
      input_type.dimensions.data + input_type.dimensions.count);
  const auto input_buffer = input_operand->buffer;
  NNADAPTER_CHECK(input_buffer);
  auto& output_type = output_operand->type;
  auto output_buffer = AllocateOperand(output_operand);
  if (input_type.precision == NNADAPTER_FLOAT32 &&
      output_type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    const auto input_data = reinterpret_cast<const float*>(input_buffer);
    auto output_data = reinterpret_cast<int8_t*>(output_buffer);
    status = math::quantize<int8_t>(
        input_data, input_shape, scale_data[0], output_data);
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
