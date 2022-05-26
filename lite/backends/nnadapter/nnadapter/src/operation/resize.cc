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

#include "core/types.h"
#include "operation/resize_nearest.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

void CopyShapeDimensionTypeToOutput(core::Operand* shape_operand,
                                    core::Operand* input_operand,
                                    core::Operand* output_operand) {
  NNAdapterOperandDimensionType shape_dims;
  if (IsConstantOperand(shape_operand)) {
    shape_dims.dynamic_count = 0;
    if (shape_operand->type.precision == NNADAPTER_INT32) {
      shape_dims.count = shape_operand->length / sizeof(int32_t);
      memcpy(shape_dims.data,
             shape_operand->buffer,
             shape_dims.count * sizeof(int32_t));
    } else if (shape_operand->type.precision == NNADAPTER_INT64) {
      shape_dims.count = shape_operand->length / sizeof(int64_t);
      auto shape_data = reinterpret_cast<int64_t*>(shape_operand->buffer);
      for (uint32_t i = 0; i < shape_dims.count; i++) {
        shape_dims.data[i] = static_cast<int32_t>(shape_data[i]);
      }
    } else {
      NNADAPTER_LOG(FATAL) << "Unsupported precision: "
                           << OperandPrecisionCodeToString(
                                  shape_operand->type.precision);
    }
  } else if (IsTemporaryShapeOperand(shape_operand)) {
    auto& temporary_shape = *(GetTemporaryShape(shape_operand));
    NNADAPTER_CHECK(temporary_shape.data);
    NNADAPTER_CHECK(temporary_shape.data[0]);
    memcpy(
        &shape_dims, &temporary_shape, sizeof(NNAdapterOperandDimensionType));
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(
                                shape_operand->type.lifetime);
  }
  memcpy(output_operand->type.dimensions.data + 2,
         shape_dims.data,
         shape_dims.count * sizeof(int32_t));
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    if (shape_dims.dynamic_count > 0) {
      memcpy(output_operand->type.dimensions.dynamic_data[i] + 2,
             shape_dims.dynamic_data[i],
             shape_dims.count * sizeof(int32_t));
    } else {
      memcpy(output_operand->type.dimensions.dynamic_data[i] + 2,
             shape_dims.data,
             shape_dims.count * sizeof(int32_t));
    }
  }
}

NNADAPTER_EXPORT bool ValidateResize(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareResize(core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_GE(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  /* Input */
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  /* Shape */
  auto shape_operand = input_operands[1];
  if (shape_operand == nullptr) {
    NNADAPTER_VLOG(5) << "Shape is null, please use scales.";
  } else {
    NNADAPTER_VLOG(5) << "shape: " << OperandToString(shape_operand);
  }
  /* Scales */
  auto scales_operand = input_operands[2];
  if (scales_operand == nullptr) {
    NNADAPTER_VLOG(5) << "Scales is null, please use shape.";
  } else {
    NNADAPTER_VLOG(5) << "scales: " << OperandToString(scales_operand);
  }
  NNADAPTER_CHECK(shape_operand != nullptr || scales_operand != nullptr)
      << "shape_operand and scales_operand should not both be null.";
  /* Output */
  auto* output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);

  if (scales_operand != nullptr) {
    std::vector<float> scales;
    if (IsConstantOperand(scales_operand)) {
      auto scales_size = scales_operand->length / sizeof(float);
      scales.resize(scales_size);
      memcpy(scales.data(), scales_operand->buffer, scales_operand->length);
    } else {
      NNADAPTER_LOG(FATAL) << "Unsupported scales lifetime: "
                           << OperandLifetimeCodeToString(
                                  scales_operand->type.lifetime);
    }
    if (scales[0] > 0 && scales[1] > 0) {
      auto infer_output_shape = [&](int32_t* input_dimensions,
                                    int32_t* output_dimensions) {
        for (size_t i = 0; i < scales.size(); i++) {
          output_dimensions[i + 2] =
              input_dimensions[i + 2] == NNADAPTER_UNKNOWN
                  ? NNADAPTER_UNKNOWN
                  : static_cast<int32_t>(
                        static_cast<float>(input_dimensions[i + 2]) *
                        scales[i]);
        }
      };
      infer_output_shape(input_operand->type.dimensions.data,
                         output_operand->type.dimensions.data);
      for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count;
           i++) {
        infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                           output_operand->type.dimensions.dynamic_data[i]);
      }
    } else {
      CopyShapeDimensionTypeToOutput(
          shape_operand, input_operand, output_operand);
    }
  } else {
    CopyShapeDimensionTypeToOutput(
        shape_operand, input_operand, output_operand);
  }
  output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteResize(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
