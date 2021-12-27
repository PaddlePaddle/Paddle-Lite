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

#include "core/operation/expand.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

static std::vector<int> expand_infer_output_shape(
    int32_t* input_dimensions_data,
    uint32_t input_dimensions_count,
    int32_t* output_dimensions_data,
    uint32_t shape_count,
    int32_t* shape_data) {
  std::vector<int> input_dims_vec;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    input_dims_vec.push_back(input_dimensions_data[i]);
  }
  auto diff = shape_count - input_dimensions_count;
  input_dims_vec.insert(input_dims_vec.begin(), diff, 1);
  std::vector<int> final_expand_shape(input_dimensions_count);
  for (uint32_t i = 0; i < input_dims_vec.size(); ++i) {
    NNADAPTER_CHECK_NE(shape_data[i], 0) << "The expanded size cannot be zero.";
    if (i < diff) {
      // shape_data = [3,4,-1,-1], X = [10,2] --> // final_expand_shape =
      // [3,4,10,2]
      NNADAPTER_CHECK_GT(shape_data[i], 0)
          << "The expanded size " << shape_data[i]
          << "for non-existing dimensions must be positive for expand_v2 op.";
      final_expand_shape[i] = shape_data[i];
    } else if (shape_data[i] > 0) {
      // shape_data = [3,4,10,4], X = [10,1] --> final_expand_shape =
      // [3,4,10,4]
      if (input_dims_vec[i] != 1) {
        NNADAPTER_CHECK_EQ(input_dims_vec[i], shape_data[i])
            << "The value " << input_dims_vec[i]
            << " of the non-singleton dimension does not match the "
               "corresponding value "
            << shape_data[i] << " in shape for expand_v2 op.";
        final_expand_shape[i] = shape_data[i];
      } else {
        final_expand_shape[i] = shape_data[i];
      }
    } else {
      // shape_data = [3,4,-1,-1], X = [10,2] --> final_expand_shape =
      // [3,4,10,2]
      NNADAPTER_CHECK_EQ(shape_data[i], -1)
          << "When the value in shape is negative for expand_v2 op, "
             "only -1 is supported, but the value received is "
          << shape_data[i];
      final_expand_shape[i] = input_dims_vec[i];
    }
  }

  for (uint32_t i = 0; i < shape_count; ++i) {
    output_dimensions_data[i] = final_expand_shape[i];
  }

  return final_expand_shape;
}

int PrepareExpand(hal::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& shape_type = shape_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);

  uint32_t shape_count;
  int32_t* shape_data;
  if (shape_type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    auto tempory_shape_info =
        *(shape_operand->hints[NNADAPTER_TEMPORY_SHAPE_INFO])
             .get_mutable<NNAdapterOperandDimensionType>();
    shape_count = tempory_shape_info.count;
    shape_data = tempory_shape_info.data;
  } else if (IsConstantOperand(shape_operand)) {
    shape_count = shape_operand->length / sizeof(int32_t);
    shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << static_cast<int32_t>(shape_type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  for (uint32_t i = 0; i < shape_count; i++) {
    NNADAPTER_VLOG(5) << "shape[" << i << "] = " << shape_data[i];
  }
  output_type.dimensions.count = shape_count;

  // Record new shape information
  if (shape_type.lifetime == NNADAPTER_TEMPORARY_SHAPE ||
      IsConstantOperand(shape_operand)) {
    NNAdapterOperandDimensionType shape_dimension_type;
    shape_dimension_type.count = shape_count;
    shape_dimension_type.dynamic_count = input_type.dimensions.dynamic_count;
    auto final_expand_shape =
        expand_infer_output_shape(input_type.dimensions.data,
                                  input_type.dimensions.count,
                                  output_type.dimensions.data,
                                  shape_count,
                                  shape_data);
    for (uint32_t i = 0; i < shape_count; i++) {
      shape_dimension_type.data[i] = final_expand_shape[i];
    }
    for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
      auto final_expand_shape =
          expand_infer_output_shape(input_type.dimensions.dynamic_data[i],
                                    input_type.dimensions.count,
                                    output_type.dimensions.dynamic_data[i],
                                    shape_count,
                                    shape_data);
      for (uint32_t j = 0; j < shape_count; j++) {
        shape_dimension_type.dynamic_data[i][j] = final_expand_shape[j];
      }
    }
    shape_operand->hints[NNADAPTER_PROCESSED_SHAPE_INFO].set(
        shape_dimension_type);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
