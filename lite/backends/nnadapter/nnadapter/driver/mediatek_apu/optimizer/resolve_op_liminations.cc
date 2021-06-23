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

#include "driver/mediatek_apu/optimizer/resolve_op_liminations.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

static void ResolveSoftmax(hal::Model* model, hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions = input_operand->type.dimensions;
  auto input_dimension_count = input_operand->type.dimension_count;
  // Axis
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_operand->type.dimension_count;
  }
  auto output_operand = output_operands[0];
  // MediaTek APU only supports 2D or 4D input
  if (input_dimension_count != 2 && input_dimension_count != 4) {
    bool is_ends_with_1 = true;
    for (uint32_t i = *axis + 1; i < input_dimension_count; i++) {
      if (input_dimensions[i] != 1) {
        is_ends_with_1 = false;
      }
    }
    auto input_count =
        ProductionOfDimensions(input_dimensions, input_dimension_count);
    auto axis_count = input_dimensions[*axis];
    auto remain_count = input_count / axis_count;
    std::vector<int32_t> output_dimensions(
        output_operand->type.dimensions,
        output_operand->type.dimensions + output_operand->type.dimension_count);
    std::vector<int32_t> reshape_input_dimensions = {
        static_cast<int32_t>(remain_count), static_cast<int32_t>(axis_count)};
    if (is_ends_with_1) {
      // Reshape the input operand to 2D and update axis to 1
      auto reshape_input_operand =
          AddReshapeOperation(model, input_operand, reshape_input_dimensions);
      ReshapeOperand(output_operand, reshape_input_dimensions);
      auto reshape_output_operand =
          AddReshapeOperation(model, output_operand, output_dimensions);
    } else {
      // Transpose (1, 192(axis), 128) to (1, 128, 192(axis))
      std::vector<int32_t> transpose_input_permutation(input_dimension_count);
      for (uint32_t i = 0; i < input_dimension_count; i++) {
        if (i < *axis) {
          transpose_input_permutation[i] = i;
        } else if (i > *axis) {
          transpose_input_permutation[i - 1] = i;
        } else {
          transpose_input_permutation[input_dimension_count - 1] = *axis;
        }
      }
      auto transpose_input_operand = AddTransposeOperation(
          model, input_operand, transpose_input_permutation);
      std::vector<int32_t> transpose_input_dimensions(
          transpose_input_operand->type.dimensions,
          transpose_input_operand->type.dimensions +
              transpose_input_operand->type.dimension_count);
      // Reshape (1, 128, 192(axis)) to (384, 192(axis))
      auto reshape_transpose_input_operand = AddReshapeOperation(
          model, transpose_input_operand, reshape_input_dimensions);
      ReshapeOperand(output_operand, reshape_input_dimensions);
      // Reshape (384, 192(axis)) back to (1, 128, 192(axis))
      auto reshape_output_operand = AddReshapeOperation(
          model, output_operand, transpose_input_dimensions);
      auto transpose_reshape_output_operand = AddTransposeOperation(
          model,
          reshape_output_operand,
          InversePermutation(transpose_input_permutation));
    }
    *axis = 1;
  }
}

void ResolveOpLiminations(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
      case NNADAPTER_SOFTMAX:
        ResolveSoftmax(model, operation);
        break;
      default:
        break;
    }
  }
}

}  // namespace mediatek_apu
}  // namespace nnadapter
