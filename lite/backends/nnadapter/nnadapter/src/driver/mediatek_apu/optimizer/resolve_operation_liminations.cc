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

#include "driver/mediatek_apu/optimizer/resolve_operation_liminations.h"
#include <cmath>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

static void ResolveSoftmax(core::Model* model, core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  auto input_operand = input_operands[0];
  auto input_dimensions_data = input_operand->type.dimensions.data;
  auto input_dimensions_count = input_operand->type.dimensions.count;
  // Axis
  auto axis = reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (*axis < 0) {
    *axis += input_operand->type.dimensions.count;
  }
  auto output_operand = output_operands[0];
  // Only supports 2D or 4D input
  if (input_dimensions_count != 2 && input_dimensions_count != 4) {
    bool is_ends_with_1 = true;
    for (uint32_t i = *axis + 1; i < input_dimensions_count; i++) {
      if (input_dimensions_data[i] != 1) {
        is_ends_with_1 = false;
      }
    }
    auto input_count =
        ProductionOfDimensions(input_dimensions_data, input_dimensions_count);
    auto axis_count = input_dimensions_data[*axis];
    auto remain_count = input_count / axis_count;
    std::vector<int32_t> reshape_input_dimensions = {
        static_cast<int32_t>(remain_count), static_cast<int32_t>(axis_count)};
    if (is_ends_with_1) {
      // Reshape the input operand to 2D and update axis to 1
      // Origin:
      // input_operand(dims=[3,4,1])->softmax(axis=1)->output_operand(dims=[3,4,1])
      // After the step:
      // input_operand(dims=[3,4,1])->reshape(shape=[3,4])->reshape_input_operand(dims=[3,4])
      auto reshape_input_operand = AppendReshapeOperation(
          model, input_operand, reshape_input_dimensions);
      // After the step:
      // input_operand(dims=[3,4,1])->reshape(shape=[3,4])->reshape_input_operand(dims=[3,4])->softmax(axis=1)->output_operand(dims=[3,4,1])
      UpdateOperationInputOperands(
          {operation}, input_operand, reshape_input_operand);
      // After the step:
      // reshape_output_operand(dims=[3,4])->reshape(shape=[3,4,1])->output_operand(dims=[3,4,1])
      auto reshape_output_operand = InsertReshapeOperation(
          model, output_operand, reshape_input_operand->type.dimensions);
      // After the step:
      // input_operand(dims=[3,4,1])->reshape(shape=[3,4])->reshape_input_operand(dims=[3,4])->softmax(axis=1)->reshape_output_operand(dims=[3,4])->reshape(shape=[3,4,1])->output_operand(dims=[3,4,1])
      UpdateOperationOutputOperands(
          operation, output_operand, reshape_output_operand);
    } else {
      // Origin:
      // input_operand(dims=[2,192,128])->softmax(axis=1)->output_operand(dims=[2,192,128])
      // After the step:
      // input_operand(dims=[2,192,128])->transpose(perm=[0,2,1])->transpose_input_operand(dims=[2,128,192])
      std::vector<int32_t> transpose_input_permutation(input_dimensions_count);
      for (uint32_t i = 0; i < input_dimensions_count; i++) {
        if (i < *axis) {
          transpose_input_permutation[i] = i;
        } else if (i > *axis) {
          transpose_input_permutation[i - 1] = i;
        } else {
          transpose_input_permutation[input_dimensions_count - 1] = *axis;
        }
      }
      auto transpose_input_operand = AppendTransposeOperation(
          model, input_operand, transpose_input_permutation);
      // After the step:
      // input_operand(dims=[2,192,128])->transpose(perm=[0,2,1])->transpose_input_operand(dims=[2,128,192])->reshape(shape=[384,192])->reshape_transpose_input_operand(dims=[384,128])
      auto reshape_transpose_input_operand = AppendReshapeOperation(
          model, transpose_input_operand, reshape_input_dimensions);
      // After the step:
      // input_operand(dims=[2,192,128])->transpose(perm=[0,2,1])->transpose_input_operand(dims=[2,128,192])->reshape(shape=[384,192])->reshape_transpose_input_operand(dims=[384,128])->softmax(axis=1)->output_operand(dims=[2,192,128])
      UpdateOperationInputOperands(
          {operation}, input_operand, reshape_transpose_input_operand);
      // After the step:
      // reshape_output_operand(dims=[2,128,192])->transpose(shape=[0,2,1])->output_operand(dims=[2,192,128])
      auto transpose_output_operand = InsertTransposeOperation(
          model,
          output_operand,
          InversePermutation(transpose_input_permutation));
      // After the step:
      // reshape_output_operand(dims=[384,192])->reshape(shape=[2,128,192])->reshape_output_operand(dims=[2,128,192])->transpose(perm=[0,2,1])->output_operand(dims=[2,192,128])
      auto reshape_transpose_output_operand = InsertReshapeOperation(
          model,
          transpose_output_operand,
          reshape_transpose_input_operand->type.dimensions);
      // After the step:
      // input_operand(dims=[2,192,128])->transpose(perm=[0,2,1])->transpose_input_operand(dims=[2,128,192])->reshape(shape=[384,192])->reshape_transpose_input_operand(dims=[384,128])->softmax(axis=1)->reshape_output_operand(dims=[384,192])->reshape(shape=[2,128,192])->reshape_output_operand(dims=[2,128,192])->transpose(perm=[0,2,1])->output_operand(dims=[2,192,128])
      UpdateOperationOutputOperands(
          operation, output_operand, reshape_transpose_output_operand);
    }
    *axis = 1;
  }
}

void ResolveOperationLiminations(core::Model* model) {
  std::vector<core::Operation*> operations =
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
