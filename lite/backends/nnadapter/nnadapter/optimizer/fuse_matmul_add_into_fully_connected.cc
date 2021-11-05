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

#include "optimizer/fuse_matmul_add_into_fully_connected.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void FuseMatMulAddIntoFullyConnected(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_MAT_MUL) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 4);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto x_operand = input_operands[0];
      auto y_operand = input_operands[1];
      if (IsConstantOperand(x_operand) || !IsConstantOperand(y_operand)) {
        NNADAPTER_VLOG(5) << "Only support x is tensor and y is persistable";
        continue;
      }
      if (x_operand->type.dimensions.count != 2 ||
          y_operand->type.dimensions.count != 2) {
        NNADAPTER_VLOG(5)
            << "Only support x's dims count and y's dims count is 2.";
        continue;
      }
      auto transpose_x_operand = input_operands[2];
      auto transpose_y_operand = input_operands[3];
      auto mat_mul_out_operand = output_operands[0];
      auto mat_mul_consumers = GetOperandConsumers(model, mat_mul_out_operand);
      if (mat_mul_consumers.size() == 0) {
        continue;
      }
      bool can_fuse = true;
      std::map<hal::Operation*, std::vector<hal::Operand*>> operation_map;
      // Process multiple add operation
      for (auto add_operation : mat_mul_consumers) {
        if (add_operation->type != NNADAPTER_ADD) {
          can_fuse = false;
          break;
        }
        auto& add_input_operands = add_operation->input_operands;
        auto& add_output_operands = add_operation->output_operands;
        NNADAPTER_CHECK_EQ(add_input_operands.size(), 3);
        NNADAPTER_CHECK_EQ(add_output_operands.size(), 1);
        auto bias_operand = add_input_operands[1];
        if (!IsConstantOperand(bias_operand)) {
          can_fuse = false;
          break;
        }
        auto fuse_code_operand = add_input_operands[2];
        if (*reinterpret_cast<bool*>(transpose_x_operand->buffer)) {
          auto x_shape_count = x_operand->type.dimensions.count;
          if (x_shape_count < 2) {
            can_fuse = false;
            break;
          }
          std::vector<int32_t> permutation;
          for (int32_t i = 0; i < x_shape_count - 2; i++) {
            permutation.push_back(i);
          }
          permutation.push_back(x_shape_count - 1);
          permutation.push_back(x_shape_count - 2);
          TransposeOperand(x_operand, permutation);
        }
        // If the transpose_y operand is false, transform the dimension to adapt
        // fc
        if (!*reinterpret_cast<bool*>(transpose_y_operand->buffer)) {
          if (y_operand->type.dimensions.count != 2) {
            can_fuse = false;
            break;
          }
          TransposeOperand(y_operand, std::vector<int32_t>({1, 0}));
        }
        // Add FC operation into map
        operation_map[add_operation] = {
            x_operand, y_operand, bias_operand, fuse_code_operand};
      }
      if (!can_fuse) {
        continue;
      }
      for (auto it = operation_map.begin(); it != operation_map.end(); ++it) {
        auto fc_operation = AddOperation(model);
        fc_operation->type = NNADAPTER_FULLY_CONNECTED;
        fc_operation->input_operands = it->second;
        fc_operation->output_operands = it->first->output_operands;
        RemoveOperation(model, it->first);
      }
      // Clean
      RemoveOperand(model, transpose_x_operand);
      RemoveOperand(model, transpose_y_operand);
      RemoveOperand(model, mat_mul_out_operand);
      RemoveOperation(model, operation);
    }
  }
}

}  // namespace nnadapter
