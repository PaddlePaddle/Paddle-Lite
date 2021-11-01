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

#include "optimizer/matmul_elementwise_add_fusion.h"
#include <algorithm>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void MatMulElementwiseAddFusion(hal::Model* model) {
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
      auto transpose_x_operand = input_operands[2];
      auto transpose_y_operand = input_operands[3];
      auto mat_mul_out_operand = output_operands[0];
      auto mat_mul_consumers = GetOperandConsumers(model, mat_mul_out_operand);
      if (mat_mul_consumers.size() != 1 ||
          mat_mul_consumers[0]->type != NNADAPTER_ADD) {
        continue;
      }
      auto eltwise_add_operation = mat_mul_consumers[0];
      auto& eltwise_add_input_operands = eltwise_add_operation->input_operands;
      auto& eltwise_add_output_operands =
          eltwise_add_operation->output_operands;
      auto bias_operand = eltwise_add_input_operands[1];
      auto fuse_code_operand = eltwise_add_input_operands[2];
      auto output_operand = eltwise_add_output_operands[0];
      // Add FC operation
      TransposeOperand(y_operand, std::vector<int32_t>({1, 0}));
      auto fc_operation = AddOperation(model);
      fc_operation->type = NNADAPTER_FULLY_CONNECTED;
      fc_operation->input_operands = {
          x_operand, y_operand, bias_operand, fuse_code_operand};
      fc_operation->output_operands = {output_operand};
      // Clean
      RemoveOperand(model, transpose_x_operand);
      RemoveOperand(model, transpose_y_operand);
      RemoveOperand(model, mat_mul_out_operand);
      RemoveOperation(model, operation);
      RemoveOperation(model, mat_mul_consumers[0]);
    }
  }
}

}  // namespace nnadapter
