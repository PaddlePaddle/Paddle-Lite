// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "optimizer/eliminate_fill_constant.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void EliminateFillConstant(hal::Model* model) {
  NNADAPTER_VLOG(5) << "Start EliminateFillConstant";
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_FILL) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 2);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto shape_operand = input_operands[0];
      auto value_operand = input_operands[1];
      auto fill_output_operand = output_operands[0];
      auto fill_consumers = GetOperandConsumers(model, fill_output_operand);
      if (fill_consumers.size() != 1) {
        continue;
      }
      if (IsConstantOperand(value_operand)) {
        fill_output_operand->type.lifetime = NNADAPTER_CONSTANT_REFERENCE;
        memcpy(fill_output_operand->buffer,
               value_operand->buffer,
               shape_operand->length);
        // Clean
        RemoveOperand(model, value_operand);
        RemoveOperand(model, shape_operand);
        RemoveOperation(model, operation);
      }
    }
  }
}

}  // namespace nnadapter
