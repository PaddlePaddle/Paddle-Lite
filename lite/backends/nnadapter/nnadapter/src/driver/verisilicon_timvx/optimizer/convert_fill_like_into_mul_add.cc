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

#include "driver/verisilicon_timvx/optimizer/convert_fill_like_into_mul_add.h"
#include <algorithm>
#include <map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

/**
 * fill_like --value==0--> eltwise_mul(zero)
 *          |
 *           --value!=0--> eltwise_mul(zero)+ eltwise_add(value)
 * Such as:
 * value: 0
 *
 *             input        zero_value(0)
 *                \            /
 *                  \        /
 *                  [eltwise_mul]
 *                      |
 *                      |
 *                    output
 *
 * value: 2
 *
 *             input        zero_value(0)
 *                \            /
 *                  \        /
 *                  [eltwise_mul]
 *                      |
 *                      |
 *                  new_input       value(2)
 *                      |          /
 *                      |        /
 *                    [elementwise_add]
 *                            |
 *                            |
 *                          output
 *
 */
NNADAPTER_EXPORT void ConvertFillLikeIntoMulAdd(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_FILL_LIKE) {
      // y = fill_like(x, value) => y = add(mul(x, 0), value)
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 2);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto input_operand = input_operands[0];
      auto value_operand = input_operands[1];
      input_operands.pop_back();
      auto output_operand = output_operands[0];
      // Multiply a zero operand
      auto zero_operand = AddOperand(model);
      CopyOperandType(&zero_operand->type, input_operand->type);
      zero_operand->type.dimensions.count = 1;
      zero_operand->type.dimensions.data[0] = 1;
      zero_operand->length =
          GetOperandPrecisionDataLength(zero_operand->type.precision);
      zero_operand->buffer = malloc(zero_operand->length);
      NNADAPTER_CHECK(zero_operand->buffer != nullptr)
          << "Failed to allocate " << zero_operand->length
          << " bytes for the buffer of an operand, out of memory!";
      memset(zero_operand->buffer, 0, zero_operand->length);
      zero_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
      input_operands.push_back(zero_operand);
      auto fuse_code_operand = AddInt32ConstantOperand(model, 0);
      input_operands.push_back(fuse_code_operand);
      operation->type = NNADAPTER_MUL;
      // Insert a Add operation to add the constant value if value is not zero
      if (!IsAllZeros(value_operand->buffer, value_operand->length)) {
        auto immediate_operand = AddOperand(model);
        CopyOperandType(&immediate_operand->type, input_operand->type);
        if (!IsTemporaryShapeOperand(input_operand)) {
          immediate_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
        }
        auto fuse_code_operand = AddInt32ConstantOperand(model, 0);
        auto add_operation = AddOperation(model);
        add_operation->type = NNADAPTER_ADD;
        add_operation->input_operands = {
            immediate_operand, value_operand, fuse_code_operand};
        add_operation->output_operands = {output_operand};
        UpdateOperationOutputOperands(
            operation, output_operand, immediate_operand);
      } else {
        RemoveOperand(model, value_operand);
      }
    }
  }
}

}  // namespace nnadapter
