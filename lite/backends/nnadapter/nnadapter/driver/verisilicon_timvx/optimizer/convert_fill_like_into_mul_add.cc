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
static void InsertAdd(hal::Model* model,
                      hal::Operand* input_operand,
                      hal::Operand* value_operand) {
  // Insert a new operand after input_operand
  auto output_operand = AddOperand(model);
  memcpy(&output_operand->type,
         &input_operand->type,
         sizeof(NNAdapterOperandType));
  InsertOperand(model, input_operand, output_operand, true);
  auto fuse_code_operand = AddInt32ConstantOperand(model, 0);
  // Insert a new ADD operation between a input operand and output_operand
  auto dummy_add_operation = AddOperation(model);
  dummy_add_operation->type = NNADAPTER_ADD;
  dummy_add_operation->input_operands = {
      input_operand, value_operand, fuse_code_operand};
  dummy_add_operation->output_operands = {output_operand};
}

NNADAPTER_EXPORT void ConvertFillLikeIntoMulAdd(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_FILL_LIKE) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 2);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto input_operand = input_operands[0];
      auto value_operand = input_operands[1];
      input_operands.pop_back();
      auto output_operand = output_operands[0];
      // Add a zero multiply operand
      auto zero_operand = AddOperand(model);
      memcpy(&zero_operand->type,
             &input_operand->type,
             sizeof(NNAdapterOperandType));
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
      // if value is not zero , insert an add operation
      if (IsUInt8AsymmPerLayerQuantType(input_operand->type.precision)) {
        uint8_t value = *static_cast<uint8_t*>(value_operand->buffer);
        if (value == 0) {
          RemoveOperand(model, value_operand);
        } else {
          InsertAdd(model, output_operand, value_operand);
        }
      } else {
        float value = *static_cast<float*>(value_operand->buffer);
        if (value == 0.f) {
          RemoveOperand(model, value_operand);
        } else {
          InsertAdd(model, output_operand, value_operand);
        }
      }
    }
  }
}

}  // namespace nnadapter
