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

#include "optimizer/remove_tempory_shape_op.h"
#include <set>
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"

namespace nnadapter {

void RemoveTemporyShapeOp(core::Model *model) {
  NNADAPTER_LOG(INFO) << "Execute RemoveTemporyShapeOp pass";
  std::vector<core::Operation *> operations =
      SortOperationsInTopologicalOrder(model);
  std::set<std::vector<core::Operand *>> remove_operands;
  std::set<core::Operation *> remove_operations;
  // whitelist operands
  std::set<core::Operand *> white_operands;
  for (auto operation : operations) {
    if (operation->type == NNADAPTER_SHAPE) {
      remove_operands.insert(operation->output_operands);
      remove_operations.insert(operation);
    } else if (operation->type == NNADAPTER_RESIZE_LINEAR ||
               operation->type == NNADAPTER_RESIZE_NEAREST) {
      auto input_operands = operation->input_operands;
      white_operands.insert(input_operands[1]);
      white_operands.insert(input_operands[2]);
    } else {
      bool is_tempory_shape_op = true;
      auto input_operands = operation->input_operands;
      auto output_operands = operation->output_operands;
      for (auto input_operand : input_operands) {
        if (IsTemporaryVariableOperand(input_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (!is_tempory_shape_op) continue;
      for (auto output_operand : output_operands) {
        if (IsTemporaryVariableOperand(output_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (is_tempory_shape_op) {
        remove_operands.insert(operation->input_operands);
        remove_operands.insert(operation->output_operands);
        remove_operations.insert(operation);
      }
    }
  }

  for (auto remove_operand_vec : remove_operands) {
    for (auto remove_operand : remove_operand_vec) {
      if (!white_operands.count(remove_operand)) {
        RemoveOperand(model, remove_operand);
        NNADAPTER_VLOG(5) << "remove_operand: "
                          << OperandIdToString(remove_operand);
      } else {
        auto &temporary_shape = *(GetTemporaryShape(remove_operand));
        auto precision = remove_operand->type.precision;
        if (precision == NNADAPTER_INT32) {
          remove_operand->length =
              temporary_shape.count * static_cast<uint32_t>(sizeof(int32_t));
        } else if (precision == NNADAPTER_FLOAT32) {
          remove_operand->length =
              temporary_shape.count * static_cast<uint32_t>(sizeof(float));
        } else if (precision == NNADAPTER_INT64) {
          remove_operand->length =
              temporary_shape.count * static_cast<uint32_t>(sizeof(int64_t));
        }
        remove_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
        remove_operand->buffer = reinterpret_cast<void *>(temporary_shape.data);
      }
    }
  }
  for (auto remove_operation : remove_operations) {
    RemoveOperation(model, remove_operation);
    NNADAPTER_VLOG(5) << "remove_operations "
                      << OperationTypeToString(remove_operation->type);
  }
}

}  // namespace nnadapter
