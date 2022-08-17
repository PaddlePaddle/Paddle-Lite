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

#include "optimizer/constant_fold_shape_and_associated_operations.h"
#include <set>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void ConstantFoldShapeAndAssociatedOperations(
    core::Model *model) {
  NNADAPTER_LOG(INFO) << "Run ConstantFoldShapeAndAssociatedOperations Pass";
  std::vector<core::Operation *> operations =
      SortOperationsInTopologicalOrder(model);
  // Check whether to support dynamic shape in the model
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    for (auto operand : input_operands) {
      if (operand && IsModelInputOperand(operand)) {
        break;
      }
      if (operand && IsTemporaryVariableOperand(operand) &&
          IsOperandWithDynamicShape(operand)) {
        NNADAPTER_LOG(WARNING)
            << "Skip if dynamic shape need to be supported in the model!";
        return;
      }
    }
  }
  // Operands that will not be removed
  std::set<core::Operand *> white_operands;
  // Operands and operations to be deleted
  std::set<core::Operand *> remove_operands;
  std::set<core::Operation *> remove_operations;
  // Collect operands and operations that need to be deleted
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    auto output_operands = operation->output_operands;
    if (operation->type == NNADAPTER_SHAPE) {
      for (auto operand : output_operands) {
        remove_operands.insert(operand);
      }
      remove_operations.insert(operation);
    } else if (operation->type == NNADAPTER_RESIZE_LINEAR ||
               operation->type == NNADAPTER_RESIZE_NEAREST) {
      white_operands.insert(input_operands[1]);
      white_operands.insert(input_operands[2]);
    } else if (operation->type == NNADAPTER_EXPAND ||
               operation->type == NNADAPTER_RESHAPE) {
      white_operands.insert(input_operands[1]);
    } else {
      bool is_tempory_shape_op = true;
      for (auto input_operand : input_operands) {
        if (IsTemporaryVariableOperand(input_operand) ||
            IsModelInputOperand(input_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (!is_tempory_shape_op) continue;
      for (auto output_operand : output_operands) {
        if (IsTemporaryVariableOperand(output_operand) ||
            IsModelOutputOperand(output_operand)) {
          is_tempory_shape_op = false;
          break;
        }
      }
      if (is_tempory_shape_op) {
        for (auto operand : input_operands) {
          remove_operands.insert(operand);
        }
        for (auto operand : output_operands) {
          remove_operands.insert(operand);
        }
        remove_operations.insert(operation);
      }
    }
  }
  // The operations cannot be deleted completely
  if (operations.size() == remove_operations.size()) {
    NNADAPTER_LOG(WARNING)
        << "Skip! The operations cannot be deleted completely.";
    return;
  }
  // Clean
  for (auto remove_operand : remove_operands) {
    if (!white_operands.count(remove_operand)) {
      RemoveOperand(model, remove_operand);
      NNADAPTER_VLOG(5) << "Operand: " << OperandIdToString(remove_operand)
                        << " is deleted!";
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
      remove_operand->buffer = malloc(remove_operand->length);
      memcpy(
          remove_operand->buffer, temporary_shape.data, remove_operand->length);
      NNADAPTER_VLOG(5) << "Operand: " << OperandIdToString(remove_operand)
                        << " is in constant folding!";
    }
  }
  for (auto remove_operation : remove_operations) {
    RemoveOperation(model, remove_operation);
    NNADAPTER_VLOG(5) << "Operation: "
                      << OperationTypeToString(remove_operation->type)
                      << " is deleted!";
  }
}

}  // namespace nnadapter
