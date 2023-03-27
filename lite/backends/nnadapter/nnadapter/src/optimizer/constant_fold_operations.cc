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

#include "optimizer/constant_fold_operations.h"
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

#define REGISTER_OPERATION(__op_type__,             \
                           __validate_func_name__,  \
                           __prepare_func_name__,   \
                           __execute_func_name__,   \
                           ...)                     \
  extern int __validate_func_name__(                \
      const nnadapter::core::Operation *operation); \
  extern int __execute_func_name__(nnadapter::core::Operation *operation);
namespace nnadapter {
namespace operation {
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_OPERATION_ALL_H__
}  // namespace operation
}  // namespace nnadapter
#undef REGISTER_OPERATION

namespace nnadapter {

// Fold operation if its inputs are all constant.
class ConstantFoldOperationsFuser : public PatternMatcher {
 public:
  ConstantFoldOperationsFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(
      core::Model *model, const std::map<std::string, Node *> &nodes) override;
};

void ConstantFoldOperationsFuser::BuildPattern() {
  // Create patterns
  CreatePattern("operation")->IsOperation();
}

bool ConstantFoldOperationsFuser::HandleMatchedResults(
    core::Model *model, const std::map<std::string, Node *> &nodes) {
  auto operation = nodes.at("operation")->operation;
  auto input_operands = operation->input_operands;
  bool is_input_constant = true;
  for (auto input_operand : input_operands) {
    if (input_operand != nullptr && !IsConstantOperand(input_operand)) {
      is_input_constant = false;
      break;
    }
  }
  // Some ops should be considered separately
  auto op_type = operation->type;
  switch (op_type) {
    case NNADAPTER_SHAPE:
      is_input_constant = !IsDynamicShapeOperandType(input_operands[0]->type);
      break;
    case NNADAPTER_FILL_LIKE:
      is_input_constant = !IsDynamicShapeOperandType(input_operands[0]->type) &&
                          IsConstantOperand(input_operands[1]);
      break;
    default:
      break;
  }
  if (!is_input_constant) return false;

  // Calculate out if it is a foldable operation.
  switch (op_type) {
#define REGISTER_OPERATION(__op_type__,                            \
                           __validate_func_name__,                 \
                           __prepare_func_name__,                  \
                           __execute_func_name__)                  \
  case NNADAPTER_##__op_type__:                                    \
    if (!operation::__validate_func_name__(operation)) {           \
      NNADAPTER_LOG(WARNING) << "Can't fold "                      \
                             << OperationTypeToString(op_type)     \
                             << " because it is not validate.";    \
      return false;                                                \
    }                                                              \
    NNADAPTER_CHECK(operation::__execute_func_name__(operation) == \
                    NNADAPTER_NO_ERROR);                           \
    break;
#include "operation/all.h"  // NOLINT
#undef __NNADAPTER_OPERATION_ALL_H__
#undef REGISTER_OPERATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported operation("
                           << OperationTypeToString(operation->type)
                           << ") is found.";
      break;
  }

  // Set outs' lifetime.
  // Remove useless input_operand.
  // Remove the foldable operation.
  for (auto output_operand : operation->output_operands) {
    output_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  }
  for (auto input_operand : input_operands) {
    if (GetOperandConsumers(model, input_operand).size() == 1) {
      RemoveOperand(model, input_operand);
    }
  }
  RemoveOperation(model, operation);
  return true;
}

NNADAPTER_EXPORT void ConstantFoldOperations(core::Model *model) {
  NNADAPTER_VLOG(5) << "Apply ConstantFoldOperationsFuser";
  bool stop;
  do {
    ConstantFoldOperationsFuser constant_fold_operations_fuser;
    stop = constant_fold_operations_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
