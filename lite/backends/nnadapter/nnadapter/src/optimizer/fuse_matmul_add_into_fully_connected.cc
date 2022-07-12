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
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class MatMulAddFuser : public PatternMatcher {
 public:
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void MatMulAddFuser::BuildPattern() {
  // Operation patterns
  auto matmul_pattern =
      CreatePattern("matmul", NNADAPTER_MAT_MUL)
          ->MatchCondition([](const Node* node) -> bool {
            auto operation = node->operation;
            return operation && operation->input_operands.size() == 4 &&
                   operation->input_operands[0]->type.dimensions.count == 2 &&
                   operation->input_operands[1]->type.dimensions.count == 2;
          })
          ->IsIntermediate();
  auto add_pattern = CreatePattern("add", NNADAPTER_ADD)->IsIntermediate();
  // Operand patterns
  auto matmul_x_pattern = CreatePattern("matmul_x")
                              ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 0)
                              ->IsVariableOperand();
  auto matmul_y_pattern = CreatePattern("matmul_y")
                              ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 1)
                              ->IsConstantOperand();
  auto matmul_transpose_x_pattern =
      CreatePattern("matmul_transpose_x")
          ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 2)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            return operand && operand->buffer &&
                   !*reinterpret_cast<bool*>(operand->buffer);
          })
          ->IsIntermediate();
  auto matmul_transpose_y_pattern =
      CreatePattern("matmul_transpose_y")
          ->IsOperationInputOperand(NNADAPTER_MAT_MUL, 3)
          ->IsConstantOperand()
          ->IsIntermediate();
  auto matmul_output_pattern = CreatePattern("matmul_output")->IsIntermediate();
  auto add_y_pattern = CreatePattern("add_y")
                           ->IsOperationInputOperand(NNADAPTER_ADD, 1)
                           ->IsConstantOperand()
                           ->MatchCondition([](const Node* node) -> bool {
                             auto operand = node->operand;
                             return operand->type.dimensions.count == 1;
                           });
  auto add_fuse_code_pattern = CreatePattern("add_fuse_code")
                                   ->IsOperationInputOperand(NNADAPTER_ADD, 2)
                                   ->IsConstantOperand();
  auto add_output_pattern = CreatePattern("add_output");
  // Create the topological connections for the above patterns
  std::vector<Pattern*> matmul_input_patterns{matmul_x_pattern,
                                              matmul_y_pattern,
                                              matmul_transpose_x_pattern,
                                              matmul_transpose_y_pattern};
  std::vector<Pattern*> add_input_patterns{
      matmul_output_pattern, add_y_pattern, add_fuse_code_pattern};
  matmul_input_patterns >> *matmul_pattern >> *matmul_output_pattern;
  add_input_patterns >> *add_pattern >> *add_output_pattern;
}

bool MatMulAddFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto matmul_operation = nodes.at("matmul")->operation;
  auto matmul_x_operand = matmul_operation->input_operands[0];
  auto matmul_y_operand = matmul_operation->input_operands[1];
  auto matmul_transpose_y_operand = matmul_operation->input_operands[3];
  auto add_operation = nodes.at("add")->operation;
  auto add_bias_operand = add_operation->input_operands[1];
  auto add_fuse_code_operand = add_operation->input_operands[2];
  auto add_output_operand = add_operation->output_operands[0];
  if (!*reinterpret_cast<bool*>(matmul_transpose_y_operand->buffer)) {
    TransposeOperand(matmul_y_operand, std::vector<int32_t>({1, 0}));
  }
  // Create a new NNADAPTER_FULLY_CONNECTED operation and replace the matched
  // subgraph nodes.
  auto* fully_connected_operation = AddOperation(model);
  fully_connected_operation->type = NNADAPTER_FULLY_CONNECTED;
  fully_connected_operation->input_operands = {matmul_x_operand,
                                               matmul_y_operand,
                                               add_bias_operand,
                                               add_fuse_code_operand};
  fully_connected_operation->output_operands = {add_output_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseMatMulAddIntoFullyConnected(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply MatMulAddFuser";
  bool stop;
  do {
    MatMulAddFuser mat_mul_add_fuser;
    stop = mat_mul_add_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
