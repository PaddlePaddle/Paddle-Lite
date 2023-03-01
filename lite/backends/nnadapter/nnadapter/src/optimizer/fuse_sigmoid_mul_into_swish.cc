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

#include "optimizer/fuse_sigmoid_mul_into_swish.h"
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

class SigmoidMulFuser : public PatternMatcher {
 public:
  SigmoidMulFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void SigmoidMulFuser::BuildPattern() {
  // Operation patterns
  auto sigmoid_pattern =
      CreatePattern("sigmoid", NNADAPTER_SIGMOID)->IsIntermediate();
  auto mul_pattern = CreatePattern("mul", NNADAPTER_MUL)->IsIntermediate();
  // Operand patterns
  auto sigmoid_input_pattern =
      CreatePattern("sigmoid_input")
          ->IsOperationInputOperand(NNADAPTER_SIGMOID, 0)
          ->IsOperationInputOperand(NNADAPTER_MUL, 0);
  auto sigmoid_output_pattern =
      CreatePattern("sigmoid_output")
          ->IsOperationOutputOperand(NNADAPTER_SIGMOID, 0)
          ->IsOperationInputOperand(NNADAPTER_MUL, 1)
          ->IsIntermediate();
  auto mul_fuse_code_pattern = CreatePattern("mul_fuse_code")
                                   ->IsOperationInputOperand(NNADAPTER_MUL, 2)
                                   ->IsIntermediate();
  auto mul_output_pattern =
      CreatePattern("mul_output")->IsOperationOutputOperand(NNADAPTER_MUL, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> mul_input_patterns{
      sigmoid_input_pattern, mul_fuse_code_pattern, sigmoid_output_pattern};
  *sigmoid_input_pattern >> *sigmoid_pattern >> *sigmoid_output_pattern;
  mul_input_patterns >> *mul_pattern >> *mul_output_pattern;
}

bool SigmoidMulFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto sigmoid_input_operand = nodes.at("sigmoid_input")->operand;
  auto mul_output_operand = nodes.at("mul_output")->operand;
  // Create a new NNADAPTER_SWISH operation and replace the matched
  // subgraph nodes.
  auto* swish_operation = AddOperation(model);
  swish_operation->type = NNADAPTER_SWISH;
  swish_operation->input_operands = {sigmoid_input_operand};
  swish_operation->output_operands = {mul_output_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseSigmoidMulIntoSwish(core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply SigmoidMulFuser";
  bool stop;
  do {
    SigmoidMulFuser sigmoid_mul_fuser;
    stop = sigmoid_mul_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
