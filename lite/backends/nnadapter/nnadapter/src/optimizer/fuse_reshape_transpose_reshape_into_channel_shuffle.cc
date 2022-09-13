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

#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
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

class ReshapeTransposeReshapeFuser : public PatternMatcher {
 public:
  ReshapeTransposeReshapeFuser() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void ReshapeTransposeReshapeFuser::BuildPattern() {
  // Operation patterns
  auto first_reshape_pattern =
      CreatePattern("first_reshape", NNADAPTER_RESHAPE)->IsIntermediate();
  auto transpose_pattern =
      CreatePattern("transpose", NNADAPTER_TRANSPOSE)->IsIntermediate();
  auto last_reshape_pattern =
      CreatePattern("last_reshape", NNADAPTER_RESHAPE)->IsIntermediate();
  // Operand patterns
  auto first_reshape_input_pattern =
      CreatePattern("first_reshape_input")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 0);
  auto first_reshape_shape_pattern =
      CreatePattern("first_reshape_shape")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 1)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            auto shape_count = operand->length / sizeof(int32_t);
            auto shape_data = reinterpret_cast<int32_t*>(operand->buffer);
            return shape_data && shape_count >= 5 && shape_data[1] > 0;
          })
          ->IsIntermediate();
  auto first_reshape_output_pattern =
      CreatePattern("first_reshape_output")
          ->IsOperationOutputOperand(NNADAPTER_RESHAPE, 0)
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  auto transpose_perm_pattern =
      CreatePattern("transpose_perm")
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            auto perm_count = operand->length / sizeof(int32_t);
            auto perm_data = reinterpret_cast<int32_t*>(operand->buffer);
            return perm_data && perm_count >= 5 && perm_data[1] == 2 &&
                   perm_data[2] == 1;
          })
          ->IsIntermediate();
  auto last_reshape_input_pattern =
      CreatePattern("last_reshape_input")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 0)
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  auto last_reshape_shape_pattern =
      CreatePattern("last_reshape_shape")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 1)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            auto shape_count = operand->length / sizeof(int32_t);
            auto shape_data = reinterpret_cast<int32_t*>(operand->buffer);
            return shape_data && shape_count >= 4;
          })
          ->IsIntermediate();
  auto last_reshape_output_pattern =
      CreatePattern("last_reshape_output")
          ->IsOperationOutputOperand(NNADAPTER_RESHAPE, 0);
  // Create the topological connections for the above patterns
  std::vector<Pattern*> first_reshape_input_patterns{
      first_reshape_input_pattern, first_reshape_shape_pattern};
  std::vector<Pattern*> transpose_input_patterns{first_reshape_output_pattern,
                                                 transpose_perm_pattern};
  std::vector<Pattern*> last_reshape_input_patterns{last_reshape_input_pattern,
                                                    last_reshape_shape_pattern};
  first_reshape_input_patterns >> *first_reshape_pattern >>
      *first_reshape_output_pattern;
  transpose_input_patterns >> *transpose_pattern >> *last_reshape_input_pattern;
  last_reshape_input_patterns >> *last_reshape_pattern >>
      *last_reshape_output_pattern;
}

bool ReshapeTransposeReshapeFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto first_reshape_input_operand = nodes.at("first_reshape_input")->operand;
  auto first_reshape_shape_operand = nodes.at("first_reshape_shape")->operand;
  auto first_reshape_shape_data =
      reinterpret_cast<int32_t*>(first_reshape_shape_operand->buffer);
  auto last_reshape_output_operand = nodes.at("last_reshape_output")->operand;
  // Create a new NNADAPTER_CHANNEL_SHUFFLE operation and replace the matched
  // subgraph nodes.
  auto* channel_shuffle_operation = AddOperation(model);
  channel_shuffle_operation->type = NNADAPTER_CHANNEL_SHUFFLE;
  auto channel_shuffle_group_operand =
      AddInt32ConstantOperand(model, first_reshape_shape_data[1]);
  channel_shuffle_operation->input_operands = {first_reshape_input_operand,
                                               channel_shuffle_group_operand};
  channel_shuffle_operation->output_operands = {last_reshape_output_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  return true;
}

NNADAPTER_EXPORT void FuseReshapeTransposeReshapeIntoChannelShuffle(
    core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply ReshapeTransposeReshapeFuser";
  bool stop;
  do {
    ReshapeTransposeReshapeFuser reshape_transpose_reshape_fuser;
    stop = reshape_transpose_reshape_fuser.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
