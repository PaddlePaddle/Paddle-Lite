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

#include <algorithm>
#include <map>
#include <vector>
#include "optimizer/fuse_reshape_transpose_reshape_into_channel_shuffle.h"
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

class ReshapeTransposeReshape5DInto4DConverter : public PatternMatcher {
 public:
  ReshapeTransposeReshape5DInto4DConverter() {}
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

void ReshapeTransposeReshape5DInto4DConverter::BuildPattern() {
  // Operation patterns
  auto first_reshape_pattern =
      CreatePattern("first_reshape", NNADAPTER_RESHAPE);
  auto transpose_pattern = CreatePattern("transpose", NNADAPTER_TRANSPOSE);
  auto last_reshape_pattern = CreatePattern("last_reshape", NNADAPTER_RESHAPE);
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
          });
  auto first_reshape_output_pattern =
      CreatePattern("first_reshape_output")
          ->IsOperationOutputOperand(NNADAPTER_RESHAPE, 0)
          ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0);
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
          });
  auto last_reshape_input_pattern =
      CreatePattern("last_reshape_input")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 0)
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0);
  auto last_reshape_shape_pattern =
      CreatePattern("last_reshape_shape")
          ->IsOperationInputOperand(NNADAPTER_RESHAPE, 1)
          ->IsConstantOperand()
          ->MatchCondition([](const Node* node) -> bool {
            auto operand = node->operand;
            auto shape_count = operand->length / sizeof(int32_t);
            auto shape_data = reinterpret_cast<int32_t*>(operand->buffer);
            return shape_data && shape_count >= 4;
          });
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

bool ReshapeTransposeReshape5DInto4DConverter::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  // Modify first-reshape
  auto first_reshape_output_operand = nodes.at("first_reshape_output")->operand;
  auto first_reshape_output_shape_data =
      first_reshape_output_operand->type.dimensions.data;
  first_reshape_output_shape_data[3] *= first_reshape_output_shape_data[4];
  auto first_reshape_output_count = 4;
  first_reshape_output_operand->type.dimensions.count =
      first_reshape_output_count;
  auto first_reshape_shape_operand = nodes.at("first_reshape_shape")->operand;
  auto first_reshape_shape_data =
      reinterpret_cast<int32_t*>(first_reshape_shape_operand->buffer);
  memcpy(first_reshape_shape_data,
         first_reshape_output_shape_data,
         first_reshape_output_count * sizeof(int32_t));
  first_reshape_shape_operand->type.precision = NNADAPTER_INT32;
  first_reshape_shape_operand->length =
      first_reshape_output_count * sizeof(int32_t);
  // Modify transpose
  auto transpose_perm = nodes.at("transpose_perm")->operand;
  auto transpose_perm_data = reinterpret_cast<int32_t*>(transpose_perm->buffer);
  transpose_perm_data[0] = 0;
  transpose_perm_data[1] = 2;
  transpose_perm_data[2] = 1;
  transpose_perm_data[3] = 3;
  transpose_perm->length = 4 * sizeof(int32_t);
  auto transpose_out_operand = nodes.at("last_reshape_input")->operand;
  transpose_out_operand->type.dimensions.count = first_reshape_output_count;
  auto transpose_out_shape_data = transpose_out_operand->type.dimensions.data;
  transpose_out_shape_data[0] = first_reshape_shape_data[0];
  transpose_out_shape_data[1] = first_reshape_shape_data[2];
  transpose_out_shape_data[2] = first_reshape_shape_data[1];
  transpose_out_shape_data[3] = first_reshape_shape_data[3];
  // Modify last-reshape
  auto last_reshape_output_operand = nodes.at("last_reshape_output")->operand;
  auto last_reshape_output_count = 4;
  last_reshape_output_operand->type.dimensions.count =
      last_reshape_output_count;
  return true;
}

NNADAPTER_EXPORT void ConvertReshapeTransposeReshape5DInto4D(
    core::Model* model) {
  NNADAPTER_VLOG(5) << "Apply ReshapeTransposeReshape5DInto4DConverter";
  bool stop;
  do {
    ReshapeTransposeReshape5DInto4DConverter converter;
    stop = converter.Apply(model) == 0;
  } while (!stop);
}

}  // namespace nnadapter
