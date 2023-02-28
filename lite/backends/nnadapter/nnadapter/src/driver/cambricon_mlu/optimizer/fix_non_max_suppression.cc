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

#include "driver/cambricon_mlu/optimizer/fix_non_max_suppression.h"
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "optimizer/pattern_matcher.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

class NMSFixer : public PatternMatcher {
 public:
  void BuildPattern() override;
  bool HandleMatchedResults(core::Model* model,
                            const std::map<std::string, Node*>& nodes) override;
};

// clang-format off
/*
*                            3*yolobox   assign                                     3*yolobox
*                               |          |                                            |
* 3*yolo_box  assign_value   transpose   transpose                    3*yolo_box    transpose
*   |           |                |         |                               |            |
*        |                          |                                      |            |
*     concat0                    concat1                                concat0      concat1
*        |                          |                                      |            |
*              |                     \                                           |
*        multiclass_nms3           squeeze                                 multiclass_nms3
*        |         |                  |                                      |       |
*        |       gather     <---   transpose                                 |       |
*        |        /                                                          |       |
*         concat2                                       ===>                  concat2
*           |                                                                    |
*         split                                                                split
*           |                                                                    |
*       roi_align                                                            roi_align
*/
// clang-format on
void NMSFixer::BuildPattern() {
  // Operation patterns
  auto concat0_pattern =
      CreatePattern("concat0", NNADAPTER_CONCAT)
          ->MatchCondition([](const Node* node) -> bool {
            auto operation = node->operation;
            return operation && operation->input_operands.size() == 5;
          });
  auto transpose0_pattern =
      CreatePattern("transpose0", NNADAPTER_TRANSPOSE)->IsIntermediate();
  auto concat1_pattern =
      CreatePattern("concat1", NNADAPTER_CONCAT)
          ->MatchCondition([](const Node* node) -> bool {
            auto operation = node->operation;
            return operation && operation->input_operands.size() == 5;
          });
  auto nms_pattern = CreatePattern("nms", NNADAPTER_NON_MAX_SUPPRESSION)
                         ->MatchCondition([](const Node* node) -> bool {
                           auto operation = node->operation;
                           return operation &&
                                  operation->input_operands.size() == 11 &&
                                  operation->output_operands.size() >= 2;
                         });
  auto squeeze0_pattern =
      CreatePattern("squeeze0", NNADAPTER_SQUEEZE)->IsIntermediate();
  auto transpose1_pattern =
      CreatePattern("transpose1", NNADAPTER_TRANSPOSE)->IsIntermediate();
  auto gather0_pattern =
      CreatePattern("gather0", NNADAPTER_GATHER)->IsIntermediate();
  auto concat2_pattern = CreatePattern("concat2", NNADAPTER_CONCAT);
  auto split0_pattern = CreatePattern("split0", NNADAPTER_SPLIT);

  // Operand patterns
  auto concat0_0 =
      CreatePattern("concat0_0")->IsOperationInputOperand(NNADAPTER_CONCAT, 0);
  auto concat0_1 =
      CreatePattern("concat0_1")->IsOperationInputOperand(NNADAPTER_CONCAT, 1);
  auto concat0_2 =
      CreatePattern("concat0_2")->IsOperationInputOperand(NNADAPTER_CONCAT, 2);
  auto concat0_3 = CreatePattern("concat0_3")
                       ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                       ->IsIntermediate();
  auto concat0_axis = CreatePattern("concat0_axis")
                          ->IsOperationInputOperand(NNADAPTER_CONCAT, 4);

  auto transpose0_input = CreatePattern("transpose0_input")
                              ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0)
                              ->IsIntermediate();
  auto transpose0_perm = CreatePattern("transpose0_perm")
                             ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
                             ->IsIntermediate();

  auto concat1_0 =
      CreatePattern("concat1_0")->IsOperationInputOperand(NNADAPTER_CONCAT, 0);
  auto concat1_1 =
      CreatePattern("concat1_1")->IsOperationInputOperand(NNADAPTER_CONCAT, 1);
  auto concat1_2 =
      CreatePattern("concat1_2")->IsOperationInputOperand(NNADAPTER_CONCAT, 2);
  auto concat1_3 = CreatePattern("concat1_3")
                       ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                       ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
                       ->IsIntermediate();
  auto concat1_axis = CreatePattern("concat1_axis")
                          ->IsOperationInputOperand(NNADAPTER_CONCAT, 4);

  auto squeeze0_axis = CreatePattern("squeeze0_axis")
                           ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 1)
                           ->IsIntermediate();

  auto transpose1_input = CreatePattern("transpose1_input")
                              ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 0)
                              ->IsOperationOutputOperand(NNADAPTER_SQUEEZE, 0)
                              ->IsIntermediate();
  auto transpose1_perm = CreatePattern("transpose1_perm")
                             ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
                             ->IsIntermediate();

  auto nms_bboxes =
      CreatePattern("nms_bboxes")
          ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
          ->IsOperationInputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 0);
  auto nms_scores =
      CreatePattern("nms_scores")
          ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
          ->IsOperationInputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 1)
          ->IsOperationInputOperand(NNADAPTER_SQUEEZE, 0);
  auto nms_out_index = CreatePattern("nms_out_index");

  auto gather0_input = CreatePattern("gather0_input")
                           ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
                           ->IsOperationInputOperand(NNADAPTER_GATHER, 0)
                           ->IsIntermediate();
  auto gather0_indices =
      CreatePattern("gather0_indices")
          ->IsOperationOutputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 2)
          ->IsOperationInputOperand(NNADAPTER_GATHER, 1);
  auto gather0_axis = CreatePattern("gather0_axis")
                          ->IsOperationInputOperand(NNADAPTER_GATHER, 2)
                          ->IsIntermediate();

  auto concat2_0 =
      CreatePattern("concat2_0")
          ->IsOperationOutputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 0)
          ->IsOperationInputOperand(NNADAPTER_CONCAT, 0);
  auto concat2_1 = CreatePattern("concat2_1")
                       ->IsOperationOutputOperand(NNADAPTER_GATHER, 0)
                       ->IsOperationInputOperand(NNADAPTER_CONCAT, 1);
  auto concat2_axis = CreatePattern("concat2_axis")
                          ->IsOperationInputOperand(NNADAPTER_CONCAT, 2);

  auto split0_input = CreatePattern("split0_input")
                          ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
                          ->IsOperationInputOperand(NNADAPTER_SPLIT, 0);
  auto split0_axis =
      CreatePattern("split0_axis")->IsOperationInputOperand(NNADAPTER_SPLIT, 1);
  auto split0_size = CreatePattern("split0_size")
                         ->IsOperationInputOperand(NNADAPTER_SPLIT, 2)
                         ->IsConstantOperand()
                         ->IsIntermediate();
  auto split0_out_0 = CreatePattern("split0_out_0")
                          ->IsOperationOutputOperand(NNADAPTER_SPLIT, 0);
  auto split0_out_1 = CreatePattern("split0_out_1")
                          ->IsOperationOutputOperand(NNADAPTER_SPLIT, 1);

  // Create the topological connections for the above patterns
  std::vector<Pattern*> concat0_input_patterns{
      concat0_0, concat0_1, concat0_2, concat0_3, concat0_axis};
  std::vector<Pattern*> transpose0_input_patterns{transpose0_input,
                                                  transpose0_perm};
  std::vector<Pattern*> concat1_input_patterns{
      concat1_0, concat1_1, concat1_2, concat1_3, concat1_axis};
  std::vector<Pattern*> squeeze0_input_patterns{nms_scores, squeeze0_axis};
  std::vector<Pattern*> transpose1_input_patterns{transpose1_input,
                                                  transpose1_perm};
  std::vector<Pattern*> gather0_input_patterns{
      gather0_input, gather0_indices, gather0_axis};
  std::vector<Pattern*> nms_input_patterns{nms_bboxes, nms_scores};
  std::vector<Pattern*> nms_output_patterns{
      concat2_0, nms_out_index, gather0_indices};

  std::vector<Pattern*> concat2_input_patterns{
      concat2_0, concat2_1, concat2_axis};

  std::vector<Pattern*> split0_input_patterns{
      split0_input, split0_axis, split0_size};

  std::vector<Pattern*> split0_output_patterns{split0_out_0, split0_out_1};

  concat0_input_patterns >> *concat0_pattern >> *nms_bboxes;
  transpose0_input_patterns >> *transpose0_pattern >> *concat1_3;
  concat1_input_patterns >> *concat1_pattern >> *nms_scores;
  squeeze0_input_patterns >> *squeeze0_pattern >> *transpose1_input;
  transpose1_input_patterns >> *transpose1_pattern >> *gather0_input;
  nms_input_patterns >> *nms_pattern >> nms_output_patterns;
  gather0_input_patterns >> *gather0_pattern >> *concat2_1;
  concat2_input_patterns >> *concat2_pattern >> *split0_input;
  split0_input_patterns >> *split0_pattern >> split0_output_patterns;
}

bool NMSFixer::HandleMatchedResults(core::Model* model,
                                    const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto concat0_operation = nodes.at("concat0")->operation;
  auto concat0_operand0 = concat0_operation->input_operands[0];
  auto concat0_operand1 = concat0_operation->input_operands[1];
  auto concat0_operand2 = concat0_operation->input_operands[2];
  auto concat0_operand4 = concat0_operation->input_operands[4];
  concat0_operation->input_operands = {
      concat0_operand0, concat0_operand1, concat0_operand2, concat0_operand4};

  auto concat1_operation = nodes.at("concat1")->operation;
  auto concat1_operand0 = concat1_operation->input_operands[0];
  auto concat1_operand1 = concat1_operation->input_operands[1];
  auto concat1_operand2 = concat1_operation->input_operands[2];
  auto concat1_operand4 = concat1_operation->input_operands[4];
  concat1_operation->input_operands = {
      concat1_operand0, concat1_operand1, concat1_operand2, concat1_operand4};

  auto concat2_operation = nodes.at("concat2")->operation;
  auto concat2_operand0 = concat2_operation->input_operands[0];
  auto concat2_operand2 = concat2_operation->input_operands[2];
  auto nms_operation = nodes.at("nms")->operation;
  auto nms_index_operand = nms_operation->output_operands[2];
  auto concat2_out_operand = concat2_operation->output_operands[0];
  concat2_operation->input_operands = {
      concat2_operand0, nms_index_operand, concat2_operand2};
  concat2_out_operand->type.dimensions.data[1] = 7;

  auto split_operation = nodes.at("split0")->operation;
  auto axis_operand = split_operation->input_operands[1];
  core::Operand* split_size_operand = AddInt32ConstantOperand(model, {2, 4, 1});
  split_operation->input_operands = {
      concat2_out_operand, axis_operand, split_size_operand};

  // The matched intermediate operands and operations will be deleted only when
  // it returns true
  return true;
}

NNADAPTER_EXPORT void FixNonMaxSuppression(core::Model* model) {
  NMSFixer nms_fixer;
  nms_fixer.Apply(model);
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
