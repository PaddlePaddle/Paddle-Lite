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
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

// clang-format off
/*
* 3*yolo_box  assign_value   3*yolo_box->transpose   assign->transpose                    3*yolo_box    3*yolo_box->transpose
*   |           |                |                       |                                    |                |
*        |                          |                                                         |                |
*     concat0                    concat1                                                   concat0          concat1
*        |                          |                                                         |                |
*              |                                                                                       |
*        multiclass_nms3                                                                        multiclass_nms3
*        |         |                                                                               |       |
*        |       gather                                                                            |       |
*        |        /                                                                                |       |
*         concat2                                                       ===>                        concat2
*           |                                                                                          |
*         split                                                                                      split
*           |                                                                                          |
*       roi_align                                                                                  roi_align
*/
// clang-format on
void FixNonMaxSuppression(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_NON_MAX_SUPPRESSION) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 11);
      NNADAPTER_CHECK_GE(output_operands.size(), 2);

      auto nms_boxes_operand = input_operands[0];
      auto nms_scores_operand = input_operands[1];
      auto nms_out_operand = output_operands[0];
      auto nms_index_operand = output_operands[2];

      std::map<std::string, core::Operation*> operation_map;
      auto nms_box_pre_operation = GetOperandProducer(model, nms_boxes_operand);
      if (nms_box_pre_operation->type != NNADAPTER_CONCAT) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(nms_box_pre_operation->type)
                          << " ...";
        continue;
      }
      if (nms_box_pre_operation->input_operands.size() != 5) {
        continue;
      }
      operation_map["concat_0"] = nms_box_pre_operation;
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(nms_box_pre_operation->type);

      auto nms_score_pre_operation =
          GetOperandProducer(model, nms_scores_operand);
      if (nms_score_pre_operation->type != NNADAPTER_CONCAT) {
        NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(
                                                  nms_score_pre_operation->type)
                          << " ...";
        continue;
      }
      if (nms_score_pre_operation->input_operands.size() != 5) {
        continue;
      }
      operation_map["concat_1"] = nms_score_pre_operation;
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(nms_score_pre_operation->type);

      auto transpose1_operation =
          GetOperandProducer(model, nms_score_pre_operation->input_operands[3]);
      if (transpose1_operation->type != NNADAPTER_TRANSPOSE) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(transpose1_operation->type)
                          << " ...";
        continue;
      }
      operation_map["transpose_1"] = transpose1_operation;
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(transpose1_operation->type);

      auto gather_operation = GetOperandConsumers(model, nms_index_operand);
      if (gather_operation[0]->type != NNADAPTER_GATHER) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(gather_operation[0]->type)
                          << " ...";
        continue;
      }
      operation_map["gather_0"] = gather_operation[0];
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(gather_operation[0]->type);

      auto transpose2_operation =
          GetOperandProducer(model, gather_operation[0]->input_operands[0]);
      if (transpose2_operation->type != NNADAPTER_TRANSPOSE) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(transpose2_operation->type)
                          << " ...";
        continue;
      }
      operation_map["transpose_2"] = transpose2_operation;
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(transpose2_operation->type);

      auto squeeze_operation =
          GetOperandProducer(model, transpose2_operation->input_operands[0]);
      if (squeeze_operation->type != NNADAPTER_SQUEEZE) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(squeeze_operation->type)
                          << " ...";
        continue;
      }
      operation_map["squeeze_0"] = squeeze_operation;
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(squeeze_operation->type);

      auto concat2_operation = GetOperandConsumers(model, nms_out_operand);
      if (concat2_operation[0]->type != NNADAPTER_CONCAT) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(concat2_operation[0]->type)
                          << " ...";
        continue;
      }
      operation_map["concat_2"] = concat2_operation[0];
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(concat2_operation[0]->type);

      auto split_operation =
          GetOperandConsumers(model, concat2_operation[0]->output_operands[0]);
      if (split_operation[0]->type != NNADAPTER_SPLIT) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(split_operation[0]->type)
                          << " ...";
        continue;
      }
      operation_map["split_0"] = split_operation[0];
      NNADAPTER_VLOG(5) << "Finding : "
                        << OperationTypeToString(split_operation[0]->type);

      auto concat0_operand0 = operation_map["concat_0"]->input_operands[0];
      auto concat0_operand1 = operation_map["concat_0"]->input_operands[1];
      auto concat0_operand2 = operation_map["concat_0"]->input_operands[2];
      auto concat0_operand3 = operation_map["concat_0"]->input_operands[3];
      auto concat0_operand4 = operation_map["concat_0"]->input_operands[4];
      RemoveOperand(model, concat0_operand3);
      operation_map["concat_0"]->input_operands = {concat0_operand0,
                                                   concat0_operand1,
                                                   concat0_operand2,
                                                   concat0_operand4};

      RemoveOperand(model, operation_map["transpose_1"]->input_operands[0]);
      RemoveOperand(model, operation_map["transpose_1"]->input_operands[1]);
      RemoveOperation(model, operation_map["transpose_1"]);
      auto concat1_operand0 = operation_map["concat_1"]->input_operands[0];
      auto concat1_operand1 = operation_map["concat_1"]->input_operands[1];
      auto concat1_operand2 = operation_map["concat_1"]->input_operands[2];
      auto concat1_operand3 = operation_map["concat_1"]->input_operands[3];
      auto concat1_operand4 = operation_map["concat_1"]->input_operands[4];
      RemoveOperand(model, concat1_operand3);
      operation_map["concat_1"]->input_operands = {concat1_operand0,
                                                   concat1_operand1,
                                                   concat1_operand2,
                                                   concat1_operand4};

      RemoveOperand(model, operation_map["squeeze_0"]->input_operands[1]);
      RemoveOperation(model, operation_map["squeeze_0"]);
      RemoveOperand(model, operation_map["transpose_2"]->input_operands[0]);
      RemoveOperand(model, operation_map["transpose_2"]->input_operands[1]);
      RemoveOperation(model, operation_map["transpose_2"]);
      RemoveOperand(model, operation_map["gather_0"]->input_operands[0]);
      RemoveOperand(model, operation_map["gather_0"]->input_operands[2]);
      RemoveOperation(model, operation_map["gather_0"]);

      auto concat2_operand2 = operation_map["concat_2"]->input_operands[2];

      RemoveOperand(model, operation_map["concat_2"]->input_operands[1]);
      operation_map["concat_2"]->input_operands = {
          nms_out_operand, nms_index_operand, concat2_operand2};
      auto concat2_out_operand = operation_map["concat_2"]->output_operands[0];
      concat2_out_operand->type.dimensions.data[1] = 7;

      RemoveOperand(
          model, operation_map["split_0"]->input_operands[2]);  // split_operand
      auto axis_operand = split_operation[0]->input_operands[1];
      core::Operand* split_operand = AddInt32ConstantOperand(model, {2, 4, 1});
      split_operation[0]->input_operands = {
          concat2_out_operand, axis_operand, split_operand};

      NNADAPTER_VLOG(5) << "fix multiclass_nms finish.";
    }
  }
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
