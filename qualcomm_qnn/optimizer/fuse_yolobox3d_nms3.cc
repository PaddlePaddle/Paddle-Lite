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

#include "driver/qualcomm_qnn/optimizer/fuse_yolobox3d_nms3.h"
#include "driver/qualcomm_qnn/operation/type.h"

namespace nnadapter {
namespace qualcomm_qnn {

// clang-format off
/*
1. before fuse:
yolo_box      yolo_box      yolo_box
   |             |             |
transpose-|   transpose-|   transpose-|
   |------|-----|-------|------|      |
          |   concat    |             |
          |-----|-------|-------------|
                |     cocnat
                |-------|
                       nms3

2. after fuse:
    NNADAPTER_YOLO_BOX_3D_NMS_FUSER               
*/
// clang-format on
void YoloBox3dNmsFuser::BuildPattern() {
  // Operation patterns
  auto yolobox_3d_0_pattern =
      CreatePattern("yolobox_3d_0", NNADAPTER_CUSTOM_YOLO_BOX_3D);
  auto yolobox_3d_1_pattern =
      CreatePattern("yolobox_3d_1", NNADAPTER_CUSTOM_YOLO_BOX_3D);
  auto yolobox_3d_2_pattern =
      CreatePattern("yolobox_3d_2", NNADAPTER_CUSTOM_YOLO_BOX_3D);

  auto concat0_pattern =
      CreatePattern("concat0", NNADAPTER_CONCAT)->IsIntermediate();
  auto concat1_pattern =
      CreatePattern("concat1", NNADAPTER_CONCAT)->IsIntermediate();
  auto concat2_pattern =
      CreatePattern("concat2", NNADAPTER_CONCAT)->IsIntermediate();
  auto concat3_pattern =
      CreatePattern("concat3", NNADAPTER_CONCAT)->IsIntermediate();
  auto concat4_pattern =
      CreatePattern("concat4", NNADAPTER_CONCAT)->IsIntermediate();

  auto tranpose0_pattern =
      CreatePattern("tranpose0", NNADAPTER_TRANSPOSE)->IsIntermediate();
  auto tranpose1_pattern =
      CreatePattern("tranpose1", NNADAPTER_TRANSPOSE)->IsIntermediate();
  auto tranpose2_pattern =
      CreatePattern("tranpose2", NNADAPTER_TRANSPOSE)->IsIntermediate();

  auto nms_pattern = CreatePattern("nms", NNADAPTER_NON_MAX_SUPPRESSION)
                         ->MatchCondition([](const Node* node) -> bool {
                           auto operation = node->operation;
                           return operation &&
                                  operation->input_operands.size() == 11 &&
                                  operation->output_operands.size() == 3;
                         });
  // Operand patterns
  auto imgsize_input =
      CreatePattern("imgsize_input")
          ->IsOperationInputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 1);
  // Yolobox_3d-0
  auto yolobox_3d_0_input_x =
      CreatePattern("yolobox_3d_0_input_x")
          ->IsOperationInputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0);
  auto yolobox_3d_0_output_boxes =
      CreatePattern("yolobox_3d_0_output_boxes")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_0_output_scores =
      CreatePattern("yolobox_3d_0_output_scores")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 1)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_0_output_location =
      CreatePattern("yolobox_3d_0_output_location")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 2)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_0_output_dim =
      CreatePattern("yolobox_3d_0_output_dim")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 3)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_0_output_alpha =
      CreatePattern("yolobox_3d_0_output_alpha")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 4)
          ->IsVariableOperand()
          ->IsIntermediate();
  // Yolobox_3d-1
  auto yolobox_3d_1_input_x =
      CreatePattern("yolobox_3d_1_input_x")
          ->IsOperationInputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0);
  auto yolobox_3d_1_output_boxes =
      CreatePattern("yolobox_3d_1_output_boxes")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_1_output_scores =
      CreatePattern("yolobox_3d_1_output_scores")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 1)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_1_output_location =
      CreatePattern("yolobox_3d_1_output_location")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 2)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_1_output_dim =
      CreatePattern("yolobox_3d_1_output_dim")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 3)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_1_output_alpha =
      CreatePattern("yolobox_3d_1_output_alpha")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 4)
          ->IsVariableOperand()
          ->IsIntermediate();
  // Yolobox_3d-2
  auto yolobox_3d_2_input_x =
      CreatePattern("yolobox_3d_2_input_x")
          ->IsOperationInputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0);
  auto yolobox_3d_2_output_boxes =
      CreatePattern("yolobox_3d_2_output_boxes")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 0)
          ->IsVariableOperand();
  auto yolobox_3d_2_output_scores =
      CreatePattern("yolobox_3d_2_output_scores")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 1)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_2_output_location =
      CreatePattern("yolobox_3d_2_output_location")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 2)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_2_output_dim =
      CreatePattern("yolobox_3d_2_output_dim")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 3)
          ->IsVariableOperand()
          ->IsIntermediate();
  auto yolobox_3d_2_output_alpha =
      CreatePattern("yolobox_3d_2_output_alpha")
          ->IsOperationOutputOperand(NNADAPTER_CUSTOM_YOLO_BOX_3D, 4)
          ->IsVariableOperand()
          ->IsIntermediate();
  // Transpose
  auto transpose_0_perm = CreatePattern("transpose_0_perm")
                              ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
                              ->IsIntermediate();
  auto transpose_1_perm = CreatePattern("transpose_1_perm")
                              ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
                              ->IsIntermediate();
  auto transpose_2_perm = CreatePattern("transpose_2_perm")
                              ->IsOperationInputOperand(NNADAPTER_TRANSPOSE, 1)
                              ->IsIntermediate();
  auto transpose_0_output =
      CreatePattern("transpose_0_output")
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  auto transpose_1_output =
      CreatePattern("transpose_1_output")
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  auto transpose_2_output =
      CreatePattern("transpose_2_output")
          ->IsOperationOutputOperand(NNADAPTER_TRANSPOSE, 0)
          ->IsIntermediate();
  // Concat
  auto concat_0_axis = CreatePattern("concat_0_axis")
                           ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                           ->IsIntermediate();
  auto concat_1_axis = CreatePattern("concat_1_axis")
                           ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                           ->IsIntermediate();
  auto concat_2_axis = CreatePattern("concat_2_axis")
                           ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                           ->IsIntermediate();
  auto concat_3_axis = CreatePattern("concat_3_axis")
                           ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                           ->IsIntermediate();
  auto concat_4_axis = CreatePattern("concat_4_axis")
                           ->IsOperationInputOperand(NNADAPTER_CONCAT, 3)
                           ->IsIntermediate();
  auto concat_0_output = CreatePattern("concat_0_output")
                             ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
                             ->IsIntermediate();
  auto concat_1_output = CreatePattern("concat_1_output")
                             ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0)
                             ->IsIntermediate();
  auto concat_2_output = CreatePattern("concat_2_output")
                             ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0);
  auto concat_3_output = CreatePattern("concat_3_output")
                             ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0);
  auto concat_4_output = CreatePattern("concat_4_output")
                             ->IsOperationOutputOperand(NNADAPTER_CONCAT, 0);
  // NMS
  auto nms_out_box =
      CreatePattern("nms_out_box")
          ->IsOperationOutputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 0);
  auto nms_out_index =
      CreatePattern("nms_out_index")
          ->IsOperationOutputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 1);
  auto nms_out_roisnum =
      CreatePattern("nms_out_roisnum")
          ->IsOperationOutputOperand(NNADAPTER_NON_MAX_SUPPRESSION, 2);

  // Create the topological connections for the above patterns
  std::vector<Pattern*> yolobox_3d_0_input_patterns{yolobox_3d_0_input_x,
                                                    imgsize_input};
  std::vector<Pattern*> yolobox_3d_1_input_patterns{yolobox_3d_1_input_x,
                                                    imgsize_input};
  std::vector<Pattern*> yolobox_3d_2_input_patterns{yolobox_3d_2_input_x,
                                                    imgsize_input};
  std::vector<Pattern*> yolobox_3d_0_output_patterns{
      yolobox_3d_0_output_boxes,
      yolobox_3d_0_output_scores,
      yolobox_3d_0_output_location,
      yolobox_3d_0_output_dim,
      yolobox_3d_0_output_alpha};
  std::vector<Pattern*> yolobox_3d_1_output_patterns{
      yolobox_3d_1_output_boxes,
      yolobox_3d_1_output_scores,
      yolobox_3d_1_output_location,
      yolobox_3d_1_output_dim,
      yolobox_3d_1_output_alpha};
  std::vector<Pattern*> yolobox_3d_2_output_patterns{
      yolobox_3d_2_output_boxes,
      yolobox_3d_2_output_scores,
      yolobox_3d_2_output_location,
      yolobox_3d_2_output_dim,
      yolobox_3d_2_output_alpha};
  std::vector<Pattern*> transpose_0_input_patterns{yolobox_3d_0_output_scores,
                                                   transpose_0_perm};
  std::vector<Pattern*> transpose_1_input_patterns{yolobox_3d_1_output_scores,
                                                   transpose_1_perm};
  std::vector<Pattern*> transpose_2_input_patterns{yolobox_3d_2_output_scores,
                                                   transpose_2_perm};
  std::vector<Pattern*> concat0_input_patterns{yolobox_3d_0_output_boxes,
                                               yolobox_3d_1_output_boxes,
                                               yolobox_3d_2_output_boxes,
                                               concat_0_axis};
  std::vector<Pattern*> concat1_input_patterns{transpose_0_output,
                                               transpose_1_output,
                                               transpose_2_output,
                                               concat_1_axis};
  std::vector<Pattern*> concat2_input_patterns{yolobox_3d_0_output_location,
                                               yolobox_3d_1_output_location,
                                               yolobox_3d_2_output_location,
                                               concat_2_axis};
  std::vector<Pattern*> concat3_input_patterns{yolobox_3d_0_output_dim,
                                               yolobox_3d_1_output_dim,
                                               yolobox_3d_2_output_dim,
                                               concat_3_axis};
  std::vector<Pattern*> concat4_input_patterns{yolobox_3d_0_output_alpha,
                                               yolobox_3d_1_output_alpha,
                                               yolobox_3d_2_output_alpha,
                                               concat_4_axis};
  std::vector<Pattern*> nms_input_patterns{concat_0_output, concat_1_output};
  std::vector<Pattern*> nms_output_patterns{
      nms_out_box, nms_out_index, nms_out_roisnum};

  yolobox_3d_0_input_patterns >> *yolobox_3d_0_pattern >>
      yolobox_3d_0_output_patterns;
  yolobox_3d_1_input_patterns >> *yolobox_3d_1_pattern >>
      yolobox_3d_1_output_patterns;
  yolobox_3d_2_input_patterns >> *yolobox_3d_2_pattern >>
      yolobox_3d_2_output_patterns;
  concat0_input_patterns >> *concat0_pattern >> *concat_0_output;
  transpose_0_input_patterns >> *tranpose0_pattern >> *transpose_0_output;
  transpose_1_input_patterns >> *tranpose1_pattern >> *transpose_1_output;
  transpose_2_input_patterns >> *tranpose2_pattern >> *transpose_2_output;
  concat1_input_patterns >> *concat1_pattern >> *concat_1_output;
  concat2_input_patterns >> *concat2_pattern >> *concat_2_output;
  concat3_input_patterns >> *concat3_pattern >> *concat_3_output;
  concat4_input_patterns >> *concat4_pattern >> *concat_4_output;
  nms_input_patterns >> *nms_pattern >> nms_output_patterns;
}

bool YoloBox3dNmsFuser::HandleMatchedResults(
    core::Model* model, const std::map<std::string, Node*>& nodes) {
  // Get the operands and operations from the matched subgraph nodes.
  auto yolobox_3d_0_operation = nodes.at("yolobox_3d_0")->operation;
  auto yolobox_3d_1_operation = nodes.at("yolobox_3d_1")->operation;
  auto yolobox_3d_2_operation = nodes.at("yolobox_3d_2")->operation;
  auto nms_operation = nodes.at("nms")->operation;
  // Input
  auto input0_operand = yolobox_3d_0_operation->input_operands[0];
  auto input1_operand = yolobox_3d_1_operation->input_operands[0];
  auto input2_operand = yolobox_3d_2_operation->input_operands[0];
  auto imgsize_operand = yolobox_3d_0_operation->input_operands[1];
  auto anchors0_operand = yolobox_3d_0_operation->input_operands[2];
  auto anchors1_operand = yolobox_3d_1_operation->input_operands[2];
  auto anchors2_operand = yolobox_3d_2_operation->input_operands[2];
  auto class_num_operand = yolobox_3d_0_operation->input_operands[3];
  auto conf_thresh_operand = yolobox_3d_0_operation->input_operands[4];
  auto downsample_ratio0_operand = yolobox_3d_0_operation->input_operands[5];
  auto downsample_ratio1_operand = yolobox_3d_1_operation->input_operands[5];
  auto downsample_ratio2_operand = yolobox_3d_2_operation->input_operands[5];
  auto scale_x_y_operand = yolobox_3d_0_operation->input_operands[6];
  auto background_label_operand = nms_operation->input_operands[3];
  auto score_threshold_operand = nms_operation->input_operands[4];
  auto nms_top_k_operand = nms_operation->input_operands[5];
  auto nms_threshold_operand = nms_operation->input_operands[6];
  auto nms_eta_operand = nms_operation->input_operands[7];
  auto keep_top_k_operand = nms_operation->input_operands[8];
  auto normalized_operand = nms_operation->input_operands[9];
  // Output
  auto out_box_operand = nms_operation->output_operands[0];
  auto out_rois_num_operand = nms_operation->output_operands[1];
  auto out_index_operand = nms_operation->output_operands[2];
  auto concat2_operation = nodes.at("concat2")->operation;
  auto concat3_operation = nodes.at("concat3")->operation;
  auto concat4_operation = nodes.at("concat4")->operation;
  auto location_operand = concat2_operation->output_operands[0];
  auto dim_operand = concat3_operation->output_operands[0];
  auto alpha_operand = concat4_operation->output_operands[0];

  // Create a new FullyConnected operation and replace the matched subgraph
  // nodes.
  auto* yolobox_3d_nms_fuser_operation = AddOperation(model);
  yolobox_3d_nms_fuser_operation->type = NNADAPTER_CUSTOM_YOLO_BOX_3D_NMS_FUSER;
  yolobox_3d_nms_fuser_operation->input_operands = {input0_operand,
                                                    input1_operand,
                                                    input2_operand,
                                                    imgsize_operand,
                                                    anchors0_operand,
                                                    anchors1_operand,
                                                    anchors2_operand,
                                                    class_num_operand,
                                                    conf_thresh_operand,
                                                    downsample_ratio0_operand,
                                                    downsample_ratio1_operand,
                                                    downsample_ratio2_operand,
                                                    scale_x_y_operand,
                                                    background_label_operand,
                                                    score_threshold_operand,
                                                    nms_top_k_operand,
                                                    nms_threshold_operand,
                                                    nms_eta_operand,
                                                    keep_top_k_operand,
                                                    normalized_operand};
  yolobox_3d_nms_fuser_operation->output_operands = {out_box_operand,
                                                     out_rois_num_operand,
                                                     out_index_operand,
                                                     location_operand,
                                                     dim_operand,
                                                     alpha_operand};
  // The matched intermediate operands and operations will be deleted only when
  // it returns true.
  RemoveOperation(model, yolobox_3d_0_operation);
  RemoveOperation(model, yolobox_3d_1_operation);
  RemoveOperation(model, yolobox_3d_2_operation);
  RemoveOperation(model, nms_operation);
  return true;
}

NNADAPTER_EXPORT void FuseYoloBox3dNms(core::Model* model) {
  NNADAPTER_LOG(INFO) << "Execute FuseYoloBox3dNms Pass";
  YoloBox3dNmsFuser yolo_box_3d_nms_fuser;
  yolo_box_3d_nms_fuser.Apply(model);
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
