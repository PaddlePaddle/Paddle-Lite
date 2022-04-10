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

#include "driver/nvidia_tensorrt/optimizer/fuse_yolo_box.h"
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include "driver/nvidia_tensorrt/operation/type.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

static bool FindYoloBoxPattern(
    core::Model* model,
    core::Operation* yolo_box_operation,
    std::map<std::string, core::Operation*>* operation_map,
    std::string yolo_box_name,
    std::string nms_name,
    int num) {
  auto yolo_box_operatio_box_operand0 = yolo_box_operation->output_operands[0];
  auto yolo_box_operation_score_operand0 =
      yolo_box_operation->output_operands[1];
  auto box_consumers =
      GetOperandConsumers(model, yolo_box_operatio_box_operand0);
  auto score_consumers =
      GetOperandConsumers(model, yolo_box_operation_score_operand0);
  if (box_consumers.size() != 1 || score_consumers.size() != 1) {
    NNADAPTER_VLOG(5) << "box_consumers size: " << box_consumers.size()
                      << " is not equal to score_consumers size : "
                      << score_consumers.size();
    return false;
  }
  (*operation_map)[yolo_box_name] = yolo_box_operation;
  auto concat_0_operation_0 = box_consumers[0];
  auto transpose2_0_operation = score_consumers[0];
  if (concat_0_operation_0->type != NNADAPTER_CONCAT &&
      transpose2_0_operation->type != NNADAPTER_TRANSPOSE) {
    NNADAPTER_VLOG(5) << "Converting "
                      << OperationTypeToString(concat_0_operation_0->type)
                      << " ...";
    NNADAPTER_VLOG(5) << "Converting "
                      << OperationTypeToString(transpose2_0_operation->type)
                      << " ...";
    return false;
  }
  NNADAPTER_CHECK_EQ(concat_0_operation_0->input_operands.size(), 4);
  (*operation_map)["concat_0"] = concat_0_operation_0;
  auto transpose2_0_output_operand = transpose2_0_operation->output_operands[0];
  auto transpose2_0_consumers =
      GetOperandConsumers(model, transpose2_0_output_operand);
  if (transpose2_0_consumers.size() != 1) {
    NNADAPTER_VLOG(5) << "transpose2_0_consumers size: "
                      << transpose2_0_consumers.size() << " is not euqal 1";
    return false;
  }
  std::string transpse2_name = "transpose2_" + std::to_string(num);
  (*operation_map)[transpse2_name] = transpose2_0_operation;
  auto concat_0_operation_1 = transpose2_0_consumers[0];
  if (concat_0_operation_1->type != NNADAPTER_CONCAT) {
    NNADAPTER_VLOG(5) << "Converting "
                      << OperationTypeToString(concat_0_operation_1->type)
                      << " ...";
    return false;
  }
  NNADAPTER_CHECK_EQ(concat_0_operation_1->input_operands.size(), 4);
  (*operation_map)["concat_1"] = concat_0_operation_1;
  auto concat_0_output_operand_1 = concat_0_operation_1->output_operands[0];
  auto concat_0_consumers =
      GetOperandConsumers(model, concat_0_output_operand_1);
  if (concat_0_consumers.size() != 1) {
    NNADAPTER_VLOG(5) << "concat_0_consumers size: "
                      << concat_0_consumers.size() << " is not euqal 1";
    return false;
  }
  auto nms_0_operation_1 = concat_0_consumers[0];
  if (nms_0_operation_1->type != NNADAPTER_MULTICLASS_NMS3) {
    NNADAPTER_VLOG(5) << "Converting "
                      << OperationTypeToString(nms_0_operation_1->type)
                      << " ...";
    return false;
  }
  NNADAPTER_CHECK_EQ(nms_0_operation_1->input_operands.size(), 11);
  (*operation_map)[nms_name] = nms_0_operation_1;
  return true;
}

static bool AddYoloHeadOperation(core::Model* model,
                                 core::Operation* yolo_box_head,
                                 core::Operation* yolo_box_operands) {
  // get input
  auto boxes = yolo_box_operands->input_operands[0];
  auto anchors = yolo_box_operands->input_operands[2];
  auto class_num = yolo_box_operands->input_operands[3];
  auto conf_thresh = yolo_box_operands->input_operands[4];
  auto downsample_ratio = yolo_box_operands->input_operands[5];
  auto clip_bbox = yolo_box_operands->input_operands[6];
  auto scale_x_y = yolo_box_operands->input_operands[7];
  // set input
  yolo_box_head->input_operands = {boxes,
                                   anchors,
                                   class_num,
                                   conf_thresh,
                                   downsample_ratio,
                                   clip_bbox,
                                   scale_x_y};
  // auto yolo_box_head0_output_operand =AddOperand(model);
  auto box_operand0 = yolo_box_operands->output_operands[0];
  auto score_operand0 = yolo_box_operands->output_operands[1];
  auto yolo_box_0_dims = box_operand0->type.dimensions.count;
  auto yolo_box_0_dims_ptr = box_operand0->type.dimensions.data;
  auto yolo_score_0_dims = score_operand0->type.dimensions.count;
  auto yolo_score_0_dims_ptr = score_operand0->type.dimensions.data;
  if (yolo_box_0_dims != yolo_score_0_dims) {
    NNADAPTER_VLOG(5) << "yolo_box_0_dims: " << yolo_box_0_dims
                      << ", is not equal to yolo_score_0_dims: "
                      << yolo_score_0_dims;
    return false;
  }
  std::vector<int32_t> yolo_box_0_output_dims;
  for (int i = 0; i < yolo_box_0_dims - 1; i++) {
    if (yolo_box_0_dims_ptr[i] != yolo_score_0_dims_ptr[i]) {
      NNADAPTER_VLOG(5) << "yolo_box_0_dims_data: " << yolo_box_0_dims_ptr[i]
                        << ", is not equal to yolo_score_0_dims_data: "
                        << yolo_score_0_dims_ptr[i];
      return false;
    }
    yolo_box_0_output_dims.push_back(yolo_box_0_dims_ptr[i]);
  }
  yolo_box_0_output_dims.push_back(yolo_box_0_dims_ptr[yolo_box_0_dims - 1] +
                                   yolo_score_0_dims_ptr[yolo_box_0_dims - 1]);
  auto yolo_box_head0_output_operand =
      AddFloat32VariableOperand(model, yolo_box_0_output_dims);
  yolo_box_head->output_operands = {yolo_box_head0_output_operand};
  return true;
}

static bool AddYoloPostOperation(core::Model* model,
                                 core::Operation* yolo_box_parse,
                                 core::Operation* yolo_box_head0,
                                 core::Operation* yolo_box_head1,
                                 core::Operation* yolo_box_head2,
                                 core::Operand* elt_div_input0_operand,
                                 core::Operand* elt_div_input1_operand,
                                 core::Operation* nms_operand) {
  // get input
  NNADAPTER_CHECK_EQ(yolo_box_head0->input_operands.size(), 7);
  NNADAPTER_CHECK_EQ(nms_operand->input_operands.size(), 11);
  auto boxes0 = yolo_box_head0->output_operands[0];
  auto boxes1 = yolo_box_head1->output_operands[0];
  auto boxes2 = yolo_box_head2->output_operands[0];
  auto anchors0 = yolo_box_head0->input_operands[1];
  auto anchors1 = yolo_box_head1->input_operands[1];
  auto anchors2 = yolo_box_head2->input_operands[1];
  auto class_num = yolo_box_head0->input_operands[2];
  auto conf_thresh = yolo_box_head0->input_operands[3];
  auto downsample_ratio0 = yolo_box_head0->input_operands[4];
  auto downsample_ratio1 = yolo_box_head1->input_operands[4];
  auto downsample_ratio2 = yolo_box_head2->input_operands[4];
  auto clip_bbox = yolo_box_head0->input_operands[5];
  auto scale_x_y = yolo_box_head0->input_operands[6];
  auto nms_thresh = nms_operand->input_operands[6];

  // set input
  yolo_box_parse->input_operands = {boxes0,
                                    boxes1,
                                    boxes2,
                                    elt_div_input0_operand,
                                    elt_div_input1_operand,
                                    anchors0,
                                    anchors1,
                                    anchors2,
                                    class_num,
                                    conf_thresh,
                                    downsample_ratio0,
                                    downsample_ratio1,
                                    downsample_ratio2,
                                    clip_bbox,
                                    scale_x_y,
                                    nms_thresh};
  yolo_box_parse->output_operands = {nms_operand->output_operands[0],
                                     nms_operand->output_operands[1]};
  return true;
}

static void DelOperation(core::Model* model,
                         core::Operation* operation,
                         std::map<std::string, core::Operation*> operation_map,
                         int num) {
  // remove yolo_box input
  RemoveOperand(model, operation->input_operands[1]);
  // remove yolo_box output == transpose2
  std::string transpose2_name = "transpose2_" + std::to_string(num);
  auto transpose2_operation = operation_map[transpose2_name];
  RemoveOperand(model, transpose2_operation->input_operands[0]);
  RemoveOperand(model, transpose2_operation->input_operands[1]);
  RemoveOperation(model, transpose2_operation);
  // remove yolo_box operation
  RemoveOperation(model, operation);
}

// clang-format off
/*
* x0  y0    input0        input1      input2                   input0     input1          input2
* |   |                                                           |         |               |
*   |
* elementwise_div
*   |
*   cast                                              ===>   yolo_head    yolo_head     yolo_head   x0     y0
*    |0       |      |1    |         |2   |                     |             |             |         |     |
*          |             |                |                                     yolo_post
*      yolo_box          yolo_box           yolo_box                                 
*       |     |0        |1       |          |2     |                            
*  transpose   concat0(0/1/2) transpose          transpose
*  |0                           |1                  |2
*                      concat0(0/1/2)              concat1(0/1/2)
*                               |                    |
*                                       nms
*/
// clang-format on
void FuseYoloBox(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    if (operation->type == NNADAPTER_DIV) {
      auto& input_operands = operation->input_operands;
      auto& output_operands = operation->output_operands;
      NNADAPTER_CHECK_EQ(input_operands.size(), 3);
      NNADAPTER_CHECK_EQ(output_operands.size(), 1);
      auto elt_div_input0_operand = input_operands[0];
      auto elt_div_input1_operand = input_operands[1];
      auto elt_div_output_operand = output_operands[0];
      if (elt_div_input0_operand->type.dimensions.count != 2 ||
          elt_div_input1_operand->type.dimensions.count != 2) {
        NNADAPTER_VLOG(5)
            << "Only support x's dims count and y's dims count is 2.";
        continue;
      }
      auto elt_div_consumers =
          GetOperandConsumers(model, elt_div_output_operand);
      if (elt_div_consumers.size() == 0 || elt_div_consumers.size() > 1) {
        continue;
      }
      auto cast_operation = elt_div_consumers[0];
      if (cast_operation->type != NNADAPTER_CAST) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(cast_operation->type)
                          << " ...";
        continue;
      }

      auto cast_input_attr_dtype = cast_operation->input_operands[1];
      auto cast_output_operand = cast_operation->output_operands[0];
      if (*reinterpret_cast<int32_t*>(cast_input_attr_dtype->buffer) !=
          NNADAPTER_INT32) {
        continue;
      }
      auto cast_consumers = GetOperandConsumers(model, cast_output_operand);
      if (cast_consumers.size() == 0 || cast_consumers.size() != 3) {
        continue;
      }
      // Process multiple yolo_box operation
      auto yolo_box_operation_0 = cast_consumers[0];
      auto yolo_box_operation_1 = cast_consumers[1];
      auto yolo_box_operation_2 = cast_consumers[2];
      if (yolo_box_operation_0->type != NNADAPTER_YOLO_BOX &&
          yolo_box_operation_1->type != NNADAPTER_YOLO_BOX &&
          yolo_box_operation_2->type != NNADAPTER_YOLO_BOX) {
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(yolo_box_operation_0->type)
                          << " ...";
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(yolo_box_operation_1->type)
                          << " ...";
        NNADAPTER_VLOG(5) << "Converting "
                          << OperationTypeToString(yolo_box_operation_2->type)
                          << " ...";
        break;
      }
      std::map<std::string, core::Operation*> operation_map;
      NNADAPTER_VLOG(5) << "FindYoloBoxPattern0 begin";
      bool find = FindYoloBoxPattern(model,
                                     yolo_box_operation_0,
                                     &operation_map,
                                     "yolo_box_0",
                                     "nms_0",
                                     0);
      if (!find) {
        NNADAPTER_VLOG(5) << "FindYoloBoxPattern0 Failed ...";
        break;
      }
      NNADAPTER_VLOG(5) << "FindYoloBoxPattern1 begin";
      find = FindYoloBoxPattern(model,
                                yolo_box_operation_1,
                                &operation_map,
                                "yolo_box_1",
                                "nms_0",
                                1);

      if (!find) {
        NNADAPTER_VLOG(5) << "FindYoloBoxPattern1 Failed ...";
        break;
      }
      NNADAPTER_VLOG(5) << "FindYoloBoxPattern2 begin";
      find = FindYoloBoxPattern(model,
                                yolo_box_operation_2,
                                &operation_map,
                                "yolo_box_2",
                                "nms_0",
                                2);
      if (!find) {
        NNADAPTER_VLOG(5) << "FindYoloBoxPattern2 Failed ...";
        break;
      }
      NNADAPTER_VLOG(5) << "FindYoloBoxPattern end";
      if (operation_map.size() != 9) {
        NNADAPTER_VLOG(5) << "operation_map size: " << operation_map.size()
                          << " is not equal 9!";
      }
      // add new operation
      auto yolo_box_head0 = AddOperation(model);
      auto yolo_box_head1 = AddOperation(model);
      auto yolo_box_head2 = AddOperation(model);
      auto yolo_box_post = AddOperation(model);
      NNADAPTER_VLOG(5) << "AddYoloHeadOperation begin";
      yolo_box_head0->type = NNADAPTER_YOLO_BOX_HEAD;
      bool add = AddYoloHeadOperation(
          model, yolo_box_head0, operation_map["yolo_box_0"]);

      if (!add) {
        NNADAPTER_VLOG(5) << "AddYoloHeadOperation0 Failed ...";
        break;
      }

      yolo_box_head1->type = NNADAPTER_YOLO_BOX_HEAD;
      add = AddYoloHeadOperation(
          model, yolo_box_head1, operation_map["yolo_box_1"]);
      if (!add) {
        NNADAPTER_VLOG(5) << "AddYoloHeadOperation1 Failed ...";
        break;
      }

      yolo_box_head2->type = NNADAPTER_YOLO_BOX_HEAD;
      add = AddYoloHeadOperation(
          model, yolo_box_head2, operation_map["yolo_box_2"]);
      if (!add) {
        NNADAPTER_VLOG(5) << "AddYoloHeadOperation2 Failed ...";
        break;
      }
      yolo_box_post->type = NNADAPTER_YOLO_BOX_POST;
      NNADAPTER_VLOG(5) << "AddYoloParseOperation begin";
      add = AddYoloPostOperation(model,
                                 yolo_box_post,
                                 yolo_box_head0,
                                 yolo_box_head1,
                                 yolo_box_head2,
                                 elt_div_input0_operand,
                                 elt_div_input1_operand,
                                 operation_map["nms_0"]);
      if (!add) {
        NNADAPTER_VLOG(5) << "AddYoloPostOperation Failed ...";
        break;
      }
      // clean operation and node
      DelOperation(model, operation_map["yolo_box_0"], operation_map, 0);
      DelOperation(model, operation_map["yolo_box_1"], operation_map, 1);
      DelOperation(model, operation_map["yolo_box_2"], operation_map, 2);
      // concat_0
      RemoveOperand(model, operation_map["concat_0"]->input_operands[0]);
      RemoveOperand(model, operation_map["concat_0"]->input_operands[1]);
      RemoveOperand(model, operation_map["concat_0"]->input_operands[2]);
      RemoveOperand(model, operation_map["concat_0"]->input_operands[3]);
      RemoveOperation(model, operation_map["concat_0"]);
      // concat_1
      RemoveOperand(model, operation_map["concat_1"]->input_operands[0]);
      RemoveOperand(model, operation_map["concat_1"]->input_operands[1]);
      RemoveOperand(model, operation_map["concat_1"]->input_operands[2]);
      RemoveOperand(model, operation_map["concat_1"]->input_operands[3]);
      RemoveOperation(model, operation_map["concat_1"]);
      // nms
      NNADAPTER_VLOG(5) << "del nms: "
                        << OperationTypeToString(operation_map["nms_0"]->type);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[0]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[1]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[2]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[3]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[4]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[5]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[7]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[8]);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[9]);
      auto return_index = *reinterpret_cast<bool*>(
          operation_map["nms_0"]->input_operands[10]->buffer);
      RemoveOperand(model, operation_map["nms_0"]->input_operands[10]);
      if (return_index) {
        RemoveOperand(model, operation_map["nms_0"]->output_operands[2]);
      }
      RemoveOperation(model, operation_map["nms_0"]);
      // cast and div
      RemoveOperand(model, cast_operation->input_operands[0]);
      RemoveOperand(model, cast_operation->input_operands[1]);
      RemoveOperation(model, cast_operation);
      RemoveOperation(model, operation);
      NNADAPTER_VLOG(5) << "fuse_yolo_box finish";
    }
  }
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
