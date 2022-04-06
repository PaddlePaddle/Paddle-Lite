// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertYoloBoxParser(Converter* converter, OpInfo* op, Scope* scope) {
  printf("是开发绿色的肌肤\n\n\n");
  // Input operand
  auto x0_name = op->Input("x0").front();
  auto x0_operand = converter->AddInputOperand(scope, x0_name, {});
  auto x1_name = op->Input("x1").front();
  auto x1_operand = converter->AddInputOperand(scope, x1_name, {});
  auto x2_name = op->Input("x2").front();
  auto x2_operand = converter->AddInputOperand(scope, x2_name, {});
  auto image_shape_name = op->Input("image_shape").front();
  auto image_shape_operand =
      converter->AddInputOperand(scope, image_shape_name, {});
  auto image_scale_name = op->Input("image_scale").front();
  auto image_scale_operand =
      converter->AddInputOperand(scope, image_scale_name, {});

  std::vector<int> anchors0;
  std::vector<int> anchors1;
  std::vector<int> anchors2;
  int class_num;
  float conf_thresh;
  int downsample_ratio0, downsample_ratio1, downsample_ratio2;
  bool clip_bbox = true;
  float scale_x_y = 1.f;
  float nms_thresh = 0.3f;

  anchors0 = op->GetAttr<std::vector<int>>("anchors0");
  anchors1 = op->GetAttr<std::vector<int>>("anchors1");
  anchors2 = op->GetAttr<std::vector<int>>("anchors2");
  class_num = op->GetAttr<int>("class_num");
  conf_thresh = op->GetAttr<float>("conf_thresh");
  downsample_ratio0 = op->GetAttr<int>("downsample_ratio0");
  downsample_ratio1 = op->GetAttr<int>("downsample_ratio1");
  downsample_ratio2 = op->GetAttr<int>("downsample_ratio2");
  if (op->HasAttr("clip_bbox")) {
    clip_bbox = op->GetAttr<bool>("clip_bbox");
  }
  if (op->HasAttr("scale_x_y")) {
    scale_x_y = op->GetAttr<float>("scale_x_y");
  }
  nms_thresh = op->GetAttr<float>("nms_thresh");
  auto anchors_operand0 = converter->AddConstantOperand(anchors0);
  auto anchors_operand1 = converter->AddConstantOperand(anchors1);
  auto anchors_operand2 = converter->AddConstantOperand(anchors2);
  auto class_num_operand = converter->AddConstantOperand(class_num);
  auto conf_thresh_operand = converter->AddConstantOperand(conf_thresh);
  auto downsample_ratio_operand0 =
      converter->AddConstantOperand(downsample_ratio0);
  auto downsample_ratio_operand1 =
      converter->AddConstantOperand(downsample_ratio1);
  auto downsample_ratio_operand2 =
      converter->AddConstantOperand(downsample_ratio2);
  auto clip_bbox_operand = converter->AddConstantOperand(clip_bbox);
  auto scale_x_y_operand = converter->AddConstantOperand(scale_x_y);
  auto nms_thresh_operand = converter->AddConstantOperand(nms_thresh);

  // Output operand
  auto boxes_scores_name = op->Output("boxes_scores").front();
  auto boxes_num_name = op->Output("boxes_num").front();
  auto boxes_scores_operand = converter->AddOutputOperand(boxes_scores_name);
  auto boxes_num_operand = converter->AddOutputOperand(boxes_num_name);

  converter->AddOperation(NNADAPTER_YOLO_BOX_PARSER,
                          {x0_operand,
                           x1_operand,
                           x2_operand,
                           image_shape_operand,
                           image_scale_operand,
                           anchors_operand0,
                           anchors_operand1,
                           anchors_operand2,
                           class_num_operand,
                           conf_thresh_operand,
                           downsample_ratio_operand0,
                           downsample_ratio_operand1,
                           downsample_ratio_operand2,
                           clip_bbox_operand,
                           scale_x_y_operand,
                           nms_thresh_operand},
                          {boxes_scores_operand, boxes_num_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
