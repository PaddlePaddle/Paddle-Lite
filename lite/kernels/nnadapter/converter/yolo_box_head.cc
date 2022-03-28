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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertYoloBoxHead(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("x").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  std::vector<int> anchors;
  int class_num;
  float conf_thresh;
  int downsample_ratio;
  bool clip_bbox = true;
  float scale_x_y = 1.f;

  anchors = op->GetAttr<std::vector<int>>("anchors");
  class_num = op->GetAttr<int>("class_num");
  conf_thresh = op->GetAttr<float>("conf_thresh");
  downsample_ratio = op->GetAttr<int>("downsample_ratio");
  if (op->HasAttr("clip_bbox")) {
    clip_bbox = op->GetAttr<bool>("clip_bbox");
  }
  if (op->HasAttr("scale_x_y")) {
    scale_x_y = op->GetAttr<float>("scale_x_y");
  }

  auto anchors_operand = converter->AddConstantOperand(anchors);
  auto class_num_operand = converter->AddConstantOperand(class_num);
  auto conf_thresh_operand = converter->AddConstantOperand(conf_thresh);
  auto downsample_ratio_operand =
      converter->AddConstantOperand(downsample_ratio);
  auto clip_bbox_operand = converter->AddConstantOperand(clip_bbox);
  auto scale_x_y_operand = converter->AddConstantOperand(scale_x_y);

  // Output operand
  auto output_name = op->Output("output").front();
  auto output_operand = converter->AddOutputOperand(output_name);

  converter->AddOperation(NNADAPTER_YOLO_BOX_HEAD,
                          {input_operand,
                           anchors_operand,
                           class_num_operand,
                           conf_thresh_operand,
                           downsample_ratio_operand,
                           clip_bbox_operand,
                           scale_x_y_operand},
                          {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
