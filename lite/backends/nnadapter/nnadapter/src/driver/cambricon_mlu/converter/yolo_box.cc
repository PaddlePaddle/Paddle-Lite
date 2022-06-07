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

#include "operation/yolo_box.h"
#include <string>
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"
namespace nnadapter {
namespace cambricon_mlu {

int ConvertYoloBox(Converter* converter, core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto imgsize_tensor = converter->GetMappedTensor(imgsize_operand);
  if (!imgsize_tensor) {
    imgsize_tensor = converter->ConvertOperand(imgsize_operand);
  }

  int image_height = 608;
  int image_width = 608;
  auto op_params = GetKeyValues(converter->op_params().c_str());
  if (op_params.count("image_height")) {
    image_height = stoi(op_params["image_height"]);
  }
  if (op_params.count("image_width")) {
    image_width = stoi(op_params["image_width"]);
  }
  auto img_shape_tensor = converter->AddInt32ConstantTensor(
      std::vector<int32_t>({image_height, image_width}).data(), {1, 2});
  auto yolo_box_node =
      converter->network()->AddIYoloBoxNode(input_tensor, img_shape_tensor);
  NNADAPTER_CHECK(yolo_box_node) << "Failed to add yolo_box node.";
  std::vector<float> anchors_fp32;
  for (int i = 0; i < anchors.size(); i++) {
    anchors_fp32.push_back(static_cast<float>(anchors[i]));
  }
  magicmind::Layout input_layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  yolo_box_node->SetLayout(input_layout);
  yolo_box_node->SetAnchorsVal(anchors_fp32);
  yolo_box_node->SetClassNumVal(static_cast<int64_t>(class_num));
  yolo_box_node->SetConfidenceThresholdVal(conf_thresh);
  yolo_box_node->SetDownsampleRatioVal(static_cast<int64_t>(downsample_ratio));
  yolo_box_node->SetClipBBoxVal(clip_bbox);
  yolo_box_node->SetScaleXYVal(scale_x_y);
  yolo_box_node->SetImageShape(image_height, image_width);
  auto boxes_tensor = yolo_box_node->GetOutput(0);
  auto scores_tensor = yolo_box_node->GetOutput(1);
  converter->UpdateTensorMap(boxes_operand, boxes_tensor);
  converter->UpdateTensorMap(scores_operand, scores_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
