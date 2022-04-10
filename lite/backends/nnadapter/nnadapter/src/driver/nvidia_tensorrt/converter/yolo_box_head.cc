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

#include "driver/nvidia_tensorrt/converter/plugin/yolo_box_head.h"
#include <iostream>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/yolo_box_head.h"
#include "utility/debug.h"
#include "utility/logging.h"
namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertYoloBoxHead(Converter* converter, core::Operation* operation) {
  YOLO_BOX_HEAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  YoloBoxHeadPlugin yolo_box_head_plugin(
      anchors, class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y);
  std::vector<nvinfer1::ITensor*> tensors{input_tensor};
  auto yolo_box_head_layer = converter->network()->addPluginV2(
      tensors.data(), 1, yolo_box_head_plugin);
  NNADAPTER_CHECK(yolo_box_head_layer);
  auto output_tensor = yolo_box_head_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
