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

#include "driver/nvidia_tensorrt/converter/plugin/yolo_box.h"
#include <iostream>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/yolo_box.h"
#include "utility/debug.h"
#include "utility/logging.h"
namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertYoloBox(Converter* converter, core::Operation* operation) {
  YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto imgsize_tensor = converter->GetMappedTensor(imgsize_operand);
  if (!imgsize_tensor) {
    imgsize_tensor = converter->ConvertOperand(imgsize_operand);
  }
  YoloBoxPluginDynamic yolo_box_plugin(anchors,
                                       class_num,
                                       conf_thresh,
                                       downsample_ratio,
                                       clip_bbox,
                                       scale_x_y,
                                       iou_aware,
                                       iou_aware_factor);

  std::vector<nvinfer1::ITensor*> tensors{input_tensor, imgsize_tensor};
  auto yolo_box_layer =
      converter->network()->addPluginV2(tensors.data(), 2, yolo_box_plugin);
  NNADAPTER_CHECK(yolo_box_layer);
  auto boxes_tensor = yolo_box_layer->getOutput(0);
  auto scores_tensor = yolo_box_layer->getOutput(1);
  converter->UpdateTensorMap(boxes_operand, boxes_tensor);
  converter->UpdateTensorMap(scores_operand, scores_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
