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
#include <vector>
namespace nnadapter {
namespace nvidia_tensorrt {

YoloBoxHeadPlugin::YoloBoxHeadPlugin(const std::vector<int32_t>& anchors,
                                     int class_num,
                                     float conf_thresh,
                                     int downsample_ratio,
                                     bool clip_bbox,
                                     float scale_x_y)
    : anchors_(anchors),
      class_num_(class_num),
      conf_thresh_(conf_thresh),
      downsample_ratio_(downsample_ratio),
      clip_bbox_(clip_bbox),
      scale_x_y_(scale_x_y) {}

YoloBoxHeadPlugin::YoloBoxHeadPlugin(const void* serial_data,
                                     size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &anchors_);
  Deserialize(&serial_data, &serial_length, &class_num_);
  Deserialize(&serial_data, &serial_length, &conf_thresh_);
  Deserialize(&serial_data, &serial_length, &downsample_ratio_);
  Deserialize(&serial_data, &serial_length, &clip_bbox_);
  Deserialize(&serial_data, &serial_length, &scale_x_y_);
}

nvinfer1::IPluginV2* YoloBoxHeadPlugin::clone() const noexcept {
  return new YoloBoxHeadPlugin(anchors_,
                               class_num_,
                               conf_thresh_,
                               downsample_ratio_,
                               clip_bbox_,
                               scale_x_y_);
}

int32_t YoloBoxHeadPlugin::enqueue(int batch_size,
                                   const void* const* inputs,
                                   void** outputs,
                                   void* workspace,
                                   cudaStream_t stream) noexcept {
  const int n = input_dims_[0].d[0];
  const int h = input_dims_[0].d[2];
  const int w = input_dims_[0].d[3];
  const int gridSizeX = w;
  const int gridSizeY = h;
  const int numBBoxes = anchors_.size() / 2;
  const float* input_data = static_cast<const float*>(inputs[0]);
  float* output_data = static_cast<float*>(outputs[0]);
  const int volume = input_dims_[0].d[1] * h * w;
  for (unsigned int batch = 0; batch < n; ++batch) {
    NNADAPTER_CHECK_EQ(YoloBoxHead(input_data + batch * volume,
                                output_data + batch * volume,
                                gridSizeX,
                                gridSizeY,
                                class_num_,
                                numBBoxes,
                                scale_x_y_,
                                stream),
                       cudaSuccess);
  }

  return 0;
}

size_t YoloBoxHeadPlugin::getSerializationSize() const noexcept {
  return SerializedSize(anchors_) + SerializedSize(class_num_) +
         SerializedSize(conf_thresh_) + SerializedSize(downsample_ratio_) +
         SerializedSize(clip_bbox_) + SerializedSize(scale_x_y_);
}

void YoloBoxHeadPlugin::serialize(void* buffer) const noexcept {
  Serialize(&buffer, anchors_);
  Serialize(&buffer, class_num_);
  Serialize(&buffer, conf_thresh_);
  Serialize(&buffer, downsample_ratio_);
  Serialize(&buffer, clip_bbox_);
  Serialize(&buffer, scale_x_y_);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(YoloBoxHeadPlugin,
                                   YoloBoxHeadPluginCreator,
                                   "yolo_box_head_plugin");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
