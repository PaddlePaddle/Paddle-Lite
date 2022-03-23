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

#include <iostream>
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/yolo_box_head.h"
namespace nnadapter {
namespace nvidia_tensorrt {

YoloBoxHeadPluginDynamic::YoloBoxHeadPluginDynamic(
    const std::vector<int32_t>& anchors,
    int class_num,
    float conf_thresh,
    int downsample_ratio,
    bool clip_bbox,
    float scale_x_y,
    bool iou_aware,
    float iou_aware_factor)
    : anchors_(anchors),
      class_num_(class_num),
      conf_thresh_(conf_thresh),
      downsample_ratio_(downsample_ratio),
      clip_bbox_(clip_bbox),
      scale_x_y_(scale_x_y),
      iou_aware_(iou_aware),
      iou_aware_factor_(iou_aware_factor) {}

YoloBoxHeadPluginDynamic::YoloBoxHeadPluginDynamic(const void* serial_data,
                                                   size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &anchors_);
  Deserialize(&serial_data, &serial_length, &class_num_);
  Deserialize(&serial_data, &serial_length, &conf_thresh_);
  Deserialize(&serial_data, &serial_length, &downsample_ratio_);
  Deserialize(&serial_data, &serial_length, &clip_bbox_);
  Deserialize(&serial_data, &serial_length, &scale_x_y_);
  Deserialize(&serial_data, &serial_length, &iou_aware_);
  Deserialize(&serial_data, &serial_length, &iou_aware_factor_);
}

nvinfer1::IPluginV2DynamicExt* YoloBoxHeadPluginDynamic::clone() const
    noexcept {
  return new YoloBoxHeadPluginDynamic(anchors_,
                                      class_num_,
                                      conf_thresh_,
                                      downsample_ratio_,
                                      clip_bbox_,
                                      scale_x_y_,
                                      iou_aware_,
                                      iou_aware_factor_);
}

nvinfer1::DimsExprs YoloBoxHeadPluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  NNADAPTER_CHECK(inputs);
  return inputs[0];
}

__global__ void yolohead_kernel(const float* input,
                                float* output,
                                const uint gridSizeX,
                                const uint gridSizeY,
                                const uint numOutputClasses,
                                const uint numBBoxes,
                                const float scale_x_y) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_id >= gridSizeX) || (y_id >= gridSizeY) || (z_id >= numBBoxes)) {
    return;
  }

  const int numGridCells = gridSizeX * gridSizeY;
  const int bbindex = y_id * gridSizeX + x_id;

  const float alpha = scale_x_y;
  const float beta = -0.5 * (scale_x_y - 1);

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)] *
          alpha +
      beta;

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)] *
          alpha +
      beta;

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)] = pow(
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)] * 2,
      2);

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)] = pow(
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)] * 2,
      2);

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];

  for (uint i = 0; i < numOutputClasses; ++i) {
    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] =
        input[bbindex +
              numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))];
  }
}

int32_t YoloBoxHeadPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  const int n = input_desc[0].dims.d[0];
  const int h = input_desc[0].dims.d[2];
  const int w = input_desc[0].dims.d[3];
  const int gridSizeX = w;
  const int gridSizeY = h;
  const int numBBoxes = anchors_.size() / 2;
  const float* input_data = static_cast<const float*>(inputs[0]);
  float* output_data = static_cast<float*>(outputs[0]);
  const int outputSize = input_desc[0].dims.d[1] * h * w;

  dim3 block(16, 16, 4);
  dim3 grid((gridSizeX / block.x) + 1,
            (gridSizeY / block.y) + 1,
            (numBBoxes / block.z) + 1);

  for (unsigned int batch = 0; batch < n; ++batch) {
    yolohead_kernel<<<grid, block, 0, stream>>>(
        input_data + batch * outputSize,
        output_data + batch * outputSize,
        gridSizeX,
        gridSizeY,
        class_num_,
        numBBoxes,
        scale_x_y_);
  }

  return 0;
}

size_t YoloBoxHeadPluginDynamic::getSerializationSize() const noexcept {
  return SerializedSize(anchors_) + sizeof(class_num_) + sizeof(conf_thresh_) +
         sizeof(downsample_ratio_) + sizeof(clip_bbox_) + sizeof(scale_x_y_) +
         sizeof(iou_aware_) + sizeof(iou_aware_factor_);
}

void YoloBoxHeadPluginDynamic::serialize(void* buffer) const noexcept {
  Serialize(&buffer, anchors_);
  Serialize(&buffer, class_num_);
  Serialize(&buffer, conf_thresh_);
  Serialize(&buffer, downsample_ratio_);
  Serialize(&buffer, clip_bbox_);
  Serialize(&buffer, scale_x_y_);
  Serialize(&buffer, iou_aware_);
  Serialize(&buffer, iou_aware_factor_);
}

int32_t YoloBoxHeadPluginDynamic::getNbOutputs() const noexcept { return 1; }

nvinfer1::DataType YoloBoxHeadPluginDynamic::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* input_types,
    int32_t nb_inputs) const noexcept {
  return input_types[0];
}

bool YoloBoxHeadPluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return true;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(YoloBoxHeadPluginDynamic,
                                   YoloBoxHeadPluginDynamicCreator,
                                   "yolo_box_head_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
