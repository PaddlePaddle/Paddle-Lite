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

#pragma once
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/plugin.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class YoloBoxPluginDynamic : public PluginDynamic {
 public:
  YoloBoxPluginDynamic(const std::vector<int32_t>& anchors,
                       int class_num,
                       float conf_thresh,
                       int downsample_ratio,
                       bool clip_bbox,
                       float scale_x_y,
                       bool iou_aware,
                       float iou_aware_factor);
  YoloBoxPluginDynamic(const void* serial_data, size_t serial_length);
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT;
  int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                  const nvinfer1::PluginTensorDesc* output_desc,
                  void const* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) TRT_NOEXCEPT;
  const char* getPluginType() const TRT_NOEXCEPT;
  size_t getSerializationSize() const TRT_NOEXCEPT;
  void serialize(void* buffer) const TRT_NOEXCEPT;

  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      const nvinfer1::DimsExprs* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT;

  int32_t getNbOutputs() const TRT_NOEXCEPT;

  nvinfer1::DataType getOutputDataType(int32_t index,
                                       const nvinfer1::DataType* input_types,
                                       int32_t nb_inputs) const TRT_NOEXCEPT;

 private:
  std::vector<int32_t> anchors_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  bool iou_aware_;
  float iou_aware_factor_;
};

class YoloBoxPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) TRT_NOEXCEPT;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
