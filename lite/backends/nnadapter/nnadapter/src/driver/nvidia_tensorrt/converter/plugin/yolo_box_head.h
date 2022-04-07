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

class YoloBoxHeadPlugin : public Plugin {
 public:
  YoloBoxHeadPlugin(const std::vector<int32_t>& anchors,
                    int class_num,
                    float conf_thresh,
                    int downsample_ratio,
                    bool clip_bbox,
                    float scale_x_y);
  YoloBoxHeadPlugin(const void* serial_data, size_t serial_length);
  const char* getPluginType() const noexcept;
  int32_t enqueue(int batch_size,
                  const void* const* inputs,
                  void** outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept;
  size_t getSerializationSize() const noexcept;
  void serialize(void* buffer) const noexcept;
  IPluginV2* clone() const noexcept;

 private:
  std::vector<int32_t> anchors_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
};

class YoloBoxHeadPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

cudaError_t YoloBoxHead(const float* input,
                     float* output,
                     const int grid_size_x,
                     const int grid_size_y,
                     const int class_num,
                     const int anchors_num,
                     const float scale_x_y,
                     cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
