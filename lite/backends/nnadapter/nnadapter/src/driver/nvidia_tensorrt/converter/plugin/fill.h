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

class FillPluginDynamic : public PluginDynamic {
 public:
  FillPluginDynamic(float value, bool value_tensor, std::vector<int32_t> shape);
  FillPluginDynamic(const void* serial_data, size_t serial_length);
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
  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int32_t nb_inputs,
                                 int32_t nb_outputs) TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      const nvinfer1::DimsExprs* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT;

 private:
  float value_;
  bool is_value_tensor_;
  std::vector<int32_t> shape_;
};

class FillPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) TRT_NOEXCEPT;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
