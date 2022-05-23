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
#include <math.h>
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/plugin.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class PriorBoxPluginDynamic : public PluginDynamic {
 public:
  PriorBoxPluginDynamic();
  PriorBoxPluginDynamic(const std::vector<float>& aspect_ratios,
                        const std::vector<int32_t>& input_dimension,
                        const std::vector<int32_t>& image_dimension,
                        float step_w,
                        float step_h,
                        const std::vector<float>& min_sizes,
                        const std::vector<float>& max_sizes,
                        float offset,
                        bool is_clip,
                        bool is_flip,
                        bool min_max_aspect_ratios_order,
                        const std::vector<float>& variances);
  PriorBoxPluginDynamic(const void* serial_data, size_t serial_length);
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
  int32_t getNbOutputs() const TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index,
      const nvinfer1::DimsExprs* inputs,
      int32_t nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT;
  nvinfer1::DataType getOutputDataType(int32_t index,
                                       const nvinfer1::DataType* input_types,
                                       int32_t nb_inputs) const TRT_NOEXCEPT;
  void ExpandAspectRatios(std::vector<float>& aspect_ratios_,  // NOLINT
                          bool flip,
                          std::vector<float>* output_aspect_ratior) {
    constexpr float epsilon = 1e-6;
    output_aspect_ratior->clear();
    output_aspect_ratior->push_back(1.0f);
    for (size_t i = 0; i < aspect_ratios_.size(); ++i) {
      float ar = aspect_ratios_[i];
      bool already_exist = false;
      for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
        if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
          already_exist = true;
          break;
        }
      }
      if (!already_exist) {
        output_aspect_ratior->push_back(ar);
        if (flip) {
          output_aspect_ratior->push_back(1.0f / ar);
        }
      }
    }
  }

 private:
  std::vector<float> aspect_ratios_;
  std::vector<int32_t> input_dimension_;
  std::vector<int32_t> image_dimension_;
  float step_w_;
  float step_h_;
  std::vector<float> min_sizes_;
  std::vector<float> max_sizes_;
  float offset_;
  bool is_clip_;
  bool is_flip_;
  bool min_max_aspect_ratios_order_;
  std::vector<float> variances_;
};

class PriorBoxPluginDynamicCreator : public PluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) TRT_NOEXCEPT;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
