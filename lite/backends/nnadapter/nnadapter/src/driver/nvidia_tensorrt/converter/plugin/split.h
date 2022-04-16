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

#include <thrust/device_vector.h>
#include <utility>
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/plugin.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class SplitPlugin : public Plugin {
 public:
  SplitPlugin() = default;

  explicit SplitPlugin(const int32_t axis, const std::vector<int>& size_splits)
      : axis_(axis), size_splits_(size_splits) {}

  SplitPlugin(const void* serial_data, size_t serial_length) {
    Deserialize(&serial_data, &serial_length, &axis_);
    Deserialize(&serial_data, &serial_length, &size_splits_);
  }

  nvinfer1::IPluginV2* clone() const noexcept {
    return new SplitPlugin(axis_, size_splits_);
  }

  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
              void* workspace,
              cudaStream_t stream) noexcept;

  const char* getPluginType() const noexcept;

  size_t getSerializationSize() const noexcept {
    return SerializedSize(axis_) + SerializedSize(size_splits_);
  }

  void serialize(void* buffer) const noexcept {
    Serialize(&buffer, axis_);
    Serialize(&buffer, size_splits_);
  };

  int32_t getNbOutputs() const noexcept { return size_splits_.size(); }

  int initialize() noexcept;

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nb_input_dims) noexcept;

 private:
  int32_t axis_;
  std::vector<int32_t> size_splits_;
  int outer_rows_;
  int inner_cols_;
  int axis_shape_;
  std::vector<int> segment_offsets_;
  thrust::device_vector<int> dev_segment_offsets_;
  thrust::device_vector<float*> dev_output_ptrs_;
};

class SplitPluginCreator : public PluginCreator {
 public:
  const char* getPluginName() const noexcept;
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length) noexcept;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
