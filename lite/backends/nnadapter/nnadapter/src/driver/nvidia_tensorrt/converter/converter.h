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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/nvidia_tensorrt/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

class Converter {
 public:
  explicit Converter(
      nvinfer1::INetworkDefinition* network,
      std::map<core::Operand*, std::vector<nvinfer1::ITensor*>>* tensors)
      : network_(network), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to a trt network
  int Apply(core::Model* model);

  nvinfer1::INetworkDefinition* network();

  nvinfer1::ITensor* GetMappedTensor(core::Operand* operand);

  void UpdateTensorMap(core::Operand* operand, nvinfer1::ITensor* tensor);

  nvinfer1::ITensor* ConvertOperand(
      core::Operand* operand, const std::vector<int32_t>& dimensions = {});

  nvinfer1::Weights OperandToWeights(core::Operand* operand);

  template <typename T>
  nvinfer1::Weights AddWeights(const T* values, size_t size);

  template <typename T>
  nvinfer1::Weights AddWeights(const std::vector<T>& values);

 private:
  nvinfer1::INetworkDefinition* network_{nullptr};
  std::map<core::Operand*, std::vector<nvinfer1::ITensor*>>* tensors_{nullptr};
  std::vector<std::vector<uint8_t>> weights_;
};

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
