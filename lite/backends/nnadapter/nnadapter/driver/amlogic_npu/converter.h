// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "driver/amlogic_npu/utility.h"

namespace nnadapter {
namespace amlogic_npu {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

 private:
  void* device_{nullptr};
  void* context_{nullptr};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(hal::Model* model, hal::Cache* cache);
  int Execute(uint32_t input_count,
              hal::Argument* input_arguments,
              uint32_t output_count,
              hal::Argument* output_arguments);

 private:
  void Clear();
  // Operand converters
  std::string GetTensorName(hal::Operand* operand);
  std::shared_ptr<aml::nn::Tensor> GetMappedTensor(hal::Operand* operand);
  std::shared_ptr<aml::nn::Tensor> UpdateTensorMap(
      hal::Operand* operand, std::shared_ptr<aml::nn::Tensor> tensor);
  std::shared_ptr<aml::nn::Tensor> ConvertOperand(
      hal::Operand* operand, std::vector<int32_t> dimensions = {});

  // Operation converters
  int ConvertActivation(hal::Operation* operation);
  int ConvertConcat(hal::Operation* operation);
  int ConvertConv2D(hal::Operation* operation);
  int ConvertConv2DTranspose(hal::Operation* operation);
  int ConvertElementwise(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertReshape(hal::Operation* operation);
  int ConvertTranspose(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to amlnpu tensor
  std::map<hal::Operand*, std::vector<std::shared_ptr<aml::nn::Tensor>>>
      tensors_;
  std::shared_ptr<aml::nn::Graph> graph_{nullptr};
  std::shared_ptr<aml::nn::Exection> execution_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
};

}  // namespace amlogic_npu
}  // namespace nnadapter
