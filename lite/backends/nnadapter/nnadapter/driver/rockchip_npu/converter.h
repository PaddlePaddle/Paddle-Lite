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
#include <vector>
#include "driver/rockchip_npu/utility.h"

namespace nnadapter {
namespace rockchip_npu {

class Context {
 public:
  Context();
  ~Context();

 private:
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
  // Operand converters
  std::shared_ptr<rk::nn::Tensor> ConvertOperand(hal::Operand* operand);

  // Operation converters
  int ConvertConv2D(hal::Operation* operation);
  int ConvertFullyConnected(hal::Operation* operation);
  int ConvertPool2D(hal::Operation* operation);
  int ConvertElementwise(hal::Operation* operation);
  int ConvertSoftmax(hal::Operation* operation);
  int ConvertActivation(hal::Operation* operation);
  int ConvertReshape(hal::Operation* operation);
  int ConvertTranspose(hal::Operation* operation);
  int ConvertConcat(hal::Operation* operation);

 private:
  Context* context_{nullptr};
  // NNAdapter operand to rknn tensor
  std::map<hal::Operand*, std::shared_ptr<rk::nn::Tensor>> tensors_;
  rk::nn::Graph* graph_{nullptr};
  rk::nn::Exection* execution_{nullptr};
  std::vector<rk::nn::InputInfo> input_info_;
  std::vector<rk::nn::OutputInfo> output_info_;
  std::vector<int32_t> input_zero_points_;
  std::vector<int32_t> output_zero_points_;
};

}  // namespace rockchip_npu
}  // namespace nnadapter
