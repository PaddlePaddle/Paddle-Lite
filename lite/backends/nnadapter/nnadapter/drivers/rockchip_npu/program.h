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
#include "../../nnadapter_driver.h"  // NOLINT
#include "context.h"                 // NOLINT
#include "rknpu/rknpu_pub.h"         // NOLINT

namespace nnadapter {
namespace driver {
namespace rockchip_npu {

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Build(driver::Model* model, driver::Cache* cache);
  int Execute(uint32_t input_count,
              driver::Argument* input_arguments,
              uint32_t output_count,
              driver::Argument* output_arguments);

 private:
  // Operation converters
  std::shared_ptr<rk::nn::Tensor> ConvertOperand(driver::Operand* operand);
  int ConvertConv2D(driver::Operation* operation);
  int ConvertFullyConnected(driver::Operation* operation);
  int ConvertAverageAndMaxPool2D(driver::Operation* operation);
  int ConvertElementwiseBinaryOperations(driver::Operation* operation);
  int ConvertSoftmax(driver::Operation* operation);
  int ConvertActivationUnaryOperations(driver::Operation* operation);

 private:
  Context* context_{nullptr};
  // NNAdapter operand to rknn tensor
  std::map<driver::Operand*, std::shared_ptr<rk::nn::Tensor>> tensors_;
  rk::nn::Graph* graph_{nullptr};
  rk::nn::Exection* execution_{nullptr};
  std::vector<rk::nn::InputInfo> input_info_;
  std::vector<rk::nn::OutputInfo> output_info_;
};

}  // namespace rockchip_npu
}  // namespace driver
}  // namespace nnadapter
