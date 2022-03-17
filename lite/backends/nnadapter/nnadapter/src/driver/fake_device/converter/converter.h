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
#include "driver/fake_device/utility.h"

namespace nnadapter {
namespace fake_device {

class Converter {
 public:
  explicit Converter(
      fakedevice::nn::Graph* graph,
      std::map<core::Operand*,
               std::vector<std::shared_ptr<fakedevice::nn::Tensor>>>* tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to fakedevice graph and tensors
  int Apply(core::Model* model);
  // Mapping a fakedevice tensor to a NNAdapter operand
  std::string GetTensorName(core::Operand* operand);
  std::shared_ptr<fakedevice::nn::Tensor> GetMappedTensor(
      core::Operand* operand);
  std::shared_ptr<fakedevice::nn::Tensor> UpdateTensorMap(
      core::Operand* operand, std::shared_ptr<fakedevice::nn::Tensor> tensor);
  // Convert a NNAdapter operand to a fakedevice tensor
  std::shared_ptr<fakedevice::nn::Tensor> ConvertOperand(
      core::Operand* operand, std::vector<int32_t> dimensions = {});
  // Add a fakedevice operator into fakedevice graph
  int AddOperator(
      fakedevice::nn::OperatorType type,
      std::vector<std::shared_ptr<fakedevice::nn::Tensor>> input_tensors,
      std::vector<std::shared_ptr<fakedevice::nn::Tensor>> output_tensors,
      void* attrs,
      std::string name = "");

 private:
  fakedevice::nn::Graph* graph_{nullptr};
  std::map<core::Operand*,
           std::vector<std::shared_ptr<fakedevice::nn::Tensor>>>* tensors_{
      nullptr};
};

}  // namespace fake_device
}  // namespace nnadapter
