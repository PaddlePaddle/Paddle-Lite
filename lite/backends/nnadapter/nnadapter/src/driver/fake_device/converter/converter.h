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
      fake_ddk::nn::Graph* graph,
      std::map<core::Operand*,
               std::vector<std::shared_ptr<fake_ddk::nn::Tensor>>>* tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to fake_ddk graph and tensors
  int Apply(core::Model* model);
  // Mapping a fake_ddk tensor to a NNAdapter operand
  std::string GetTensorName(core::Operand* operand);
  std::shared_ptr<fake_ddk::nn::Tensor> GetMappedTensor(core::Operand* operand);
  std::shared_ptr<fake_ddk::nn::Tensor> UpdateTensorMap(
      core::Operand* operand, std::shared_ptr<fake_ddk::nn::Tensor> tensor);
  // Convert a NNAdapter operand to a fake_ddk tensor
  std::shared_ptr<fake_ddk::nn::Tensor> ConvertOperand(
      core::Operand* operand, std::vector<int32_t> dimensions = {});
  // Add a fake_ddk operator into fake_ddk graph
  int AddOperator(
      fake_ddk::nn::OperatorType type,
      std::vector<std::shared_ptr<fake_ddk::nn::Tensor>> input_tensors,
      std::vector<std::shared_ptr<fake_ddk::nn::Tensor>> output_tensors,
      void* attrs,
      std::string name = "");

 private:
  fake_ddk::nn::Graph* graph_{nullptr};
  std::map<core::Operand*, std::vector<std::shared_ptr<fake_ddk::nn::Tensor>>>*
      tensors_{nullptr};
};

}  // namespace fake_device
}  // namespace nnadapter
