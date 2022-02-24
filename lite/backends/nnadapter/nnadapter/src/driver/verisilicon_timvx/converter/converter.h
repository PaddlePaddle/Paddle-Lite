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
#include "driver/verisilicon_timvx/utility.h"

namespace nnadapter {
namespace verisilicon_timvx {

class Converter {
 public:
  explicit Converter(
      tim::vx::Graph* graph,
      std::map<core::Operand*, std::vector<std::shared_ptr<tim::vx::Tensor>>>*
          tensors)
      : graph_(graph), tensors_(tensors) {}
  ~Converter() {}

  // Convert a NNAdapter model to tim-vx graph and tensors
  int Apply(core::Model* model);
  tim::vx::Graph* graph() { return graph_; }
  // Mapping a tim-vx tensor to a NNAdapter operand
  std::shared_ptr<tim::vx::Tensor> GetMappedTensor(core::Operand* operand);
  std::shared_ptr<tim::vx::Tensor> UpdateTensorMap(
      core::Operand* operand, std::shared_ptr<tim::vx::Tensor> tensor);
  // Create and add a anonymous tim-vx tensor into the tensor pool
  std::shared_ptr<tim::vx::Tensor> AddTensor(
      const NNAdapterOperandType* type,
      void* buffer = nullptr,
      std::vector<int32_t> dimensions = {});
  // Convert a NNAdapter operand to a tim-vx tensor, and map it to the NNAdapter
  // operand
  std::shared_ptr<tim::vx::Tensor> ConvertOperand(
      core::Operand* operand, std::vector<int32_t> dimensions = {});

 private:
  tim::vx::Graph* graph_{nullptr};
  std::map<core::Operand*, std::vector<std::shared_ptr<tim::vx::Tensor>>>*
      tensors_{nullptr};
};

}  // namespace verisilicon_timvx
}  // namespace nnadapter
