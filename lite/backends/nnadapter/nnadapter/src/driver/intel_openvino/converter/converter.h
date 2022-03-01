// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "driver/intel_openvino/utility.h"

namespace nnadapter {
namespace intel_openvino {

class Converter {
 public:
  explicit Converter(
      std::vector<std::shared_ptr<default_opset::Parameter>>* paramter_nodes,
      std::map<core::Operand*, std::vector<std::shared_ptr<OutputNode>>>*
          output_nodes)
      : parameter_nodes_(paramter_nodes), output_nodes_(output_nodes) {}

  ~Converter() {}

  // Convert a NNAdapter model to an intel openvino graph
  int Apply(core::Model* model);

  // Convert a NNAdapter operand to an intel openvino OutputNode
  std::shared_ptr<OutputNode> ConvertToOutputNode(
      core::Operand* operand, std::vector<int32_t> dimensions = {});

  std::shared_ptr<OutputNode> UpdateOutputNodeMap(
      core::Operand* operand, std::shared_ptr<OutputNode> output_node);

  std::shared_ptr<OutputNode> GetMappedOutputNode(core::Operand* operand);

  template <typename T>
  std::shared_ptr<OutputNode> AddUnsqueezeOutputNode(
      core::Operand* operand,
      std::vector<size_t> dimensions,
      std::vector<T> axes) {
    auto axes_node = AddConstOutputNode<T>(dimensions, axes);
    auto y_node = ConvertToOutputNode(operand);
    auto unsqueeze_node =
        std::make_shared<default_opset::Unsqueeze>(*y_node, *axes_node);
    return std::make_shared<OutputNode>(unsqueeze_node->output(0));
  }

 private:
  std::vector<std::shared_ptr<default_opset::Parameter>>* parameter_nodes_;
  std::map<core::Operand*, std::vector<std::shared_ptr<OutputNode>>>*
      output_nodes_;
};

}  // namespace intel_openvino
}  // namespace nnadapter
