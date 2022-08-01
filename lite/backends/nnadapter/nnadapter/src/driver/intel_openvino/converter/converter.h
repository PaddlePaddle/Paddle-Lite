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
      std::map<core::Operand*, std::shared_ptr<default_opset::Parameter>>*
          paramter_node_map,
      std::map<core::Operand*, std::vector<std::shared_ptr<Tensor>>>*
          tensor_map)
      : parameter_node_map_(paramter_node_map), tensor_map_(tensor_map) {}

  ~Converter() {}

  // Convert a NNAdapter model to OpenVINO graph
  int Apply(core::Model* model);

  // Convert a NNAdapter operand to OpenVINO Tensor
  std::shared_ptr<Tensor> ConvertOperand(core::Operand* operand,
                                         std::vector<int32_t> dimensions = {});

  std::shared_ptr<Tensor> UpdateTensorMap(core::Operand* operand,
                                          std::shared_ptr<Tensor> tensor);

  std::shared_ptr<Tensor> GetMappedTensor(core::Operand* operand);

  // Convert NNAdapter filter's shape layout to openvino's
  // layout for conv-like operater.
  std::shared_ptr<Operator> GetGroupConvFilterShape(
      std::shared_ptr<Tensor> filter, const int groups);

  template <typename T>
  std::shared_ptr<Operator> AddUnsqueezeOperator(
      std::shared_ptr<Tensor> input_tensor, std::vector<T> axes) {
    auto axes_tensor = AddConstantTensor<T>(axes);
    return std::make_shared<default_opset::Unsqueeze>(*input_tensor,
                                                      *axes_tensor);
  }

  template <typename T>
  std::shared_ptr<Tensor> AddConstantTensor(
      std::vector<T> values, std::vector<size_t> dimensions = {}) {
    if (dimensions.empty()) {
      dimensions = std::vector<size_t>(1, values.size());
    }
    auto constant_op = std::make_shared<default_opset::Constant>(
        GetElementType<T>(), Shape(dimensions), values);
    return std::make_shared<Tensor>(constant_op->output(0));
  }

  template <typename T>
  std::shared_ptr<Tensor> AddConstantTensor(T value) {
    auto constant_op = std::make_shared<default_opset::Constant>(
        GetElementType<T>(), Shape({}), value);
    return std::make_shared<Tensor>(constant_op->output(0));
  }

 private:
  std::map<core::Operand*, std::shared_ptr<default_opset::Parameter>>*
      parameter_node_map_;
  std::map<core::Operand*, std::vector<std::shared_ptr<Tensor>>>* tensor_map_;
};

#define MAP_OUTPUT(output_operand, operator, output_index)         \
  ({                                                               \
    converter->UpdateTensorMap(                                    \
        output_operand,                                            \
        std::make_shared<Tensor>(operator->output(output_index))); \
  })

}  // namespace intel_openvino
}  // namespace nnadapter
