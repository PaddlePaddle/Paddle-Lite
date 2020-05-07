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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/backends/apu/neuron_adapter.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace apu {

// Graph and node is defined to collect all of converted HiAI IR nodes
class Node {
 public:
  Node(int32_t operand_idx, std::vector<uint32_t> shape)
      : idx_(operand_idx), shape_(shape) {}

  void set_shape(std::vector<uint32_t> shape) { shape_ = shape; }

  uint32_t index() { return idx_; }
  std::vector<uint32_t> shape() const { return shape_; }
  void set_data(std::shared_ptr<Tensor> data) { data_ = data; }

 private:
  int32_t idx_;
  std::vector<uint32_t> shape_;
  std::shared_ptr<Tensor> data_{nullptr};
};

class Graph {
 public:
  int Add(const std::string& name, std::shared_ptr<Node> node);

  // Variable, const or data node
  std::shared_ptr<Node> Add(const std::string& name,
                            std::vector<uint32_t> shape) {
    CHECK(shape.size()) << name << " : " << shape.size();
    auto node = std::make_shared<Node>(operandIdx_, shape);
    auto idx = Add(name, node);
    CHECK_GE(idx, 1);

    return node;
  }

  void set_model(NeuronModel* model) { model_ = model; }
  NeuronModel* model() { return model_; }

  void set_input_names(const std::vector<std::string> input_names) {
    input_names_ = input_names;
  }

  bool IsInput(const std::string& name) {
    for (int i = 0; i < input_names_.size(); i++) {
      if (input_names_[i] == name) return true;
    }
    return false;
  }

  bool IsOutput(const std::string& name) {
    for (int i = 0; i < output_names_.size(); i++) {
      if (output_names_[i] == name) return true;
    }
    return false;
  }

  void set_output_names(const std::vector<std::string> output_names) {
    output_names_ = output_names;
  }

  std::shared_ptr<Node> Get(std::string name) {
    CHECK(Has(name)) << "[APU] Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 private:
  NeuronModel* model_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
  int32_t operandIdx_ = 0;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace apu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
