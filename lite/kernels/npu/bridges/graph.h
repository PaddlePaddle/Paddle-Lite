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
#include <utility>
#include <vector>
#include "graph/compatible/all_ops.h"
#include "graph/op/all_ops.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

// Graph and node is defined to collect all of converted HiAI IR nodes
class Node {
 public:
  enum class Role {
    kVar = 0,
    kConst,
    kData,
  };

  Node(std::shared_ptr<ge::Operator> data,
       PrecisionType precision,
       DataLayoutType layout,
       Role role)
      : data_(data), precision_(precision), layout_(layout), role_(role) {}
  Node(PrecisionType precision, DataLayoutType layout, Role role)
      : precision_(precision), layout_(layout), role_(role) {}

  void set_data(std::shared_ptr<ge::Operator> data) { data_ = data; }
  void set_precision(PrecisionType precision) { precision_ = precision; }
  void set_layout(DataLayoutType layout) { layout_ = layout; }
  void set_role(Role role) { role_ = role; }

  template <typename T>
  std::shared_ptr<T> data() {
    return std::static_pointer_cast<T>(data_);
  }
  std::shared_ptr<ge::Operator> data() { return data_; }
  PrecisionType precision() const { return precision_; }
  DataLayoutType layout() const { return layout_; }
  bool is_var() const { return role_ == Role::kVar; }
  bool is_const() const { return role_ == Role::kConst; }
  bool is_data() const { return role_ == Role::kData; }

 private:
  std::shared_ptr<ge::Operator> data_{nullptr};
  PrecisionType precision_{PRECISION(kFloat)};
  DataLayoutType layout_{DATALAYOUT(kNCHW)};
  Role role_{Role::kVar};
};

class Graph {
 public:
  int Add(const std::string& name, std::shared_ptr<Node> node);

  // Variable, const or data node
  template <typename T>
  std::shared_ptr<Node> Add(const std::string& name,
                            PrecisionType precision = PRECISION(kFloat),
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    Node::Role role = Node::Role::kVar;
    if (FastTypeId<T>() == FastTypeId<ge::op::Const>()) {
      role = Node::Role::kConst;
    } else if (FastTypeId<T>() == FastTypeId<ge::op::Data>()) {
      role = Node::Role::kData;
    }
    auto node = std::make_shared<Node>(precision, layout, role);
    auto idx = Add(name, node);
    CHECK_GE(idx, 1);
    // Generate a unique name for the created HiAI IR
    node->set_data(
        std::make_shared<T>(name + "__" + paddle::lite::to_string(idx)));
    return node;
  }

  // Const or data node
  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            std::vector<int64_t> shape,
                            DataLayoutType layout = DATALAYOUT(kNCHW));

  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    return Add(name, tensor, tensor.dims().Vectorize(), layout);
  }

  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            DDim dims,
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    return Add(name, tensor, dims.Vectorize(), layout);
  }

  // Const node
  template <typename T>
  std::shared_ptr<Node> Add(const std::string& name,
                            const std::vector<T>& data,
                            std::vector<int64_t> shape = {},
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    if (shape.empty()) {
      shape = {static_cast<int64_t>(data.size())};
    } else {
      int size = 1;
      for (auto i : shape) {
        size *= i;
      }
      CHECK_EQ(data.size(), size);
    }
    Tensor tensor;
    tensor.Resize(shape);
    tensor.set_persistable(true);
    std::memcpy(reinterpret_cast<uint8_t*>(tensor.mutable_data<T>()),
                reinterpret_cast<const uint8_t*>(data.data()),
                data.size() * sizeof(T));
    return Add(name, tensor, layout);
  }

  template <typename T>
  std::shared_ptr<Node> Add(const std::string& name,
                            const std::vector<T>& data,
                            DDim dims,
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    return Add(name, data, dims.Vectorize(), layout);
  }

  template <typename T>
  std::shared_ptr<Node> Add(const std::string& name,
                            T value,
                            std::vector<int64_t> shape = {1},
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    int64_t size = 1;
    for (auto i : shape) {
      size *= i;
    }
    std::vector<T> data(size, value);
    return Add(name, data, shape, layout);
  }

  template <typename T>
  std::shared_ptr<Node> Add(const std::string& name,
                            T value,
                            DDim dims,
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    return Add(name, value, dims.Vectorize(), layout);
  }

  // Data node
  std::shared_ptr<Node> Add(const std::string& name,
                            std::vector<int64_t> shape,
                            PrecisionType precision = PRECISION(kFloat),
                            DataLayoutType layout = DATALAYOUT(kNCHW));

  std::shared_ptr<Node> Add(const std::string& name,
                            DDim dims,
                            PrecisionType precision = PRECISION(kFloat),
                            DataLayoutType layout = DATALAYOUT(kNCHW)) {
    return Add(name, dims.Vectorize(), precision, layout);
  }

  std::shared_ptr<Node> Get(std::string name) {
    CHECK(Has(name)) << "[NPU] Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 private:
  std::map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
};

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
