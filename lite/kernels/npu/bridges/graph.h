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
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

// Type and registers of converters for converting Paddle Ops to HiAI IR graph
class Graph {
 public:
  template <typename T>
  std::shared_ptr<T> AddNode(const std::string& name) {
    auto unique_name = [&](const std::string& key) {
      int idx = 1;
      auto it = counts_.find(key);
      if (it == counts_.end()) {
        counts_.insert(std::make_pair(key, idx));
      } else {
        idx = ++(it->second);
      }
      return key + "_" + std::to_string(idx);
    };
    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
      // Generate a new unique name as the key to bind the origin node:
      // new_name->node
      nodes_.insert(std::make_pair(unique_name(name + "_var"), it->second));
      nodes_.erase(it);
    }
    // Create a new node and bind with the name: name->new_node
    auto node = std::make_shared<T>(unique_name(name + "_op"));
    nodes_.insert(std::make_pair(name, node));
    return node;
  }

  // Const node
  std::shared_ptr<ge::op::Const> AddNode(
      const std::string& name,
      const Tensor& tensor,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  std::shared_ptr<ge::op::Const> AddNode(
      const std::string& name,
      const Tensor& tensor,
      std::vector<int64_t> shape,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  template <typename T>
  std::shared_ptr<ge::op::Const> AddNode(
      const std::string& name,
      const std::vector<T>& data,
      std::vector<int64_t> shape = {},
      DataLayoutType ltype = DATALAYOUT(kNCHW)) {
    const std::type_info& info = typeid(T);
    PrecisionType ptype = PRECISION(kFloat);
    if (info == typeid(float)) {
      ptype = PRECISION(kFloat);
    } else if (info == typeid(int8_t)) {
      ptype = PRECISION(kFloat);
    } else if (info == typeid(int32_t)) {
      ptype = PRECISION(kInt32);
    } else {
      LOG(FATAL) << "[NPU] Unknow data type " << info.name();
    }
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
    std::memcpy(reinterpret_cast<uint8_t*>(tensor.mutable_data<T>()),
                reinterpret_cast<const uint8_t*>(data.data()),
                data.size() * sizeof(T));
    return AddNode(name, tensor, ptype, ltype);
  }

  template <typename T>
  std::shared_ptr<ge::op::Const> AddNode(
      const std::string& name,
      T value,
      std::vector<int64_t> shape = {1},
      DataLayoutType ltype = DATALAYOUT(kNCHW)) {
    int64_t size = 1;
    for (auto i : shape) {
      size *= i;
    }
    std::vector<T> data(size, value);
    return AddNode(name, data, shape, ltype);
  }

  // Data node
  std::shared_ptr<ge::op::Data> AddNode(
      const std::string& name,
      std::vector<int64_t> shape,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  std::shared_ptr<ge::Operator> GetNode(std::string name) {
    CHECK(HasNode(name)) << "[NPU] Node " << name << " not found.";
    return nodes_.at(name);
  }

  bool HasNode(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<ge::Operator>> nodes_;
  std::unordered_map<std::string, int> counts_;
};

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
