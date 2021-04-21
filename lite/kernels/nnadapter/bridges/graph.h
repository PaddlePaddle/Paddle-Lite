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
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

class Node {
 public:
  enum class Role {
    kVar = 0,
    kConst,
    kData,
  };

  Node(NNAdapterOperand* data, Role role) : data_(data), role_(role) {}

  void set_data(NNAdapterOperand* data) { data_ = data; }
  void set_role(Role role) { role_ = role; }

  NNAdapterOperand* data() { return data_; }
  bool is_var() const { return role_ == Role::kVar; }
  bool is_const() const { return role_ == Role::kConst; }
  bool is_data() const { return role_ == Role::kData; }

 private:
  NNAdapterOperand* data_{nullptr};
  Role role_{Role::kVar};
};

class Graph {
 public:
  explicit Graph(const std::vector<std::string>& device_names) {
    for (auto& device_name : device_names) {
      NNAdapterDevice* device = nullptr;
      int result = NNAdapter::Global().NNAdapterDevice_acquire(
          device_name.c_str(), &device);
      bool found = result == NNADAPTER_NO_ERROR && device != nullptr;
      if (found) {
        const char* name = nullptr;
        NNAdapter::Global().NNAdapterDevice_getName(device, &name);
        const char* vendor = nullptr;
        NNAdapter::Global().NNAdapterDevice_getVendor(device, &vendor);
        NNAdapterDeviceType type = 0;
        NNAdapter::Global().NNAdapterDevice_getType(device, &type);
        int32_t version = 0;
        NNAdapter::Global().NNAdapterDevice_getVersion(device, &version);
        LOG(INFO) << device_name << "(" << name << ":" << vendor << ":" << type
                  << ":" << version << ")";
        devices_.push_back(device);
      }
    }
    NNAdapter::Global().NNAdapterGraph_create(&graph_);
  }

  ~Graph() {
    NNAdapter::Global().NNAdapterGraph_destroy(graph_);
    for (auto* device : devices_) {
      NNAdapter::Global().NNAdapterDevice_release(device);
    }
  }

 public:
  NNAdapterGraph* Handle() { return graph_; }

  int Add(const std::string& name, std::shared_ptr<Node> node);
  std::shared_ptr<Node> Add(const std::string& name, NNAdapterOperand* operand);

  std::shared_ptr<Node> Get(std::string name) {
    CHECK(Has(name)) << "Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 private:
  std::map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
  NNAdapterGraph* graph_{nullptr};
  std::vector<NNAdapterDevice*> devices_;
};

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
