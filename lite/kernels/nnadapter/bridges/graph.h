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

class Node {};

class Graph {
 public:
  explicit Graph(const std::vector<std::string>& device_names) {
    for (auto& device_name : device_names) {
      NNAdapterDevice* device = nullptr;
      int result = NNAdapter::Global().NNAdapterDevice_acquire(
          device_name.c_str(), &device);
      if (device != nullptr) {
        const char* name = nullptr;
        result |= NNAdapter::Global().NNAdapterDevice_getName(device, &name);
        const char* vendor = nullptr;
        result |=
            NNAdapter::Global().NNAdapterDevice_getVendor(device, &vendor);
        NNAdapterDeviceType type = 0;
        result |= NNAdapter::Global().NNAdapterDevice_getType(device, &type);
        int32_t version = 0;
        result |=
            NNAdapter::Global().NNAdapterDevice_getVersion(device, &version);
        LOG(INFO) << "[NNAdapter] " << device_name << " result=" << result
                  << ", name=" << name << " vendor=" << vendor
                  << " type=" << type << " version=" << version;
        devices_.push_back(device);
      }
    }
    NNAdapter::Global().NNAdapterNetwork_create(&network_);
  }
  ~Graph() { NNAdapter::Global().NNAdapterNetwork_free(network_); }

 public:
  std::shared_ptr<Node> Get(std::string name) {
    CHECK(Has(name)) << "[NNAdapter] Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 private:
  std::map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
  NNAdapterNetwork* network_{nullptr};
  std::vector<NNAdapterDevice*> devices_;
};

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
