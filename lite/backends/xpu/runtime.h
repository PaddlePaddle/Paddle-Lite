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

#include <xtcl/xtcl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace xpu {

class DeviceInfo {
 public:
  static DeviceInfo& Global() {
    static DeviceInfo x;
    return x;
  }
  DeviceInfo() {}

  void Insert(const std::string& name,
              std::shared_ptr<xtcl::network::xRuntimeInstance> runtime) {
    if (runtimes_.find(name) != runtimes_.end()) {
      LOG(WARNING) << "[XPU] Model " << name << " already exists.";
      return;
    }
    runtimes_.emplace(std::make_pair(name, runtime));
  }

  void Clear() { runtimes_.clear(); }

  std::shared_ptr<xtcl::network::xRuntimeInstance> Find(
      const std::string& name) const {
    if (runtimes_.find(name) != runtimes_.end()) {
      return runtimes_.at(name);
    } else {
      return nullptr;
    }
  }

 private:
  int device_id_{0};
  std::string device_name_{"default"};
  std::unordered_map<std::string,
                     std::shared_ptr<xtcl::network::xRuntimeInstance>>
      runtimes_;
};

bool LoadModel(const lite::Tensor& model,
               std::shared_ptr<xtcl::network::xRuntimeInstance>* runtime);

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
