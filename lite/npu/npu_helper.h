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
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace npu {

class DeviceInfo {
 public:
  static DeviceInfo& Global() {
    static DeviceInfo x;
    return x;
  }
  DeviceInfo() {}
  void Insert(const std::string& name,
              std::unique_ptr<hiai::AiModelMngerClient> client) {
    if (clients_.find(name) != clients_.end()) {
      LOG(WARNING) << "[NPU] Already insert " << name;
      return;
    }
    clients_.emplace(std::make_pair(name, std::move(client)));
  }

  void Clear() { clients_.clear(); }

  hiai::AiModelMngerClient* client(const std::string& model_name) const {
    if (clients_.find(model_name) != clients_.end()) {
      return clients_.at(model_name).get();
    } else {
      return nullptr;
    }
  }
  std::vector<std::string> AllClientNames() {
    std::vector<std::string> names;
    for (auto& i : clients_) {
      names.push_back(i.first);
    }
    return names;
  }

  int freq_level() { return freq_level_; }
  int framework_type() { return framework_type_; }
  int model_type() { return model_type_; }
  int device_type() { return device_type_; }

 private:
  int freq_level_{3};
  int framework_type_{0};
  int model_type_{0};
  int device_type_{0};
  // TODO(TJ): find better place
  std::unordered_map<std::string, std::unique_ptr<hiai::AiModelMngerClient>>
      clients_;
};

class OpList {
 public:
  static OpList& Global() {
    static thread_local OpList x;
    return x;
  }
  void clear() { lists_.clear(); }
  void add(std::shared_ptr<ge::Operator> p) { lists_.push_back(p); }

 private:
  std::vector<std::shared_ptr<ge::Operator>> lists_;
};

bool SaveNPUModel(const void* om_model_data,
                  const size_t om_model_size,
                  const std::string& om_file_path);

// If build from inputs and outputs will save the npu offline model
bool BuildNPUClient(std::vector<ge::Operator>& inputs,   // NOLINT
                    std::vector<ge::Operator>& outputs,  // NOLINT
                    const std::string& name);

// If build from path will not save the npu offline model
bool BuildNPUClient(const std::string& om_model_file_path,
                    const std::string& name);

bool BuildNPUClient(const void* om_model_data,
                    const size_t om_model_size,
                    const std::string& name);

}  // namespace npu
}  // namespace lite
}  // namespace paddle
