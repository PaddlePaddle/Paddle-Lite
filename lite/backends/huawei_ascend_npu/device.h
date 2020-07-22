// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <mutex>  // NOLINT
#include <string>
#include <vector>
#include "lite/backends/huawei_ascend_npu/model_client.h"

namespace paddle {
namespace lite {
namespace huawei_ascend_npu {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() { InitOnce(); }

  ~Device() { DestroyOnce(); }

  std::shared_ptr<AclModelClient> LoadFromMem(
      const std::vector<char>& model_buffer, const int device_id);
  std::shared_ptr<AclModelClient> LoadFromFile(const std::string& model_path,
                                               const int device_id);
  // Build the ACL IR graph to the ACL om model
  bool Build(std::vector<ge::Operator>& input_nodes,   // NOLINT
             std::vector<ge::Operator>& output_nodes,  // NOLINT
             std::vector<char>* model_buffer);         // NOLINT

 private:
  void InitOnce();
  void DestroyOnce();
  bool runtime_inited_{false};
  static std::mutex device_mutex_;
};

}  // namespace huawei_ascend_npu
}  // namespace lite
}  // namespace paddle
