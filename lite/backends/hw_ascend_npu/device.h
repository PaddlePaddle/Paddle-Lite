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
#include <vector>
#include "ge/ge_ir_build.h"
#include "lite/backends/hw_ascend_npu/runtime.h"
#include "lite/utils/cp_logging.h"
namespace paddle {
namespace lite {
namespace hw_ascend_npu {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() : inited_(false) {}

  ~Device() { ReleaseDevice(); }

  bool is_device() const { return is_devcie_; }

  // Build the IR graph to om model, return a HWAscendNPURuntime instance to
  // load om model and run inference.
  std::shared_ptr<HWAscendNPURuntime> Build(
      std::vector<ge::Operator>& input_nodes,  // NOLINT
      std::vector<ge::Operator>& output_nodes  // NOLINT
      );                                       // NOLINT

 private:
  int InitDevice();
  void ReleaseDevice();

 private:
  bool inited_{false};
  int device_id_{0};
  bool is_devcie_{false};
  aclrtContext context_ptr_{nullptr};
  aclrtStream stream_ptr_{nullptr};
};

}  // namespace hw_ascend_npu
}  // namespace lite
}  // namespace paddle
