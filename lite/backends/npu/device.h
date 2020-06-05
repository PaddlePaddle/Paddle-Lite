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
#include <vector>
#include "HiAiModelManagerService.h"  // NOLINT
#include "hiai_ir_build.h"            // NOLINT

namespace paddle {
namespace lite {
namespace npu {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() {}

  int freq_level() { return freq_level_; }
  int framework_type() { return framework_type_; }
  int model_type() { return model_type_; }
  int device_type() { return device_type_; }

  // Build the HiAI IR graph to om model, return HiAI model manager client to
  // load om model and run inference.
  std::shared_ptr<hiai::AiModelMngerClient> Build(
      const std::string model_name,             // NOLINT
      std::vector<ge::Operator>& input_nodes,   // NOLINT
      std::vector<ge::Operator>& output_nodes,  // NOLINT
      const std::string model_cache_name        // NOLINT
      );                                        // NOLINT

 private:
  int freq_level_{3};
  int framework_type_{0};
  int model_type_{0};
  int device_type_{0};
};

}  // namespace npu
}  // namespace lite
}  // namespace paddle
