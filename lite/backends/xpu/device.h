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

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {
namespace xpu {

class Device {
 public:
  static Device& Global() {
    static Device x;
    return x;
  }
  Device() {
    char* name = std::getenv("XPU_DEVICE_NAME");
    if (name) {
      name_ = std::string(name);
    }
    // XPU_DEVICE_TARGET for XPU model building, which supports 'llvm' and 'xpu
    // -libs=xdnn'
    char* target = std::getenv("XPU_DEVICE_TARGET");
    if (target) {
      target_ = std::string(target);
    }
  }

  // Build the XPU graph to the XPU runtime, return the XPU runtime which can be
  // used to run inference.
  std::unique_ptr<xtcl::network::xRuntimeInstance> Build(
      xtcl::network::xNetworkBuilder* builder,
      xtcl::network::xTensorCompiler::ParamNDArrayMap* params,
      std::vector<xtcl::xExpr*>* outputs);

  const std::string name() const { return name_; }
  const std::string target() const { return target_; }

 private:
  std::string name_{""};
  std::string target_{""};
};

}  // namespace xpu
}  // namespace lite
}  // namespace paddle
