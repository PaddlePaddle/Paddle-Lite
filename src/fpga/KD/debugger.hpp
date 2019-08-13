/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdio.h>
#include <string>
#include <unordered_map>

#include "llapi/filter.h"
#include "llapi/zynqmp_api.h"
#include "tensor.hpp"

namespace paddle_mobile {
namespace zynqmp {

class Debugger {
 public:
  static Debugger& get_instance() {
    static Debugger s_instance;
    return s_instance;
  }

  void registerOutput(std::string op_type, Tensor* tensor) {
    tensor->saveToFile(op_type, true);
  }

 private:
  std::unordered_map<std::string, bool> op_config;
  Debugger() {
    op_config["concat"] = true;
    op_config["conv_add_bn"] = true;
    op_config["conv_add_bn_relu"] = true;
    op_config["conv_add"] = true;
    op_config["conv_add_relu"] = true;
    op_config["conv_bn"] = true;
    op_config["conv_bn_relu"] = true;
    op_config["crop"] = true;
  }
};
}  // namespace zynqmp
}  // namespace paddle_mobile
