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

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "lite/core/program.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

class Monitor {
 public:
  static Monitor& get_instance() {
    static Monitor s_instance;
    return s_instance;
  }

  void inferStart() {}

  void preRun(Instruction& inst) {  // NOLINT
    auto op = const_cast<OpLite*>(inst.op());
    auto op_type = op->Type();

    VLOG(4) << "Running op:" << op_type << " on "
            << op->kernel_place().DebugString();
  }

  void postRun(Instruction& inst) {  // NOLINT
    auto op = const_cast<OpLite*>(inst.op());
    auto op_info = op->op_info();
    auto in_names = op_info->input_names();

    for (auto name : in_names) {
      // auto *var = op->scope()->FindVar(name);
      // CHECK(var) << "no variable called " << name << " found";
      // auto tensor = var->Get<lite::Tensor>();
    }

    auto out_args = op_info->output_names();
    for (auto name : out_args) {
      VLOG(4) << "\n out_tensor:" << name;
      auto* var = op->scope()->FindVar(name);
      if (var->IsType<lite::Tensor>()) {
        lite::Tensor* tensor =
            const_cast<lite::Tensor*>(&var->Get<lite::Tensor>());
        if (tensor->ZynqTensor() != nullptr) {
          // tensor->ZynqTensor()->saveToFile(name, true);
        }
      }
    }
  }

  void inferEnd() {}

 private:
};

}  // namespace lite
}  // namespace paddle
