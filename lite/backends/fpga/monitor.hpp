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

  void preRun(Instruction& inst) {
    VLOG(4)  << "Running op:" << const_cast<OpLite*>(inst.op())->Type();
  }

  void postRun(Instruction& inst) {}

  void inferEnd() {}

 private:
};

}  // namespace lite
}  // namespace paddle
