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

#include <vector>
#include "runtime/compilation.h"

namespace nnadapter {
namespace runtime {

class Execution {
 public:
  explicit Execution(Compilation* compilation) : compilation_(compilation) {}
  int SetInput(int32_t index,
               void* memory,
               void* (*access)(void* memory,
                               NNAdapterOperandType* type,
                               void* device_buffer));
  int SetOutput(int32_t index,
                void* memory,
                void* (*access)(void* memory,
                                NNAdapterOperandType* type,
                                void* device_buffer));
  int Compute();

 private:
  Compilation* compilation_{nullptr};
  std::vector<core::Argument> input_arguments_;
  std::vector<core::Argument> output_arguments_;
};

}  // namespace runtime
}  // namespace nnadapter
