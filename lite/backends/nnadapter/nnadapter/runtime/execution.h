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
               const int32_t* dimensions,
               uint32_t dimension_count,
               void* buffer,
               uint32_t length);
  int SetOutput(int32_t index,
                const int32_t* dimensions,
                uint32_t dimensionCount,
                void* buffer,
                uint32_t length);
  int Compute();

 private:
  Compilation* compilation_{nullptr};
  std::vector<hal::Argument> input_arguments_;
  std::vector<hal::Argument> output_arguments_;
};

}  // namespace runtime
}  // namespace nnadapter
