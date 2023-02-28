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

#include "runtime/execution.h"
#include "utility/logging.h"

namespace nnadapter {
namespace runtime {

int Execution::SetInput(int32_t index,
                        void* memory,
                        void* (*access)(void* memory,
                                        NNAdapterOperandType* type,
                                        void* device_buffer)) {
  core::Argument* argument = nullptr;
  for (auto& input_argument : input_arguments_) {
    if (input_argument.index == index) {
      argument = &input_argument;
      break;
    }
  }
  if (!argument) {
    input_arguments_.emplace_back();
    argument = &input_arguments_.back();
    argument->index = index;
  }
  argument->memory = memory;
  argument->access = access;
  return NNADAPTER_NO_ERROR;
}

int Execution::SetOutput(int32_t index,
                         void* memory,
                         void* (*access)(void* memory,
                                         NNAdapterOperandType* type,
                                         void* device_buffer)) {
  core::Argument* argument = nullptr;
  for (auto& output_argument : output_arguments_) {
    if (output_argument.index == index) {
      argument = &output_argument;
      break;
    }
  }
  if (!argument) {
    output_arguments_.emplace_back();
    argument = &output_arguments_.back();
    argument->index = index;
  }
  argument->memory = memory;
  argument->access = access;
  return NNADAPTER_NO_ERROR;
}

int Execution::Compute() {
  // TODO(hong19860320) support asynchronously execution
  return compilation_->Execute(&input_arguments_, &output_arguments_);
}

}  // namespace runtime
}  // namespace nnadapter
