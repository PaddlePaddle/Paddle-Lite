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

#include "runtime/compilation.h"
#include <string>
#include "utility/logging.h"

namespace nnadapter {
namespace runtime {

Compilation::Compilation(Model* model,
                         const char* cache_key,
                         void* cache_buffer,
                         uint32_t cache_length,
                         const char* cache_dir,
                         Context* context)
    : model_(model), program_(nullptr), context_(context), completed_(false) {
  cache_.cache_key = std::string(cache_key);
  cache_.cache_buffer = cache_buffer;
  cache_.cache_length = cache_length;
  cache_.cache_dir = std::string(cache_dir);
}

Compilation::~Compilation() {
  if (program_) {
    auto first_device = context_->GetFirstDevice();
    first_device.second->DestroyProgram(program_);
  }
}

int Compilation::Execute(std::vector<hal::Argument>* input_arguments,
                         std::vector<hal::Argument>* output_arguments) {
  // Execute generated program on target device asynchronously or synchronously
  auto first_device = context_->GetFirstDevice();
  // TODO(hong19860320) support asynchronously execution
  return first_device.second->ExecuteProgram(program_,
                                             input_arguments->size(),
                                             &((*input_arguments)[0]),
                                             output_arguments->size(),
                                             &((*output_arguments)[0]));
}

int Compilation::Finish() {
  // Start to build program from model or cache
  completed_ = true;
  auto first_device = context_->GetFirstDevice();
  // TODO(hong19860320) Support the task partition for multi-devices
  int result = first_device.second->CreateProgram(
      first_device.first, &model_->model_, &cache_, &program_);
  return result;
}

int Compilation::QueryInputsAndOutputs(uint32_t* input_count,
                                       NNAdapterOperandType** input_types,
                                       uint32_t* output_count,
                                       NNAdapterOperandType** output_types) {
  if (!input_count || !output_count) {
    return NNADAPTER_INVALID_PARAMETER;
  }
  if (model_) {
    // From model
    *input_count = static_cast<uint32_t>(model_->model_.input_operands.size());
    *output_count =
        static_cast<uint32_t>(model_->model_.output_operands.size());
    if (input_types && output_types) {
      for (uint32_t i = 0; i < *input_count; i++) {
        input_types[i] = &model_->model_.input_operands[i]->type;
      }
      for (uint32_t i = 0; i < *output_count; i++) {
        output_types[i] = &model_->model_.output_operands[i]->type;
      }
    }
  } else {
    // From cache
    *input_count = static_cast<uint32_t>(cache_.input_types.size());
    *output_count = static_cast<uint32_t>(cache_.output_types.size());
    if (input_types && output_types) {
      for (uint32_t i = 0; i < *input_count; i++) {
        input_types[i] = &cache_.input_types[i];
      }
      for (uint32_t i = 0; i < *output_count; i++) {
        output_types[i] = &cache_.output_types[i];
      }
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace runtime
}  // namespace nnadapter
