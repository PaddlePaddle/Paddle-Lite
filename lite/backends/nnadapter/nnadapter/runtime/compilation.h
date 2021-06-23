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
#include "runtime/context.h"
#include "runtime/model.h"

namespace nnadapter {
namespace runtime {

class Compilation {
 public:
  Compilation(Model* model,
              const char* cache_key,
              void* cache_buffer,
              uint32_t cache_length,
              const char* cache_dir,
              Context* context);
  ~Compilation();
  int Finish();
  int QueryInputsAndOutputs(uint32_t* input_count,
                            NNAdapterOperandType** input_types,
                            uint32_t* output_count,
                            NNAdapterOperandType** output_types);
  int Execute(std::vector<hal::Argument>* input_arguments,
              std::vector<hal::Argument>* output_arguments);

 private:
  Model* model_{nullptr};
  hal::Cache cache_;
  Context* context_{nullptr};
  void* program_{nullptr};
  bool completed_{false};
};

}  // namespace runtime
}  // namespace nnadapter
