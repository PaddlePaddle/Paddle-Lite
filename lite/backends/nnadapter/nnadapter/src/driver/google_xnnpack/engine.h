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
#include "driver/google_xnnpack/utility.h"

namespace nnadapter {
namespace google_xnnpack {

class Device {
 public:
  Device();
  ~Device();
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  bool num_threads() { return num_threads_; }
  pthreadpool_t threadpool() { return threadpool_; }
  ~Context();

 private:
  void* device_{nullptr};
  int num_threads_{0};
  pthreadpool_t threadpool_{nullptr};
};

class Program {
 public:
  explicit Program(Context* context) : context_(context) {}
  ~Program();

  int Validate(const core::Model* model, bool* supported_operations);
  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  void Clear();
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  Context* context_{nullptr};
  // Map NNAdapter operand to XNNPACK tensor value id
  std::map<core::Operand*, std::vector<uint32_t>> tensor_value_ids_;
  xnn_subgraph_t subgraph_{nullptr};
  xnn_runtime_t runtime_{nullptr};
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::vector<xnn_external_value> external_values_;
};

}  // namespace google_xnnpack
}  // namespace nnadapter
