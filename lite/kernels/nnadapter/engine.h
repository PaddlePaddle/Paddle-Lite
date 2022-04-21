// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/program.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

typedef struct {
  std::string name;
  Tensor* value{nullptr};
  std::vector<std::vector<int64_t>> dynamic_dimensions{};
  float quant_scale{-1.0f};
  int32_t quant_zero_point{0};
} Variable;

class Program {
 public:
  explicit Program(::NNAdapterContext* context) : context_(context) {}
  ~Program();
  // Load the compiled device program from buffer or file
  bool LoadFromCache(const std::string& model_cache_token,
                     std::vector<char>* model_cache_buffer,
                     const std::string& model_cache_dir);
  // Build the model online, cache the compiled device program to file if
  // model_cache_dir is provided
  bool BuildAndCacheToFile(const cpp::BlockDesc* block_desc,
                           Scope* exec_scope,
                           const std::vector<Variable>& input_vars,
                           std::vector<Variable>* output_vars,
                           const std::string& model_cache_token,
                           const std::string& model_cache_dir);
  // Create an execution, set the model input and output variables and the
  // functions to access them
  bool SetInputsAndOutputs(std::vector<Variable>* input_vars,
                           std::vector<Variable>* output_vars);
  int Execute();
  bool IsValid() { return context_ && compilation_; }
  bool IsReady() { return IsValid() && execution_; }

 public:
  NNAdapterModel* model_{nullptr};
  NNAdapterCompilation* compilation_{nullptr};
  NNAdapterExecution* execution_{nullptr};
  ::NNAdapterContext* context_{nullptr};
};

class Engine {
 public:
  Engine(KernelContext* ctx,
         const cpp::BlockDesc* block_desc,
         Scope* exec_scope,
         const std::vector<std::string>& input_names,
         const std::vector<std::string>& output_names,
         const std::vector<float>& input_scales,
         const std::vector<float>& output_scales);
  ~Engine();
  bool Run();

 private:
  KernelContext* ctx_{nullptr};
  const cpp::BlockDesc* block_desc_{nullptr};
  Scope* exec_scope_{nullptr};
  std::vector<Variable> input_vars_;
  std::vector<Variable> output_vars_;
  std::vector<NNAdapterDevice*> devices_;
  ::NNAdapterContext* context_{nullptr};
  std::vector<std::shared_ptr<Program>> programs_;
  std::string model_cache_dir_{""};
};

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
