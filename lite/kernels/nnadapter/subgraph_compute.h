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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/nnadapter/nnadapter_wrapper.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

class Program {
 public:
  explicit Program(::NNAdapterContext* context) : context_(context) {}
  ~Program();
  // Load the compiled device program from buffer or file
  bool LoadFromCache(const std::string& model_cache_token,
                     std::vector<char>* model_cache_buffer,
                     const std::string& model_cache_dir);
  // Build the model online, cache the compiled device program to file if
  // model_cache_dir is given
  bool BuildAndCacheToFile(
      int block_idx,
      const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
      Scope* exec_scope,
      const std::vector<std::string>& input_names,
      std::vector<std::string>* output_names,
      const std::string& model_cache_token,
      const std::string& model_cache_dir);
  // Create an execution and set the buffer of inputs and outputs
  bool SetInputsAndOutputs(const std::vector<Tensor*>& input_tensors,
                           const std::vector<Tensor*>& output_tensors);
  bool Execute();
  bool IsValid() { return context_ && compilation_; }
  bool IsReady() { return IsValid() && execution_; }

 public:
  NNAdapterModel* model_{nullptr};
  NNAdapterCompilation* compilation_{nullptr};
  NNAdapterExecution* execution_{nullptr};
  ::NNAdapterContext* context_{nullptr};
};

class SubgraphEngine {
 public:
  SubgraphEngine(KernelContext* ctx,
                 int block_idx,
                 const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
                 Scope* exec_scope,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names);
  ~SubgraphEngine();
  bool Run();

 protected:
  std::shared_ptr<Program> Build(
      const std::vector<std::vector<int64_t>>& input_dims);

 private:
  KernelContext* ctx_{nullptr};
  int block_idx_{-1};
  const std::shared_ptr<const cpp::ProgramDesc> program_desc_{nullptr};
  Scope* exec_scope_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> input_dims_;
  std::vector<Tensor*> input_tensors_;
  std::vector<Tensor*> output_tensors_;
  std::vector<NNAdapterDevice*> devices_;
  ::NNAdapterContext* context_{nullptr};
  std::map<std::vector<std::vector<int64_t>>, std::shared_ptr<Program>>
      programs_;
  std::string model_cache_dir_{""};
};

class SubgraphCompute : public KernelLite<TARGET(kNNAdapter),
                                          PRECISION(kAny),
                                          DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override;

  void Run() override;

  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine> engine_;
};

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
