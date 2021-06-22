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
#include "lite/core/subgraph_engine_base.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

class DeviceProgram {
 public:
  explicit DeviceProgram(const std::string& model_cache_key,
                         ::NNAdapterContext* context)
      : model_cache_key_(model_cache_key), context_(context) {}
  ~DeviceProgram();
  // Load the compiled device program from the buffers or files
  bool LoadFromCache(std::vector<char>* model_cache_buffer,
                     const std::string& model_cache_dir);
  // Build the model online and cache the compiled device program to the file
  // system if model_cache_dir is given
  bool BuildAndCacheToFiles(RuntimeProgram* origin_program,
                            const std::vector<std::string>& input_names,
                            const std::vector<std::string>& output_names,
                            const std::string& model_cache_dir);
  // Create an execution and set the buffer of inputs and outputs
  bool SetInputsAndOutputs(std::vector<Tensor*>* origin_itensors,
                           std::vector<Tensor*>* origin_otensors);
  bool Execute();
  bool IsValid() { return context_ && compilation_; }
  bool IsReady() { return IsValid() && execution_; }

 public:
  std::string model_cache_key_{""};
  NNAdapterModel* model_{nullptr};
  NNAdapterCompilation* compilation_{nullptr};
  NNAdapterExecution* execution_{nullptr};
  ::NNAdapterContext* context_{nullptr};
};

class SubgraphEngine : public subgraph::SubgraphEngineBase {
 public:
  SubgraphEngine(KernelContext* ctx,
                 int block_idx,
                 const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
                 Scope* exec_scope,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names);
  ~SubgraphEngine();

 protected:
  bool BuildDeviceProgram() override;
  bool LaunchDeviceProgram() override;

 private:
  std::string model_cache_dir_{""};
  std::vector<NNAdapterDevice*> devices_;
  ::NNAdapterContext* context_{nullptr};
  std::map<std::vector<std::vector<int64_t>>, std::shared_ptr<DeviceProgram>>
      device_programs_;
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
