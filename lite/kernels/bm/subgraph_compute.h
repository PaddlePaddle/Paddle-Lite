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

#include <bmcompiler_if.h>
#include <bmruntime_interface.h>
#include <bmruntime_legacy.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/core/subgraph/subgraph_engine_base.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

class SubgraphEngine : public subgraph::SubgraphEngineBase {
 public:
  SubgraphEngine(KernelContext *ctx,
                 int block_idx,
                 const std::shared_ptr<const cpp::ProgramDesc> &program_desc,
                 Scope *exec_scope,
                 const std::vector<std::string> &input_names,
                 const std::vector<std::string> &output_names)
      : subgraph::SubgraphEngineBase(ctx,
                                     block_idx,
                                     program_desc,
                                     exec_scope,
                                     input_names,
                                     output_names) {}

 protected:
  bool BuildDeviceProgram() override;
  bool LaunchDeviceProgram() override;
  bool InputShapeChanged() override;

 private:
  void *bmrt_hd_ = nullptr;
  std::vector<bm_tensor_t> device_inputs_;
  std::vector<bm_tensor_t> device_outputs_;
  std::map<std::string, int> outname_map_;
  const char **net_names_;
  const bm_net_info_t *net_info_;
  bm_handle_t bm_hd_;
};

class SubgraphCompute : public KernelLite<TARGET(kBM), PRECISION(kFloat)> {
 public:
  using param_t = operators::SubgraphParam;
  void PrepareForRun() override;
  void Run() override;
  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine> engine_;
};

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
