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

#include "lite/kernels/bm/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/paddle_use_bridges.h"
#include "lite/kernels/bm/bridges/utility.h"
#include "bmcompiler_if.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

int SubgraphEngine::BuildDeviceProgram() {
  
  int status = 0;
  subgraph::bm::Graph graph;
  const auto& bridges = subgraph::Registry::Instance();
  graph.CreateCompilerHandle();

  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists("BM", op_type)) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select("BM", op_type)(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op),
                                             const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }

  std::string net_name = "paddle_bitmain";
  __bmcompile_opt(graph.GetCompilerHandle(), const_cast<char*>(net_name.c_str()), 2);
  finish_bmcompiler(graph.GetCompilerHandle());
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.sub_block_idx,
                                   param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kBM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::bm::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
