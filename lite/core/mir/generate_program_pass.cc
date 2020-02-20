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

#include "lite/core/mir/generate_program_pass.h"
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/target_wrapper.h"
#endif

namespace paddle {
namespace lite {
namespace mir {

void GenerateProgramPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  VLOG(4) << "final program \n" << Visualize(graph.get());
#ifdef LITE_WITH_CUDA
  for (auto& item : graph->GetNodesInOrder()) {
#else
  for (auto& item : graph->StmtTopologicalOrder()) {
#endif
    if (item->IsStmt()) {
      auto& stmt = item->AsStmt();
      VLOG(4) << stmt;
#ifdef LITE_WITH_CUDA
      if (stmt.kernels().front()->target() == TargetType::kCUDA) {
        stmt.kernels()
            .front()
            ->mutable_context()
            ->As<CUDAContext>()
            .set_need_sync(stmt.need_sync_);
        stmt.kernels()
            .front()
            ->mutable_context()
            ->As<CUDAContext>()
            .set_sync_streams(stmt.sync_streams_);
      }
      insts_.emplace_back(stmt.op(), std::move(stmt.kernels().front()));
#else
      insts_.emplace_back(stmt.op(), std::move(stmt.kernels().front()));
#endif
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(generate_program_pass, paddle::lite::mir::GenerateProgramPass)
    .BindTargets({TARGET(kAny)});
