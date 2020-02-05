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

#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class RuntimeContextAssignPass : public StmtPass {
 public:
  RuntimeContextAssignPass() {}

  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt()) continue;
      auto& inst = node.AsStmt();

      // #ifdef LITE_WITH_CUDA
      bool sync = inst.need_sync_;
      int stream_id = inst.stream_id_;
      // auto& ins = node.inlinks;
      // std::vector<uint32_t> lanes;
      // for (auto& in_arg : ins) {
      //   // Weight parameter does not involve stream id, so just skip it.
      //   if (in_arg->AsArg().is_weight || in_arg->AsArg().is_persist) {
      //     continue;
      //   }
      //   if (std::find(lanes.begin(), lanes.end(), in_arg->AsArg().lane) ==
      //       lanes.end()) {
      //     lanes.push_back(in_arg->AsArg().lane);
      //   }
      // }
      // #endif
      inst.picked_kernel().SetContext(
          ContextScheduler::Global().NewContext(inst.picked_kernel().target()),
          stream_id);
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(runtime_context_assign_pass,
                  paddle::lite::mir::RuntimeContextAssignPass)
    .BindTargets({TARGET(kAny)});
