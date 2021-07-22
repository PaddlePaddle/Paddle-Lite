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
#ifdef LITE_WITH_OPENCL
    using OpenCLContext = Context<TargetType::kOpenCL>;
    std::unique_ptr<KernelContext> local_ctx(new KernelContext());
    local_ctx->As<OpenCLContext>().InitOnce();
#endif
    for (auto& node : graph->mutable_nodes()) {
      if (!node.IsStmt()) continue;
      auto& inst = node.AsStmt();

#ifdef LITE_WITH_OPENCL
      if (inst.picked_kernel().target() == TARGET(kOpenCL)) {
        std::unique_ptr<KernelContext> ctx(new KernelContext());
        (*local_ctx)
            .As<OpenCLContext>()
            .CopySharedTo(&ctx->As<OpenCLContext>());
        inst.picked_kernel().SetContext(std::move(ctx));
      } else {
        inst.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
            inst.picked_kernel().target()));
      }
#elif LITE_WITH_MLU
      inst.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
          inst.picked_kernel().target(),
          static_cast<int>(reinterpret_cast<int64_t>(graph.get()))));
#else
      int stream_id = inst.stream_id_;

      inst.picked_kernel().SetContext(ContextScheduler::Global().NewContext(
          inst.picked_kernel().target(), stream_id));
#endif
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(runtime_context_assign_pass,
                  paddle::lite::mir::RuntimeContextAssignPass)
    .BindTargets({TARGET(kAny)});
